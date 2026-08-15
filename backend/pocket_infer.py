import json
import math
import random
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


class PocketTokenizer:
    """SentencePiece-style BPE tokenizer built from an ONNX exporter's
    vocab.json + token_scores.json. Adapted from soniqo/speech-core
    (PocketTtsTokenizer) and sherpa-onnx's SentencePieceTokenizer."""

    def __init__(self, vocab_path, scores_path):
        vocab = _load_json(vocab_path)
        scores = _load_json(scores_path)
        if len(vocab) != len(scores):
            raise ValueError("vocab.json and token_scores.json sizes differ")

        size = len(vocab)
        self.id_to_token = [""] * size
        self.token_to_id = {}
        self.token_to_score = {}
        for token, tid in vocab.items():
            tid = int(tid)
            self.token_to_id[token] = tid
            self.id_to_token[tid] = token
        for token, score in scores.items():
            self.token_to_score[token] = float(score)

        self.trie = [{"next": {}, "id": -1, "score": 0.0}]
        for token, tid in self.token_to_id.items():
            node = 0
            for byte in token.encode("utf-8"):
                nxt = self.trie[node]["next"]
                if byte not in nxt:
                    nxt[byte] = len(self.trie)
                    self.trie.append({"next": {}, "id": -1, "score": 0.0})
                node = nxt[byte]
            self.trie[node]["id"] = tid
            self.trie[node]["score"] = self.token_to_score[token]

        self.byte_id = [-1] * 256
        self.byte_score = [float("-inf")] * 256
        for byte in range(256):
            name = "<0x%02X>" % byte
            if name in self.token_to_id:
                self.byte_id[byte] = self.token_to_id[name]
                self.byte_score[byte] = self.token_to_score[name]

    def encode_ids(self, text: str):
        text = text.replace(" ", "\u2581")
        if not text.startswith("\u2581"):
            text = "\u2581" + text
        data = text.encode("utf-8")
        length = len(data)
        neg_inf = float("-inf")
        best = [neg_inf] * (length + 1)
        best[length] = 0.0
        back = [-1] * (length + 1)
        back_id = [-1] * (length + 1)

        for start in range(length - 1, -1, -1):
            node = 0
            for end in range(start, length):
                nxt = self.trie[node]["next"]
                nb = data[end]
                if nb not in nxt:
                    break
                node = nxt[nb]
                cand = self.trie[node]
                if cand["id"] >= 0:
                    score = cand["score"] + best[end + 1]
                    if score > best[start]:
                        best[start] = score
                        back[start] = end + 1
                        back_id[start] = cand["id"]
            if back[start] < 0:
                b = data[start]
                if self.byte_id[b] >= 0:
                    best[start] = self.byte_score[b] + best[start + 1]
                    back_id[start] = self.byte_id[b]
                back[start] = start + 1

        ids = []
        offset = 0
        while offset < length:
            ids.append(back_id[offset])
            offset = back[offset]
        return ids


def _zero_state(io_spec):
    shape = io_spec.shape
    shape = [1 if (d is None or d < 0) else int(d) for d in shape]
    dtype = io_spec.type
    if "bool" in dtype:
        return np.zeros(shape, dtype=np.bool_)
    if "int64" in dtype:
        return np.zeros(shape, dtype=np.int64)
    return np.zeros(shape, dtype=np.float32)


class PocketInferenceEngine:
    """Low-level synchronous ONNX inference loop for the soniqo
    Pocket-TTS-100M-ONNX-INT8 bundle (fixed Alba voice)."""

    def __init__(self, model_dir, threads: int = 2):
        model_dir = Path(model_dir)

        def session(name):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            return ort.InferenceSession(
                str(model_dir / name),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

        self.tokenizer = PocketTokenizer(
            model_dir / "vocab.json", model_dir / "token_scores.json"
        )
        self.conditioner = session("text_conditioner.onnx")
        self.encoder = session("encoder.onnx")
        self.lm = session("lm_main.int8.onnx")
        self.flow = session("lm_flow.int8.onnx")
        self.decoder = session("decoder.int8.onnx")

        self.flow_steps = 4
        self.max_frames = 500
        self.frames_after_eos = 3
        self.eos_threshold = -4.0
        self.temperature = 0.7

        self.lm_inputs = [i.name for i in self.lm.get_inputs()]
        self.dec_inputs = [i.name for i in self.decoder.get_inputs()]
        self.lm_init = []
        for i in range(2, len(self.lm.get_inputs())):
            self.lm_init.append(_zero_state(self.lm.get_inputs()[i]))
        self.dec_init = []
        for i in range(1, len(self.decoder.get_inputs())):
            self.dec_init.append(_zero_state(self.decoder.get_inputs()[i]))

        self.cache_len = self.lm.get_inputs()[2].shape[2]

        zero = np.zeros((1, 1, 1), dtype=np.float32)
        self.voice = self.encoder.run(None, {"audio": zero})[0]
        self.voice_tokens = int(self.voice.shape[1])

    def _run_lm(self, seq, embeddings, states):
        feed = {self.lm_inputs[0]: seq, self.lm_inputs[1]: embeddings}
        for name, state in zip(self.lm_inputs[2:], states):
            feed[name] = state
        outputs = self.lm.run(None, feed)
        states[:] = outputs[2:]
        return outputs[0], outputs[1]

    def _run_flow(self, conditioning, noise2d):
        latent = noise2d
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            s = step / self.flow_steps
            t = s + dt
            out = self.flow.run(
                None,
                {
                    "c": conditioning,
                    "s": np.array([[s]], dtype=np.float32),
                    "t": np.array([[t]], dtype=np.float32),
                    "x": latent,
                },
            )[0]
            latent = latent + out * dt
        return latent

    def _run_decoder(self, latent, states):
        feed = {self.dec_inputs[0]: latent}
        for name, state in zip(self.dec_inputs[1:], states):
            feed[name] = state
        outputs = self.decoder.run(None, feed)
        states[:] = outputs[1:]
        return outputs[0][0][0]

    def generate_frames(self, text: str, seed: int = -1):
        """Yield float32 numpy arrays of FRAME_SAMPLES (1920) samples each.

        Long text is split into sentence-sized chunks. Each chunk runs its own
        autoregressive pass with a fresh LM state and voice/text warm-up (this
        prevents the degeneration small autoregressive TTS models suffer from
        on very long single passes), while the Mimi decoder state is carried
        across chunks so the audio stays continuous.
        """

        chunks = self._split_text(text)
        if not chunks:
            return

        if seed < 0:
            seed = random.randrange(1 << 31)
        rng = random.Random(seed)
        stddev = math.sqrt(self.temperature)

        dec_states = [s.copy() for s in self.dec_init]
        for chunk in chunks:
            yield from self._generate_chunk_frames(chunk, dec_states, rng, stddev)

    def _generate_chunk_frames(self, text, dec_states, rng, stddev):
        ids = self.tokenizer.encode_ids(text)
        if not ids:
            return

        remaining = self.cache_len - self.voice_tokens - len(ids)
        if remaining <= 0:
            raise ValueError(
                "Pocket TTS text conditioning exceeds the LM cache; "
                "the chunk could not be split further."
            )
        frame_limit = min(self.max_frames, remaining)

        text_emb = self.conditioner.run(
            None, {"token_ids": np.array([ids], dtype=np.int64)}
        )[0]

        lm_states = [s.copy() for s in self.lm_init]
        empty_voice_seq = np.zeros((1, 0, 32), dtype=np.float32)
        empty_text = np.zeros((1, 0, 1024), dtype=np.float32)

        self._run_lm(empty_voice_seq.copy(), self.voice, lm_states)
        self._run_lm(empty_voice_seq.copy(), text_emb, lm_states)

        current = np.full((1, 1, 32), np.nan, dtype=np.float32)
        eos_frame = -1

        for frame in range(frame_limit):
            conditioning, eos_logit = self._run_lm(current, empty_text, lm_states)
            eos_score = float(eos_logit[0, 0])
            if eos_frame < 0 and eos_score > self.eos_threshold:
                eos_frame = frame
            if eos_frame >= 0 and frame >= eos_frame + self.frames_after_eos:
                break

            noise = np.array(
                [rng.gauss(0.0, stddev) for _ in range(32)], dtype=np.float32
            )
            current = self._run_flow(conditioning, noise.reshape(1, 32))[None]
            audio = self._run_decoder(current, dec_states)

            if audio.shape[0] != FRAME_SAMPLES:
                raise RuntimeError(
                    f"Pocket TTS decoder returned {audio.shape[0]} samples, "
                    f"expected {FRAME_SAMPLES}"
                )
            yield audio

    def _split_text(self, text):
        """Split text into chunks for separate autoregressive passes.

        Paragraphs (blank-line separated) are the primary tonal unit: each
        multi-sentence paragraph becomes its own chunk so the LM is
        re-conditioned only at paragraph boundaries, keeping prosody steady.
        Consecutive single-sentence ("short") paragraphs are merged into the
        buffer for extra consistency. Oversized paragraphs or heavily merged
        runs are capped at ``MAX_SENTENCES_PER_CHUNK``.
        """
        max_sentences = 6
        max_chars = 900
        text = text.replace("\r", "").strip()
        if not text:
            return []

        paragraphs = []
        for block in re.split(r"\n\s*\n", text):
            para = block.replace("\n", " ").strip()
            if not para:
                continue
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", para)
                if s.strip()
            ]
            if sentences:
                paragraphs.append(sentences)

        chunks = []
        buffer = []
        buffer_chars = 0

        def flush_buffer():
            nonlocal buffer, buffer_chars
            if buffer:
                chunks.append(" ".join(buffer))
                buffer = []
                buffer_chars = 0

        def emit_group(sentences):
            # split a sentence group into <= max_sentences chunks
            group = []
            gchars = 0
            for s in sentences:
                if len(s) > max_chars:
                    if group:
                        chunks.append(" ".join(group))
                        group, gchars = [], 0
                    for i in range(0, len(s), max_chars):
                        chunks.append(s[i : i + max_chars])
                    continue
                if (
                    group
                    and (len(group) >= max_sentences or gchars + len(s) > max_chars)
                ):
                    chunks.append(" ".join(group))
                    group, gchars = [], 0
                group.append(s)
                gchars += len(s)
            if group:
                chunks.append(" ".join(group))

        for sentences in paragraphs:
            if len(sentences) > 1:
                # genuine tonal unit: start its own chunk(s)
                flush_buffer()
                emit_group(sentences)
            else:
                # single-sentence paragraph: merge into buffer if it fits
                s = sentences[0]
                if (
                    buffer
                    and (len(buffer) >= max_sentences or buffer_chars + len(s) > max_chars)
                ):
                    flush_buffer()
                if len(s) > max_chars:
                    flush_buffer()
                    for i in range(0, len(s), max_chars):
                        chunks.append(s[i : i + max_chars])
                else:
                    buffer.append(s)
                    buffer_chars += len(s)

        flush_buffer()
        return chunks
