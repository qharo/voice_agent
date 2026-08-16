import asyncio
import logging

import numpy as np

from pocket_infer import PocketInferenceEngine, SAMPLE_RATE, FRAME_SAMPLES

logger = logging.getLogger(__name__)

_DONE = object()


class PocketTTSStreamEngine:
    """Async adapter over PocketInferenceEngine.

    Pulls each 80 ms frame (1920 Float32 samples) from the synchronous ONNX
    loop on demand via the executor, and streams it back to the caller as
    Int16 PCM bytes. Generation pauses while the consumer awaits, giving
    natural backpressure so playback can never outrun slow network sends.
    """

    sample_rate = SAMPLE_RATE
    num_channels = 1

    def __init__(self, model_dir: str = "/models/pocket-tts", threads: int = 4):
        self.engine = PocketInferenceEngine(model_dir, threads=threads)

    @staticmethod
    def _pull_frame(gen, done):
        try:
            return next(gen)
        except StopIteration:
            return done

    async def stream_int16(self, text: str):
        gen = self.engine.generate_frames(text)
        while True:
            frame = await asyncio.to_thread(self._pull_frame, gen, _DONE)
            if frame is _DONE:
                break
            pcm = (np.clip(frame, -1.0, 1.0).astype(np.float32) * 32767.0).astype(
                np.int16
            ).tobytes()
            yield pcm

