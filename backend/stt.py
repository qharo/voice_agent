import asyncio
import numpy as np
import transcribe_cpp

MODEL_PATH = "/models/moonshine-streaming-tiny-Q8_0.gguf"

_model = transcribe_cpp.Model(MODEL_PATH)


class TranscribeSession:
    def __init__(self):
        self.session = None
        self.stream = None
        self._text = ""
        self._stopped = asyncio.Event()

    def start(self):
        self.session = _model.session()
        self.stream = self.session.stream()

    def feed_audio(self, audio_bytes: bytes):
        audio_int16 = np.frombuffer(audio_bytes, dtype="<i2")
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        self.stream.feed(audio_float32)

    async def read_events(self):
        last_text = ""
        while not self._stopped.is_set():
            await asyncio.sleep(0.2)
            if self.stream:
                t = self.stream.text()
                current = (t.full or (t.committed + " " + t.tentative)).strip()
                if current and current != last_text:
                    last_text = current
                    yield "text_changed", current

    def stop(self):
        try:
            if self.stream:
                self.stream.finalize()
                t = self.stream.text()
                self._text = (t.full or (t.committed + " " + t.tentative)).strip()
                self.stream.reset()
                self.stream = None
                self.session.close()
                self.session = None
        finally:
            self._stopped.set()

    def get_final_text(self) -> str:
        return self._text


class STT:
    def create_session(self) -> TranscribeSession:
        return TranscribeSession()
