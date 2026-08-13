import asyncio
import subprocess
import tempfile
import os


class TTSEngine:
    def __init__(self):
        self.voice = "en_US-amy-medium"

    async def synthesize(self, sentence: str) -> bytes:
        def _run():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                subprocess.run(
                    [
                        "piper",
                        "--model", f"/models/{self.voice}.onnx",
                        "--output_file", tmp.name,
                    ],
                    input=sentence.encode(),
                    check=True,
                    capture_output=True,
                )
                with open(tmp.name, "rb") as f:
                    wav_data = f.read()
                os.unlink(tmp.name)
                return wav_data

        return await asyncio.get_event_loop().run_in_executor(None, _run)
