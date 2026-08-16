import asyncio
import os
import subprocess


class TTSEngine:
    """Piper TTS that synthesizes the whole input as one streaming pass.

    Each ``stream_pcm`` call spawns a single Piper process, feeds it the
    entire text, and yields raw Int16 PCM chunks as Piper produces them
    (streaming). End-of-speech is detected when the pipe reaches EOF, which
    Piper emits once it has finished synthesizing the supplied text.
    """

    def __init__(self, voice="en_US-amy-medium", length_scale=None, sample_rate=None):
        self.voice = voice
        if length_scale is None:
            length_scale = float(os.getenv("PIPER_LENGTH_SCALE", "0.9"))
        self.length_scale = length_scale
        self.sample_rate = sample_rate or int(os.getenv("PIPER_SAMPLE_RATE", "22050"))

    def _build_cmd(self):
        return [
            "piper",
            "--model", f"/models/{self.voice}.onnx",
            "--output-raw",
            "--length-scale", str(self.length_scale),
        ]

    async def stream_pcm(self, text: str):
        """Yield raw Int16 mono PCM chunks for ``text`` as Piper streams them."""
        loop = asyncio.get_event_loop()

        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.Popen(
                self._build_cmd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ),
        )

        def _write():
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()

        await loop.run_in_executor(None, _write)

        try:
            while True:
                chunk = await loop.run_in_executor(None, proc.stdout.read, 8192)
                if not chunk:
                    break
                yield chunk
        finally:
            def _cleanup():
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.wait()
                except Exception:
                    pass

            await loop.run_in_executor(None, _cleanup)
