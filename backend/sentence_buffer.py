import asyncio
from typing import List, Callable, Awaitable


class SentenceBuffer:
    def __init__(self, on_sentence: Callable[[str], Awaitable[None]], on_token: Callable[[str], Awaitable[None]] = None):
        self.buffer = ""
        self.on_sentence = on_sentence
        self.on_token = on_token
        self.flush_task = None

    async def add_token(self, token: str):
        self.buffer += token
        if self.on_token:
            await self.on_token(token)

        if any(self.buffer.rstrip().endswith(p) for p in [".", "!", "?"]):
            await self._flush()
        else:
            self._reset_flush_timer()

    async def _flush(self):
        if self.buffer.strip():
            sentence = self.buffer.strip()
            self.buffer = ""
            if self.flush_task:
                self.flush_task.cancel()
            try:
                await self.on_sentence(sentence)
            except Exception as e:
                print(f"TTS error: {e}")

    def _reset_flush_timer(self):
        if self.flush_task:
            self.flush_task.cancel()
        self.flush_task = asyncio.create_task(self._delayed_flush())

    async def _delayed_flush(self):
        await asyncio.sleep(1.5)
        await self._flush()

    async def finalize(self):
        if self.buffer.strip():
            await self._flush()
