import os
from groq import Groq
from typing import AsyncGenerator
import asyncio

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your answers are read aloud to the user by "
    "text-to-speech, so always respond the way you would speak out loud. Use natural, "
    "conversational sentences. Never use markdown, bullet points, lists, headings, or "
    "code blocks. Write out numbers, units, and abbreviations in full (e.g. 'twenty-two "
    "degrees Celsius', not '22C'). Avoid symbols like *, -, / and emojis. When you use a "
    "tool, present the result naturally as if giving the answer verbally. Keep answers "
    "concise but complete."
)


def _build_messages(prompt: str, history: list[dict] | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


class LLMClient:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    async def get_response(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ):
        kwargs = {
            "model": "qwen/qwen3.6-27b",
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(**kwargs),
        )

    async def stream_response(self, prompt: str, history: list[dict] | None = None) -> AsyncGenerator[str, None]:
        messages = _build_messages(prompt, history)

        loop = asyncio.get_event_loop()
        stream = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                stream=True,
                max_tokens=512,
            )
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
