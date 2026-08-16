import os
import re

from groq import Groq
from typing import AsyncGenerator
import asyncio


def strip_markdown(text: str) -> str:
    # Remove bold/italic markers
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bullet points and dashes
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    # Remove numbered lists
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Remove emoji and symbol characters (keep accented letters)
    text = re.sub(r'[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D\u20E3]', '', text)
    # Remove control characters (keep tabs and newlines)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# SYSTEM_PROMPT = (
#     "You are a helpful voice assistant. Your answers are read aloud to the user by "
#     "text-to-speech, so always respond the way you would speak out loud. Use natural, "
#     "conversational sentences. Never use markdown, bullet points, lists, headings, or "
#     "code blocks. Write out numbers, units, and abbreviations in full (e.g. 'twenty-two "
#     "degrees Celsius', not '22C'). Avoid symbols like *, -, / and emojis. When you use a "
#     "tool, present the result naturally as if giving the answer verbally. Keep answers "
#     "concise but complete."
# )

SYSTEM_PROMPT = ("""You are a helpful voice assistant. Your responses will be spoken aloud by a text-to-speech engine.

Rules you must follow:
- Never use markdown, asterisks, bullet points, headers, or any formatting symbols
- Never write lists with dashes or numbers — instead say "first... second... third"
- Never use abbreviations like km/h, °C, or USD — say "kilometers per hour", "degrees Celsius", "US dollars"
- Keep responses short and conversational, like you are speaking to someone face to face
- Never say "Certainly!" or "Great question!" — just answer directly
- If giving multiple points, connect them with words like "and", "also", "plus" rather than listing them""")

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
