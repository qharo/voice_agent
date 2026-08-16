import asyncio
import re

import requests

schema = {
    "type": "function",
    "function": {
        "name": "wikipedia",
        "description": "Look up a factual topic on Wikipedia and return a short summary. Use this for factual questions, historical events, people, places, and any topic where the answer should be accurate.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic to look up (e.g. 'Eiffel Tower', 'Albert Einstein', 'photosynthesis')",
                }
            },
            "required": ["query"],
        },
    },
}

API_URL = "https://en.wikipedia.org/w/api.php"

MAX_CHARS = 800

HEADERS = {
    "User-Agent": "voice-agent/1.0 (conversational voice assistant; for text-to-speech summarization)"
}


def _clean(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", text)  # bracketed IPA
    text = re.sub(r"/\s*[^/\s][^/]*/", "", text)  # slash-delimited IPA
    text = re.sub(r"\(\s*[^)]*;[^)]*\)", "", text)  # gloss parentheticals
    return " ".join(text.split())


async def execute(query: str) -> str:
    return await asyncio.to_thread(_run, query)


def _run(query: str) -> str:
    params = {
        "action": "opensearch",
        "search": query,
        "limit": 1,
        "format": "json",
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if not data or len(data) < 2 or not data[1]:
        return f"Could not find anything about {query} on Wikipedia."

    title = data[1][0]

    extract_params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "redirects": 1,
        "titles": title,
        "format": "json",
    }
    resp2 = requests.get(API_URL, params=extract_params, headers=HEADERS, timeout=15)
    resp2.raise_for_status()
    pages = resp2.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    extract = _clean(page.get("extract", ""))

    if not extract:
        return f"Could not find a summary for {query}."

    if len(extract) > MAX_CHARS:
        extract = extract[:MAX_CHARS].rstrip() + "..."

    return f"Wikipedia says about {title}: {extract}"
