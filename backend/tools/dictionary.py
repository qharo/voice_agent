import httpx

schema = {
    "type": "function",
    "function": {
        "name": "dictionary",
        "description": "Define an English word. Use when the user asks what a word means, or for its meaning, definition, or part of speech.",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "The word to define (e.g. 'serendipity', 'ubiquitous')",
                }
            },
            "required": ["word"],
        },
    },
}

API_URL = "https://api.datamuse.com/words"

POS = {
    "n": "noun",
    "v": "verb",
    "adj": "adjective",
    "a": "adjective",
    "adv": "adverb",
    "r": "adverb",
    "prep": "preposition",
    "conj": "conjunction",
    "pron": "pronoun",
    "interj": "interjection",
    "u": "word",
}

MAX_DEFS = 2


async def execute(word: str) -> str:
    target = word.strip().lower()

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            API_URL,
            params={"sp": target, "md": "d", "max": 1},
        )

    resp.raise_for_status()
    entries = resp.json()

    if not entries:
        return f"No definition found for {target}."

    entry = entries[0]
    definitions = entry.get("defs", [])
    if not definitions:
        return f"No definition found for {target}."

    parts = []
    for defi in definitions[:MAX_DEFS]:
        if not defi:
            continue
        if "\t" in defi:
            pos_code, definition = defi.split("\t", 1)
        else:
            pos_code, definition = "", defi
        pos = POS.get(pos_code, "word")
        article = "an" if pos[0].lower() in "aeiou" else "a"
        definition = definition.rstrip(".").rstrip()
        parts.append(f"used as {article} {pos}, meaning {definition}")

    if not parts:
        return f"No definition found for {target}."

    return f"{target} is defined as {'; and also, '.join(parts)}."
