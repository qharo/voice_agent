import json
from pathlib import Path

PRESETS_DIR = Path(__file__).parent

_catalog: dict | None = None


def _load_catalog() -> dict:
    global _catalog
    if _catalog is None:
        with open(PRESETS_DIR / "presets.json", "r", encoding="utf-8") as f:
            _catalog = json.load(f)
    return _catalog


def all_presets() -> dict:
    return _load_catalog()


def get_preset(slug: str) -> dict | None:
    return _load_catalog().get(slug)


def read_document(slug: str) -> bytes:
    preset = get_preset(slug)
    if not preset:
        raise KeyError(slug)
    return (PRESETS_DIR / preset["document"]).read_bytes()