import io
from typing import Optional

from fastembed import TextEmbedding
from pypdf import PdfReader

MAX_PAGES = 10
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MAX_FILE_BYTES = 20 * 1024 * 1024

_embedding: Optional[TextEmbedding] = None
_doc: Optional[dict] = None


def _get_embedding() -> TextEmbedding:
    global _embedding
    if _embedding is None:
        _embedding = TextEmbedding(EMBED_MODEL)
    return _embedding


def _chunk(text: str) -> list[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_CHARS, n)
        if end < n:
            split = text.rfind("\n", start, end)
            if split > start + CHUNK_CHARS // 2:
                end = split
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def load_document(filename: str, data: bytes) -> dict:
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit")

    reader = PdfReader(io.BytesIO(data))
    pages = min(MAX_PAGES, len(reader.pages))

    parts = []
    for i in range(pages):
        text = reader.pages[i].extract_text() or ""
        parts.append(f"[Page {i + 1}]\n{text}")
    full = "\n\n".join(parts).strip()

    if not full:
        raise ValueError("No extractable text found in the first pages. The PDF may be scanned or image-based.")

    chunks = _chunk(full)
    model = _get_embedding()
    embeddings = list(model.embed(chunks))

    global _doc
    _doc = {
        "filename": filename,
        "pages": pages,
        "chunks": [
            {"text": chunk, "emb": emb}
            for chunk, emb in zip(chunks, embeddings)
        ],
    }
    return {"filename": filename, "pages": pages, "chunks": len(chunks)}


def clear_document() -> None:
    global _doc
    _doc = None


def has_document() -> bool:
    return _doc is not None


def get_doc_name() -> str:
    return _doc["filename"] if _doc else ""


def get_doc_info() -> Optional[dict]:
    if not _doc:
        return None
    return {
        "filename": _doc["filename"],
        "pages": _doc["pages"],
        "chunks": len(_doc["chunks"]),
    }


def search(query: str, top_k: int = TOP_K) -> str:
    if not _doc:
        return ""
    model = _get_embedding()
    qemb = list(model.query_embed([query]))[0]

    ranked = sorted(
        ((_cosine(qemb, c["emb"]), c) for c in _doc["chunks"]),
        key=lambda t: t[0],
        reverse=True,
    )[:top_k]

    results = []
    for score, chunk in ranked:
        if score <= 0:
            continue
        results.append(chunk["text"])
    return "\n\n".join(results)