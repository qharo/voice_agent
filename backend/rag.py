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
PDF_MEDIA_TYPE = "application/pdf"
TXT_MEDIA_TYPE = "text/plain; charset=utf-8"
SUPPORTED_EXTENSIONS = (".pdf", ".txt")

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
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _media_type_for_filename(filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        return PDF_MEDIA_TYPE
    return TXT_MEDIA_TYPE


def _extract_pdf_text(data: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(data))
    pages = min(MAX_PAGES, len(reader.pages))

    parts = []
    for i in range(pages):
        text = reader.pages[i].extract_text() or ""
        parts.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(parts).strip(), pages


def _decode_text_document(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig").strip()
    except UnicodeDecodeError as e:
        raise ValueError("Could not decode that TXT file. Please use UTF-8 text.") from e


def load_document(filename: str, data: bytes) -> dict:
    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        raise ValueError("Only PDF and TXT files are supported")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit")

    if filename.lower().endswith(".pdf"):
        full, pages = _extract_pdf_text(data)
        if not full:
            raise ValueError("No extractable text found in the first pages. The PDF may be scanned or image-based.")
    else:
        full = _decode_text_document(data)
        pages = 1
        if not full:
            raise ValueError("No text found in that TXT file.")

    chunks = _chunk(full)
    model = _get_embedding()
    embeddings = list(model.embed(chunks))

    global _doc
    _doc = {
        "filename": filename,
        "media_type": _media_type_for_filename(filename),
        "data": bytes(data),
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


def get_download() -> Optional[dict]:
    if not _doc or not _doc.get("data"):
        return None
    return {
        "filename": _doc["filename"],
        "media_type": _doc.get("media_type", PDF_MEDIA_TYPE),
        "data": _doc["data"],
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