import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import Document, Settings

DEFAULT_OCR_LANGS = "ita+eng"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_VERSION = 1


def configure_embedding(model_name: Optional[str] = None) -> Dict[str, str]:
    if not model_name:
        model_name = os.getenv("DOC_INDEX_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except Exception as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "Missing HuggingFace embeddings. Install 'llama-index-embeddings-huggingface' "
            "and 'sentence-transformers', or set DOC_INDEX_EMBED_MODEL to a valid model."
        ) from exc
    Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
    Settings.llm = None
    return {"type": "huggingface", "model": model_name}


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_binary_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(2048)
    except Exception:
        return False
    if not chunk:
        return False
    if b"\0" in chunk:
        return True
    nontext = 0
    for byte in chunk:
        if byte in (9, 10, 13) or 32 <= byte <= 126:
            continue
        nontext += 1
    return (nontext / len(chunk)) > 0.3


def save_docs_cache(
    path: Path, docs: List[Document], file_path: str, file_type: str
) -> None:
    payload: Dict[str, Any] = {
        "version": CACHE_VERSION,
        "file_path": file_path,
        "file_type": file_type,
        "docs": [{"text": doc.text, "metadata": doc.metadata} for doc in docs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))


def load_docs_cache(path: Path) -> List[Document]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    docs = []
    for item in payload.get("docs", []):
        docs.append(Document(text=item.get("text", ""), metadata=item.get("metadata") or {}))
    return docs
