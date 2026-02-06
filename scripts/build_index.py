import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from llama_index.core import Document, StorageContext, VectorStoreIndex

from doc_index_common import (
    DEFAULT_OCR_LANGS,
    configure_embedding,
    compute_sha256,
    is_binary_file,
    load_docs_cache,
    save_docs_cache,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PDF_EXTS = {".pdf"}
XLSX_EXTS = {".xlsx"}
IGNORED_DIRS = {
    ".git",
    ".codex",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
}
IGNORED_EXTS = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".dat",
    ".class",
    ".jar",
    ".pyc",
    ".o",
    ".a",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".sqlite",
    ".db",
}
MAX_LINES_PER_DOC = 200
PDF_OCR_DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or update the Doc Master index.")
    parser.add_argument("--root", default=os.getcwd(), help="Workspace root (default: cwd).")
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Index directory (default: <root>/.codex/doc-master).",
    )
    parser.add_argument(
        "--languages",
        default=None,
        help="OCR languages (default: ita+eng).",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and folders.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild index even if no changes are detected.",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="HuggingFace embedding model id.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def storage_ready(storage_dir: Path) -> bool:
    required = (
        "docstore.json",
        "index_store.json",
        "default__vector_store.json",
    )
    return all((storage_dir / name).exists() for name in required)


def should_skip_dir(path: Path, index_dir: Path, include_hidden: bool) -> bool:
    name = path.name
    if name in IGNORED_DIRS:
        return True
    if not include_hidden and name.startswith("."):
        return True
    if is_under(path, index_dir):
        return True
    return False


def should_skip_file(path: Path, include_hidden: bool) -> bool:
    name = path.name
    if not include_hidden and name.startswith("."):
        return True
    if path.suffix.lower() in IGNORED_EXTS:
        return True
    return False


def collect_files(root: Path, index_dir: Path, include_hidden: bool) -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        dirnames[:] = [
            name
            for name in dirnames
            if not should_skip_dir(dir_path / name, index_dir, include_hidden)
        ]
        for name in filenames:
            path = dir_path / name
            if should_skip_file(path, include_hidden):
                continue
            files.append(path)
    files.sort()
    return files


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_docs(text: str, rel_path: str, file_type: str) -> List[Document]:
    docs: List[Document] = []
    lines = text.splitlines()
    if not lines:
        return docs
    for start in range(0, len(lines), MAX_LINES_PER_DOC):
        chunk_lines = lines[start : start + MAX_LINES_PER_DOC]
        chunk_text = "\n".join(chunk_lines).strip()
        if not chunk_text:
            continue
        line_range = f"{start + 1}-{start + len(chunk_lines)}"
        metadata = {
            "file_path": rel_path,
            "file_type": file_type,
            "line_range": line_range,
        }
        docs.append(Document(text=chunk_text, metadata=metadata))
    return docs


def ensure_ocr_ready() -> Tuple[object, object]:
    try:
        import pytesseract
    except Exception as exc:
        raise RuntimeError("Missing pytesseract. Install it to enable OCR.") from exc
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Missing Pillow. Install it to enable OCR.") from exc
    return pytesseract, Image


def ocr_image(image, languages: str) -> str:
    pytesseract, _ = ensure_ocr_ready()
    return pytesseract.image_to_string(image, lang=languages)


def extract_pdf_docs(path: Path, rel_path: str, languages: str) -> List[Document]:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("Missing PyMuPDF (pymupdf). Install it to read PDFs.") from exc
    docs: List[Document] = []
    with fitz.open(path) as pdf_doc:
        for index in range(pdf_doc.page_count):
            page = pdf_doc.load_page(index)
            text = page.get_text("text") or ""
            method = "text"
            if len(text.strip()) < 20:
                _, Image = ensure_ocr_ready()
                pix = page.get_pixmap(dpi=PDF_OCR_DPI)
                mode = "RGB" if pix.alpha == 0 else "RGBA"
                image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                text = ocr_image(image, languages)
                method = "ocr"
            if not text.strip():
                continue
            metadata = {
                "file_path": rel_path,
                "file_type": "pdf",
                "page_number": index + 1,
                "extract_method": method,
            }
            docs.append(Document(text=text, metadata=metadata))
    return docs


def extract_image_docs(path: Path, rel_path: str, languages: str) -> List[Document]:
    _, Image = ensure_ocr_ready()
    with Image.open(path) as image:
        text = ocr_image(image, languages)
    if not text.strip():
        return []
    metadata = {"file_path": rel_path, "file_type": "image"}
    return [Document(text=text, metadata=metadata)]


def extract_xlsx_docs(path: Path, rel_path: str) -> List[Document]:
    try:
        import openpyxl
    except Exception as exc:
        raise RuntimeError("Missing openpyxl. Install it to read XLSX files.") from exc
    from openpyxl.utils.cell import get_column_letter, range_boundaries

    docs: List[Document] = []
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    for sheet in workbook.worksheets:
        dimension = sheet.calculate_dimension()
        if dimension == "A1" and sheet["A1"].value is None:
            continue
        min_col, min_row, max_col, max_row = range_boundaries(dimension)
        col_start = get_column_letter(min_col)
        col_end = get_column_letter(max_col)
        row = min_row
        while row <= max_row:
            row_end = min(row + 49, max_row)
            lines = []
            for values in sheet.iter_rows(
                min_row=row,
                max_row=row_end,
                min_col=min_col,
                max_col=max_col,
                values_only=True,
            ):
                line = "\t".join("" if value is None else str(value) for value in values)
                if line.strip():
                    lines.append(line)
            text = "\n".join(lines).strip()
            if text:
                cell_range = f"{col_start}{row}:{col_end}{row_end}"
                metadata = {
                    "file_path": rel_path,
                    "file_type": "xlsx",
                    "sheet": sheet.title,
                    "cell_range": cell_range,
                }
                docs.append(Document(text=text, metadata=metadata))
            row = row_end + 1
    workbook.close()
    return docs


def extract_docs_for_file(path: Path, rel_path: str, languages: str) -> Tuple[List[Document], str]:
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        return extract_pdf_docs(path, rel_path, languages), "pdf"
    if ext in IMAGE_EXTS:
        return extract_image_docs(path, rel_path, languages), "image"
    if ext in XLSX_EXTS:
        return extract_xlsx_docs(path, rel_path), "xlsx"
    if is_binary_file(path):
        return [], "binary"
    text = read_text_file(path)
    return extract_text_docs(text, rel_path, "text"), "text"


def load_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"version": 1, "files": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(path: Path, manifest: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2)


def write_index_summary(
    path: Path,
    root: Path,
    index_dir: Path,
    languages: str,
    stats: Dict[str, int],
    doc_count: int,
    last_log: str,
    embed_info: Dict[str, str],
) -> None:
    lines = [
        "# Doc Master",
        "",
        f"Root: {root}",
        f"Index dir: {index_dir}",
        f"Last run: {now_iso()}",
        f"OCR languages: {languages}",
        f"Embedding: {embed_info.get('type')}:{embed_info.get('model')}",
        "",
        "Counts:",
        f"- total files: {stats.get('total_files', 0)}",
        f"- processed: {stats.get('processed', 0)}",
        f"- skipped: {stats.get('skipped', 0)}",
        f"- errors: {stats.get('errors', 0)}",
        f"- removed: {stats.get('removed', 0)}",
        f"- documents: {doc_count}",
        "",
        "Locations:",
        f"- storage: {index_dir / 'storage'}",
        f"- cache: {index_dir / 'cache'}",
        f"- logs: {index_dir / 'logs'}",
        "",
        f"Last log: {last_log}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    index_dir = Path(args.index_dir).resolve() if args.index_dir else root / ".codex" / "doc-master"
    index_dir = index_dir.resolve()

    if not is_under(index_dir, root):
        raise SystemExit("Index directory must be inside the workspace root.")

    languages = args.languages or os.getenv("DOC_INDEX_OCR_LANGS", DEFAULT_OCR_LANGS)

    index_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = index_dir / "cache"
    log_dir = index_dir / "logs"
    storage_dir = index_dir / "storage"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)

    embed_info = configure_embedding(args.embed_model)

    manifest_path = index_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    previous_files: Dict[str, Dict[str, object]] = manifest.get("files", {})  # type: ignore[assignment]

    log_entries: List[Dict[str, object]] = []
    stats = {"total_files": 0, "processed": 0, "skipped": 0, "errors": 0, "removed": 0}
    documents: List[Document] = []
    new_files: Dict[str, Dict[str, object]] = {}
    reprocessed = 0

    files = collect_files(root, index_dir, args.include_hidden)

    for path in files:
        rel_path = path.relative_to(root).as_posix()
        stats["total_files"] += 1
        prev = previous_files.get(rel_path)
        stat = path.stat()
        prev_cache_path = None
        if prev:
            prev_cache_rel = prev.get("cache_path")
            if prev_cache_rel:
                candidate = index_dir / str(prev_cache_rel)
                if candidate.is_file():
                    prev_cache_path = candidate
        if (
            prev
            and prev.get("mtime") == stat.st_mtime
            and prev.get("size") == stat.st_size
            and prev_cache_path
        ):
            cache_docs = load_docs_cache(prev_cache_path)
            documents.extend(cache_docs)
            log_entries.append(
                {"path": rel_path, "status": "skipped", "reason": "unchanged"}
            )
            stats["skipped"] += 1
            updated = dict(prev)
            updated["doc_count"] = len(cache_docs)
            new_files[rel_path] = updated
            continue

        file_hash = compute_sha256(path)
        new_cache_path = cache_dir / f"{file_hash}.json"
        if prev and prev.get("hash") == file_hash and new_cache_path.exists():
            cache_docs = load_docs_cache(new_cache_path)
            documents.extend(cache_docs)
            log_entries.append(
                {"path": rel_path, "status": "skipped", "reason": "hash match"}
            )
            stats["skipped"] += 1
            new_files[rel_path] = {
                **prev,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "cache_path": str(new_cache_path.relative_to(index_dir)),
                "doc_count": len(cache_docs),
            }
            continue

        try:
            docs, file_type = extract_docs_for_file(path, rel_path, languages)
            save_docs_cache(new_cache_path, docs, rel_path, file_type)
            reprocessed += 1
            if docs:
                documents.extend(docs)
                stats["processed"] += 1
                reason = "extracted"
            else:
                stats["skipped"] += 1
                reason = "binary" if file_type == "binary" else "no text"
            log_entries.append(
                {
                    "path": rel_path,
                    "status": "processed" if docs else "skipped",
                    "reason": reason,
                    "file_type": file_type,
                }
            )
            new_files[rel_path] = {
                "hash": file_hash,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "cache_path": str(new_cache_path.relative_to(index_dir)),
                "file_type": file_type,
                "doc_count": len(docs),
            }
        except Exception as exc:
            if prev and prev_cache_path:
                cache_docs = load_docs_cache(prev_cache_path)
                documents.extend(cache_docs)
                updated = dict(prev)
                updated["doc_count"] = len(cache_docs)
                new_files[rel_path] = updated
                log_entries.append(
                    {
                        "path": rel_path,
                        "status": "error",
                        "reason": str(exc),
                        "used_cache": True,
                    }
                )
            else:
                log_entries.append(
                    {"path": rel_path, "status": "error", "reason": str(exc)}
                )
            stats["errors"] += 1

    removed = set(previous_files.keys()) - set(new_files.keys())
    for rel_path in sorted(removed):
        stats["removed"] += 1
        log_entries.append(
            {"path": rel_path, "status": "skipped", "reason": "removed"}
        )

    changed = reprocessed > 0 or stats["removed"] > 0
    storage_ok = storage_ready(storage_dir)
    if documents and (changed or args.force_rebuild or not storage_ok):
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        storage_context.persist(persist_dir=str(storage_dir))
    elif not documents:
        log_entries.append(
            {"status": "error", "reason": "no documents extracted; index not rebuilt"}
        )

    log_name = f"index-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jsonl"
    log_path = log_dir / log_name
    with log_path.open("w", encoding="utf-8") as handle:
        for entry in log_entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    manifest["root"] = str(root)
    manifest["index_dir"] = str(index_dir)
    manifest["ocr_languages"] = languages
    manifest["last_run"] = now_iso()
    manifest["embed"] = embed_info
    manifest["files"] = new_files
    manifest["stats"] = stats
    save_manifest(manifest_path, manifest)

    write_index_summary(
        index_dir / "INDEX.md",
        root,
        index_dir,
        languages,
        stats,
        len(documents),
        str(log_path),
        embed_info,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
