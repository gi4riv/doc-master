import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from llama_index.core import StorageContext, load_index_from_storage

from doc_index_common import configure_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the Doc Master index.")
    parser.add_argument("query", nargs="+", help="Query text.")
    parser.add_argument("--root", default=os.getcwd(), help="Workspace root (default: cwd).")
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Index directory (default: <root>/.codex/doc-master).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results.")
    parser.add_argument(
        "--show-snippets",
        action="store_true",
        help="Include text snippets in the response.",
    )
    parser.add_argument(
        "--max-snippet-chars",
        type=int,
        default=280,
        help="Max characters per snippet when showing snippets.",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="HuggingFace embedding model id.",
    )
    return parser.parse_args()


def format_citation(metadata: dict) -> str:
    path = metadata.get("file_path", "unknown")
    if metadata.get("file_type") == "pdf" and metadata.get("page_number"):
        return f"{path}:page={metadata.get('page_number')}"
    if metadata.get("file_type") == "xlsx" and metadata.get("sheet"):
        cell_range = metadata.get("cell_range")
        if cell_range:
            return f"{path}:sheet={metadata.get('sheet')} range={cell_range}"
        return f"{path}:sheet={metadata.get('sheet')}"
    if metadata.get("line_range"):
        return f"{path}:lines={metadata.get('line_range')}"
    return path


def format_snippet(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def get_node_text(node) -> str:
    if hasattr(node, "get_text"):
        try:
            return node.get_text() or ""
        except Exception:
            return ""
    if hasattr(node, "node") and hasattr(node.node, "get_text"):
        try:
            return node.node.get_text() or ""
        except Exception:
            return ""
    return getattr(node, "text", "") or ""


def get_node_metadata(node) -> dict:
    metadata = getattr(node, "metadata", None)
    if metadata:
        return metadata
    if hasattr(node, "node"):
        return getattr(node.node, "metadata", {}) or {}
    return {}


def load_manifest_embed_model(index_dir: Path) -> Optional[str]:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception:
        return None
    embed = manifest.get("embed") or {}
    return embed.get("model")


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    index_dir = Path(args.index_dir).resolve() if args.index_dir else root / ".codex" / "doc-master"
    storage_dir = index_dir / "storage"

    if not storage_dir.exists():
        raise SystemExit("Index storage not found. Run build_index.py first.")

    embed_model = args.embed_model or load_manifest_embed_model(index_dir)
    configure_embedding(embed_model)

    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    index = load_index_from_storage(storage_context)

    query_text = " ".join(args.query).strip()
    if not query_text:
        raise SystemExit("Query text is required.")

    retriever = index.as_retriever(similarity_top_k=args.top_k)
    nodes = retriever.retrieve(query_text)
    if not nodes:
        print("No matches found.")
        return 0

    lines: List[str] = []
    lines.append("Answer:")
    for i, node in enumerate(nodes, start=1):
        snippet = format_snippet(get_node_text(node), args.max_snippet_chars)
        lines.append(f"[{i}] {snippet}")

    lines.append("")
    lines.append("Citations:")
    for i, node in enumerate(nodes, start=1):
        citation = format_citation(get_node_metadata(node))
        if args.show_snippets:
            snippet = format_snippet(get_node_text(node), args.max_snippet_chars)
            lines.append(f"[{i}] {citation} :: {snippet}")
        else:
            lines.append(f"[{i}] {citation}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
