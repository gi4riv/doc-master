---
name: doc-master
description: Build and query a local, persistent LlamaIndex index for a workspace directory, including OCR for scanned PDFs and raster images. Use when asked to create/update an index, search the entire folder, make documents searchable, or build a knowledge base from a directory (including scanned PDFs and images).
---

# Doc Master

## Overview
Build, update, and query a persistent local index of the current workspace (recursive), including OCR for scanned PDFs and images and structured extraction for XLSX.

## Quick start
- Build or update index:
  - `python3 .codex/skills/doc-master/scripts/build_index.py`
- Query:
  - `python3 .codex/skills/doc-master/scripts/query_index.py "your question"`

## Build or update the index
- Run `build_index.py` to scan the workspace and store outputs under `<root>/.codex/doc-master/`.
- Use incremental processing based on mtime/size and content hash, reusing cached extraction results when possible.
- Default OCR languages are `ita+eng`. Override with `--languages` or `DOC_INDEX_OCR_LANGS`.

Flags:
- `--root <path>`: workspace root (default: cwd)
- `--index-dir <path>`: override index directory (must stay inside root)
- `--languages <langcodes>`: OCR languages for tesseract
- `--include-hidden`: include hidden files and folders
- `--force-rebuild`: rebuild index even if no changes detected
- `--embed-model <model-id>`: HuggingFace embedding model id

## Query the index
- Run `query_index.py` to retrieve relevant passages and citations.
- Citations include file path and position: `page` for PDF, `sheet` and `range` for XLSX, or `lines` for text.

Flags:
- `--top-k <n>`: number of results
- `--show-snippets`: include snippet text inline with citations
- `--max-snippet-chars <n>`: snippet length
- `--embed-model <model-id>`: must match the build embedding model

## Outputs
The index is persisted under `<root>/.codex/doc-master/`:
- `storage/` (LlamaIndex storage)
- `cache/` (extracted text and OCR cache)
- `manifest.json` (state for incremental updates)
- `logs/` (per-run file status logs)
- `INDEX.md` (summary of last run)

## Workspace hygiene
When creating ad-hoc tools or temporary artifacts for personal use, keep them inside the Doc Master area:
- Create a dedicated folder at `<root>/.codex/doc-master/scratch/`.
- Store any throwaway Python tools, scripts, notebooks, or experiments under `scratch/`.
- Store temporary files and outputs under `scratch/tmp/`.
- Store virtual environments or dependency caches (e.g., `.env`, `.venv`, `venv/`) under `scratch/env/`.
- Avoid creating these artifacts in the workspace root or alongside project sources.

## Dependencies
Install Python deps from `scripts/requirements.txt`. OCR also requires the system `tesseract` binary.
