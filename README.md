# Doc Master Skill

**Doc Master** is a Codex skill for local document indexing and search, built on **LlamaIndex**.
It scans a workspace, extracts content from common file types, builds a persistent vector
index, and enables fast, cited retrieval for questions about your files.

Key capabilities:
- LlamaIndex-backed vector index persisted under `.codex/doc-master/`
- Hybrid PDF OCR: native extraction + OCRmyPDF/unpaper fallback for scanned pages
- OCR tool auto-install (multi-OS) with graceful legacy fallback
- OCR for images (via Tesseract)
- Structured extraction for XLSX sheets
- Incremental rebuilds using file hash/mtime cache
- Query results include citations (file + page/line/sheet ranges)

## Contents
```
doc-master/
├── SKILL.md
├── scripts/
├── agents/
│   └── openai.yaml
└── assets/
```

## Requirements
- Python deps: doc-master/scripts/requirements.txt
- Tesseract binary for OCR (system install)

## Advanced OCR auto-setup
Doc Master auto-installs OCR tools when PDF OCR needs them and the tools are missing.
If installation fails, indexing continues with the legacy OCR path.

| OS | Install strategy |
| --- | --- |
| macOS | `brew install ocrmypdf unpaper` |
| Linux | `apt-get` or `dnf` or `pacman` install path (with optional `sudo` retry) |
| Windows | `winget` for Ghostscript/Tesseract + `pip` for OCRmyPDF; `unpaper` native auto-install unsupported |

Relevant knobs:
- CLI: `--pdf-ocr-engine`, `--pdf-clean`, `--pdf-unpaper-args`, `--no-auto-install-ocr-tools`
- Env: `DOC_INDEX_PDF_OCR_ENGINE`, `DOC_INDEX_PDF_CLEAN`, `DOC_INDEX_PDF_UNPAPER_ARGS`, `DOC_INDEX_AUTO_INSTALL_OCR_TOOLS`

## Install (Codex)
Replace <REPO_URL> with your GitHub repo URL:

```
$skill-installer install <REPO_URL>/tree/main/doc-master
```

## Usage
Build/update index:

```
python3 .codex/skills/doc-master/scripts/build_index.py
```

Query index:

```
python3 .codex/skills/doc-master/scripts/query_index.py "your question"
```

## Notes
Index data is stored under `<root>/.codex/doc-master/`.
