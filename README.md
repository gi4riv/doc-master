# Doc Master Skill

**Doc Master** is a Codex skill for local document indexing and search, built on **LlamaIndex**.
It scans a workspace, extracts content from common file types, builds a persistent vector
index, and enables fast, cited retrieval for questions about your files.

Key capabilities:
- LlamaIndex-backed vector index persisted under `.codex/doc-master/`
- OCR for scanned PDFs and images (via Tesseract)
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
