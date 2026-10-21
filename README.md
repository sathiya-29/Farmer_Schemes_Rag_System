# 🌾 Farmer Schemes RAG Assistant

A **fully local, open-source Retrieval-Augmented-Generation (RAG) system** for
Indian farmer / agriculture scheme PDFs. It ingests **text, tables, and
images** from scheme guideline PDFs, indexes them in a local **ChromaDB**
vector store, retrieves with **hybrid search (BM25 keyword + dense vector)**,
refines results with a **cross-encoder reranker**, and answers questions
through a **Streamlit** chat UI backed by a **local LLM (Ollama)**.

No cloud API keys. No external services. Nothing leaves your machine.

```
PDF folder ──▶ extract (text/tables/images) ──▶ chunk ──▶ embed ──▶ Chroma
                                                                       │
User question ──▶ hybrid search (BM25 + vector) ──▶ rerank ──▶ local LLM ──▶ answer
```

---

## Table of contents

1. [Features](#-features)
2. [Project structure — every file explained](#-project-structure--every-file-explained)
3. [How it works end to end](#-how-it-works-end-to-end)
4. [Prerequisites](#-prerequisites)
5. [Full local setup, step by step](#-full-local-setup-step-by-step)
6. [Ingesting your PDFs](#-ingesting-your-pdfs)
7. [Running the app](#-running-the-app)
8. [Example questions](#-example-questions-to-try)
9. [Configuration reference](#-configuration-reference-env)
10. [Adding / updating schemes later](#-adding--updating-schemes-later)
11. [Troubleshooting](#-troubleshooting)
12. [Tech stack](#-tech-stack)
13. [Roadmap ideas](#-roadmap-ideas)

---

## ✨ Features

- **Multi-modal PDF ingestion** — extracts plain text (PyMuPDF), tables
  (pdfplumber, converted to markdown), and embedded images (PyMuPDF +
  Tesseract OCR captions) from every PDF, not just raw text.
- **Rich per-scheme metadata** — each PDF has a companion `*_metadata.json`
  (scheme name, source organization, level, state/district, status, last
  updated, etc.) that gets attached to every chunk for citation, filtering,
  and display.
- **Hybrid search** — combines BM25 keyword search (great for exact terms,
  scheme names, numbers) with dense embedding search (great for paraphrased,
  semantic questions), fused with LangChain's `EnsembleRetriever`.
- **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2`
  re-scores the retrieved candidates jointly with the query for much higher
  precision before anything reaches the LLM.
- **Intelligent fallback answering** — if confidence is low, the system does
  **not** just say "I don't know." It widens the search, answers with the
  best partial information available, transparently flags reduced
  confidence, and lists which schemes are actually in the knowledge base.
- **Bulk-ready ingestion** — point it at a folder with hundreds of PDFs.
  Unchanged files are skipped automatically (content-hash manifest), changed
  files have their stale chunks cleaned up before re-ingesting, and a single
  bad PDF never stops the rest of the batch.
- **100% local & open source** — HuggingFace `sentence-transformers` for
  embeddings, a local cross-encoder for reranking, ChromaDB for vector
  storage, and Ollama for LLM generation.
- **Streamlit chat UI** — ask questions conversationally, see cited sources
  (scheme, page, chunk type, relevance score), and view extracted images
  inline.

---

## 📁 Project structure — every file explained

```
farmer_schemes_rag/
│
├── app.py                      Streamlit chat UI — the frontend entry point
├── check_setup.py               Standalone script that verifies your local
│                                 environment (Python deps, tesseract, ollama)
│                                 before you try to run anything
├── requirements.txt              Pinned Python dependencies
├── .env.example                  Template for all configurable settings —
│                                 copy to `.env` and edit
├── .gitignore                    Excludes .env, caches, the vector DB, and
│                                 extracted images/pickles from version control
├── README.md                     You are here
│
├── data/
│   ├── pdfs/                     Put your source scheme PDFs here
│   │   └── Agriculture_Infrastructure_Fund.pdf     (sample PDF, included)
│   ├── metadata/                 One JSON per PDF, same base filename as the PDF
│   │   └── Agriculture_Infrastructure_Fund_metadata.json  (sample, included)
│   └── images/                   Auto-populated at ingestion time — images
│                                 extracted from inside the PDFs, later shown
│                                 in the Streamlit UI next to relevant answers
│
├── chroma_db/                    Auto-created — the persistent local Chroma
│                                 vector database (this is where embeddings live)
│
├── storage/
│   ├── bm25_docs.pkl              Auto-created — the full document corpus,
│   │                              cached so the BM25 keyword retriever can be
│   │                              rebuilt instantly on every app/ingest run
│   └── ingest_manifest.json        Auto-created — tracks which PDFs have been
│                                   ingested and their content hash, so re-runs
│                                   skip unchanged files and clean up changed ones
│
└── src/
    ├── __init__.py                Marks `src` as a Python package (empty)
    │
    ├── config.py                   Central configuration. Reads every setting
    │                               from `.env` (with safe defaults if absent)
    │                               and resolves all folder paths, model names,
    │                               chunk sizes, and retrieval parameters used
    │                               throughout the project.
    │
    ├── pdf_extractor.py             Low-level PDF parsing. Three extractors:
    │                               `extract_text_by_page()` (PyMuPDF),
    │                               `extract_tables()` (pdfplumber → markdown),
    │                               `extract_images()` (PyMuPDF, saves images to
    │                               disk, OCRs them with pytesseract if the
    │                               tesseract binary is installed — degrades
    │                               gracefully to "no caption" if not).
    │                               `extract_all()` runs all three and returns
    │                               a single dict.
    │
    ├── chunking.py                  Turns raw extracted content into LangChain
    │                               `Document` objects with consistent, rich
    │                               metadata. Text is split with
    │                               `RecursiveCharacterTextSplitter`; tables and
    │                               images become their own chunk types
    │                               (`chunk_type: "text" | "table" | "image"`).
    │                               Every chunk is stamped with scheme_id,
    │                               scheme_name, source_file, source_url,
    │                               source_organization, scheme_level, state,
    │                               district, status, last_updated, page.
    │
    ├── embeddings.py                 Returns a singleton local HuggingFace
    │                               embedding function
    │                               (`sentence-transformers/all-MiniLM-L6-v2`),
    │                               used identically at ingestion and query time.
    │
    ├── vectorstore.py                 Manages the persistent Chroma collection
    │                               and the hybrid retriever. Key functions:
    │                               `get_vectorstore()`, `save_bm25_corpus()` /
    │                               `load_bm25_corpus()` / `append_bm25_corpus()`
    │                               (the BM25 keyword index cache),
    │                               `get_hybrid_retriever()` (fuses BM25 +
    │                               Chroma via `EnsembleRetriever`),
    │                               `delete_doc_chunks()` (removes a scheme's
    │                               old chunks when its PDF is updated),
    │                               `get_all_scheme_names()` (powers the sidebar
    │                               and the "here's what IS in the KB" fallback).
    │
    ├── reranker.py                    Loads a local `CrossEncoder`
    │                               (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and
    │                               exposes `rerank(query, documents, top_n)`,
    │                               which de-duplicates near-identical chunks,
    │                               scores each (query, chunk) pair jointly, and
    │                               returns the top-N most relevant chunks with
    │                               normalized [0,1] confidence scores.
    │
    ├── rag_chain.py                    The end-to-end answer pipeline:
    │                               hybrid retrieval → rerank → confidence
    │                               check → (high confidence: answer normally
    │                               with citations) or (low/no confidence:
    │                               widen the search, answer with the best
    │                               available info, flag reduced confidence,
    │                               and list which schemes are actually
    │                               indexed). Talks to the local LLM through
    │                               `langchain_community.llms.Ollama`.
    │                               Exposes `answer_query(question) -> dict`
    │                               with `answer`, `sources`, `confidence`.
    │
    └── ingest.py                       CLI ingestion pipeline (bulk-ready).
                                    Matches each PDF to its metadata JSON,
                                    extracts + chunks + embeds + upserts into
                                    Chroma, and appends to the BM25 corpus.
                                    Tracks a content-hash manifest so re-runs
                                    skip unchanged files and clean up stale
                                    chunks for changed files. Supports
                                    `--dir`, `--recursive`, `--file`, `--force`,
                                    `--reset` (see "Ingesting your PDFs" below).
```

---

## 🔍 How it works end to end

**Ingestion (`python -m src.ingest`):**

```
PDF file
   │
   ├─▶ extract_text_by_page()   (PyMuPDF)        ─┐
   ├─▶ extract_tables()         (pdfplumber)       ├─▶ build_all_documents()  ─▶  LangChain Documents
   └─▶ extract_images()         (PyMuPDF + OCR)   ─┘        (chunking.py)          + rich metadata
                                                                    │
                                                                    ▼
                                              embed (HuggingFace, local) + upsert into Chroma
                                                                    │
                                                                    ▼
                                          append to BM25 corpus cache (storage/bm25_docs.pkl)
                                                                    │
                                                                    ▼
                                      record content hash in storage/ingest_manifest.json
```

**Querying (Streamlit app → `answer_query()`):**

```
User question
      │
      ▼
┌───────────────────────────────┐
│   Hybrid Retriever (top-K)     │
│  BM25 (keyword) + Chroma       │
│  (dense vector), fused via     │
│  LangChain EnsembleRetriever   │
└───────────────┬─────────────────┘
                ▼
┌───────────────────────────────┐
│   Cross-Encoder Reranker        │
│  (ms-marco-MiniLM-L-6-v2)       │
│  scores (query, chunk) pairs    │
│  jointly → top-N by relevance   │
└───────────────┬─────────────────┘
                ▼
        Confidence check
   (top rerank score vs.
    CONFIDENCE_SCORE_THRESHOLD)
       │                  │
     HIGH              LOW / NONE
       │                  │
       ▼                  ▼
  Answer normally,   Widen search (no filters, larger k)
  grounded in the    → answer with best available info,
  top chunks, with   clearly flag reduced confidence, and
  citations.         list which schemes ARE indexed.
       │                  │
       └────────┬─────────┘
                ▼
     Local LLM (Ollama) generates the final answer,
     strictly grounded in the retrieved context.
                ▼
     Streamlit renders the answer + an expandable
     "Sources" panel (scheme, page, type, score, image).
```

---

## ✅ Prerequisites

| Requirement | Why | Get it |
|---|---|---|
| **Python 3.10 – 3.12** | Runs the whole pipeline | [python.org](https://www.python.org/downloads/) |
| **Ollama** | Serves the local LLM that generates answers | [ollama.com](https://ollama.com/download) |
| **Tesseract OCR** *(optional but recommended)* | Extracts text captions from images/figures inside PDFs | see OS-specific commands below |
| ~4–6 GB free disk | Embedding model, reranker model, Ollama model, and the vector DB all live on disk | — |

No GPU is required — everything defaults to CPU. No internet is required
*after* the one-time model downloads (embedding model, reranker model, and
the Ollama model are all cached locally the first time you use them).

---

## 🛠 Full local setup, step by step

### Step 1 — Clone the project

```bash
git clone https://github.com/sathiya-29/Farmer_Schemes_Rag_System.git
cd Farmer_Schemes_Rag_System
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd.exe)
.venv\Scripts\activate.bat
```

### Step 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Install Ollama and pull a local LLM

Download and install Ollama from **https://ollama.com/download** for your OS,
then:

```bash
ollama pull llama3.1
```

You can substitute any other local model (`mistral`, `qwen2.5`, `phi3`, etc.)
— just make sure the same name is set in `OLLAMA_MODEL` in your `.env` later.

Ollama runs a local server on `http://localhost:11434`. On macOS/Windows it
typically starts automatically after installation; on Linux you may need to
start it manually:

```bash
ollama serve
```

Leave that running in its own terminal (or run it as a background
service/systemd unit) while you use the app.

### Step 5 — (Optional but recommended) Install Tesseract OCR

This lets the pipeline pull text captions out of images/figures embedded in
your PDFs. Without it, images are still extracted and shown in the UI — they
just won't have a searchable OCR caption.

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr

# macOS (Homebrew)
brew install tesseract

# Windows
# Install from https://github.com/UB-Mannheim/tesseract/wiki
# and make sure the install folder is added to your PATH
```

### Step 6 — Configure environment variables

```bash
cp .env.example .env
```

The defaults in `.env.example` work out of the box. Open `.env` in an editor
only if you want to change models, chunk sizes, or retrieval parameters (see
the [Configuration reference](#-configuration-reference-env) below).

### Step 7 — Verify your environment

```bash
python check_setup.py
```

You should see `[OK]` next to each Python dependency. `tesseract` and
`ollama` are flagged as optional/required respectively — install anything
marked `MISSING` before continuing.

Expected output looks like:

```
[OK     ] langchain
[OK     ] chromadb
[OK     ] sentence-transformers
[OK     ] PyMuPDF (fitz)
[OK     ] pdfplumber
[OK     ] tesseract-ocr binary (optional, for image OCR captions)
[OK     ] ollama binary (required for local LLM generation)

Python dependencies look good. Next: `python -m src.ingest` then `streamlit run app.py`.
```

### Step 8 — Ingest the sample PDF (first run)

A sample scheme PDF and its metadata are already included in `data/pdfs/`
and `data/metadata/`, so you can test the whole pipeline immediately:

```bash
python -m src.ingest
```

The **first run downloads the embedding model and reranker model** from
HuggingFace (a few hundred MB total) — this needs internet access once, then
both models are cached locally under your HuggingFace cache directory
(`~/.cache/huggingface` by default) and never re-downloaded.

You should see a summary like:

```
======================================================================
INGESTION SUMMARY
======================================================================
  Processed (new/updated) : 1
  Skipped (unchanged)     : 0
  Failed                  : 0
  New/updated chunks      : 30
  Total chunks in index   : 30
======================================================================
```

### Step 9 — Launch the app

```bash
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) — open it in
your browser. The sidebar shows which schemes are currently indexed; type a
question in the chat box at the bottom.

**That's it — the full pipeline (extraction → chunking → embedding → hybrid
retrieval → reranking → local LLM generation) is now running end to end on
your machine.**

---

## 📥 Ingesting your PDFs

### Add your own scheme PDFs

1. Drop scheme PDFs into `data/pdfs/`.
2. Drop a matching metadata JSON into `data/metadata/`, using the **same base
   filename** as the PDF:
   - `MySchemeName.pdf` ↔ `MySchemeName_metadata.json` (preferred), or
   - `MySchemeName.pdf` ↔ `MySchemeName.json`

   Metadata shape (see `data/metadata/Agriculture_Infrastructure_Fund_metadata.json`
   for a real example):

   ```json
   {
     "document_id": "AIF_GUIDELINES_2026",
     "document_name": "Agriculture Infrastructure Fund",
     "document_type": "scheme_guideline",
     "source_organization": "Ministry of Agriculture, Government of India",
     "source_url": "https://vikaspedia.in",
     "language": "English",
     "country": "India",
     "publication_date": "2020-01-01",
     "last_updated": "2023-10-01",
     "version": "1.0",
     "status": "active",
     "scheme_id": "AIF_2026",
     "scheme_level": "Central",
     "applicable_scope": "India",
     "state": null,
     "district": null
   }
   ```

   A PDF with **no matching metadata file is still ingested** — the pipeline
   auto-generates minimal metadata from the filename and logs a warning, so a
   missing metadata file never blocks ingestion.

3. Run:

   ```bash
   python -m src.ingest
   ```

### Ingesting a large folder of PDFs (bulk mode)

If your PDFs live somewhere else entirely, point `--dir` straight at that
folder — no need to copy files into `data/pdfs/` first:

```bash
python -m src.ingest --dir /path/to/your/pdf/folder
python -m src.ingest --dir /path/to/your/pdf/folder --recursive   # also scan subfolders
```

Metadata JSONs still go in `data/metadata/` (matched purely by filename,
regardless of which folder the PDF itself lives in).

Bulk ingestion is **safe to re-run** any time, including as a scheduled job
over a folder that keeps growing:

- **Unchanged files are skipped automatically** — a content-hash manifest at
  `storage/ingest_manifest.json` means re-running over a folder of hundreds
  of PDFs after adding a handful of new ones only processes those new ones.
- **Changed files are cleanly updated** — if a previously-ingested PDF's
  content changes (a new version, edited guidelines, etc.), its old chunks
  are automatically removed from both Chroma and the BM25 index before the
  new version is ingested. No stale data, no duplicates.
- **One bad file never stops the batch** — a corrupted or unreadable PDF is
  logged and skipped; every other file still gets processed. A summary at
  the end tells you exactly what succeeded, what was skipped, and what
  failed (and why).
- **Fast at scale** — the BM25 keyword index is written **once** at the end
  of the whole run, not once per file.

All ingestion flags:

| Flag | Effect |
|---|---|
| `--dir PATH` | Ingest PDFs from `PATH` instead of the default `data/pdfs/` |
| `--recursive` | Also scan subfolders of `--dir` for PDFs |
| `--file NAME.pdf` | Ingest only this single PDF (from `--dir` or `data/pdfs/`) |
| `--force` | Re-ingest every file even if its content hash is unchanged |
| `--reset` | Wipe the vector DB, BM25 cache, and manifest, then ingest everything fresh |

Examples:

```bash
python -m src.ingest --file Agriculture_Infrastructure_Fund.pdf
python -m src.ingest --force
python -m src.ingest --reset
```

---

## 💬 Running the app

```bash
streamlit run app.py
```

- **Sidebar** — shows how many schemes are indexed, lists them by name, and
  lets you inspect the raw metadata JSON of any ingested scheme.
- **Chat box** — ask a question in plain English (or whatever language your
  PDFs are in).
- **Sources panel** — every answer includes an expandable panel showing
  which scheme(s), page number(s), chunk type (text/table/image), and
  relevance score backed the answer — plus the actual extracted image when a
  figure from the PDF was used.

---

## 🗣 Example questions to try

With the included Agriculture Infrastructure Fund sample:

- "What is the interest subvention rate under the Agriculture Infrastructure Fund?"
- "Who is eligible to apply under AIF?"
- "What documents are needed for the loan application?"
- "What's the credit guarantee coverage limit for FPOs?"
- "What is the moratorium period for repayment?"
- "List the eligible projects for building community farming assets."

---

## ⚙️ Configuration reference (`.env`)

All settings live in `.env` (copy from `.env.example`). Every value has a
sensible default, so you never *have* to touch this file.

| Variable | Default | Purpose |
|---|---|---|
| `CHROMA_DIR` | `./chroma_db` | Where the persistent vector DB is stored |
| `COLLECTION_NAME` | `farmer_schemes` | Chroma collection name |
| `PDF_DIR` | `./data/pdfs` | Default folder ingestion reads PDFs from |
| `METADATA_DIR` | `./data/metadata` | Folder with per-PDF metadata JSON files |
| `IMAGE_DIR` | `./data/images` | Where extracted images are saved |
| `BM25_STORE_PATH` | `./storage/bm25_docs.pkl` | Cached corpus for the BM25 keyword retriever |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any HuggingFace sentence-transformers model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Any HuggingFace CrossEncoder model |
| `OLLAMA_MODEL` | `llama3.1` | Any model you've pulled into local Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `CHUNK_SIZE` | `900` | Characters per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `VECTOR_TOP_K` | `8` | Candidates pulled from Chroma before fusion |
| `BM25_TOP_K` | `8` | Candidates pulled from BM25 before fusion |
| `HYBRID_VECTOR_WEIGHT` | `0.6` | Weight given to dense/vector search in fusion |
| `HYBRID_BM25_WEIGHT` | `0.4` | Weight given to keyword/BM25 search in fusion |
| `RERANK_TOP_N` | `5` | Chunks kept after reranking, passed to the LLM |
| `CONFIDENCE_SCORE_THRESHOLD` | `0.15` | Below this reranker score, the fallback strategy kicks in |

---

## ➕ Adding / updating schemes later

1. Drop the new (or updated) PDF into `data/pdfs/` (or any folder you pass
   via `--dir`).
2. Drop/update its metadata JSON in `data/metadata/` using the same base
   filename.
3. Run `python -m src.ingest` (safe to run repeatedly — see bulk-mode notes
   above; updated files are automatically detected and re-indexed cleanly).
4. Refresh the Streamlit app — the sidebar's scheme list updates
   automatically on the next question or page reload.

---

## 🐛 Troubleshooting

| Symptom | Fix |
|---|---|
| `⚠️ Could not reach the local LLM (Ollama)` in an answer | Make sure `ollama serve` is running and you've pulled the model named in `.env` (`ollama pull llama3.1`) |
| `ModuleNotFoundError` when running anything | Activate your virtualenv and re-run `pip install -r requirements.txt` |
| No images extracted from a PDF | Expected for PDFs that are pure vector graphics with no embedded raster images |
| OCR captions on images are empty | Install `tesseract-ocr` (Step 5 above); the app still works without it, images just won't have a searchable text caption |
| A specific PDF always shows up as "Failed" in the ingestion summary | The file is likely corrupted or not a real PDF — check it opens in a normal PDF viewer |
| Answers seem to ignore a document you ingested | Confirm it shows up in the Streamlit sidebar; if something looks out of sync run `python -m src.ingest --reset` to rebuild cleanly |
| First `ingest` or first app launch is slow | The embedding and reranker models download from HuggingFace on first use (a few hundred MB total) and are cached locally afterward |
| `pip install` fails on `chromadb` with a build error | Make sure you're on Python 3.10–3.12 and have a recent `pip`; on some Linux distros you may need `build-essential` / a C compiler installed |

---

## 🧩 Tech stack

| Layer | Technology |
|---|---|
| PDF text/table/image extraction | PyMuPDF, pdfplumber, Pillow, pytesseract |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, open source) |
| Vector store | ChromaDB (persistent, local) |
| Keyword search | BM25 (`rank_bm25` via LangChain `BM25Retriever`) |
| Hybrid fusion | LangChain `EnsembleRetriever` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, open source) |
| LLM generation | Ollama (any local open-source model, e.g. Llama 3.1) |
| Frontend | Streamlit |

Everything above runs entirely on your local machine — no API keys, no cloud
services, no data leaving your environment.

---

## 🗺 Roadmap ideas

- State/district/scheme-level filters in the Streamlit sidebar (metadata is
  already captured on every chunk — this is mostly UI work)
- A `/health` or CLI command to validate that every ingested PDF still has a
  corresponding, valid metadata file
- Optional swap-in of a stronger local vision-language model for image
  understanding beyond OCR text
- Multi-turn conversational memory in the chat UI (currently each question
  is answered independently)

Contributions and issues are welcome — this is a starting point meant to be
extended for your own scheme collection.
