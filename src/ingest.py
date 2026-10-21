"""
Ingestion pipeline — designed to correctly and efficiently bulk-ingest a
whole folder of scheme PDFs (dozens, hundreds, whatever you have).

Key properties for bulk use:
  - SKIPS unchanged files automatically (content-hash based manifest), so
    re-running on a growing folder only processes what's new/changed.
  - CLEANS UP stale chunks when a previously-ingested PDF changes (a new
    version with fewer/different pages won't leave orphaned old chunks
    behind in Chroma or in the BM25 keyword index).
  - Batches the BM25 corpus write ONCE at the end of the whole run instead
    of once per file — ingesting 200 PDFs no longer means 200 full
    load-modify-save cycles of the keyword index.
  - Keeps going on a per-file failure (corrupt/unreadable PDF, missing
    metadata, etc.) and prints a clear summary of what succeeded, what was
    skipped, and what failed, instead of stopping the whole batch.
  - Works on any folder, not just data/pdfs/, and can recurse into
    subfolders (handy if you organise PDFs by state/category).

Run:
    python -m src.ingest                          # ingest data/pdfs (default)
    python -m src.ingest --dir /path/to/folder     # ingest a custom folder
    python -m src.ingest --recursive               # also scan subfolders
    python -m src.ingest --file NAME.pdf           # ingest just one file
    python -m src.ingest --force                   # re-ingest even unchanged files
    python -m src.ingest --reset                   # wipe DB + manifest, ingest everything fresh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

from tqdm import tqdm

from src.config import PDF_DIR, METADATA_DIR, IMAGE_DIR, CHROMA_DIR, BM25_STORE_PATH
from src.pdf_extractor import extract_all
from src.chunking import build_all_documents
from src.vectorstore import get_vectorstore, load_bm25_corpus, save_bm25_corpus, delete_doc_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ingest")

MANIFEST_PATH = BM25_STORE_PATH.parent / "ingest_manifest.json"


# ---------------------------------------------------------------------------
# Manifest: tracks what's already been ingested and its content hash, so
# re-running ingestion on a large, growing folder is fast and safe.
# ---------------------------------------------------------------------------
def _load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: Dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_metadata_for_pdf(pdf_path: Path) -> Optional[Dict[str, Any]]:
    stem = pdf_path.stem
    candidates = [
        METADATA_DIR / f"{stem}_metadata.json",
        METADATA_DIR / f"{stem}.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["_source_file"] = pdf_path.name
            return meta

    logger.warning(
        f"No metadata JSON found for '{pdf_path.name}' "
        f"(looked for {candidates[0].name} or {candidates[1].name}). "
        f"Ingesting with minimal auto-generated metadata instead."
    )
    return {
        "document_id": stem,
        "document_name": stem.replace("_", " "),
        "scheme_id": stem,
        "_source_file": pdf_path.name,
        "status": "unknown",
    }


def _collect_pdf_files(source_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(p for p in source_dir.glob(pattern) if p.is_file())


def ingest_pdf(pdf_path: Path, vectorstore) -> List:
    """Extracts + chunks a single PDF. Returns the list of new Documents (does NOT write BM25)."""
    doc_meta = _find_metadata_for_pdf(pdf_path)
    doc_id = doc_meta.get("document_id", pdf_path.stem)

    extracted = extract_all(str(pdf_path), str(IMAGE_DIR), doc_id)
    logger.info(
        f"  -> {len(extracted['text_pages'])} text pages, "
        f"{len(extracted['tables'])} tables, "
        f"{len(extracted['images'])} images extracted"
    )

    documents = build_all_documents(
        extracted["text_pages"], extracted["tables"], extracted["images"], doc_meta
    )
    if not documents:
        return []

    ids = [
        f"{doc_id}_{d.metadata['chunk_type']}_p{d.metadata.get('page','0')}_{i}"
        for i, d in enumerate(documents)
    ]
    vectorstore.add_documents(documents, ids=ids)
    return documents


def run(
    reset: bool = False,
    single_file: Optional[str] = None,
    source_dir: Optional[str] = None,
    recursive: bool = False,
    force: bool = False,
) -> None:
    src_dir = Path(source_dir).resolve() if source_dir else PDF_DIR

    if reset:
        logger.info("Resetting vector store, BM25 cache, and manifest...")
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        if BM25_STORE_PATH.exists():
            BM25_STORE_PATH.unlink()
        if MANIFEST_PATH.exists():
            MANIFEST_PATH.unlink()
        save_bm25_corpus([])
        force = True  # everything needs to be reprocessed after a reset

    vectorstore = get_vectorstore()
    manifest = _load_manifest()
    bm25_corpus = load_bm25_corpus()  # load ONCE, mutate in memory, save ONCE at the end

    if single_file:
        pdf_files = [src_dir / single_file]
        if not pdf_files[0].exists():
            logger.error(f"File not found: {pdf_files[0]}")
            sys.exit(1)
    else:
        pdf_files = _collect_pdf_files(src_dir, recursive)

    if not pdf_files:
        logger.warning(
            f"No PDF files found in {src_dir}"
            f"{' (recursive)' if recursive else ''}. Add PDFs there and re-run."
        )
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s) in {src_dir}.")

    processed, skipped, failed = [], [], []
    total_new_chunks = 0

    for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs"):
        rel_key = str(pdf_path.resolve())
        try:
            current_hash = _file_hash(pdf_path)
        except Exception as e:
            logger.error(f"Could not read '{pdf_path.name}': {e}")
            failed.append((pdf_path.name, str(e)))
            continue

        prior = manifest.get(rel_key)
        if prior and prior.get("hash") == current_hash and not force:
            skipped.append(pdf_path.name)
            continue

        logger.info(f"Processing: {pdf_path.name}" + (" (changed, re-ingesting)" if prior else " (new)"))

        # If this file was ingested before under a different content hash,
        # remove its old chunks first so updates don't leave stale data behind.
        if prior:
            old_doc_id = prior.get("doc_id")
            if old_doc_id:
                removed_vec = delete_doc_chunks(vectorstore, old_doc_id)
                before = len(bm25_corpus)
                bm25_corpus = [d for d in bm25_corpus if d.metadata.get("doc_id") != old_doc_id]
                removed_bm25 = before - len(bm25_corpus)
                if removed_vec or removed_bm25:
                    logger.info(f"  -> Removed {removed_vec} stale vector chunks / {removed_bm25} stale BM25 chunks")

        try:
            new_docs = ingest_pdf(pdf_path, vectorstore)
        except Exception as e:
            logger.exception(f"Failed to ingest {pdf_path.name}: {e}")
            failed.append((pdf_path.name, str(e)))
            continue

        if not new_docs:
            logger.warning(f"  No content extracted from {pdf_path.name}, skipping.")
            failed.append((pdf_path.name, "No extractable content"))
            continue

        bm25_corpus.extend(new_docs)
        doc_id = new_docs[0].metadata.get("doc_id", pdf_path.stem)
        manifest[rel_key] = {
            "doc_id": doc_id,
            "hash": current_hash,
            "chunk_count": len(new_docs),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        processed.append(pdf_path.name)
        total_new_chunks += len(new_docs)
        logger.info(f"  -> Ingested {len(new_docs)} chunks.")

    # Single batched write at the end — much faster than per-file writes for large folders.
    save_bm25_corpus(bm25_corpus)
    _save_manifest(manifest)

    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"  Processed (new/updated) : {len(processed)}")
    print(f"  Skipped (unchanged)     : {len(skipped)}")
    print(f"  Failed                  : {len(failed)}")
    print(f"  New/updated chunks      : {total_new_chunks}")
    print(f"  Total chunks in index   : {len(bm25_corpus)}")
    if failed:
        print("\n  Failed files:")
        for name, reason in failed:
            print(f"    - {name}: {reason}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk-ingest farmer scheme PDFs into the local RAG store.")
    parser.add_argument("--reset", action="store_true", help="Wipe the DB + manifest before ingesting.")
    parser.add_argument("--file", type=str, default=None, help="Ingest only this single PDF filename.")
    parser.add_argument("--dir", type=str, default=None, help="Custom folder to ingest PDFs from (default: data/pdfs).")
    parser.add_argument("--recursive", action="store_true", help="Also scan subfolders of --dir for PDFs.")
    parser.add_argument("--force", action="store_true", help="Re-ingest files even if unchanged since last run.")
    args = parser.parse_args()
    run(
        reset=args.reset,
        single_file=args.file,
        source_dir=args.dir,
        recursive=args.recursive,
        force=args.force,
    )
