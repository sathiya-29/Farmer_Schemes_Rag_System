"""
Turns raw extracted content (text pages / tables / images) into LangChain
`Document` objects with rich, consistent metadata, so retrieval + citation +
UI rendering can all rely on the same fields.

Metadata attached to EVERY chunk:
    doc_id, scheme_id, scheme_name, source_file, source_url,
    source_organization, scheme_level, state, district, status,
    last_updated, chunk_type ("text" | "table" | "image"), page
"""
from __future__ import annotations

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def _base_metadata(doc_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Pulls the fields we care about out of a metadata.json, with safe defaults."""
    return {
        "doc_id": doc_meta.get("document_id", "unknown"),
        "scheme_id": doc_meta.get("scheme_id", doc_meta.get("document_id", "unknown")),
        "scheme_name": doc_meta.get("document_name", "Unknown Scheme"),
        "source_file": doc_meta.get("_source_file", ""),
        "source_url": doc_meta.get("source_url", ""),
        "source_organization": doc_meta.get("source_organization", ""),
        "scheme_level": doc_meta.get("scheme_level", ""),
        "state": doc_meta.get("state") or "All India",
        "district": doc_meta.get("district") or "All",
        "status": doc_meta.get("status", "unknown"),
        "last_updated": doc_meta.get("last_updated", ""),
    }


def build_text_documents(text_pages: List[Dict[str, Any]], doc_meta: Dict[str, Any]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    base_meta = _base_metadata(doc_meta)
    docs: List[Document] = []

    for page in text_pages:
        chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(chunks):
            meta = {
                **base_meta,
                "chunk_type": "text",
                "page": page["page"],
                "chunk_index": i,
            }
            docs.append(Document(page_content=chunk, metadata=meta))
    return docs


def build_table_documents(tables: List[Dict[str, Any]], doc_meta: Dict[str, Any]) -> List[Document]:
    base_meta = _base_metadata(doc_meta)
    docs: List[Document] = []
    for t in tables:
        # Give the LLM/embedder context about *what* this table is, not just raw cells.
        content = (
            f"Table from '{base_meta['scheme_name']}' (page {t['page']}):\n\n{t['markdown']}"
        )
        meta = {
            **base_meta,
            "chunk_type": "table",
            "page": t["page"],
            "chunk_index": t["table_index"],
        }
        docs.append(Document(page_content=content, metadata=meta))
    return docs


def build_image_documents(images: List[Dict[str, Any]], doc_meta: Dict[str, Any]) -> List[Document]:
    """
    Images become searchable text chunks via their OCR caption (if any).
    Even without OCR text, we still index a descriptive placeholder so the
    image is discoverable and can be shown to the user alongside a text answer.
    """
    base_meta = _base_metadata(doc_meta)
    docs: List[Document] = []
    for img in images:
        ocr_text = img.get("ocr_text", "").strip()
        if ocr_text:
            content = (
                f"Image/figure from '{base_meta['scheme_name']}' (page {img['page']}). "
                f"Extracted text from the image:\n{ocr_text}"
            )
        else:
            content = (
                f"Image/figure from '{base_meta['scheme_name']}' (page {img['page']}). "
                f"No text could be extracted from this image (likely a photo, logo, or icon)."
            )
        meta = {
            **base_meta,
            "chunk_type": "image",
            "page": img["page"],
            "image_path": img["image_path"],
        }
        docs.append(Document(page_content=content, metadata=meta))
    return docs


def build_all_documents(
    text_pages: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
    doc_meta: Dict[str, Any],
) -> List[Document]:
    docs = []
    docs += build_text_documents(text_pages, doc_meta)
    docs += build_table_documents(tables, doc_meta)
    docs += build_image_documents(images, doc_meta)
    return docs
