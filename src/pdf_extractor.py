"""
Extracts three kinds of content from a PDF:
  1. Plain text, page by page                (PyMuPDF / fitz)
  2. Tables, converted to markdown            (pdfplumber)
  3. Images, saved to disk + OCR'd if possible (PyMuPDF + pytesseract)

Each extractor returns a list of plain dicts so the rest of the pipeline
does not need to know anything about PDF-parsing internals.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Dict, Any

import pymupdf as fitz  # PyMuPDF (import name kept as `fitz` for readability below)
import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import pytesseract
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

try:
    # Only works if the tesseract binary is actually installed on the system.
    if _OCR_AVAILABLE:
        pytesseract.get_tesseract_version()
except Exception:
    _OCR_AVAILABLE = False
    logger.warning(
        "Tesseract binary not found. Image OCR will be skipped; images will "
        "still be extracted and stored, just without a text caption. "
        "Install tesseract-ocr to enable OCR captions for scanned figures."
    )


def extract_text_by_page(pdf_path: str) -> List[Dict[str, Any]]:
    """Returns [{"page": int, "text": str}, ...] using PyMuPDF."""
    results = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                results.append({"page": page_num, "text": text})
    return results


def extract_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Returns [{"page": int, "table_index": int, "markdown": str}, ...]
    Tables are converted to markdown so they can be embedded and later
    rendered nicely in the Streamlit UI / passed to the LLM as readable text.
    """
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                    md = _table_to_markdown(table)
                    if md.strip():
                        results.append(
                            {"page": page_num, "table_index": t_idx, "markdown": md}
                        )
    except Exception as e:
        logger.warning(f"Table extraction failed for {pdf_path}: {e}")
    return results


def _table_to_markdown(table: List[List[Any]]) -> str:
    rows = [[("" if c is None else str(c).replace("\n", " ").strip()) for c in row] for row in table]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        # pad/truncate row to header length to keep the markdown table valid
        row = (row + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_images(pdf_path: str, output_dir: str, doc_id: str) -> List[Dict[str, Any]]:
    """
    Extracts embedded images with PyMuPDF, saves each to `output_dir`, and
    (if tesseract is installed) OCRs the image to produce a text caption
    that can be embedded and searched like any other chunk.

    Returns [{"page": int, "image_path": str, "ocr_text": str}, ...]
    """
    results = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            image_list = page.get_images(full=True)
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image.get("ext", "png")
                except Exception as e:
                    logger.warning(f"Could not extract image xref {xref} on page {page_num}: {e}")
                    continue

                # Skip tiny decorative images (icons/logos) that add noise, not signal.
                try:
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    if pil_img.width < 120 or pil_img.height < 120:
                        continue
                except Exception:
                    pil_img = None

                fname = f"{doc_id}_p{page_num}_img{img_idx}.{ext}"
                fpath = out_dir / fname
                with open(fpath, "wb") as f:
                    f.write(image_bytes)

                ocr_text = ""
                if _OCR_AVAILABLE and pil_img is not None:
                    try:
                        ocr_text = pytesseract.image_to_string(pil_img).strip()
                    except Exception as e:
                        logger.warning(f"OCR failed for {fpath}: {e}")

                results.append(
                    {"page": page_num, "image_path": str(fpath), "ocr_text": ocr_text}
                )
    return results


def extract_all(pdf_path: str, image_output_dir: str, doc_id: str) -> Dict[str, Any]:
    """Convenience wrapper returning text + tables + images in one call."""
    return {
        "text_pages": extract_text_by_page(pdf_path),
        "tables": extract_tables(pdf_path),
        "images": extract_images(pdf_path, image_output_dir, doc_id),
    }
