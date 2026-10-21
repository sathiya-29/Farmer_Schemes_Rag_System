"""
Quick sanity check you can run after `pip install -r requirements.txt` to
confirm your local environment is ready before ingesting/running the app.

Usage:
    python check_setup.py
"""
import shutil
import sys


def check(name, ok, hint=""):
    status = "OK" if ok else "MISSING"
    print(f"[{status:7}] {name}" + (f"  -> {hint}" if not ok and hint else ""))
    return ok


def main():
    all_ok = True

    try:
        import langchain  # noqa
        all_ok &= check("langchain", True)
    except ImportError:
        all_ok &= check("langchain", False, "pip install -r requirements.txt")

    try:
        import chromadb  # noqa
        all_ok &= check("chromadb", True)
    except ImportError:
        all_ok &= check("chromadb", False, "pip install -r requirements.txt")

    try:
        import sentence_transformers  # noqa
        all_ok &= check("sentence-transformers", True)
    except ImportError:
        all_ok &= check("sentence-transformers", False, "pip install -r requirements.txt")

    try:
        import fitz  # PyMuPDF  # noqa
        all_ok &= check("PyMuPDF (fitz)", True)
    except ImportError:
        all_ok &= check("PyMuPDF (fitz)", False, "pip install -r requirements.txt")

    try:
        import pdfplumber  # noqa
        all_ok &= check("pdfplumber", True)
    except ImportError:
        all_ok &= check("pdfplumber", False, "pip install -r requirements.txt")

    tesseract_ok = shutil.which("tesseract") is not None
    check(
        "tesseract-ocr binary (optional, for image OCR captions)",
        tesseract_ok,
        "Install via apt/brew: 'sudo apt-get install tesseract-ocr' -- app still works without it.",
    )

    ollama_ok = shutil.which("ollama") is not None
    check(
        "ollama binary (required for local LLM generation)",
        ollama_ok,
        "Install from https://ollama.com then run: ollama pull llama3.1",
    )

    print()
    if all_ok:
        print("Python dependencies look good. Next: `python -m src.ingest` then `streamlit run app.py`.")
    else:
        print("Some Python dependencies are missing — run: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
