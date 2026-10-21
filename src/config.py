"""
Central configuration for the Farmer Schemes RAG project.
Everything is read from environment variables (.env) with sane defaults,
so the project works out-of-the-box even if the user hasn't created a .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present (falls back to defaults otherwise)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

def _path(env_key: str, default: str) -> Path:
    return (BASE_DIR / os.getenv(env_key, default)).resolve()

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
PDF_DIR = _path("PDF_DIR", "./data/pdfs")
METADATA_DIR = _path("METADATA_DIR", "./data/metadata")
IMAGE_DIR = _path("IMAGE_DIR", "./data/images")
CHROMA_DIR = _path("CHROMA_DIR", "./chroma_db")
BM25_STORE_PATH = _path("BM25_STORE_PATH", "./storage/bm25_docs.pkl")

for d in (PDF_DIR, METADATA_DIR, IMAGE_DIR, CHROMA_DIR, BM25_STORE_PATH.parent):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "farmer_schemes")

# ---------------------------------------------------------------------------
# Models (open source, run locally)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# ---------------------------------------------------------------------------
# Retrieval / hybrid search / reranking
# ---------------------------------------------------------------------------
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", 8))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", 8))
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", 0.6))
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", 0.4))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 5))

# Below this normalized reranker confidence, the answer engine switches
# to its "low-confidence, best-effort" fallback strategy instead of
# retrieving nothing / refusing outright.
CONFIDENCE_SCORE_THRESHOLD = float(os.getenv("CONFIDENCE_SCORE_THRESHOLD", 0.15))
