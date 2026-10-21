"""
Open-source, fully local embedding function used for both ingestion and
querying. sentence-transformers/all-MiniLM-L6-v2 is small (~80MB), fast on
CPU, and downloads once from HuggingFace on first use (no API key needed).
"""
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL

_embedding_singleton = None


def get_embedding_function() -> HuggingFaceEmbeddings:
    global _embedding_singleton
    if _embedding_singleton is None:
        _embedding_singleton = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_singleton
