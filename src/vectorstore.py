"""
Manages the persistent local Chroma vector store, and builds a hybrid
retriever that combines:
    - Dense vector search (Chroma, via HuggingFace embeddings)
    - Sparse keyword search (BM25, via rank_bm25 / langchain BM25Retriever)
using LangChain's EnsembleRetriever (reciprocal-rank fusion under the hood).

BM25 needs the full in-memory document collection (it is not something
Chroma persists for us), so we cache the ingested Documents to a pickle
file at ingestion time and reload them here on app startup.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        try:
            from langchain.retrievers.ensemble import EnsembleRetriever
        except ImportError:
            from langchain_community.retrievers import EnsembleRetriever
        
from langchain_core.documents import Document

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    BM25_STORE_PATH,
    VECTOR_TOP_K,
    BM25_TOP_K,
    HYBRID_VECTOR_WEIGHT,
    HYBRID_BM25_WEIGHT,
)
from src.embeddings import get_embedding_function


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=str(CHROMA_DIR),
    )


def save_bm25_corpus(all_docs: List[Document]) -> None:
    """Persist the full document set to disk so BM25 can be rebuilt without re-ingesting."""
    Path(BM25_STORE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_STORE_PATH, "wb") as f:
        pickle.dump(all_docs, f)


def load_bm25_corpus() -> List[Document]:
    if not Path(BM25_STORE_PATH).exists():
        return []
    with open(BM25_STORE_PATH, "rb") as f:
        return pickle.load(f)


def append_bm25_corpus(new_docs: List[Document]) -> None:
    """Used during incremental ingestion — merges new docs into the existing pickle."""
    existing = load_bm25_corpus()
    existing.extend(new_docs)
    save_bm25_corpus(existing)


def get_hybrid_retriever(vectorstore: Chroma | None = None, k: int | None = None):
    """
    Returns a LangChain EnsembleRetriever mixing dense (Chroma) and sparse
    (BM25) retrieval. Falls back gracefully to vector-only search if the
    BM25 corpus hasn't been built yet (e.g. fresh clone before ingestion).
    """
    vectorstore = vectorstore or get_vectorstore()
    top_k = k or VECTOR_TOP_K

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    bm25_docs = load_bm25_corpus()
    if not bm25_docs:
        # No cached corpus yet -> vector search only.
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = BM25_TOP_K

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[HYBRID_BM25_WEIGHT, HYBRID_VECTOR_WEIGHT],
    )


def get_all_scheme_names() -> List[str]:
    """Used by the UI sidebar and by the fallback-answer logic."""
    docs = load_bm25_corpus()
    names = sorted({d.metadata.get("scheme_name", "Unknown") for d in docs})
    return names


def delete_doc_chunks(vectorstore: Chroma, doc_id: str) -> int:
    """
    Removes every chunk belonging to `doc_id` from Chroma. Used when a PDF is
    re-ingested (e.g. an updated version) so stale chunks don't linger
    alongside the new ones. Returns the number of chunks deleted.
    """
    try:
        existing = vectorstore.get(where={"doc_id": doc_id})
        ids = existing.get("ids", [])
        if ids:
            vectorstore.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0
