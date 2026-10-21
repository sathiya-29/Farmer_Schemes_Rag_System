"""
Local, open-source cross-encoder reranker. After hybrid retrieval pulls a
candidate pool of chunks (recall-focused, sometimes noisy), the reranker
scores each (query, chunk) pair jointly for much higher precision than
either BM25 or embedding-cosine-similarity alone, then we keep only the
top-N most relevant chunks to pass to the LLM.
"""
from __future__ import annotations

from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from src.config import RERANKER_MODEL, RERANK_TOP_N

_reranker_singleton = None


def get_reranker() -> CrossEncoder:
    global _reranker_singleton
    if _reranker_singleton is None:
        _reranker_singleton = CrossEncoder(RERANKER_MODEL)
    return _reranker_singleton


def rerank(query: str, documents: List[Document], top_n: int = None) -> List[Tuple[Document, float]]:
    """
    Returns a list of (Document, score) tuples, sorted by descending
    relevance score, truncated to `top_n`. Scores are raw CrossEncoder
    logits passed through a sigmoid so they land roughly in [0, 1].
    """
    if not documents:
        return []

    top_n = top_n or RERANK_TOP_N
    model = get_reranker()

    # De-duplicate near-identical chunks (can happen when BM25 + vector both
    # return the same chunk) before we spend compute reranking them.
    seen = set()
    deduped = []
    for d in documents:
        key = (d.metadata.get("doc_id"), d.metadata.get("page"), d.page_content[:120])
        if key not in seen:
            seen.add(key)
            deduped.append(d)

    pairs = [[query, d.page_content] for d in deduped]
    raw_scores = model.predict(pairs)

    # sigmoid normalize
    import math
    scores = [1 / (1 + math.exp(-s)) for s in raw_scores]

    scored = list(zip(deduped, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]
