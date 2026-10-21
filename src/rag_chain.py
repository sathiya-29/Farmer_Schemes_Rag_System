"""
The end-to-end answer pipeline:

    query
      -> hybrid retrieval (BM25 + Chroma vector search)
      -> cross-encoder reranking
      -> confidence check
           - HIGH confidence -> answer normally, grounded strictly in the
             retrieved chunks, with citations.
           - LOW / NO confidence -> do NOT just say "I don't know". Instead:
               1. Widen the search (more candidates, no scheme filter).
               2. If something loosely relevant turns up, answer from it but
                  clearly flag reduced confidence.
               3. If truly nothing relevant exists, tell the user plainly,
                  AND proactively list which schemes actually ARE in the
                  knowledge base + suggest a rephrase, so they aren't stuck
                  with a dead end.

This module never returns a bare "I don't know" — it always tries to be
maximally useful with whatever the retriever could find.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    CONFIDENCE_SCORE_THRESHOLD,
    RERANK_TOP_N,
)
from src.vectorstore import get_vectorstore, get_hybrid_retriever, get_all_scheme_names
from src.reranker import rerank

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """You are an expert assistant on Indian government agriculture \
and farmer welfare schemes. You answer ONLY using the CONTEXT provided below, which was \
retrieved from official scheme guideline documents.

Rules:
1. Be specific: cite scheme name, numbers, eligibility, cost norms, subsidy percentages, \
deadlines, and procedural steps exactly as given in the context. Do not invent figures.
2. If the context fully answers the question, answer confidently and clearly, using bullet \
points for lists (eligibility, benefits, documents, steps).
3. If the context only PARTIALLY answers the question, answer with what IS available, then \
explicitly state what is missing and suggest how the user can find that specific detail \
(e.g. "check the official portal" or "this isn't covered in the ingested documents").
4. If the context is only loosely related to the question, do NOT simply refuse. Instead, \
explain what related information IS available, and clearly state the exact question could \
not be fully answered from the current documents.
5. Never fabricate scheme names, numbers, or eligibility criteria that are not in the context.
6. Keep answers well-organized and easy to scan.
"""

_llm_singleton = None


def get_llm() -> Ollama:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    return _llm_singleton


def _format_context(scored_docs: List[Tuple[Document, float]]) -> str:
    blocks = []
    for i, (doc, score) in enumerate(scored_docs, start=1):
        meta = doc.metadata
        header = (
            f"[Source {i} | Scheme: {meta.get('scheme_name')} | "
            f"Type: {meta.get('chunk_type')} | Page: {meta.get('page')} | "
            f"Relevance: {score:.2f}]"
        )
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _build_prompt(question: str, context: str, low_confidence: bool, available_schemes: List[str]) -> str:
    confidence_note = ""
    if low_confidence:
        scheme_list = ", ".join(available_schemes[:15]) if available_schemes else "none ingested yet"
        confidence_note = f"""
NOTE: The retrieved context below has LOW relevance confidence for this exact question. \
Do your best with what's available, be upfront that this may not fully answer the question, \
and mention that the following schemes ARE covered in the knowledge base in case one of them \
is what the user actually meant: {scheme_list}.
"""

    return f"""{SYSTEM_INSTRUCTIONS}
{confidence_note}
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def answer_query(question: str, k: int = 15) -> Dict[str, Any]:
    """
    Returns:
        {
          "answer": str,
          "sources": [ {scheme_name, page, chunk_type, image_path?, score}, ... ],
          "confidence": "high" | "low" | "none",
        }
    """
    vectorstore = get_vectorstore()
    retriever = get_hybrid_retriever(vectorstore, k=k)

    try:
        candidates: List[Document] = retriever.invoke(question)
    except Exception as e:
        logger.exception(f"Retrieval failed: {e}")
        candidates = []

    scored = rerank(question, candidates, top_n=RERANK_TOP_N)

    confidence = "none"
    if scored:
        top_score = scored[0][1]
        confidence = "high" if top_score >= CONFIDENCE_SCORE_THRESHOLD else "low"

    # --- Fallback path: widen the net if nothing useful came back ---------
    if confidence == "none":
        logger.info("No candidates from hybrid retriever, widening search...")
        try:
            wide_candidates = vectorstore.similarity_search(question, k=25)
        except Exception:
            wide_candidates = []
        scored = rerank(question, wide_candidates, top_n=RERANK_TOP_N)
        if scored:
            confidence = "low" if scored[0][1] < CONFIDENCE_SCORE_THRESHOLD else "high"

    available_schemes = get_all_scheme_names()

    if not scored:
        # Truly nothing in the DB is related — be transparent AND useful.
        scheme_list = ", ".join(available_schemes) if available_schemes else None
        if scheme_list:
            answer = (
                "I couldn't find information in the ingested documents that directly answers "
                "this question. The knowledge base currently only contains details on the "
                f"following scheme(s): {scheme_list}. Could you rephrase your question around "
                "one of these, or upload the relevant scheme PDF and metadata so I can index it?"
            )
        else:
            answer = (
                "No documents have been ingested into the knowledge base yet, so I can't answer "
                "this question. Please run the ingestion pipeline (`python -m src.ingest`) after "
                "adding scheme PDFs to `data/pdfs/` and matching metadata to `data/metadata/`."
            )
        return {"answer": answer, "sources": [], "confidence": "none"}

    context = _format_context(scored)
    prompt = _build_prompt(question, context, low_confidence=(confidence == "low"), available_schemes=available_schemes)

    try:
        llm = get_llm()
        raw_answer = llm.invoke(prompt)
    except Exception as e:
        logger.exception(f"LLM generation failed: {e}")
        raw_answer = (
            "⚠️ Could not reach the local LLM (Ollama). Showing the most relevant retrieved "
            "passages instead so you still get useful information:\n\n" + context[:2000]
        )

    sources = []
    for doc, score in scored:
        meta = doc.metadata
        src = {
            "scheme_name": meta.get("scheme_name"),
            "page": meta.get("page"),
            "chunk_type": meta.get("chunk_type"),
            "score": round(float(score), 3),
            "source_file": meta.get("source_file"),
        }
        if meta.get("chunk_type") == "image":
            src["image_path"] = meta.get("image_path")
        sources.append(src)

    return {"answer": raw_answer, "sources": sources, "confidence": confidence}
