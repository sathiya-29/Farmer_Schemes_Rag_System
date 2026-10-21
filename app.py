"""
Streamlit frontend for the Farmer Schemes RAG assistant.

Run with:
    streamlit run app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from src.config import METADATA_DIR, PDF_DIR
from src.vectorstore import get_all_scheme_names
from src.rag_chain import answer_query

st.set_page_config(page_title="Farmer Schemes Assistant", page_icon="🌾", layout="wide")

# --------------------------------------------------------------------------
# Sidebar: knowledge base overview
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🌾 Knowledge Base")
    st.caption("Local, fully offline RAG over Indian farmer/agriculture scheme PDFs.")

    schemes = get_all_scheme_names()
    st.metric("Ingested schemes", len(schemes))

    if schemes:
        with st.expander("Schemes currently indexed", expanded=True):
            for s in schemes:
                st.markdown(f"- {s}")
    else:
        st.warning(
            "No documents ingested yet.\n\n"
            "Add PDFs to `data/pdfs/` and matching metadata JSON to "
            "`data/metadata/`, then run:\n\n`python -m src.ingest`"
        )

    st.divider()
    st.subheader("Raw metadata files")
    meta_files = sorted(Path(METADATA_DIR).glob("*.json"))
    if meta_files:
        chosen = st.selectbox("Inspect a metadata file", [m.name for m in meta_files])
        if chosen:
            with open(Path(METADATA_DIR) / chosen, "r", encoding="utf-8") as f:
                st.json(json.load(f))
    else:
        st.caption("No metadata files found yet.")

    st.divider()
    st.caption(
        "Pipeline: Hybrid Search (BM25 + Chroma vectors) → Cross-Encoder "
        "Reranker → Local LLM (Ollama). Everything runs on your machine."
    )

# --------------------------------------------------------------------------
# Main chat area
# --------------------------------------------------------------------------
st.title("🌾 Farmer Schemes Assistant")
st.caption(
    "Ask about eligibility, benefits, subsidy amounts, required documents, "
    "or how to apply for any ingested agriculture / farmer welfare scheme."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources = msg["sources"]
            with st.expander(f"📚 Sources ({len(_render_sources)}) — confidence: {msg.get('confidence','n/a')}"):
                for src in _render_sources:
                    st.markdown(
                        f"**{src['scheme_name']}** — page {src['page']} · "
                        f"type: `{src['chunk_type']}` · relevance score: {src['score']}"
                    )
                    if src.get("image_path") and Path(src["image_path"]).exists():
                        st.image(src["image_path"], width=300)

query = st.chat_input("e.g. What is the interest subvention under the Agriculture Infrastructure Fund?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching schemes and generating answer..."):
            result = answer_query(query)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander(f"📚 Sources ({len(result['sources'])}) — confidence: {result['confidence']}"):
                for src in result["sources"]:
                    st.markdown(
                        f"**{src['scheme_name']}** — page {src['page']} · "
                        f"type: `{src['chunk_type']}` · relevance score: {src['score']}"
                    )
                    if src.get("image_path") and Path(src["image_path"]).exists():
                        st.image(src["image_path"], width=300)
        elif result["confidence"] == "none":
            st.info(
                "No matching content was found. Try rephrasing, or check the sidebar "
                "for which schemes are currently ingested."
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
        }
    )
