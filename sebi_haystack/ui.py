"""
ui.py — Streamlit demo UI for the SEBI RAG system.

Run:
    streamlit run ui.py
"""

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="SEBI RAG Assistant", page_icon="📋", layout="wide")
st.title("📋 SEBI Regulatory Assistant")
st.caption("Answers grounded in SEBI documents only. Not investment advice.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=5)
    category_override = st.selectbox(
        "Force category (optional)",
        options=["Auto-detect", "sebi_retail", "sebi_aif", "sebi_fpi", "sebi_general"],
    )
    show_chunks = st.checkbox("Show retrieved chunks", value=True)

# ── Query input ───────────────────────────────────────────────────────────────
query = st.text_area("Ask a SEBI regulation question:", height=100,
                     placeholder="e.g. What are the KYC requirements for retail investors?")

if st.button("Ask", type="primary") and query.strip():
    payload: dict = {"query": query.strip(), "top_k": top_k}
    if category_override != "Auto-detect":
        payload["category"] = category_override

    with st.spinner("Retrieving and generating answer…"):
        try:
            resp = httpx.post(f"{API_URL}/query", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            st.error(f"API error {e.response.status_code}: {e.response.text}")
            st.stop()
        except httpx.RequestError as e:
            st.error(f"Could not reach API at {API_URL}: {e}")
            st.stop()

    # ── Answer ────────────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.info(data["answer"])
    st.caption(f"Routed to collection: `{data['category']}`")

    # ── Sources ───────────────────────────────────────────────────────────────
    if show_chunks and data.get("sources"):
        st.subheader(f"Retrieved chunks ({len(data['sources'])})")
        for i, src in enumerate(data["sources"], 1):
            with st.expander(f"[{i}] {src['source']}  —  page {src['page']}  (score: {src['score']:.4f})"):
                st.write(src["preview"])
