"""
rag_pipeline.py — Haystack RAG pipeline: embed query → retrieve → generate.

Exposes a single function:
    ask(query: str, top_k: int = 5, category: str | None = None) -> dict
"""

import logging
import os
import re

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# ── Query router ──────────────────────────────────────────────────────────────

COLLECTION_KEYWORDS: dict[str, list[str]] = {
    "sebi_retail": [
        "retail investor", "mutual fund", "kyc", "demat", "ipo", "equity",
        "sip", "stock broker", "nse", "bse", "dividend", "salary", "portfolio",
        "advisor",
    ],
    "sebi_aif": [
        "aif", "alternative investment fund", "venture capital", "hedge fund",
        "private equity",
    ],
    "sebi_fpi": [
        "fpi", "foreign portfolio investor", "fii", "p-note", "overseas",
        "foreign institutional",
    ],
}


def classify_query(query: str) -> str:
    q = query.lower()
    for category, keywords in COLLECTION_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return category
    return "sebi_general"


# ── Safety filter ─────────────────────────────────────────────────────────────

_ADVISORY_PATTERNS = [
    (r"you should invest\b", "I can only provide educational information, not investment advice."),
    (r"recommended allocation\b", "I can only provide educational information, not investment advice."),
    (r"i recommend (investing|buying|selling)", "I can only provide educational information, not investment advice."),
    (r"best investment\b", "I can only provide educational information, not investment advice."),
]


def apply_safety_filter(text: str) -> str:
    for pattern, replacement in _ADVISORY_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a SEBI regulatory assistant. Answer ONLY based on the provided context.

STRICT RULES:
- DO NOT provide investment advice
- DO NOT suggest allocations or strategies
- ONLY explain regulations and risks
- ALWAYS assume the user is an Indian retail investor
- If the answer is not found in the context, say exactly:
  "Information not found in SEBI documents."

Context:
{% for doc in documents %}
[Source: {{ doc.meta.source }}, Page {{ doc.meta.page }}]
{{ doc.content }}
---
{% endfor %}

Question: {{ query }}
Answer:"""


# ── Pipeline factory ──────────────────────────────────────────────────────────

def _build_pipeline(store: ChromaDocumentStore, top_k: int) -> Pipeline:
    pipe = Pipeline()

    pipe.add_component("embedder", OllamaTextEmbedder(
        model=EMBED_MODEL,
        url=f"{OLLAMA_URL}/api/embed",
    ))
    pipe.add_component("retriever", ChromaEmbeddingRetriever(
        document_store=store,
        top_k=top_k,
    ))
    pipe.add_component("prompt_builder", PromptBuilder(template=SYSTEM_PROMPT))
    pipe.add_component("llm", OllamaGenerator(
        model=LLM_MODEL,
        url=f"{OLLAMA_URL}/api/generate",
        generation_kwargs={
            "temperature": 0.1,
            "num_predict": 512,
        },
    ))

    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    return pipe


# ── Singleton store + pipeline ────────────────────────────────────────────────

_store: ChromaDocumentStore | None = None
_pipeline: Pipeline | None = None


def _get_pipeline(top_k: int = 5) -> tuple[Pipeline, ChromaDocumentStore]:
    global _store, _pipeline
    if _store is None:
        _store = ChromaDocumentStore(
            persist_path=CHROMA_DB_PATH,
            collection_name="sebi_docs",
            embedding_function=None,
        )
    if _pipeline is None:
        _pipeline = _build_pipeline(_store, top_k)
    return _pipeline, _store


# ── Public API ────────────────────────────────────────────────────────────────

def ask(query: str, top_k: int = 5, category: str | None = None) -> dict:
    """
    Run the RAG pipeline for a query.

    Returns:
        {
            "answer": str,
            "category": str,
            "sources": [{"source": str, "page": int, "score": float, "preview": str}]
        }
    """
    resolved_category = category or classify_query(query)
    pipe, store = _get_pipeline(top_k)

    # Build metadata filter for ChromaDB
    filters = {"field": "meta.category", "operator": "==", "value": resolved_category}

    result = pipe.run(
        {
            "embedder": {"text": query},
            "retriever": {"filters": filters, "top_k": top_k},
            "prompt_builder": {"query": query},
        }
    )

    docs = result.get("retriever", {}).get("documents", [])
    raw_answer = result.get("llm", {}).get("replies", [""])[0]

    # Log retrieved chunks for debugging
    if docs:
        logger.debug("Retrieved %d chunks for query: %r", len(docs), query[:80])
        for i, doc in enumerate(docs):
            score = doc.score if doc.score is not None else 0.0
            logger.debug("  [%d] score=%.4f source=%s page=%s",
                         i + 1, score, doc.meta.get("source"), doc.meta.get("page"))
    else:
        logger.warning("No chunks retrieved for query: %r", query[:80])
        raw_answer = "No relevant SEBI regulation found."

    # Check similarity threshold — if best score is too low, say so
    if docs:
        best_score = max((d.score or 0.0) for d in docs)
        if best_score < SIMILARITY_THRESHOLD:
            logger.warning("Best score %.4f below threshold %.4f", best_score, SIMILARITY_THRESHOLD)
            raw_answer = "No relevant SEBI regulation found."

    safe_answer = apply_safety_filter(raw_answer)

    sources = [
        {
            "source": doc.meta.get("source", ""),
            "page": doc.meta.get("page", 0),
            "score": round(doc.score or 0.0, 4),
            "preview": doc.content[:300] if doc.content else "",
        }
        for doc in docs
    ]

    return {
        "answer": safe_answer,
        "category": resolved_category,
        "sources": sources,
    }
