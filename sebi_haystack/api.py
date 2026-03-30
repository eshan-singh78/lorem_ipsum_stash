"""
api.py — FastAPI layer for the SEBI Haystack RAG system.

Endpoints:
    GET  /health
    POST /query  →  { answer, category, sources }
"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pipeline import ask, classify_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="SEBI RAG API (Haystack)", version="2.0.0")


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    category: str | None = Field(default=None)


class SourceChunk(BaseModel):
    source: str
    page: int
    score: float
    preview: str


class QueryResponse(BaseModel):
    answer: str
    category: str
    sources: list[SourceChunk]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_sebi(req: QueryRequest):
    try:
        result = ask(query=req.query, top_k=req.top_k, category=req.category)
    except Exception as exc:
        logger.exception("RAG pipeline error: %s", exc)
        raise HTTPException(status_code=503, detail="RAG pipeline unavailable") from exc

    return QueryResponse(
        answer=result["answer"],
        category=result["category"],
        sources=[SourceChunk(**s) for s in result["sources"]],
    )
