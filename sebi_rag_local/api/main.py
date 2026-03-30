"""
SEBI RAG Query API
POST /query  →  classify → retrieve from correct collection → generate answer
"""

import os
import re
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

R2R_BASE = os.getenv("R2R_BASE_URL", "http://localhost:7272")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")

# ─── Collection routing ───────────────────────────────────────────────────────
COLLECTION_MAP = {
    "sebi_retail": [
        "retail investor", "mutual fund", "kyc", "demat", "ipo", "equity",
        "sip", "stock broker", "nse", "bse", "dividend",
    ],
    "sebi_aif": [
        "aif", "alternative investment fund", "venture capital", "hedge fund",
        "private equity", "cat i", "cat ii", "cat iii",
    ],
    "sebi_fpi": [
        "fpi", "foreign portfolio investor", "fii", "p-note", "overseas",
        "foreign institutional", "participatory note",
    ],
    "sebi_general": [],
}

ADVISORY_PATTERNS = [
    (r"you should invest\b", "I can only provide educational information, not investment advice."),
    (r"recommended allocation\b", "I can only provide educational information, not investment advice."),
    (r"i recommend (investing|buying|selling)", "I can only provide educational information, not investment advice."),
    (r"best investment\b", "I can only provide educational information, not investment advice."),
]

SYSTEM_PROMPT = (
    "You are a SEBI-aware financial information assistant.\n"
    "STRICT RULES:\n"
    "- DO NOT give investment advice\n"
    "- DO NOT suggest allocations or strategies\n"
    "- ALWAYS assume user is an Indian retail investor\n"
    "YOU MAY:\n"
    "- Explain asset classes\n"
    "- Explain risks\n"
    "- Explain regulations\n"
    "If user asks for advice: Refuse politely and provide general information only."
)


# ─── Models ───────────────────────────────────────────────────────────────────
class SourceChunk(BaseModel):
    document_id: str
    chunk_id: str
    score: float
    text_preview: str
    metadata: dict


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    collection: str
    sources: list[SourceChunk]


app = FastAPI(title="SEBI RAG API", version="1.0.0")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def classify_query(query: str) -> str:
    """Route query to the most relevant SEBI collection."""
    q = query.lower()
    for collection, keywords in COLLECTION_MAP.items():
        if any(kw in q for kw in keywords):
            return collection
    return "sebi_general"


def apply_safety_filter(text: str) -> str:
    """Replace any investment-advice-like phrases."""
    for pattern, replacement in ADVISORY_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


async def get_collection_id(client: httpx.AsyncClient, name: str) -> str | None:
    """Resolve collection name → R2R collection id."""
    resp = await client.get(f"{R2R_BASE}/v3/collections", params={"limit": 100})
    resp.raise_for_status()
    for col in resp.json().get("results", []):
        if col.get("name") == name:
            return col["id"]
    return None


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_sebi(req: QueryRequest):
    collection_name = classify_query(req.query)

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Resolve collection id
            col_id = await get_collection_id(client, collection_name)
            if col_id is None:
                # Fall back to sebi_general
                col_id = await get_collection_id(client, "sebi_general")

            # Build RAG request
            rag_payload = {
                "query": req.query,
                "rag_generation_config": {
                    "model": "openai/qwen2.5:3b-instruct",
                    "temperature": 0.1,
                    "max_tokens_to_sample": 512,
                    "stream": False,
                    "api_base": OLLAMA_BASE,
                },
                "vector_search_settings": {
                    "search_limit": req.top_k,
                    "use_hybrid_search": True,
                },
                "system_prompt": SYSTEM_PROMPT,
            }

            if col_id:
                rag_payload["collection_ids"] = [col_id]

            resp = await client.post(f"{R2R_BASE}/v3/retrieval/rag", json=rag_payload)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)

            data = resp.json()
            raw_answer = data.get("results", {}).get("generated_answer", "")
            safe_answer = apply_safety_filter(raw_answer)

            # Extract source chunks
            sources = []
            for chunk in data.get("results", {}).get("search_results", {}).get("chunk_search_results", []):
                sources.append(SourceChunk(
                    document_id=chunk.get("document_id", ""),
                    chunk_id=chunk.get("id", ""),
                    score=chunk.get("score", 0.0),
                    text_preview=chunk.get("text", "")[:200],
                    metadata=chunk.get("metadata", {}),
                ))

    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        raise HTTPException(status_code=503, detail="Upstream service unavailable") from exc

    return QueryResponse(answer=safe_answer, collection=collection_name, sources=sources)
