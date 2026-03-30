# SEBI RAG System — Haystack + Ollama + ChromaDB

Local RAG pipeline for ~100 SEBI PDF regulations.  
No cloud. No R2R. Pure Haystack + Ollama + ChromaDB.

## Stack

| Component   | Tool                          |
|-------------|-------------------------------|
| Chunking    | `sebiv2/pipeline1.py` (done)  |
| Embeddings  | Ollama `nomic-embed-text`     |
| Vector DB   | ChromaDB (local file)         |
| Retrieval   | Haystack `ChromaEmbeddingRetriever` |
| Generation  | Ollama `qwen2.5:3b-instruct`  |
| API         | FastAPI (`api.py`)            |
| UI          | Streamlit (`ui.py`)           |

## Prerequisites

1. **Ollama running locally**
   ```bash
   ollama serve
   ollama pull nomic-embed-text
   ollama pull qwen2.5:3b-instruct
   ```

2. **Python 3.10+** with a virtual environment
   ```bash
   cd sebi_haystack
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Step 1 — Chunk PDFs (already done if `sebiv2/chunks/` is populated)

```bash
cd sebiv2
python pipeline1.py
# Chunks written to sebiv2/chunks/*.jsonl
```

## Step 2 — Index chunks into ChromaDB

```bash
cd sebi_haystack
python indexer.py --chunks_dir ../sebiv2/chunks --db_path ./chroma_db
```

This embeds every chunk with `nomic-embed-text` and stores vectors in `./chroma_db`.  
Takes ~5–15 min for 100 PDFs depending on hardware.

## Step 3 — Start the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Query via curl

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the KYC requirements for retail investors?"}'
```

Response:
```json
{
  "answer": "According to SEBI regulations...",
  "category": "sebi_retail",
  "sources": [
    { "source": "1492004818999.pdf", "page": 3, "score": 0.87, "preview": "..." }
  ]
}
```

## Step 4 — Start the Streamlit UI (optional)

```bash
streamlit run ui.py
```

Open http://localhost:8501

## Environment variables

| Variable             | Default                  | Description                    |
|----------------------|--------------------------|--------------------------------|
| `OLLAMA_URL`         | `http://localhost:11434` | Ollama base URL                |
| `EMBED_MODEL`        | `nomic-embed-text`       | Embedding model                |
| `LLM_MODEL`          | `qwen2.5:3b-instruct`    | Generation model               |
| `CHROMA_DB_PATH`     | `./chroma_db`            | ChromaDB persistence directory |
| `SIMILARITY_THRESHOLD` | `0.3`                  | Min score; below → "not found" |
| `CHUNKS_DIR`         | `../sebiv2/chunks`       | Input for indexer              |

## Query routing

Queries are automatically routed to the most relevant collection:

| Keyword in query                          | Collection     |
|-------------------------------------------|----------------|
| advisor, retail investor, kyc, salary…    | `sebi_retail`  |
| aif, alternative investment fund…         | `sebi_aif`     |
| fpi, foreign portfolio investor…          | `sebi_fpi`     |
| anything else                             | `sebi_general` |

## Safety

All LLM output is post-processed to replace investment-advice phrases with a
standard disclaimer. The system prompt strictly forbids advice generation.
