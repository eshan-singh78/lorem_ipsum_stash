# SEBI Local RAG System

Full local RAG stack: R2R + Ollama + PostgreSQL/pgvector + custom query API.

## Folder structure

```
sebi_rag_local/
├── docker-compose.yml          # All services
├── config/
│   └── sebi_ollama.toml        # R2R config (Ollama, chunking, hybrid search)
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py                 # FastAPI query layer (POST /query)
├── ingestion/
│   └── ingest.py               # PDF ingestion script
└── scripts/
    └── setup_collections.sh    # Creates the 4 SEBI collections
```

## Services & ports

| Service         | Port  | Description                        |
|-----------------|-------|------------------------------------|
| R2R Engine      | 7272  | Core RAG API                       |
| R2R Dashboard   | 3000  | Web UI                             |
| Ollama          | 11434 | Local LLM + embeddings             |
| PostgreSQL      | 5432  | Vector store (pgvector)            |
| SEBI Query API  | 8000  | Custom POST /query endpoint        |

## Quick start

### 1. Clone repos (optional — we use the published Docker images)

```bash
# Only needed if you want to build from source
git clone https://github.com/SciPhi-AI/R2R
git clone https://github.com/SciPhi-AI/R2R-Application
```

### 2. Start the stack

```bash
cd sebi_rag_local
docker compose up -d
```

The `ollama-pull` container will automatically pull `qwen2.5:3b-instruct` and
`nomic-embed-text` on first run. This takes a few minutes depending on your
connection. Watch progress with:

```bash
docker logs -f sebi_ollama_pull
```

### 3. Create collections

```bash
bash scripts/setup_collections.sh
```

### 4. Ingest PDFs

```bash
# Install ingestion deps locally (or run inside a venv)
pip install httpx

# Ingest into a specific collection
python ingestion/ingest.py --pdf_dir /path/to/sebi_retail_pdfs --collection sebi_retail
python ingestion/ingest.py --pdf_dir /path/to/sebi_aif_pdfs    --collection sebi_aif
python ingestion/ingest.py --pdf_dir /path/to/sebi_fpi_pdfs    --collection sebi_fpi
python ingestion/ingest.py --pdf_dir /path/to/general_pdfs     --collection sebi_general
```

### 5. Query via API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the KYC requirements for retail investors in India?"}'
```

Response:
```json
{
  "answer": "According to SEBI regulations, KYC (Know Your Customer) ...",
  "collection": "sebi_retail",
  "sources": [
    {
      "document_id": "...",
      "chunk_id": "...",
      "score": 0.91,
      "text_preview": "...",
      "metadata": { "page_number": 3, "filename": "..." }
    }
  ]
}
```

### 6. Query via R2R Dashboard UI

Open http://localhost:3000 in your browser.
Default login: `admin@example.com` / `change_me_immediately`

### 7. Query directly via R2R API

```bash
curl -X POST http://localhost:7272/v3/retrieval/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain SEBI AIF regulations",
    "vector_search_settings": { "search_limit": 5, "use_hybrid_search": true }
  }'
```

## Chunking strategy

| Setting              | Value                          |
|----------------------|--------------------------------|
| Strategy             | `hi_res` + `by_title`          |
| Chunk size           | ~800 tokens (3200 chars)       |
| Overlap              | ~150 tokens (600 chars)        |
| Split on             | Numbered sections, headings    |
| Metadata             | filename, page_number, chunk_id|

## Models

| Role        | Model                    |
|-------------|--------------------------|
| Generation  | `qwen2.5:3b-instruct`    |
| Embeddings  | `nomic-embed-text` (768d)|

## Safety layer

The `/query` endpoint post-processes all LLM output and replaces any
investment-advice-like phrases (e.g. "you should invest", "recommended
allocation") with a standard disclaimer.

## Stopping the stack

```bash
docker compose down
# To also remove volumes (wipes DB + model cache):
docker compose down -v
```
