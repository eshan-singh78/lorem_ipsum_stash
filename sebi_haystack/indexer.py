"""
indexer.py — Load pre-chunked JSONL files into ChromaDB via Haystack.

Usage:
    python indexer.py --chunks_dir ../sebiv2/chunks --db_path ./chroma_db

Reads every *.jsonl file produced by pipeline1.py and embeds each chunk
using Ollama nomic-embed-text, storing vectors + metadata in ChromaDB.
"""

import argparse
import json
import logging
import os
from pathlib import Path

from haystack import Document
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def build_store(db_path: str) -> ChromaDocumentStore:
    return ChromaDocumentStore(
        persist_path=db_path,
        collection_name="sebi_docs",
        embedding_function=None,  # Haystack handles embeddings
    )


def load_chunks(chunks_dir: str) -> list[Document]:
    docs: list[Document] = []
    jsonl_files = sorted(Path(chunks_dir).glob("*.jsonl"))
    logger.info("Found %d JSONL files in %s", len(jsonl_files), chunks_dir)

    for path in jsonl_files:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed line %d in %s: %s", lineno, path.name, e)
                    continue

                meta = chunk.get("metadata", {})
                docs.append(Document(
                    id=chunk["id"],
                    content=chunk["text"],
                    meta={
                        "source": meta.get("source", path.name),
                        "page": meta.get("page", 0),
                        "category": meta.get("category", "sebi_general"),
                        "chunk_id": meta.get("chunk_id", chunk["id"]),
                    },
                ))

    logger.info("Loaded %d chunks total", len(docs))
    return docs


def run_indexer(chunks_dir: str, db_path: str, batch_size: int = 32) -> None:
    store = build_store(db_path)

    embedder = OllamaDocumentEmbedder(
        model=EMBED_MODEL,
        url=f"{OLLAMA_URL}/api/embed",
        batch_size=batch_size,
    )

    writer = DocumentWriter(document_store=store, policy=DuplicatePolicy.SKIP)

    docs = load_chunks(chunks_dir)
    if not docs:
        logger.warning("No documents to index.")
        return

    logger.info("Embedding %d documents (batch_size=%d)…", len(docs), batch_size)
    embed_result = embedder.run(documents=docs)
    embedded_docs = embed_result["documents"]

    logger.info("Writing to ChromaDB at %s…", db_path)
    writer.run(documents=embedded_docs)
    logger.info("✅ Indexed %d documents into ChromaDB", len(embedded_docs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index SEBI chunks into ChromaDB")
    parser.add_argument("--chunks_dir", default=os.getenv("CHUNKS_DIR", "../sebiv2/chunks"))
    parser.add_argument("--db_path", default=os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_indexer(args.chunks_dir, args.db_path, args.batch_size)
