"""
SEBI Chunk Ingestion Script
Usage:
  python ingest.py --chunks_dir ../../sebiv2/chunks --r2r_url http://localhost:7272

Set INGEST_CONCURRENCY env var to control parallel threads (default: 4).
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

R2R_BASE = os.getenv("R2R_BASE_URL", "http://localhost:7272")
COLLECTIONS = ["sebi_retail", "sebi_aif", "sebi_fpi", "sebi_general"]
CONCURRENCY = int(os.getenv("INGEST_CONCURRENCY", "4"))


def get_or_create_collection(client: httpx.Client, name: str) -> str:
    """Return existing collection id or create a new one. Idempotent."""
    resp = client.get("/v3/collections", params={"limit": 100})
    resp.raise_for_status()
    for col in resp.json().get("results", []):
        if col.get("name") == name:
            logger.info("Reusing collection '%s': %s", name, col["id"])
            return col["id"]
    resp = client.post("/v3/collections", json={"name": name})
    resp.raise_for_status()
    col_id = resp.json()["results"]["id"]
    logger.info("Created collection '%s': %s", name, col_id)
    return col_id


def _make_client(r2r_url: str) -> httpx.Client:
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=10.0)
    return httpx.Client(base_url=r2r_url, timeout=timeout)


def ingest_chunk(r2r_url: str, chunk: dict, collection_id: str) -> bool:
    """POST one chunk to /v3/documents. Each call creates its own client (thread-safe)."""
    form_data = {
        "raw_text": chunk["text"],
        "id": chunk["id"],
        "metadata": json.dumps(chunk["metadata"]),
        "collection_ids": json.dumps([collection_id]),
        "run_with_orchestration": "false",
    }
    with _make_client(r2r_url) as client:
        resp = client.post("/v3/documents", data=form_data)

    if resp.status_code in (200, 201, 202):
        return True

    logger.error("Failed chunk %s: HTTP %s — %s",
                 chunk.get("id", "?"), resp.status_code, resp.text[:300])
    return False


def ingest_jsonl_file(r2r_url: str, jsonl_path: str, collection_map: dict) -> tuple[int, int]:
    """Read JSONL and ingest all chunks concurrently. Returns (success, fail)."""
    chunks, fail = [], 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.error("Malformed line %d in %s: %s", lineno, jsonl_path, exc)
                fail += 1

    if not chunks:
        return 0, fail

    success = 0

    def _submit(chunk):
        category = chunk.get("metadata", {}).get("category", "sebi_general")
        col_id = collection_map.get(category)
        if not col_id:
            logger.error("Unknown category '%s' — skipping chunk %s", category, chunk.get("id"))
            return False
        return ingest_chunk(r2r_url, chunk, col_id)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(_submit, c): c for c in chunks}
        for fut in as_completed(futures):
            if fut.result():
                success += 1
            else:
                fail += 1

    return success, fail


def main(chunks_dir: str, r2r_url: str, limit: int = 0) -> None:
    logger.info("Connecting to R2R at %s  (concurrency=%d)", r2r_url, CONCURRENCY)

    with _make_client(r2r_url) as client:
        collection_map = {name: get_or_create_collection(client, name) for name in COLLECTIONS}

    jsonl_files = sorted(Path(chunks_dir).glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No JSONL files found in %s", chunks_dir)
        return

    logger.info("Found %d JSONL files", len(jsonl_files))

    total_ok = total_fail = 0
    chunks_remaining = limit if limit > 0 else float("inf")

    for i, path in enumerate(jsonl_files, 1):
        if chunks_remaining <= 0:
            break
        logger.info("[%d/%d] %s", i, len(jsonl_files), path.name)

        # Read only as many chunks as we still need
        chunks, fail = [], 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if chunks_remaining <= 0:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(json.loads(line))
                    chunks_remaining -= 1
                except json.JSONDecodeError as exc:
                    logger.error("Malformed line in %s: %s", path.name, exc)
                    fail += 1

        ok = 0

        def _submit(chunk):
            category = chunk.get("metadata", {}).get("category", "sebi_general")
            col_id = collection_map.get(category)
            if not col_id:
                return False
            return ingest_chunk(r2r_url, chunk, col_id)

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {pool.submit(_submit, c): c for c in chunks}
            for fut in as_completed(futures):
                if fut.result():
                    ok += 1
                else:
                    fail += 1

        total_ok += ok
        total_fail += fail
        logger.info("  → %d ok, %d failed  (total: %d ok)", ok, fail, total_ok)

    print(f"\nIngestion complete. Success: {total_ok}  Failed: {total_fail}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--chunks_dir", default=os.getenv("PIPELINE_OUTPUT_DIR", "chunks"))
    p.add_argument("--r2r_url", default=R2R_BASE)
    p.add_argument("--limit", type=int, default=0, help="Max chunks to ingest (0 = all)")
    args = p.parse_args()
    main(args.chunks_dir, args.r2r_url, args.limit)
