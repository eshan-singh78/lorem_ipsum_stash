"""
Unit tests for ingestion edge cases.
Requirements: 10.2, 10.3, 11.5, 11.6, 17.2
"""

import json
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Ensure the project root is on sys.path so we can import sebi_rag_local.ingestion.ingest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sebi_rag_local.ingestion.ingest import (
    get_or_create_collection,
    ingest_chunk,
    ingest_jsonl_file,
    main,
)


# ---------------------------------------------------------------------------
# Req 10.2 — Existing collection is reused (no duplicate POST)
# ---------------------------------------------------------------------------

def test_existing_collection_reused_no_post():
    """get_or_create_collection returns existing id without calling POST."""
    client = MagicMock()
    client.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"results": [{"name": "sebi_retail", "id": "col-123"}]},
    )
    client.get.return_value.raise_for_status = MagicMock()

    result = get_or_create_collection(client, "sebi_retail")

    assert result == "col-123"
    client.post.assert_not_called()


# ---------------------------------------------------------------------------
# Req 10.3 — Missing collection triggers POST
# ---------------------------------------------------------------------------

def test_missing_collection_triggers_post():
    """get_or_create_collection creates a new collection when none exists."""
    client = MagicMock()

    get_resp = MagicMock(status_code=200)
    get_resp.raise_for_status = MagicMock()
    get_resp.json = lambda: {"results": []}
    client.get.return_value = get_resp

    post_resp = MagicMock(status_code=201)
    post_resp.raise_for_status = MagicMock()
    post_resp.json = lambda: {"results": {"id": "new-col-456"}}
    client.post.return_value = post_resp

    result = get_or_create_collection(client, "sebi_aif")

    assert result == "new-col-456"
    client.post.assert_called_once()


# ---------------------------------------------------------------------------
# Req 11.5 — Failed upload logs chunk_id + status + body and continues
# ---------------------------------------------------------------------------

def test_failed_upload_logs_and_returns_false(caplog):
    """ingest_chunk returns False and logs chunk_id, status, and error body on failure."""
    import logging

    client = MagicMock()
    resp = MagicMock(status_code=500, text="Internal Server Error")
    client.post.return_value = resp

    chunk = {
        "id": "chunk-abc",
        "text": "some text",
        "metadata": {"category": "sebi_retail"},
    }

    with caplog.at_level(logging.ERROR, logger="sebi_rag_local.ingestion.ingest"):
        result = ingest_chunk(client, chunk, "col-id")

    assert result is False
    # Verify the log message contains chunk_id, status code, and error body
    assert any(
        "chunk-abc" in record.message and "500" in record.message and "Internal Server Error" in record.message
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Req 17.2 — Malformed JSONL line is skipped with log
# ---------------------------------------------------------------------------

def test_malformed_jsonl_line_skipped(caplog, tmp_path):
    """ingest_jsonl_file skips malformed lines and returns (1, 1)."""
    import logging

    valid_chunk = {
        "id": "chunk-valid",
        "text": "valid text",
        "metadata": {"category": "sebi_retail"},
    }
    jsonl_file = tmp_path / "test.jsonl"
    jsonl_file.write_text(
        json.dumps(valid_chunk) + "\n" + "not json\n",
        encoding="utf-8",
    )

    client = MagicMock()
    ok_resp = MagicMock(status_code=201)
    client.post.return_value = ok_resp

    collection_map = {"sebi_retail": "col-retail-id"}

    with caplog.at_level(logging.ERROR, logger="sebi_rag_local.ingestion.ingest"):
        success, fail = ingest_jsonl_file(client, str(jsonl_file), collection_map)

    assert success == 1
    assert fail == 1
    # Confirm a log entry was emitted for the malformed line
    assert any("Malformed" in record.message or "malformed" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# Req 11.6 — Final summary printed with correct counts
# ---------------------------------------------------------------------------

def test_main_prints_correct_summary(tmp_path, capsys):
    """main() prints 'Success: 2' and 'Failed: 0' when both chunks ingest successfully."""
    chunk1 = {"id": "c1", "text": "text one", "metadata": {"category": "sebi_retail"}}
    chunk2 = {"id": "c2", "text": "text two", "metadata": {"category": "sebi_retail"}}

    jsonl_file = tmp_path / "docs.jsonl"
    jsonl_file.write_text(
        json.dumps(chunk1) + "\n" + json.dumps(chunk2) + "\n",
        encoding="utf-8",
    )

    with (
        patch("sebi_rag_local.ingestion.ingest.get_or_create_collection", return_value="col-id") as mock_get_col,
        patch("sebi_rag_local.ingestion.ingest.ingest_chunk", return_value=True) as mock_ingest,
        patch("sebi_rag_local.ingestion.ingest.httpx.Client") as mock_client_cls,
    ):
        # Make the context manager return a usable mock client
        mock_client_instance = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        main(str(tmp_path), "http://localhost:7272")

    captured = capsys.readouterr()
    assert "Success: 2" in captured.out
    assert "Failed: 0" in captured.out
