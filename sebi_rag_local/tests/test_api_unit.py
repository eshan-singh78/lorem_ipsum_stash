"""
Unit tests for SEBI RAG Query API edge cases.
Requirements: 13.3, 14.3, 14.4, 16.3, 16.4
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.testclient import TestClient

from sebi_rag_local.api.main import app

client = TestClient(app)


def test_post_query_missing_query_field_returns_422():
    """
    POST /query with missing query field → HTTP 422.
    Validates: Req 16.3
    """
    resp = client.post("/query", json={"top_k": 5})
    assert resp.status_code == 422


def test_get_health_returns_200_ok():
    """
    GET /health → HTTP 200 {"status": "ok"}.
    Validates: Req 16.4
    """
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_r2r_unavailable_returns_503():
    """
    R2R unavailable (ConnectError) → HTTP 503.
    Validates: Req 14.4
    """
    mock_async_client = AsyncMock()
    mock_async_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("sebi_rag_local.api.main.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = mock_async_client
        resp = client.post("/query", json={"query": "what is a mutual fund?", "top_k": 5})

    assert resp.status_code == 503


def test_collection_not_found_falls_back_to_sebi_general():
    """
    Collection not found → falls back to sebi_general.
    The response collection field reflects classify_query result, not the fallback name.
    Validates: Req 13.3
    """
    # First call: sebi_retail not found (empty results)
    # Second call: sebi_general found
    mock_get_resp_empty = MagicMock()
    mock_get_resp_empty.status_code = 200
    mock_get_resp_empty.raise_for_status = MagicMock()
    mock_get_resp_empty.json = lambda: {"results": []}

    mock_get_resp_general = MagicMock()
    mock_get_resp_general.status_code = 200
    mock_get_resp_general.raise_for_status = MagicMock()
    mock_get_resp_general.json = lambda: {"results": [{"name": "sebi_general", "id": "col-gen-id"}]}

    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.json = lambda: {
        "results": {
            "generated_answer": "Here is some info about retail investing.",
            "search_results": {"chunk_search_results": []},
        }
    }

    mock_async_client = AsyncMock()
    mock_async_client.get = AsyncMock(side_effect=[mock_get_resp_empty, mock_get_resp_general])
    mock_async_client.post = AsyncMock(return_value=mock_post_resp)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("sebi_rag_local.api.main.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = mock_async_client
        # "retail investor" keyword routes to sebi_retail via classify_query
        resp = client.post("/query", json={"query": "retail investor guide", "top_k": 5})

    assert resp.status_code == 200
    body = resp.json()
    # collection in response is from classify_query, not the fallback
    assert body["collection"] == "sebi_retail"


def test_generation_config_has_correct_model_temperature_max_tokens():
    """
    Generation config includes temperature=0.1, max_tokens=512, correct model name.
    Validates: Req 14.3
    """
    captured_payload = {}

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.raise_for_status = MagicMock()
    mock_get_resp.json = lambda: {"results": [{"name": "sebi_general", "id": "col-gen-id"}]}

    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.json = lambda: {
        "results": {
            "generated_answer": "answer",
            "search_results": {"chunk_search_results": []},
        }
    }

    async def capture_post(url, **kwargs):
        captured_payload.update(kwargs.get("json", {}))
        return mock_post_resp

    mock_async_client = AsyncMock()
    mock_async_client.get = AsyncMock(return_value=mock_get_resp)
    mock_async_client.post = AsyncMock(side_effect=capture_post)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("sebi_rag_local.api.main.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = mock_async_client
        resp = client.post("/query", json={"query": "what is sebi?", "top_k": 5})

    assert resp.status_code == 200
    gen_config = captured_payload.get("rag_generation_config", {})
    assert gen_config["model"] == "openai/qwen2.5:3b-instruct"
    assert gen_config["temperature"] == 0.1
    assert gen_config["max_tokens_to_sample"] == 512
