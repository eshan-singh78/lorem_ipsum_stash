# Feature: sebi-rag-system, Property 14: Ingestion chunk-to-collection routing

import sys
import os
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sebi_rag_local.ingestion.ingest import ingest_chunk  # noqa: E402

VALID_CATEGORIES = ["sebi_retail", "sebi_aif", "sebi_fpi", "sebi_general"]


@st.composite
def chunk_and_collection(draw):
    chunk_id = str(draw(st.uuids()))
    text = draw(st.text(min_size=1))
    category = draw(st.sampled_from(VALID_CATEGORIES))
    chunk = {
        "id": chunk_id,
        "text": text,
        "metadata": {"category": category},
    }
    collection_id = str(draw(st.uuids()))
    return chunk, collection_id


@given(chunk_and_collection())
@settings(max_examples=30)
def test_ingestion_chunk_routing(chunk_and_col):
    """
    **Validates: Requirements 11.2, 11.3**

    Property 14: For any valid chunk and collection_id, ingest_chunk must:
    - Return True on a 201 response
    - POST exactly once to a URL containing /v3/documents
    - Send content, collection_ids, and metadata correctly in the payload
    """
    chunk, collection_id = chunk_and_col

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.text = ""

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    result = ingest_chunk(mock_client, chunk, collection_id)

    assert result is True
    mock_client.post.assert_called_once()

    call_args = mock_client.post.call_args
    url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
    assert "/v3/documents" in url

    json_payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
    assert json_payload["content"] == chunk["text"]
    assert json_payload["collection_ids"] == [collection_id]
    assert json_payload["metadata"] == chunk["metadata"]
