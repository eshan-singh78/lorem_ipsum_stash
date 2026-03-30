# Feature: sebi-rag-system, Property 15: Router query classification

import sys
import os

# sys.path manipulation so we can import from sebi_rag_local.api.main
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sebi_rag_local.api.main import classify_query, COLLECTION_MAP

from hypothesis import given, settings, strategies as st

VALID_CATEGORIES = {"sebi_retail", "sebi_aif", "sebi_fpi", "sebi_general"}

RETAIL_KEYWORDS = COLLECTION_MAP["sebi_retail"]
AIF_KEYWORDS = COLLECTION_MAP["sebi_aif"]
FPI_KEYWORDS = COLLECTION_MAP["sebi_fpi"]


@st.composite
def query_strategy(draw):
    has_retail = draw(st.booleans())
    has_aif = draw(st.booleans())
    has_fpi = draw(st.booleans())

    parts = []

    if has_retail:
        parts.append(draw(st.sampled_from(RETAIL_KEYWORDS)))
    if has_aif:
        # Only use AIF keywords that don't contain retail keywords
        safe_aif = [k for k in AIF_KEYWORDS if not any(rk in k for rk in RETAIL_KEYWORDS)]
        if safe_aif:
            parts.append(draw(st.sampled_from(safe_aif)))
        else:
            has_aif = False
    if has_fpi:
        # Only use FPI keywords that don't contain retail or AIF keywords
        safe_fpi = [k for k in FPI_KEYWORDS if not any(rk in k for rk in RETAIL_KEYWORDS) and not any(ak in k for ak in AIF_KEYWORDS)]
        if safe_fpi:
            parts.append(draw(st.sampled_from(safe_fpi)))
        else:
            has_fpi = False

    filler = draw(st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")), min_size=0, max_size=20))
    parts.append(filler)

    query = " ".join(parts)
    return (query, has_retail, has_aif, has_fpi)


@given(query_strategy())
@settings(max_examples=30)
def test_router_classification(args):
    """
    **Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6**

    Property 15: Router query classification
    - Result is always one of the 4 valid categories.
    - Priority order is respected:
        has_retail → "sebi_retail"
        elif has_aif → "sebi_aif"
        elif has_fpi → "sebi_fpi"
        else → "sebi_general"
    """
    query, has_retail, has_aif, has_fpi = args

    result = classify_query(query)

    # Result must be one of the 4 valid categories
    assert result in VALID_CATEGORIES

    # Priority order must be respected
    if has_retail:
        assert result == "sebi_retail"
    elif has_aif:
        assert result == "sebi_aif"
    elif has_fpi:
        assert result == "sebi_fpi"
    else:
        assert result == "sebi_general"


# Feature: sebi-rag-system, Property 16: Safety filter replaces advisory phrases case-insensitively

from sebi_rag_local.api.main import apply_safety_filter, ADVISORY_PATTERNS


ADVISORY_PHRASES = [
    "you should invest",
    "recommended allocation",
    "I recommend investing",
    "I recommend buying",
    "I recommend selling",
    "best investment",
]


@st.composite
def advisory_text_strategy(draw):
    phrase = draw(st.sampled_from(ADVISORY_PHRASES))
    # Randomly vary case of each character
    varied = "".join(
        draw(st.sampled_from([c.upper(), c.lower()]))
        for c in phrase
    )
    base = draw(st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs")),
        min_size=0,
        max_size=50,
    ))
    text = base + " " + varied + " " + base
    return text, phrase


@given(advisory_text_strategy())
@settings(max_examples=30)
def test_safety_filter_replaces_advisory_phrases(args):
    """
    **Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6**

    Property 16: Safety filter replaces advisory phrases case-insensitively.
    - Output does NOT contain the original advisory phrase (case-insensitive).
    - Output DOES contain the disclaimer text.
    """
    text, phrase = args
    result = apply_safety_filter(text)

    assert phrase.lower() not in result.lower()
    assert "I can only provide educational information, not investment advice." in result


# Feature: sebi-rag-system, Property 17: Safety filter preserves non-advisory content

@given(st.text(alphabet=st.characters(whitelist_categories=("Nd",)), min_size=0, max_size=100))
@settings(max_examples=30)
def test_safety_filter_preserves_non_advisory_content(text):
    """
    **Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6**

    Property 17: Safety filter preserves non-advisory content unchanged.
    - Output equals the input when no advisory phrases are present.
    """
    result = apply_safety_filter(text)
    assert result == text


# Feature: sebi-rag-system, Property 18: Query API response schema

from unittest.mock import AsyncMock, MagicMock, patch
from starlette.testclient import TestClient
from sebi_rag_local.api.main import app

VALID_COLLECTION_NAMES = {"sebi_retail", "sebi_aif", "sebi_fpi", "sebi_general"}


@given(st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs")), min_size=1, max_size=100).filter(lambda s: s.strip()))
@settings(max_examples=10)
def test_query_api_response_schema(query_string):
    """
    **Validates: Requirements 16.2, 13.4**

    Property 18: Query API response schema
    - Response status is 200.
    - Response JSON has keys: "answer", "collection", "sources".
    - "answer" is a string.
    - "collection" is one of the 4 valid categories.
    - "sources" is a list.
    - If sources non-empty, each source has: "document_id", "chunk_id", "score", "text_preview", "metadata".
    """
    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.raise_for_status = MagicMock()
    mock_get_resp.json = lambda: {"results": [{"name": "sebi_general", "id": "col-gen-id"}]}

    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.json = lambda: {
        "results": {
            "generated_answer": "test answer",
            "search_results": {
                "chunk_search_results": [
                    {
                        "document_id": "doc1",
                        "id": "chunk1",
                        "score": 0.9,
                        "text": "some text",
                        "metadata": {},
                    }
                ]
            },
        }
    }

    mock_async_client = AsyncMock()
    mock_async_client.get = AsyncMock(return_value=mock_get_resp)
    mock_async_client.post = AsyncMock(return_value=mock_post_resp)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("sebi_rag_local.api.main.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = mock_async_client
        client = TestClient(app)
        resp = client.post("/query", json={"query": query_string})

    assert resp.status_code == 200

    body = resp.json()
    assert "answer" in body
    assert "collection" in body
    assert "sources" in body

    assert isinstance(body["answer"], str)
    assert body["collection"] in VALID_COLLECTION_NAMES
    assert isinstance(body["sources"], list)

    for source in body["sources"]:
        assert "document_id" in source
        assert "chunk_id" in source
        assert "score" in source
        assert "text_preview" in source
        assert "metadata" in source
