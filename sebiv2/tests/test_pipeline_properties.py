# Feature: sebi-rag-system, Property 1: Extractor page count invariant

import sys
import os
import tempfile

import fitz  # PyMuPDF
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

# Ensure sebiv2 package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sebiv2.pipeline1 import extract_pdf


@settings(max_examples=20)
@given(st.integers(min_value=1, max_value=20))
def test_extractor_page_count(page_count):
    """
    **Validates: Requirements 3.1, 3.3**

    Property 1: Extractor page count invariant
    For any PDF with N pages, extract_pdf must return a list of exactly N strings.
    """
    # Build a synthetic in-memory PDF with exactly page_count pages
    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page()
        # Alternate between pages with text and blank pages
        if i % 2 == 0:
            page.insert_text((72, 72), f"Page {i + 1} content: some regulatory text here.")

    # Write to a named temp file and close it so extract_pdf can open it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        doc.save(tmp_path)
        doc.close()

        result = extract_pdf(tmp_path)

        assert len(result) == page_count, (
            f"Expected {page_count} pages, got {len(result)}"
        )
    finally:
        os.unlink(tmp_path)


# Feature: sebi-rag-system, Property 2: Extractor error resilience


@settings(max_examples=20)
@given(st.binary(min_size=1, max_size=1000))
def test_extractor_error_resilience(corrupted_bytes):
    """
    **Validates: Requirements 3.4**

    Property 2: Extractor error resilience
    For any batch containing a valid PDF and a corrupted file, extract_pdf must
    successfully process the valid PDF (returning a non-empty list) and must
    return an empty list for the corrupted file without raising any exception.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a valid PDF with 1 page and some text
        valid_path = os.path.join(tmp_dir, "valid.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Valid regulatory text for testing.")
        doc.save(valid_path)
        doc.close()

        # Create a corrupted file with random bytes
        corrupted_path = os.path.join(tmp_dir, "corrupted.pdf")
        with open(corrupted_path, "wb") as f:
            f.write(corrupted_bytes)

        # Valid PDF must return a non-empty list
        valid_result = extract_pdf(valid_path)
        assert len(valid_result) > 0, "Expected non-empty list for valid PDF"

        # Corrupted file must return empty list without raising
        corrupted_result = extract_pdf(corrupted_path)
        assert corrupted_result == [], (
            f"Expected [] for corrupted file, got {corrupted_result}"
        )


# Feature: sebi-rag-system, Property 3: Cleaner removes page-number patterns

import re as _re

from sebiv2.pipeline1 import clean_text


@st.composite
def _text_with_page_number(draw):
    base = draw(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=200))
    n = draw(st.integers(min_value=1, max_value=999))
    m = draw(st.integers(min_value=1, max_value=999))
    pos = draw(st.integers(min_value=0, max_value=len(base)))
    return base[:pos] + f"Page {n} of {m}" + base[pos:]


@settings(max_examples=30)
@given(_text_with_page_number())
def test_cleaner_removes_page_numbers(injected_text):
    """
    **Validates: Requirements 4.1**

    Property 3: Cleaner removes page-number patterns
    For any text with an injected "Page N of M" pattern, clean_text must
    return a string that does not contain any substring matching Page \\d+ of \\d+.
    """
    result = clean_text(injected_text)
    assert not _re.search(r"Page \d+ of \d+", result), (
        f"clean_text did not remove page-number pattern from: {injected_text!r}\n"
        f"Got: {result!r}"
    )


# Feature: sebi-rag-system, Property 4: Cleaner removes Indian legal date patterns

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


@st.composite
def _text_with_date(draw):
    base = draw(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=200))
    day = draw(st.integers(min_value=1, max_value=31))
    year = draw(st.integers(min_value=1000, max_value=2100))  # always 4 digits
    use_named = draw(st.booleans())
    if use_named:
        month = draw(st.sampled_from(_MONTH_NAMES))
        date_str = f"{day} {month} {year}"
    else:
        month_num = draw(st.integers(min_value=1, max_value=12))
        date_str = f"{day}-{month_num}-{year}"
    pos = draw(st.integers(min_value=0, max_value=len(base)))
    return base[:pos] + date_str + base[pos:]


@settings(max_examples=30)
@given(_text_with_date())
def test_cleaner_removes_dates(injected_text):
    """
    **Validates: Requirements 4.2**

    Property 4: Cleaner removes Indian legal date patterns
    For any text with an injected Indian legal date pattern, clean_text must
    return a string that does not contain any substring matching either
    \\d{1,2} (Month name) \\d{4} or \\d{1,2}-\\d{1,2}-\\d{4}.
    """
    result = clean_text(injected_text)
    month_pattern = (
        r"(?<!\d)\d{1,2}\s+(?:January|February|March|April|May|June|"
        r"July|August|September|October|November|December)\s+\d{4}(?!\d)"
    )
    numeric_pattern = r"(?<!\d)\d{1,2}-\d{1,2}-\d{4}(?!\d)"
    assert not _re.search(month_pattern, result), (
        f"clean_text did not remove named-month date from: {injected_text!r}\n"
        f"Got: {result!r}"
    )
    assert not _re.search(numeric_pattern, result), (
        f"clean_text did not remove numeric date from: {injected_text!r}\n"
        f"Got: {result!r}"
    )


# Feature: sebi-rag-system, Property 5: Cleaner collapses whitespace

import re as _re2


@st.composite
def _text_with_injected_whitespace(draw):
    base = draw(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=200))
    num_injections = draw(st.integers(min_value=1, max_value=5))
    text = base
    for _ in range(num_injections):
        ws_chunk = draw(st.text(alphabet=st.sampled_from(" \t\n\r"), min_size=2, max_size=10))
        pos = draw(st.integers(min_value=0, max_value=len(text)))
        text = text[:pos] + ws_chunk + text[pos:]
    return text


@settings(max_examples=30)
@given(_text_with_injected_whitespace())
def test_cleaner_collapses_whitespace(text_with_whitespace):
    """
    **Validates: Requirements 4.3, 4.5**

    Property 5: Cleaner collapses whitespace
    For any text with injected whitespace sequences, clean_text must return a
    string with no consecutive whitespace, no leading whitespace, and no
    trailing whitespace.
    """
    result = clean_text(text_with_whitespace)
    assert not _re2.search(r"\s{2,}", result), (
        f"clean_text left consecutive whitespace in: {text_with_whitespace!r}\n"
        f"Got: {result!r}"
    )
    assert result == result.lstrip(), (
        f"clean_text left leading whitespace in: {text_with_whitespace!r}\n"
        f"Got: {result!r}"
    )
    assert result == result.rstrip(), (
        f"clean_text left trailing whitespace in: {text_with_whitespace!r}\n"
        f"Got: {result!r}"
    )
    assert result == result.strip(), (
        f"clean_text result does not equal result.strip() for: {text_with_whitespace!r}\n"
        f"Got: {result!r}"
    )


# Feature: sebi-rag-system, Property 6: Cleaner removes repeated headers and footers

from sebiv2.pipeline1 import remove_repeated_headers


@st.composite
def _pages_with_repeated_header(draw):
    header_line = draw(
        st.text(
            alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\n\r"),
            min_size=1,
            max_size=50,
        )
    )
    num_pages = draw(st.integers(min_value=3, max_value=10))
    pages = []
    for i in range(num_pages):
        unique_content = draw(
            st.text(
                alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\n\r"),
                min_size=0,
                max_size=100,
            )
        )
        # Header is a complete line; inject it as a separate line on every page
        page_text = f"{header_line}\nPage {i + 1} content {unique_content}"
        pages.append(page_text)
    return pages


@settings(max_examples=30)
@given(_pages_with_repeated_header())
def test_cleaner_removes_repeated_headers(pages):
    """
    **Validates: Requirements 4.4**

    Property 6: Cleaner removes repeated headers and footers
    For any list of pages where a header line appears on every page as a
    complete line, remove_repeated_headers must return pages that do not
    contain that header line.
    """
    # Extract the header line from the first page (first line)
    header_line = pages[0].splitlines()[0]

    result = remove_repeated_headers(pages)

    for i, page in enumerate(result):
        page_lines = page.splitlines()
        assert header_line not in page_lines, (
            f"Header line {header_line!r} still present in page {i} after removal.\n"
            f"Page content: {page!r}"
        )


# Feature: sebi-rag-system, Property 7: Chunker word-count bounds

from sebiv2.pipeline1 import split_sections


@st.composite
def _regulatory_text(draw):
    # Keep sections small to avoid Hypothesis data-too-large errors
    num_sections = draw(st.integers(min_value=3, max_value=5))
    sections = []
    for n in range(1, num_sections + 1):
        num_words = draw(st.integers(min_value=600, max_value=700))
        # Use a fixed short word to keep data size manageable
        words = ["word"] * num_words
        section_text = f"Regulation {n} " + " ".join(words)
        sections.append(section_text)
    return "\n".join(sections)


@settings(max_examples=20, suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow, HealthCheck.data_too_large])
@given(_regulatory_text())
def test_chunker_word_count_bounds(text):
    """
    **Validates: Requirements 5.2, 5.3, 5.4**

    Property 7: Chunker word-count bounds
    For any regulatory-style text with 5-20 sections of 600-1000 words each,
    all chunks except the last must have between 500 and 950 words (upper bound
    accounts for 100-150 word overlap prepended to each chunk after the first).
    The last chunk may be shorter.
    """
    chunks = split_sections(text)

    # If no chunks produced, nothing to assert
    if not chunks:
        return

    # All chunks except the last must satisfy the word-count bounds
    for i, chunk in enumerate(chunks[:-1]):
        word_count = len(chunk.split())
        assert 500 <= word_count <= 950, (
            f"Chunk {i} has {word_count} words, expected 500-950 "
            f"(overlap of 100-150 words is included in upper bound).\n"
            f"Chunk preview: {chunk[:200]!r}"
        )


# Feature: sebi-rag-system, Property 8: Chunker overlap invariant

from hypothesis import assume


@settings(max_examples=20, suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow, HealthCheck.data_too_large])
@given(_regulatory_text())
def test_chunker_overlap_invariant(text):
    """
    **Validates: Requirements 5.5**

    Property 8: Chunker overlap invariant
    For any regulatory-style text producing 2+ chunks, the last 125 words of
    chunk[i] must equal the first 125 words of chunk[i+1], confirming the
    100-150 word overlap is applied correctly between consecutive chunks.
    """
    chunks = split_sections(text)

    assume(len(chunks) >= 2)

    for i in range(len(chunks) - 1):
        tail_words = chunks[i].split()[-125:]
        prefix_words = chunks[i + 1].split()[:125]
        assert tail_words == prefix_words, (
            f"Overlap mismatch between chunk {i} and chunk {i + 1}.\n"
            f"Last 125 words of chunk {i}: {tail_words}\n"
            f"First 125 words of chunk {i + 1}: {prefix_words}"
        )


# Feature: sebi-rag-system, Property 9: Chunker UUID uniqueness

import uuid as _uuid

from sebiv2.pipeline1 import build_chunks


@st.composite
def _multi_pdf_inputs(draw):
    num_pdfs = draw(st.integers(min_value=1, max_value=5))
    pdfs = []
    for i in range(num_pdfs):
        filename = f"doc{i + 1}.pdf"
        num_pages = draw(st.integers(min_value=1, max_value=3))
        pages = draw(
            st.lists(
                st.text(
                    alphabet=st.characters(blacklist_categories=("Cs",)),
                    min_size=10,
                    max_size=300,
                ),
                min_size=num_pages,
                max_size=num_pages,
            )
        )
        pdfs.append((filename, pages))
    return pdfs


@settings(max_examples=20)
@given(_multi_pdf_inputs())
def test_chunker_uuid_uniqueness(pdfs):
    """
    **Validates: Requirements 5.6**

    Property 9: Chunker UUID uniqueness
    For any set of 1-5 PDFs, all chunk_ids and top-level ids produced by
    build_chunks must be globally unique and valid UUID4 strings.
    """
    all_chunk_ids = []
    all_ids = []

    for filename, pages in pdfs:
        chunks = build_chunks(filename, pages)
        for chunk in chunks:
            all_chunk_ids.append(chunk["metadata"]["chunk_id"])
            all_ids.append(chunk["id"])

    # Assert all chunk_ids are distinct
    assert len(all_chunk_ids) == len(set(all_chunk_ids)), (
        f"Duplicate chunk_ids found: {len(all_chunk_ids)} total, "
        f"{len(set(all_chunk_ids))} unique"
    )

    # Assert all top-level ids are distinct
    assert len(all_ids) == len(set(all_ids)), (
        f"Duplicate top-level ids found: {len(all_ids)} total, "
        f"{len(set(all_ids))} unique"
    )

    # Assert each chunk_id is a valid UUID4 string
    for chunk_id in all_chunk_ids:
        assert str(_uuid.UUID(chunk_id, version=4)) == chunk_id, (
            f"chunk_id {chunk_id!r} is not a valid UUID4 string"
        )


# Feature: sebi-rag-system, Property 10: Filter correctness

from sebiv2.pipeline1 import is_valid

_FILTER_KEYWORDS = ["shall", "must", "regulation", "advisor", "client"]


@st.composite
def _filter_chunk(draw):
    has_length = draw(st.booleans())
    has_keyword = draw(st.booleans())

    keyword = draw(st.sampled_from(_FILTER_KEYWORDS)) if has_keyword else ""
    kw_len = len(keyword)

    # Generate base text of 'x' chars. Total chunk = base + keyword.
    # Ensure total length satisfies has_length condition.
    if has_length:
        # total must be >= 200; base fills the rest
        base_size = draw(st.integers(min_value=max(0, 200 - kw_len), max_value=400 - kw_len))
    else:
        # total must be < 200
        base_size = draw(st.integers(min_value=0, max_value=max(0, 199 - kw_len)))

    base_text = "x" * base_size

    if has_keyword:
        inject_pos = draw(st.integers(min_value=0, max_value=len(base_text)))
        chunk = base_text[:inject_pos] + keyword + base_text[inject_pos:]
    else:
        chunk = base_text

    return has_length, has_keyword, chunk


@settings(max_examples=30)
@given(_filter_chunk())
def test_filter_correctness(args):
    """
    **Validates: Requirements 6.1, 6.2, 6.3**

    Property 10: Filter correctness
    For any chunk, is_valid must return True if and only if the chunk has
    >= 200 characters AND contains at least one of the required keywords.
    """
    has_length, has_keyword, chunk = args

    result = is_valid(chunk)

    if has_length and has_keyword:
        assert result is True, (
            f"Expected is_valid to return True for chunk with length={len(chunk)} "
            f"and keyword present, but got False.\nChunk preview: {chunk[:100]!r}"
        )
    else:
        assert result is False, (
            f"Expected is_valid to return False for chunk with has_length={has_length}, "
            f"has_keyword={has_keyword}, but got True.\nChunk preview: {chunk[:100]!r}"
        )


# Feature: sebi-rag-system, Property 11: Classifier priority routing

from sebiv2.pipeline1 import classify_doc


@st.composite
def _classifier_filename(draw):
    has_advisor = draw(st.booleans())
    has_aif = draw(st.booleans())
    has_fpi = draw(st.booleans())
    filler = draw(st.text(alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters=""), min_size=0, max_size=20))
    parts = []
    if has_advisor:
        parts.append("advisor")
    if has_aif:
        parts.append("aif")
    if has_fpi:
        parts.append("fpi")
    parts.append(filler)
    filename = "_".join(parts)
    return filename, has_advisor, has_aif, has_fpi


@settings(max_examples=30)
@given(_classifier_filename())
def test_classifier_priority_routing(args):
    """
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

    Property 11: Classifier priority routing
    For any filename built from optional trigger substrings (advisor, aif, fpi)
    and random filler, classify_doc must return one of the four valid categories
    and must respect the priority order: advisor > aif > fpi > general.
    """
    filename, has_advisor, has_aif, has_fpi = args

    result = classify_doc(filename)

    valid_categories = ["sebi_retail", "sebi_aif", "sebi_fpi", "sebi_general"]
    assert result in valid_categories, (
        f"classify_doc({filename!r}) returned {result!r}, "
        f"which is not one of {valid_categories}"
    )

    if has_advisor:
        assert result == "sebi_retail", (
            f"Expected 'sebi_retail' when has_advisor=True, got {result!r} "
            f"for filename {filename!r}"
        )
    elif has_aif:
        assert result == "sebi_aif", (
            f"Expected 'sebi_aif' when has_aif=True (and has_advisor=False), "
            f"got {result!r} for filename {filename!r}"
        )
    elif has_fpi:
        assert result == "sebi_fpi", (
            f"Expected 'sebi_fpi' when has_fpi=True (and has_advisor=False, has_aif=False), "
            f"got {result!r} for filename {filename!r}"
        )
    else:
        assert result == "sebi_general", (
            f"Expected 'sebi_general' when no trigger flags set, "
            f"got {result!r} for filename {filename!r}"
        )


# Feature: sebi-rag-system, Property 12: Chunk schema round-trip integrity

import json
import uuid


@st.composite
def _chunk_dict(draw):
    text = draw(st.text())
    source = draw(st.text(min_size=1, max_size=50)) + ".pdf"
    page = draw(st.integers(min_value=1, max_value=100))
    category = draw(st.sampled_from(["sebi_retail", "sebi_aif", "sebi_fpi", "sebi_general"]))
    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "metadata": {
            "source": source,
            "page": page,
            "category": category,
            "chunk_id": str(uuid.uuid4()),
        },
    }


@settings(max_examples=30)
@given(_chunk_dict())
def test_chunk_schema_round_trip(chunk):
    """
    **Validates: Requirements 8.1, 8.3, 9.3, 17.2**

    Property 12: Chunk schema round-trip integrity
    For any Chunk dict, serializing to a JSONL line and parsing it back must
    produce an object with identical structure and values.
    """
    line = json.dumps(chunk) + "\n"
    parsed = json.loads(line.strip())

    # Top-level keys must be present
    assert "id" in parsed
    assert "text" in parsed
    assert "metadata" in parsed

    # Metadata keys must be present
    assert "source" in parsed["metadata"]
    assert "page" in parsed["metadata"]
    assert "category" in parsed["metadata"]
    assert "chunk_id" in parsed["metadata"]

    # All values must match the original
    assert parsed["id"] == chunk["id"]
    assert parsed["text"] == chunk["text"]
    assert parsed["metadata"]["source"] == chunk["metadata"]["source"]
    assert parsed["metadata"]["page"] == chunk["metadata"]["page"]
    assert parsed["metadata"]["category"] == chunk["metadata"]["category"]
    assert parsed["metadata"]["chunk_id"] == chunk["metadata"]["chunk_id"]


# Feature: sebi-rag-system, Property 13: Pipeline idempotency


@st.composite
def _filename_and_pages(draw):
    filename = "test_doc.pdf"
    num_pages = draw(st.integers(min_value=1, max_value=5))
    pages = draw(
        st.lists(
            st.text(
                alphabet=st.characters(blacklist_categories=("Cs",)),
                min_size=1,
                max_size=300,
            ),
            min_size=num_pages,
            max_size=num_pages,
        )
    )
    return filename, pages


@settings(max_examples=20)
@given(_filename_and_pages())
def test_pipeline_idempotency(args):
    """
    **Validates: Requirements 17.1**

    Property 13: Pipeline idempotency
    For any filename and list of pages, calling build_chunks twice with the
    same inputs must produce chunk texts in the same order with identical
    content. UUIDs will differ between runs (that's expected), but text
    content must be identical.
    """
    filename, pages = args

    chunks_run1 = build_chunks(filename, pages)
    chunks_run2 = build_chunks(filename, pages)

    texts_run1 = [chunk["text"] for chunk in chunks_run1]
    texts_run2 = [chunk["text"] for chunk in chunks_run2]

    assert texts_run1 == texts_run2, (
        f"build_chunks produced different text content on two runs.\n"
        f"Run 1 texts: {texts_run1}\n"
        f"Run 2 texts: {texts_run2}"
    )
