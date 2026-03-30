"""
Unit tests for Extractor and Cleaner edge cases.

Validates: Requirements 3.3, 4.5
"""
import sys
import os
import tempfile

import fitz
import pytest

# Ensure sebiv2 package root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sebiv2.pipeline1 import extract_pdf, clean_text


# ---------------------------------------------------------------------------
# Req 3.3 – Blank page returns empty string
# ---------------------------------------------------------------------------

def test_blank_page_returns_empty_string():
    """A PDF with a single blank page should yield an empty string for that page."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create a one-page PDF with no text content
        doc = fitz.open()
        doc.new_page()          # blank page – no text inserted
        doc.save(tmp_path)
        doc.close()

        result = extract_pdf(tmp_path)

        assert len(result) == 1, "Expected exactly one page in result"
        assert result[0] == "", f"Expected empty string for blank page, got: {result[0]!r}"
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Req 4.5 – clean_text strips leading/trailing whitespace
# ---------------------------------------------------------------------------

def test_clean_text_strips_spaces():
    """Leading and trailing spaces should be removed."""
    assert clean_text("  hello world  ") == "hello world"


def test_clean_text_strips_newlines_and_tabs():
    """Leading/trailing newlines and tabs should be removed."""
    assert clean_text("\n\thello\n") == "hello"


def test_clean_text_whitespace_only_returns_empty():
    """A string of only whitespace should return an empty string."""
    assert clean_text("   ") == ""


# ---------------------------------------------------------------------------
# Imports for Chunker, Filter, Classifier, and Runner edge case tests
# ---------------------------------------------------------------------------
import unittest.mock

from sebiv2.pipeline1 import split_sections, is_valid, classify_doc, process_file


# ---------------------------------------------------------------------------
# Req 5.3 – Short segment merging
# ---------------------------------------------------------------------------

def test_short_segment_merged_with_next():
    """A section with < 500 words should be merged with the following section."""
    # First section: short (< 500 words), starts with a legal marker
    short_words = " ".join(["word"] * 50)
    short_section = f"Regulation 1 {short_words}"

    # Second section: long enough to stand alone (> 500 words)
    long_words = " ".join(["word"] * 600)
    long_section = f"Regulation 2 {long_words}"

    text = short_section + " " + long_section

    result = split_sections(text)

    # The short section should have been merged, so fewer chunks than raw sections
    assert len(result) < 2, (
        f"Expected short section to be merged, but got {len(result)} chunk(s)"
    )


# ---------------------------------------------------------------------------
# Req 5.4 – Long segment splitting at sentence boundary
# ---------------------------------------------------------------------------

def test_long_segment_split_at_sentence_boundary():
    """A section > 800 words should be split into multiple chunks at sentence boundaries."""
    # Build a single section with > 800 words and clear sentence boundaries
    sentence = "This is a valid regulatory sentence. "
    # Each sentence is 6 words; 150 sentences = 900 words
    body = sentence * 150
    text = f"Regulation 1 {body}"

    result = split_sections(text)

    assert len(result) > 1, "Expected long section to be split into multiple chunks"

    # Each non-last chunk should be <= 950 words (800 + up to 150 overlap)
    for chunk in result[:-1]:
        word_count = len(chunk.split())
        assert word_count <= 950, (
            f"Non-last chunk has {word_count} words, expected <= 950"
        )


# ---------------------------------------------------------------------------
# Req 6.1 – Filter: exact 200-char boundary
# ---------------------------------------------------------------------------

def test_filter_exactly_200_chars_passes():
    """A chunk with exactly 200 chars and a keyword should pass the filter."""
    # "shall " = 6 chars, pad with 194 'x' chars → total 200
    chunk = "shall " + "x" * 194
    assert len(chunk) == 200
    assert is_valid(chunk) is True


def test_filter_199_chars_fails():
    """A chunk with 199 chars should fail the filter even with a keyword."""
    # "shall " = 6 chars, pad with 193 'x' chars → total 199
    chunk = "shall " + "x" * 193
    assert len(chunk) == 199
    assert is_valid(chunk) is False


# ---------------------------------------------------------------------------
# Req 7.5 – Classifier: filename matching multiple substrings
# ---------------------------------------------------------------------------

def test_classify_advisor_wins_over_aif_and_fpi():
    """'advisor' in filename should return 'sebi_retail' (highest priority)."""
    assert classify_doc("advisor_aif_fpi.pdf") == "sebi_retail"


def test_classify_aif_wins_over_fpi():
    """'aif' in filename (no advisor) should return 'sebi_aif'."""
    assert classify_doc("aif_fpi.pdf") == "sebi_aif"


def test_classify_fpi_only():
    """'fpi' in filename (no advisor, no aif) should return 'sebi_fpi'."""
    assert classify_doc("fpi_doc.pdf") == "sebi_fpi"


# ---------------------------------------------------------------------------
# Req 17.3 – Atomic write: no partial file on mid-write failure
# ---------------------------------------------------------------------------

def test_atomic_write_no_partial_file_on_failure():
    """If os.replace raises, neither the .tmp nor the final .jsonl should remain."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Point the pipeline at a real PDF and a temp output dir
        pdf_name = "1289549364138.pdf"
        pdf_src = os.path.join(
            os.path.dirname(__file__), "..", "raw_pdfs", pdf_name
        )

        # Patch global INPUT_DIR / OUTPUT_DIR inside pipeline1 and mock os.replace
        import sebiv2.pipeline1 as pipeline_module

        original_input = pipeline_module.INPUT_DIR
        original_output = pipeline_module.OUTPUT_DIR

        pipeline_module.INPUT_DIR = os.path.join(
            os.path.dirname(__file__), "..", "raw_pdfs"
        )
        pipeline_module.OUTPUT_DIR = tmp_dir

        try:
            with unittest.mock.patch("sebiv2.pipeline1.os.replace",
                                     side_effect=OSError("simulated failure")):
                process_file(pdf_name)

            tmp_path = os.path.join(tmp_dir, pdf_name + ".jsonl.tmp")
            final_path = os.path.join(tmp_dir, pdf_name + ".jsonl")

            assert not os.path.exists(tmp_path), (
                ".tmp file should not exist after a failed atomic write"
            )
            assert not os.path.exists(final_path), (
                ".jsonl file should not exist after a failed atomic write"
            )
        finally:
            pipeline_module.INPUT_DIR = original_input
            pipeline_module.OUTPUT_DIR = original_output
