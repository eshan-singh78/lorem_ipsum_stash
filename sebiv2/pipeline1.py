import os
import re
import json
import uuid
import logging
import fitz
from multiprocessing import Pool

logger = logging.getLogger(__name__)

# OPTIONAL: enable if using R2R ingestion
USE_R2R = False

if USE_R2R:
    from r2r import R2RClient
    r2r_client = R2RClient(base_url="http://localhost:7272")

# -----------------------------
# CONFIG
# -----------------------------

INPUT_DIR = "raw_pdfs"
OUTPUT_DIR = "chunks"
NUM_WORKERS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# STEP 1: CLEAN TEXT
# -----------------------------

def clean_text(text: str) -> str:
    # Remove "Page N of M" patterns
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text)
    # Remove standalone numeric page markers (a line/token that is just a number)
    text = re.sub(r"(?<!\S)\d+(?!\S)", "", text)
    # Remove Indian legal date strings: "DD Month YYYY" e.g. "31 March 2014"
    text = re.sub(
        r"(?<!\d)\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}(?!\d)",
        "",
        text,
    )
    # Remove Indian legal date strings: "DD-M-YYYY" / "DD-MM-YYYY" e.g. "31-3-2014"
    text = re.sub(r"(?<!\d)\d{1,2}-\d{1,2}-\d{4}(?!\d)", "", text)
    # Collapse all whitespace sequences into a single space and strip
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# STEP 2: EXTRACT PDF
# -----------------------------

def extract_pdf(file_path: str) -> list[str]:
    """
    Open a PDF with PyMuPDF and return one raw text string per page.
    Returns empty string for blank pages. Logs and skips corrupted files,
    returning an empty list rather than raising.
    """
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error("Failed to open PDF %s: %s", file_path, e)
        return []

    pages = []
    try:
        for page in doc:
            try:
                text = page.get_text("text")
            except Exception as e:
                logger.error("Failed to extract page from %s: %s", file_path, e)
                text = ""
            pages.append(text if text.strip() else "")
    finally:
        doc.close()

    return pages


# -----------------------------
# STEP 2b: REMOVE REPEATED HEADERS
# -----------------------------

def remove_repeated_headers(pages: list[str]) -> list[str]:
    """
    Detect lines appearing verbatim on 3 or more consecutive pages and
    remove them from ALL pages in the document.
    """
    if not pages:
        return pages

    # Split each page into lines
    split_pages = [page.splitlines() for page in pages]
    n = len(split_pages)

    # Find lines that appear on 3+ consecutive pages
    repeated_lines: set[str] = set()

    # For each line in each page, check if it appears on the next 2 pages as well
    for i in range(n):
        for line in split_pages[i]:
            if not line.strip():
                continue
            if line in repeated_lines:
                continue
            # Count consecutive pages (starting at i) that contain this line
            consecutive = 0
            for j in range(i, n):
                if line in split_pages[j]:
                    consecutive += 1
                else:
                    break
            if consecutive >= 3:
                repeated_lines.add(line)

    if not repeated_lines:
        return pages

    # Remove repeated lines from all pages
    result = []
    for page_lines in split_pages:
        filtered = [line for line in page_lines if line not in repeated_lines]
        result.append("\n".join(filtered))

    return result

# -----------------------------
# STEP 3: STRUCTURED SPLIT
# -----------------------------

def split_sections(text: str) -> list[str]:
    """
    Split on legal section markers: "Regulation N", "Clause (X)", "N." at line start.
    Merge segments < 500 words with the next; split segments > 800 words at sentence boundary.
    Apply 100-150 word overlap between consecutive chunks.
    """
    # Split on legal section markers, keeping the delimiter as part of the following segment
    pattern = r"(?=Regulation\s+\d+|Clause\s+\([a-zA-Z0-9]+\)|^\d+\.\s)"
    raw_segments = re.split(pattern, text, flags=re.MULTILINE)

    # Strip and discard empty segments
    segments = [s.strip() for s in raw_segments if s.strip()]

    # --- Merge segments < 500 words with the NEXT segment ---
    merged: list[str] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        while len(seg.split()) < 500 and i + 1 < len(segments):
            i += 1
            seg = seg + " " + segments[i]
        merged.append(seg.strip())
        i += 1

    # --- Split segments > 800 words at nearest sentence boundary below 800 words ---
    split_chunks: list[str] = []
    for seg in merged:
        words = seg.split()
        if len(words) <= 800:
            split_chunks.append(seg)
            continue
        # Need to split; find sentence boundaries
        remaining = seg
        while len(remaining.split()) > 800:
            # Find the last ". " or ".\n" within the first 800 words
            # Reconstruct the first ~800 words as a substring to search in
            approx_800 = " ".join(remaining.split()[:800])
            # Find the last sentence boundary at or before 800 words
            cut = -1
            for m in re.finditer(r"\.\s", remaining):
                if m.end() <= len(approx_800) + 1:
                    cut = m.end()
                else:
                    break
            if cut == -1:
                # No sentence boundary found; hard-cut at 800 words
                cut = len(approx_800)
            split_chunks.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            split_chunks.append(remaining)

    # --- Apply 100-150 word overlap between consecutive chunks ---
    if len(split_chunks) <= 1:
        return split_chunks

    overlapped: list[str] = [split_chunks[0]]
    for i in range(1, len(split_chunks)):
        prev_words = overlapped[i - 1].split()
        # Take last 100-150 words of the previous chunk (use 125 as midpoint)
        overlap_count = min(125, len(prev_words))
        overlap_count = max(overlap_count, min(100, len(prev_words)))
        overlap_text = " ".join(prev_words[-overlap_count:])
        overlapped.append(overlap_text + " " + split_chunks[i])

    return overlapped

# -----------------------------
# STEP 4: FILTER CHUNKS
# -----------------------------

def is_valid(chunk: str) -> bool:
    if len(chunk) < 200:
        return False
    keywords = ["shall", "must", "regulation", "advisor", "client"]
    return any(k in chunk.lower() for k in keywords)

# -----------------------------
# STEP 5: CLASSIFY DOCUMENT
# -----------------------------

def classify_doc(filename: str) -> str:
    name = filename.lower()
    if "advisor" in name:
        return "sebi_retail"
    if "aif" in name:
        return "sebi_aif"
    if "fpi" in name:
        return "sebi_fpi"
    return "sebi_general"

# -----------------------------
# STEP 6: BUILD CHUNKS
# -----------------------------

def build_chunks(file_name: str, pages: list[str]) -> list[dict]:
    """
    Orchestrate clean → split → filter → classify → attach metadata.
    Returns list of Chunk dicts.
    """
    category = classify_doc(file_name)

    # Clean each page and track page boundaries (cumulative word counts)
    cleaned_pages: list[str] = [clean_text(page) for page in pages]

    # Build cumulative word-count boundaries so we can map a text position back to a page
    page_word_counts: list[int] = []
    cumulative = 0
    for cp in cleaned_pages:
        cumulative += len(cp.split()) if cp else 0
        page_word_counts.append(cumulative)

    # Concatenate all cleaned pages into one text
    full_text = " ".join(cp for cp in cleaned_pages if cp)

    # Split into sections
    sections = split_sections(full_text)

    # Track word offset into full_text to determine page number for each section
    word_offset = 0
    all_chunks: list[dict] = []

    for section in sections:
        if not is_valid(section):
            word_offset += len(section.split())
            continue

        # Determine 1-based page number: find which page this section starts on
        page_num = 1
        for idx, boundary in enumerate(page_word_counts):
            if word_offset < boundary:
                page_num = idx + 1
                break
        else:
            page_num = len(pages)

        chunk = {
            "id": str(uuid.uuid4()),
            "text": section,
            "metadata": {
                "source": file_name,
                "page": page_num,
                "category": category,
                "chunk_id": str(uuid.uuid4()),
            },
        }
        all_chunks.append(chunk)
        word_offset += len(section.split())

    return all_chunks

# -----------------------------
# STEP 7: SAVE JSONL
# -----------------------------

def save_chunks(file_name, chunks):
    output_file = os.path.join(OUTPUT_DIR, file_name + ".jsonl")
    with open(output_file, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

# -----------------------------
# STEP 8: INGEST TO R2R
# -----------------------------

def ingest_to_r2r(chunks):
    for chunk in chunks:
        try:
            r2r_client.documents.create(
                content=chunk["text"],
                metadata=chunk["metadata"],
                collection=chunk["metadata"]["category"]
            )
        except Exception as e:
            print("R2R ingestion error:", e)

# -----------------------------
# STEP 9: PROCESS SINGLE FILE
# -----------------------------

def process_file(file_name: str) -> None:
    if not file_name.endswith(".pdf"):
        return
    file_path = os.path.join(INPUT_DIR, file_name)
    output_path = os.path.join(OUTPUT_DIR, file_name + ".jsonl")
    tmp_path = output_path + ".tmp"
    try:
        pages = extract_pdf(file_path)
        if not pages:
            logger.warning("No pages extracted from %s, skipping", file_name)
            return
        chunks = build_chunks(file_name, pages)
        # Atomic write: write to temp file, then rename
        with open(tmp_path, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
        os.replace(tmp_path, output_path)
        logger.info("Done: %s | Chunks: %d", file_name, len(chunks))
    except Exception as e:
        logger.error("Error processing %s: %s", file_name, e)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# -----------------------------
# MAIN PIPELINE
# -----------------------------

def run_pipeline(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR, num_workers: int = NUM_WORKERS) -> None:
    # Read from env vars, falling back to parameters
    actual_input = os.environ.get("PIPELINE_INPUT_DIR", input_dir)
    actual_output = os.environ.get("PIPELINE_OUTPUT_DIR", output_dir)
    actual_workers = int(os.environ.get("PIPELINE_NUM_WORKERS", str(num_workers)))

    os.makedirs(actual_output, exist_ok=True)

    # Update globals so process_file uses the right dirs
    global INPUT_DIR, OUTPUT_DIR
    INPUT_DIR = actual_input
    OUTPUT_DIR = actual_output

    files = os.listdir(actual_input)
    with Pool(actual_workers) as pool:
        pool.map(process_file, files)
    print("\n✅ Pipeline completed for all PDFs")

# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    run_pipeline(
        input_dir=os.environ.get("PIPELINE_INPUT_DIR", INPUT_DIR),
        output_dir=os.environ.get("PIPELINE_OUTPUT_DIR", OUTPUT_DIR),
        num_workers=int(os.environ.get("PIPELINE_NUM_WORKERS", str(NUM_WORKERS))),
    )
