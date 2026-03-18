import fitz
import os
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


PDF_FOLDER = "pdfs"


# -----------------------------
# Step 1: Extract text page-wise
# -----------------------------
def extract_pdf_documents():
    documents = []

    for file in os.listdir(PDF_FOLDER):
        if not file.endswith(".pdf"):
            continue

        path = os.path.join(PDF_FOLDER, file)
        pdf = fitz.open(path)

        for page_num, page in enumerate(pdf):
            text = page.get_text()

            if len(text.strip()) < 100:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file,
                        "page": page_num + 1
                    }
                )
            )

    return documents


# -----------------------------
# Step 2: Regulation-aware split
# -----------------------------
def split_regulations(text):
    """
    Split on patterns like:
    1. ...
    2. ...
    (1) ...
    (a) ...
    """

    # Primary split on numbered regulations
    chunks = re.split(r"\n\s*\d+\.\s", text)

    refined_chunks = []

    for chunk in chunks:
        chunk = chunk.strip()

        if len(chunk) < 200:
            continue

        refined_chunks.append(chunk)

    return refined_chunks


# -----------------------------
# Step 3: Fallback splitter
# -----------------------------
fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)


# -----------------------------
# Step 4: Build structured chunks
# -----------------------------
def build_chunks(documents):
    final_docs = []
    global_chunk_id = 0

    for doc in documents:
        text = doc.page_content

        # Try regulation-aware split
        reg_chunks = split_regulations(text)

        if not reg_chunks:
            reg_chunks = [text]

        for rc in reg_chunks:
            # If too large → fallback split
            if len(rc) > 1200:
                sub_chunks = fallback_splitter.split_text(rc)
            else:
                sub_chunks = [rc]

            for i, chunk in enumerate(sub_chunks):
                final_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": doc.metadata["source"],
                            "page": doc.metadata["page"],
                            "chunk_id": global_chunk_id,
                            "chunk_local_id": i,
                            "type": "regulation_chunk"
                        }
                    )
                )
                global_chunk_id += 1

    return final_docs


# -----------------------------
# MAIN PIPELINE
# -----------------------------
print("📄 Extracting PDFs...")
documents = extract_pdf_documents()

print(f"Total pages extracted: {len(documents)}")

print("✂️ Creating smart chunks...")
chunked_docs = build_chunks(documents)

print(f"Total chunks created: {len(chunked_docs)}")


# -----------------------------
# Step 5: Store in Chroma
# -----------------------------
print("🧠 Creating embeddings...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ⚠️ Clear old DB manually before running
# rm -rf pdf_index

vector_db = Chroma.from_documents(
    chunked_docs,
    embedding=embeddings,
    persist_directory="./pdf2_index"
)



print("✅ Done! Vector DB created with structured chunks.")