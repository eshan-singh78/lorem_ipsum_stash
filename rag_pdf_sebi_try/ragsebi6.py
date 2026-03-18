import os
import fitz

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

pdf_folder = "pdfs"

documents = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=200
)

for file in os.listdir(pdf_folder):

    if not file.endswith(".pdf"):
        continue

    path = os.path.join(pdf_folder, file)
    pdf = fitz.open(path)

    print("Indexing:", file)

    for page_num, page in enumerate(pdf):

        text = page.get_text()

        if not text:
            continue

        # -------- basic TOC filtering --------
        if text.count("Chapter") > 5:
            continue

        if text.count("Schedule") > 5:
            continue

        if "Page" in text and "of" in text:
            if len(text) < 500:
                continue
        # ------------------------------------

        chunks = splitter.split_text(text)

        for chunk in chunks:

            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file,
                        "page": page_num + 1
                    }
                )
            )

print("Total chunks prepared:", len(documents))

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma.from_documents(
    documents,
    embedding=embeddings,
    persist_directory="./pdf_index"
)

print("Index built successfully.")