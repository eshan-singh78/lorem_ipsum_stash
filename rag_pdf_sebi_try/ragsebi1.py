import fitz
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

pdf_folder = "pdfs"

docs = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_folder, file)
        pdf = fitz.open(path)
        text = ""
        for page in pdf:
            text += page.get_text()
        docs.append(text)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = []
for doc in docs:
    chunks.extend(splitter.split_text(doc))

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma.from_texts(
    chunks,
    embedding=embeddings,
    persist_directory="./pdf_index"
)

vector_db.persist()