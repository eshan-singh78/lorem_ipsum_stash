from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="./pdf_index",
    embedding_function=embeddings
)

print("Number of chunks:", db._collection.count())