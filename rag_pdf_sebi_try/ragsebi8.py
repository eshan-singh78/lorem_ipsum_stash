from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma(
    persist_directory="./pdf_index",
    embedding_function=embeddings
)

collection = vector_db._collection

data = collection.get(limit=5, include=["documents", "metadatas"])

for i in range(len(data["documents"])):
    print("\n--- CHUNK ---")
    print("Text:", data["documents"][i][:200])
    print("Metadata:", data["metadatas"][i])