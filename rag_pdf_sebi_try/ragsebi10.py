from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="./pdf2_index",
    embedding_function=embeddings
)

docs = db.similarity_search("intermediary registration sebi", k=5)

for d in docs:
    print("\n--- CHUNK ---")
    print("Text:", d.page_content[:200])
    print("Metadata:", d.metadata)