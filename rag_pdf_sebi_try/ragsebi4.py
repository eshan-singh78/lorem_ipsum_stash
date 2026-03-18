from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Load embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load vector database
vector_db = Chroma(
    persist_directory="./pdf_index",
    embedding_function=embeddings
)

query = "SEBI merchant banker regulations"

print("\nQuery:", query)
print("\nSearching...\n")

# Retrieve chunks
docs = vector_db.similarity_search(query, k=5)

# Print retrieved chunks
for i, d in enumerate(docs, 1):
    print("=" * 80)
    print(f"Result {i}")
    print("-" * 80)

    print("Metadata:", d.metadata)

    print("\nContent Preview:\n")
    print(d.page_content[:800])   # first 800 chars

    print("\n")