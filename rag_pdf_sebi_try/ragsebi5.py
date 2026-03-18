import random
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Load embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load vector database
vector_db = Chroma(
    persist_directory="./pdf_index",
    embedding_function=embeddings
)

collection = vector_db._collection

# Total chunks
total = collection.count()
print("\nTotal chunks in index:", total)

# Fetch random samples
sample_size = 5
offset = random.randint(0, max(0, total - sample_size))

results = collection.get(
    limit=sample_size,
    offset=offset,
    include=["documents", "metadatas"]
)

docs = results["documents"]
metas = results["metadatas"]

print("\nShowing random chunk samples:\n")

for i, (doc, meta) in enumerate(zip(docs, metas), 1):
    print("=" * 80)
    print(f"Sample {i}")

    print("\nMetadata:")
    print(meta)

    print("\nChunk length:", len(doc))

    print("\nText preview:\n")
    print(doc[:1000])   # preview first 1000 characters

    print("\n")