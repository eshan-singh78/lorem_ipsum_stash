from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# -----------------------------
# Load embedding model
# -----------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# -----------------------------
# Load vector database
# -----------------------------
vector_db = Chroma(
    persist_directory="./pdf_index",
    embedding_function=embeddings
)

# -----------------------------
# Load LLM
# -----------------------------
llm = OllamaLLM(model="qwen2.5:3b-instruct")

# -----------------------------
# Query
# -----------------------------
query = "What are the disclosure requirements for issuers under SEBI regulations?"

# Retrieve more chunks for regulatory docs
docs = vector_db.similarity_search(query, k=8)

# -----------------------------
# Build context
# -----------------------------
context_parts = []
sources = []

for d in docs:
    context_parts.append(d.page_content)

    # Collect source info if available
    if "source" in d.metadata:
        src = d.metadata.get("source")
        page = d.metadata.get("page", "unknown")
        sources.append(f"{src} (page {page})")

context = "\n\n".join(context_parts)

# -----------------------------
# Prompt
# -----------------------------
prompt = f"""
You are analyzing SEBI regulatory documents.

Answer the question using ONLY the context provided.
If the answer is not present in the context, say:
"The provided documents do not contain this information."

Provide a concise explanation.

Context:
{context}

Question:
{query}
"""

# -----------------------------
# Generate response
# -----------------------------
response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response)

print("\nSources:\n")
for s in set(sources):
    print("-", s)