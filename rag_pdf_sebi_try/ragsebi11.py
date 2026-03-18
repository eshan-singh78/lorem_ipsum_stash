import json
import hashlib
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# -----------------------------
# Load DB + Models
# -----------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma(
    persist_directory="./pdf2_index",
    embedding_function=embeddings
)

# Fast generator
main_llm = OllamaLLM(model="qwen2.5:3b-instruct")

# Guardrail model (stronger)
guard_llm = OllamaLLM(model="qwen2.5:3b-instruct")


# -----------------------------
# Dedup + Filter
# -----------------------------
def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()


def is_useful_chunk(text):
    if len(text) < 200:
        return False

    keywords = ["shall", "must", "required", "penalty", "registration"]
    return any(k in text.lower() for k in keywords)


def get_context(query, k=10):
    docs = vector_db.similarity_search(query, k=k)

    seen = set()
    selected = []

    for d in docs:
        text = d.page_content.strip()
        key = hash_text(text[:300])

        if key not in seen and is_useful_chunk(text):
            seen.add(key)
            selected.append(d)

    return selected[:5]


# -----------------------------
# Prompts
# -----------------------------
GEN_PROMPT = """
You are a financial assistant.

Answer the user's question clearly.
Do not assume compliance.

User:
{query}
"""


GUARD_PROMPT = """
You are a strict SEBI compliance auditor.

Regulations:
{context}

User query:
{query}

Model response:
{answer}

Tasks:
1. Check compliance
2. If violation → fix it
3. Return structured JSON

Format:

{{
  "status": "ok" or "violation",
  "final_answer": "...",
  "issues": ["..."],
  "severity": "low | medium | high",
  "violated_chunks": [
    {{
      "chunk_id": ...,
      "source": "...",
      "page": ...
    }}
  ]
}}
"""


# -----------------------------
# Safe JSON Parse
# -----------------------------
def safe_parse(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        return {
            "status": "error",
            "final_answer": text,
            "issues": ["parse_error"],
            "severity": "low",
            "violated_chunks": []
        }


# -----------------------------
# Chat Pipeline
# -----------------------------
def chat(query):
    # Step 1: Generate answer
    raw_answer = main_llm.invoke(GEN_PROMPT.format(query=query))

    # Step 2: Retrieve context
    docs = get_context(query)

    context = "\n\n".join([
        f"[Chunk {d.metadata['chunk_id']} | Page {d.metadata['page']}]\n{d.page_content[:500]}"
        for d in docs
    ])

    # Step 3: Guardrail check
    guard_prompt = GUARD_PROMPT.format(
        context=context,
        query=query,
        answer=raw_answer
    )

    guard_output = guard_llm.invoke(guard_prompt)
    result = safe_parse(guard_output)

    return raw_answer, result, docs


# -----------------------------
# CLI Loop
# -----------------------------
def main():
    print("\n⚖️ SEBI-Compliant Chatbot (RAG + Guardrails)")
    print("Type 'exit' to quit\n")

    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            break

        raw, result, docs = chat(query)

        print("\n💡 Final Answer:")
        print(result["final_answer"])

        if result["status"] == "violation":
            print("\n⚠️ Issues:")
            for i in result["issues"]:
                print(f"- {i}")
            print("Severity:", result["severity"])

        print("\n📚 Sources:")
        for d in docs:
            print(f"- Chunk {d.metadata['chunk_id']} | Page {d.metadata['page']} | {d.metadata['source']}")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()