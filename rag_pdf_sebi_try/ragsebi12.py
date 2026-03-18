import json
import hashlib
import re
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

# Models
main_llm = OllamaLLM(model="qwen2.5:3b-instruct")
guard_llm = OllamaLLM(model="qwen2.5:3b-instruct")


# -----------------------------
# Utility
# -----------------------------
def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()


def is_useful_chunk(text):
    if len(text) < 200:
        return False
    keywords = ["shall", "must", "required", "penalty", "registration"]
    return any(k in text.lower() for k in keywords)


# -----------------------------
# Relevance Gate (CRITICAL)
# -----------------------------
def is_sebi_related(query):
    prompt = f"""
    Determine if the query is related to:
    - SEBI regulations
    - financial compliance
    - investments, securities, intermediaries

    Query: "{query}"

    Answer ONLY "yes" or "no".
    """

    result = main_llm.invoke(prompt).lower()
    return "yes" in result


# -----------------------------
# Context Retrieval
# -----------------------------
def get_context(query, k=10):
    docs = vector_db.similarity_search(query, k=k)

    seen = set()
    selected = []

    query_words = query.lower().split()

    for d in docs:
        text = d.page_content.lower()
        key = hash_text(text[:300])

        # keyword overlap score
        score = sum(1 for w in query_words if w in text)

        if key not in seen and is_useful_chunk(text) and score >= 2:
            seen.add(key)
            selected.append((d, score))

    selected = sorted(selected, key=lambda x: x[1], reverse=True)

    return [d for d, _ in selected[:5]]


# -----------------------------
# Prompts
# -----------------------------
GEN_PROMPT = """
You are a helpful financial assistant.

- Give clear, direct, practical answers
- If it's personal finance → give realistic guidance
- If it's regulatory → explain clearly

User:
{query}
"""


GUARD_PROMPT = """
You are a strict SEBI compliance auditor.

Only use the provided regulations.
Do NOT invent laws.

Regulations:
{context}

User query:
{query}

Model response:
{answer}

Tasks:
1. Check compliance
2. If violation → fix it properly
3. Be precise and confident

Return ONLY JSON:

{{
  "status": "ok" or "violation",
  "final_answer": "...",
  "issues": ["..."],
  "severity": "low | medium | high"
}}
"""


# -----------------------------
# Safe JSON Parse
# -----------------------------
def safe_parse(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return {
        "status": "error",
        "final_answer": text,
        "issues": ["parse_error"],
        "severity": "low"
    }


# -----------------------------
# Chat Pipeline
# -----------------------------
def chat(query):
    # Step 0: Relevance check
    if not is_sebi_related(query):
        answer = main_llm.invoke(GEN_PROMPT.format(query=query))
        return {
            "mode": "normal",
            "final_answer": answer,
            "issues": [],
            "severity": "none",
            "sources": []
        }

    # Step 1: Generate
    raw_answer = main_llm.invoke(GEN_PROMPT.format(query=query))

    # Step 2: Retrieve context
    docs = get_context(query)

    context = "\n\n".join([
        f"[Chunk {d.metadata['chunk_id']} | Page {d.metadata['page']}]\n{d.page_content[:400]}"
        for d in docs
    ])

    # Step 3: Guardrail
    guard_prompt = GUARD_PROMPT.format(
        context=context,
        query=query,
        answer=raw_answer
    )

    guard_output = guard_llm.invoke(guard_prompt)
    result = safe_parse(guard_output)

    return {
        "mode": "guarded",
        "final_answer": result.get("final_answer", raw_answer),
        "issues": result.get("issues", []),
        "severity": result.get("severity", "low"),
        "sources": docs
    }


# -----------------------------
# CLI
# -----------------------------
def main():
    print("\n⚖️ SEBI Compliance Chatbot (Smart Mode)")
    print("Type 'exit' to quit\n")

    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            break

        result = chat(query)

        print("\n💡 Answer:")
        print(result["final_answer"])

        # Guardrail output
        if result["mode"] == "guarded":
            if result["issues"]:
                print("\n⚠️ Compliance Issues:")
                for i in result["issues"]:
                    print(f"- {i}")
                print("Severity:", result["severity"])

            print("\n📚 Sources:")
            for d in result["sources"]:
                print(f"- Chunk {d.metadata['chunk_id']} | Page {d.metadata['page']}")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()