import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# -----------------------------
# Config
# -----------------------------
BATCH_SIZE = 15
MAX_WORKERS = 5   # ⚠️ Tune this (start with 3–5 for Ollama)

# -----------------------------
# Safe JSON extractor
# -----------------------------
def extract_json(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return []
    return []


# -----------------------------
# Load embeddings + DB
# -----------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma(
    persist_directory="./pdf_index",
    embedding_function=embeddings
)

collection = vector_db._collection
total = collection.count()

print("Total chunks:", total)


# -----------------------------
# Load LLM (IMPORTANT: per-thread instance safer)
# -----------------------------
def get_llm():
    return OllamaLLM(model="qwen2.5:3b-instruct")


# -----------------------------
# Prompt
# -----------------------------
RULE_PROMPT = """You are a SEBI regulatory compliance analyst.

Extract enforceable regulatory rules from the following text.

Extract only:
- obligations
- restrictions
- compliance requirements
- eligibility conditions
- penalties
- disclosure requirements

Ignore:
- examples
- explanations
- amendment notes
- footnotes
- page numbers

Return ONLY a JSON array.

Do not include explanations.
Do not include markdown.

Each rule must have:

rule_id
rule_text
category (licensing, disclosure, eligibility, reporting, transaction_limits, penalties, compliance_procedure)
applies_to
condition
restriction (true/false)
severity (informational, moderate, critical)

Text:
"""


# -----------------------------
# Worker function
# -----------------------------
def process_batch(offset):
    try:
        llm = get_llm()

        batch = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas"]
        )

        docs = batch["documents"]
        metas = batch["metadatas"]

        texts = [d for d in docs if len(d) > 200]

        if not texts:
            return []

        combined_text = "\n\n---\n\n".join(texts)
        prompt = RULE_PROMPT + combined_text

        response = llm.invoke(prompt)
        extracted = extract_json(response)

        for r in extracted:
            r["source_document"] = metas[0].get("source")
            r["source_page"] = metas[0].get("page")

        return extracted

    except Exception as e:
        print(f"Error at offset {offset}: {e}")
        return []


# -----------------------------
# Parallel execution
# -----------------------------
rules = []

offsets = list(range(0, total, BATCH_SIZE))

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_batch, off): off for off in offsets}

    pbar = tqdm(total=len(offsets), desc="Extracting rules")

    for future in as_completed(futures):
        result = future.result()
        rules.extend(result)

        pbar.update(1)

        # periodic save
        if len(rules) % 200 == 0:
            with open("rules_partial.json", "w") as f:
                json.dump(rules, f, indent=2)

    pbar.close()


# -----------------------------
# Save final
# -----------------------------
with open("rules.json", "w") as f:
    json.dump(rules, f, indent=2)

print("Total rules extracted:", len(rules))