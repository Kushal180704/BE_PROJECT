import os
from dotenv import load_dotenv
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

# Get API key safely
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY not found in .env file")

# Configure Gemini
genai.configure(api_key=API_KEY)

# CONFIG
USE_GEMINI_FOR_EMBEDDING = True   # True = Gemini embeddings, False = local model
FAISS_DIR = Path("data/processed/faiss_gemini")  # since we built with local SBERT
TOP_K = 4

# Gemini models
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"   # or "gemini-1.5-pro"

# Load FAISS + metadata
index = faiss.read_index(str(FAISS_DIR / "faiss_index_gemini.idx"))
ids = json.loads((FAISS_DIR / "ids.json").read_text(encoding="utf-8"))
metas = json.loads((FAISS_DIR / "metas.json").read_text(encoding="utf-8"))

def embed_query_gemini(text):
    resp = genai.embed_content(model=EMBED_MODEL, content=text)
    return np.array(resp["embedding"], dtype="float32")

def embed_query_local(model, text):
    vec = model.encode([text], convert_to_numpy=True)[0]
    return np.array(vec, dtype="float32")

def retrieve(query_vec, k=TOP_K):
    q = query_vec.reshape(1, -1)
    distances, indices = index.search(q, k)
    inds = indices[0].tolist()
    ds = distances[0].tolist()
    results = []
    for i, d in zip(inds, ds):
        if i < 0 or i >= len(ids):
            continue
        results.append({
            "id": ids[i],
            "score": float(d),
            "metadata": metas[i],
        })
    return results, inds

def load_chunk_texts():
    chunk_file = Path("data/processed/kb_chunks.jsonl")
    by_id = {}
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            by_id[str(o["id"])] = o
    return by_id

def make_prompt(user_question, retrieved_chunks):
    ctx_parts = []
    for c in retrieved_chunks:
        ctx_parts.append(f"---\n{c['text']}\n")
    context = "\n".join(ctx_parts)
    prompt = (
        "You are an expert DBMS tutor. Use the provided knowledge snippets to answer the user's question clearly and step-by-step. "
        "If there is conflicting or insufficient information, mention it and give best-practice guidance.\n\n"
        f"Knowledge snippets:\n{context}\n\n"
        f"User question: {user_question}\n\n"
        "Answer concisely but thoroughly, include examples if helpful, and label sections (Explanation, Example, Key Points).\n"
    )
    return prompt

def call_gemini_chat(prompt):
    model = genai.GenerativeModel(CHAT_MODEL)
    resp = model.generate_content(prompt)
    return resp.text

def main():
    user_q = input("Ask a DBMS question: ").strip()
    if USE_GEMINI_FOR_EMBEDDING:
        qvec = embed_query_gemini(user_q)
    else:
        local_model = SentenceTransformer("all-MiniLM-L6-v2")
        qvec = embed_query_local(local_model, user_q)

    results, inds = retrieve(qvec, k=TOP_K)
    by_id = load_chunk_texts()
    retrieved_chunks = []
    for r, i in zip(results, inds):
        cid = r["id"]
        if str(cid) in by_id:
            retrieved_chunks.append(by_id[str(cid)])
        else:
            retrieved_chunks.append({
                "id": cid,
                "text": f"(missing chunk text for id {cid})",
                "metadata": r["metadata"]
            })

    prompt = make_prompt(user_q, retrieved_chunks)
    print("\n--- Prompt sent to Gemini (truncated) ---\n")
    print(prompt[:2000])

    answer = call_gemini_chat(prompt)
    print("\n--- Gemini Answer ---\n")
    print(answer)

if __name__ == "__main__":
    main()
