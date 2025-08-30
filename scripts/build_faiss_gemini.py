import json
from pathlib import Path
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY missing in .env")
genai.configure(api_key=API_KEY)

CHUNKS_PATH = Path("data/processed/kb_chunks.jsonl")
OUT_DIR = Path("data/processed/faiss_gemini")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "models/embedding-001"

def embed_text(text):
    resp = genai.embed_content(model=EMBED_MODEL, content=text)
    return resp["embedding"]

def main():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    texts = [c["text"] for c in chunks]
    ids = [str(c["id"]) for c in chunks]
    metas = [c.get("metadata", {}) for c in chunks]

    vectors = [embed_text(t) for t in texts]
    vectors = np.array(vectors, dtype="float32")

    dim = vectors.shape[1]
    print("Embedding dimension:", dim)

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    print("Added vectors:", index.ntotal)

    faiss.write_index(index, str(OUT_DIR / "faiss_index_gemini.idx"))
    (OUT_DIR / "ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "metas.json").write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    print("Saved Gemini FAISS index to", OUT_DIR)

if __name__ == "__main__":
    main()
