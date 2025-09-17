import json, re, hashlib, unicodedata
from pathlib import Path
from typing import Dict, Any, List
from unidecode import unidecode

# ---------- Config paths ----------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
CONFIG_DIR = ROOT / "config"

TAXONOMY_PATH = CONFIG_DIR / "taxonomy.json"
TOPIC_RULES_PATH = CONFIG_DIR / "topic_rules.json"

OUT_CLEAN_JSON = PROC_DIR / "kb_clean.json"
OUT_CHUNKS_JSONL = PROC_DIR / "kb_chunks.jsonl"
OUT_KG_EDGES_CSV = PROC_DIR / "kg_edges.csv"

PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
def strip_html(text: str) -> str:
    # quick HTML strip (lightweight)
    text = re.sub(r"<\s*br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return text

def normalize_text(s: str) -> str:
    s = unidecode(s)
    s = strip_html(s)
    s = s.replace("\xa0", " ")
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n\s*", "\n\n", s)  # compact multiple blank lines
    return s.strip()

def stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def load_json_files() -> List[Dict[str, Any]]:
    items = []
    for fp in RAW_DIR.glob("*.json"):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # in case the file is a dict with a key holding the array
            # try to find first list value
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
        if not isinstance(data, list):
            raise ValueError(f"{fp.name} must be a JSON array")
        for obj in data:
            items.append(obj)
    return items

def load_or_default_taxonomy() -> Dict[str, Any]:
    if TAXONOMY_PATH.exists():
        return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    # minimal default
    return {"DBMS": {"Basics": ["Normalization"]}}

def load_topic_rules() -> List[Dict[str, Any]]:
    if TOPIC_RULES_PATH.exists():
        return json.loads(TOPIC_RULES_PATH.read_text(encoding="utf-8"))
    return []

def assign_topic(q: str, a: str, rules: List[Dict[str, Any]]):
    text = f"{q} {a}".lower()
    best = None
    for rule in rules:
        for kw in rule["keywords"]:
            if kw.lower() in text:
                best = (rule["topic"], rule.get("subtopic"))
                break
        if best:
            break
    if best:
        return {"topic": best[0], "subtopic": best[1]}
    return {"topic": "Uncategorized", "subtopic": None}

def difficulty_heuristic(q: str, a: str) -> str:
    t = f"{q} {a}".lower()
    hard_hits = any(x in t for x in [
        "serializable", "2pl", "mvcc", "query optimization", "distributed", "cap", "sharding",
        "write-ahead logging", "recovery", "b+ tree", "isolation level",
        "multiple inheritance", "mro", "smart pointers", "raii", "virtual functions",
        "pimpl", "copy elision", "serialization", "thread safety",
        "deadlock avoidance", "banker's algorithm", "context switching", "virtual memory",
        "page replacement", "synchronization primitives", "race conditions", "critical sections"
    ])
    easy_hits = any(x in t for x in [
        "define", "what is", "basic", "1nf", "primary key",
        "object", "class", "encapsulation", "inheritance", "polymorphism", "abstraction",
        "operating system", "process", "thread", "file system", "memory", "cpu scheduling"
    ])
    if hard_hits and not easy_hits:
        return "Advanced"
    if easy_hits and not hard_hits:
        return "Beginner"
    return "Intermediate"

def taxonomy_to_edges(tax: Dict[str, Any]) -> List[tuple]:
    edges = []
    def walk(parent, subtree):
        if isinstance(subtree, dict):
            for k, v in subtree.items():
                edges.append((k, parent, "subtopic_of"))
                walk(k, v)
        elif isinstance(subtree, list):
            for leaf in subtree:
                edges.append((leaf, parent, "subtopic_of"))
    for root, subtree in tax.items():
        walk(root, subtree)
    return edges

# ---------- Main ----------
def main():
    raw_items = load_json_files()
    topic_rules = load_topic_rules()
    taxonomy = load_or_default_taxonomy()

    cleaned = []
    seen = set()  # for deduping on normalized question+answer

    for obj in raw_items:
        q = normalize_text(str(obj.get("question", "")))
        a = normalize_text(str(obj.get("answer", "")))
        if not q or not a:
            continue

        key = (q.lower(), a.lower())
        if key in seen:
            continue
        seen.add(key)

        _id = str(obj.get("id") or f"auto-{stable_id(q+a)}")
        meta = assign_topic(q, a, topic_rules)
        difficulty = difficulty_heuristic(q, a)

        cleaned.append({
            "id": _id,
            "question": q,
            "answer": a,
            "topic": meta["topic"],
            "subtopic": meta["subtopic"],
            "difficulty": difficulty,
            "source": obj.get("source") or None
        })

    # Save clean JSON
    OUT_CLEAN_JSON.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build chunks (the text you will embed)
    with OUT_CHUNKS_JSONL.open("w", encoding="utf-8") as f:
        for row in cleaned:
            chunk_text = (
                f"Topic: {row['topic']}" + (f" > {row['subtopic']}" if row['subtopic'] else "") + "\n"
                f"Difficulty: {row['difficulty']}\n\n"
                f"Q: {row['question']}\n"
                f"A: {row['answer']}"
            )
            out = {
                "id": row["id"],
                "text": chunk_text,
                "metadata": {
                    "topic": row["topic"],
                    "subtopic": row["subtopic"],
                    "difficulty": row["difficulty"],
                    "source": row["source"]
                }
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Export KG edges from taxonomy
    edges = taxonomy_to_edges(taxonomy)
    with OUT_KG_EDGES_CSV.open("w", encoding="utf-8") as f:
        f.write("child,parent,relation\n")
        for child, parent, rel in edges:
            f.write(f'"{child}","{parent}","{rel}"\n')

    print(f"✅ Cleaned Q&A: {OUT_CLEAN_JSON}")
    print(f"✅ Chunks for embedding: {OUT_CHUNKS_JSONL}")
    print(f"✅ Knowledge graph edges: {OUT_KG_EDGES_CSV}")

if __name__ == "__main__":
    main()