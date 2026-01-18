import os
from typing import List
from .agents.memory import MemoryManager, CHROMA_AVAILABLE


def print_agent_memories(agent_name: str, limit: int = 5):
    print(f"\n=== Agent: {agent_name} ===")
    mm = MemoryManager(agent_name)
    collection = getattr(mm, "collection", None)
    if collection is None:
        print("No collection or ChromaDB unavailable.")
        return
    try:
        data = collection.get(limit=limit)
    except Exception as e:
        print(f"Error reading collection: {e}")
        return
    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    if not ids:
        print("No memories stored.")
        return
    for i, mid in enumerate(ids):
        doc = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        ts = meta.get("timestamp", "N/A")
        decision = meta.get("decision", "N/A")
        outcome = meta.get("outcome", "N/A")
        preview = str(doc).replace("\n", " ")[:160]
        print(f"- [{i+1}] id={mid}")
        print(f"  time: {ts}")
        print(f"  decision: {decision}, outcome: {outcome}")
        print(f"  text: {preview}")


def main():
    if not CHROMA_AVAILABLE:
        print("chromadb not installed or unavailable.")
        return
    default_agents: List[str] = [
        "WarrenBuffett",
        "GeorgeSoros",
        "RayDalio",
        "JimSimons",
        "SentimentAnalyzer",
    ]
    storage_dir = os.path.abspath("agent_memory_db")
    print(f"Reading memories from: {storage_dir}")
    for name in default_agents:
        print_agent_memories(name, limit=5)


if __name__ == "__main__":
    main()

