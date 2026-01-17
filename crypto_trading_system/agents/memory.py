import json
import os
import uuid
import datetime
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: chromadb not found. Please pip install chromadb")

import threading

_EF_LOCK = threading.Lock()
_SHARED_EF = None

class MemoryManager:
    """
    Manages long-term memory for agents using ChromaDB.
    """
    def __init__(self, agent_name: str, storage_dir: str = "agent_memory_db"):
        self.agent_name = agent_name
        # Sanitize collection name: lowercase, replace spaces, keep only alphanumeric and _-
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in agent_name).lower()
        while "__" in safe_name: safe_name = safe_name.replace("__", "_")
        safe_name = safe_name.strip("_")
        if len(safe_name) < 3: safe_name = f"agent_{safe_name}"
        if len(safe_name) > 63: safe_name = safe_name[:63]
        
        self.collection_name = safe_name
        self.storage_dir = os.path.abspath(storage_dir)
        
        self.client = None
        self.collection = None
        
        if CHROMA_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=self.storage_dir)
                
                # Use shared embedding function to avoid redundant downloads/memory usage
                global _SHARED_EF
                with _EF_LOCK:
                    if _SHARED_EF is None:
                        # Use default embedding function (all-MiniLM-L6-v2)
                        # Note: This requires internet access on first run to download the model (~80MB)
                        _SHARED_EF = embedding_functions.DefaultEmbeddingFunction()
                
                self.ef = _SHARED_EF
                
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.ef
                )
                # print(f"DEBUG: MemoryManager connected to ChromaDB collection: {self.collection_name}")
            except Exception as e:
                print(f"Error initializing ChromaDB: {e}")

    def clear_memory(self):
        """
        Clear all memories from the collection.
        Useful for resetting before a fresh backtest.
        """
        if not self.collection:
            return
            
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.ef
            )
            print(f"Memory cleared for agent: {self.agent_name}")
        except Exception as e:
            print(f"Error clearing memory for {self.agent_name}: {e}")

    def add_memory(self, context: str, decision: str, outcome: str, reasoning: str):
        """
        Add a reflection/memory item to ChromaDB.
        """
        if not self.collection:
            return

        # Prepare metadata and document
        timestamp = datetime.datetime.now().isoformat()
        
        # We store the full "story" as the document for embedding/retrieval
        # But we also store structured data in metadata
        document_text = f"Context: {context}\nDecision: {decision}\nOutcome: {outcome}\nReasoning: {reasoning}"
        
        metadata = {
            "timestamp": timestamp,
            "decision": decision,
            "outcome": outcome,
            "agent": self.agent_name
            # ChromaDB metadata must be int, float, str, bool
        }
        
        unique_id = str(uuid.uuid4())
        
        try:
            self.collection.add(
                documents=[document_text],
                metadatas=[metadata],
                ids=[unique_id]
            )
            # print(f"Memory stored for {self.agent_name}. ID: {unique_id}")
        except Exception as e:
            print(f"Failed to store memory: {e}")

    def retrieve_relevant(self, current_context: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant past memories using semantic search.
        Focuses on 'error' or 'loss' cases if possible by filtering, 
        but semantic search naturally finds similar contexts.
        """
        if not self.collection:
            return []

        try:
            # Query ChromaDB
            # We can also add a where clause if we ONLY want to find past errors
            # where={"outcome": "loss"} or similar, if we strictly followed that format.
            # For now, let's trust semantic search to find relevant past situations.
            # If the user specifically wants to avoid errors, we might prefer retrieving error cases.
            
            # Strategy: Retrieve similar memories, then prioritize those that were "mistakes"
            # OR: Retrieve specifically memories where outcome='loss' that are similar to current context.
            
            # User requirement: "retrieve 'previously made similar mistakes' to avoid falling into same trap"
            # So let's filter for negative outcomes.
            
            results = self.collection.query(
                query_texts=[current_context],
                n_results=limit,
                where={"outcome": "loss"} # Only retrieve lessons learned from losses
            )
            
            # Parse results
            memories = []
            if results['documents']:
                # results is a dict of lists (batch format)
                docs = results['documents'][0]
                metas = results['metadatas'][0]
                
                for doc, meta in zip(docs, metas):
                    memories.append({
                        "context": doc, # Or reconstruct from meta if needed, but doc has the full text
                        "decision": meta.get("decision", "N/A"),
                        "outcome": meta.get("outcome", "N/A"),
                        "reasoning": "See context/doc for details" 
                    })
            
            return memories
            
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []

    def format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return "No relevant past mistakes found."
        
        output = "Reflecting on similar past MISTAKES (Losses):\n"
        for i, m in enumerate(memories):
            # The 'context' in memory dict is actually the full document text we stored
            content = m.get('context', '')
            output += f"{i+1}. {content}\n"
        return output
