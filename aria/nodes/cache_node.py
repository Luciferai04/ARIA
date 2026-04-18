# aria/nodes/cache_node.py
# Semantic cache layer to short-circuit the LangGraph if questions are identical.

from aria.state import ARIAState
import aria.knowledge_base

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def cache_node(state: ARIAState) -> dict:
    """
    Check if the current question's semantic embedding matches any perfectly cached store.
    """
    # Obtain global embedding instance safely
    vs = aria.knowledge_base.get_vectorstore()
    emb_fn = getattr(vs, "embeddings", None) or getattr(vs, "_embedding_function", None)
    
    if not emb_fn:
        return {"cache_hit": False}

    # Generate current query embedding
    try:
        current_emb = emb_fn.embed_query(state["question"])
    except Exception as e:
        print(f"[CacheNode] Embedding failed: {e}")
        return {"cache_hit": False}

    cache_store = state.get("cache_store", [])
    if not cache_store:
        return {"cache_hit": False, "current_embedding": current_emb}

    # Fast dot product check against all store vectors
    best_score = -1.0
    best_entry = None
    
    for entry in cache_store:
        score = dot_product(current_emb, entry["question_embedding"])
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score > 0.92 and best_entry is not None:
        print(f"[CacheNode] Cache hit! Sim Score: {best_score:.3f}")
        return {
            "cache_hit": True,
            "report": best_entry["report"],
            "answer": best_entry["answer"],
            "sources": best_entry["sources"],
            "faithfulness": 1.0,
            "route": "cache",
            "current_embedding": current_emb
        }
    
    print(f"[CacheNode] Cache miss. Max Sim Score: {best_score:.3f}")
    return {"cache_hit": False, "current_embedding": current_emb}
