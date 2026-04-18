# aria/nodes/retrieve_node.py
# Queries ChromaDB for each planner sub-query and deduplicates results.
# Now includes: session collection merge, cross-encoder reranking, comparison mode.

from aria.state import ARIAState
from aria.knowledge_base import query_kb


def _query_session_kb(query: str, thread_id: str, top_k: int = 5):
    """Query the session-specific ChromaDB collection for user-uploaded docs."""
    try:
        from aria.knowledge_base import get_vectorstore
        import chromadb
        from pathlib import Path

        chroma_path = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")
        client = chromadb.PersistentClient(path=chroma_path)
        collection_name = f"aria_kb_{thread_id[:8]}"

        # Check if session collection exists
        existing = [c.name for c in client.list_collections()]
        if collection_name not in existing:
            return []

        vs = get_vectorstore()
        emb_fn = getattr(vs, "embeddings", None) or getattr(vs, "_embedding_function", None)
        if not emb_fn:
            return []

        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma

        session_vs = Chroma(
            collection_name=collection_name,
            embedding_function=emb_fn,
            persist_directory=chroma_path,
        )
        results = session_vs.similarity_search_with_score(query, k=top_k)
        return [
            (doc.page_content, doc.metadata)
            for doc, score in results
            if score < 1.5
        ]
    except Exception as e:
        print(f"[SessionKB] query error: {e}")
        return []


def retrieve_node(state: ARIAState) -> dict:
    """
    For each sub-query, retrieve top-10 chunks from ChromaDB + session KB.
    Deduplicate by content prefix, then rerank with cross-encoder, use top 5.
    Skip gracefully if route is 'tool' (no KB needed).
    """
    if state.get("route", "both") == "tool":
        return {
            "retrieval_artifact": {},
            "retrieved": "",
            "sources": [],
            "reranker_scores": []
        }

    thread_id = state.get("thread_id", "")
    all_chunks: list = []
    all_sources: list = []
    seen: set = set()

    # ── Comparison mode: separate queries per concept ────────
    if state.get("comparison_mode") and state.get("comparison_concepts"):
        for concept in state["comparison_concepts"]:
            results = query_kb(concept, top_k=5)
            results += _query_session_kb(concept, thread_id, top_k=3)
            for doc, metadata in results:
                key = doc[:100]
                if key not in seen:
                    seen.add(key)
                    all_chunks.append((doc, metadata))
                    all_sources.append(metadata.get("source", "Unknown"))
    else:
        # ── Normal retrieval ─────────────────────────────────
        for query in state.get("sub_queries", []):
            results = query_kb(query, top_k=10)
            results += _query_session_kb(query, thread_id, top_k=3)
            for doc, metadata in results:
                key = doc[:100]
                if key not in seen:
                    seen.add(key)
                    all_chunks.append((doc, metadata))
                    all_sources.append(metadata.get("source", "Unknown"))

    # ── Cross-encoder reranking ──────────────────────────────
    reranker_scores = []
    try:
        from aria.reranker import rerank
        reranked, reranker_scores = rerank(state["question"], all_chunks, top_k=5)
        all_chunks = reranked
        all_sources = [m.get("source", "Unknown") for _, m in reranked]
    except Exception as e:
        print(f"[Reranker] Skipped: {e}")
        # If reranker fails, just take first 5
        all_chunks = all_chunks[:5]
        all_sources = all_sources[:5]

    artifact_chunks = []
    for doc, metadata in all_chunks:
        artifact_chunks.append({
            "text": doc,
            "source": metadata.get("source", "Unknown"),
            "metadata": metadata
        })

    return {
        "retrieval_artifact": {
            "chunks": artifact_chunks,
            "total_candidates": len(seen),
            "reranked_to": len(all_chunks)
        },
        "retrieved": "\n\n---\n\n".join(doc for doc, _ in all_chunks),
        "sources": list(dict.fromkeys(all_sources)),
        "reranker_scores": reranker_scores
    }
