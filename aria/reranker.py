# aria/reranker.py
# Cross-encoder reranker for RAG pipeline precision improvement.
# Uses ms-marco-MiniLM-L-6-v2 to rescore bi-encoder results.

from typing import List, Tuple

_cross_encoder = None  # module-level singleton


def _get_cross_encoder():
    """Load CrossEncoder once and cache at module level."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank(query: str, chunks: List[Tuple[str, dict]], top_k: int = 5) -> Tuple[List[Tuple[str, dict]], List[float]]:
    """
    Rerank retrieved chunks using a cross-encoder model.

    Args:
        query: The original search query.
        chunks: List of (page_content, metadata) tuples from bi-encoder retrieval.
        top_k: Number of top results to return after reranking.

    Returns:
        Tuple of (reranked_chunks, scores) — both sorted by descending relevance.
    """
    if not chunks:
        return [], []

    ce = _get_cross_encoder()
    pairs = [(query, chunk[0]) for chunk in chunks]
    scores = ce.predict(pairs).tolist()

    # Pair chunks with scores and sort descending
    scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    reranked_chunks = [item[0] for item in top]
    reranked_scores = [round(item[1], 4) for item in top]

    return reranked_chunks, reranked_scores
