# aria/knowledge_base.py
# ChromaDB vectorstore wrapper — used by retrieve_node.

import os
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # type: ignore
from pathlib import Path

CHROMA_PATH = str(Path(__file__).parent.parent / "data" / "chroma_db")
COLLECTION  = "aria_kb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_vectorstore = None   # module-level singleton to avoid re-loading on every call


def get_vectorstore() -> Chroma:
    """Return (and cache) the ChromaDB vectorstore with HuggingFace embeddings."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
    return _vectorstore


def query_kb(query: str, top_k: int = 5) -> list[tuple[str, dict]]:
    """
    Query ChromaDB for the most relevant chunks.

    Returns a list of (page_content, metadata) tuples filtered by a
    cosine-distance threshold of < 1.5 (closer = more relevant).
    """
    try:
        vs = get_vectorstore()
        results = vs.similarity_search_with_score(query, k=top_k)
        return [
            (doc.page_content, doc.metadata)
            for doc, score in results
            if score < 1.5
        ]
    except Exception as e:
        # KB not yet ingested or Chroma unavailable — return empty gracefully
        print(f"[KB] query_kb error: {e}")
        return []


def get_collection_count() -> int:
    """Return number of documents in the KB (useful for health-checks)."""
    try:
        vs = get_vectorstore()
        return vs._collection.count()
    except Exception:
        return 0
