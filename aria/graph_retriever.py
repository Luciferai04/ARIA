# aria/graph_retriever.py
# GraphRAG supplementary retrieval path.
# Falls back gracefully if graphrag index hasn't been built.

from pathlib import Path

GRAPHRAG_DIR = Path(__file__).parent.parent / "data" / "graphrag"


def query_graph(question: str) -> str:
    """
    Run graphrag local search and return entity-relationship context.
    Returns empty string if graphrag index hasn't been built or library not available.
    """
    if not GRAPHRAG_DIR.exists():
        return ""

    try:
        # Try to use graphrag library for local search
        import subprocess
        result = subprocess.run(
            ["python", "-m", "graphrag.query",
             "--root", str(GRAPHRAG_DIR),
             "--method", "local",
             "--query", question],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()[:2000]
    except Exception as e:
        print(f"[GraphRAG] Query failed: {e}")

    return ""
