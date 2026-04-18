# aria/tools/arxiv_tool.py
# LangChain @tool wrapper around the ArxivAPIWrapper.

from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.tools import tool


@tool
def arxiv_search(query: str) -> str:
    """
    Search arXiv for recent academic papers.
    Returns titles, authors, and abstracts for the top 5 results.
    Use this for questions about ML/AI research papers, algorithms, or recent advances.
    """
    try:
        wrapper = ArxivAPIWrapper(
            top_k_results=5,
            doc_content_chars_max=800,
            load_all_available_meta=False,
        )
        return wrapper.run(query)
    except Exception as e:
        return f"arXiv search unavailable: {e}"
