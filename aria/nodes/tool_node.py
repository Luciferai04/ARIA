# aria/nodes/tool_node.py
# Runs arXiv and DuckDuckGo searches on the planned sub-queries.
# Now extracts arXiv paper metadata for research timeline visualization.

import re
import time
from aria.state import ARIAState
from aria.tools.arxiv_tool import arxiv_search
from aria.tools.web_search_tool import web_search


def _extract_arxiv_papers(text: str) -> list:
    """Extract publication metadata from arXiv search results."""
    papers = []
    # Match arXiv IDs like 2401.12345
    arxiv_ids = re.findall(r'(\d{4})\.\d{4,5}', text)
    # Match dates like 2024-01-15
    dates = re.findall(r'(\d{4}-\d{2}-\d{2})', text)
    # Match titles (heuristic: lines that look like titles)
    titles = re.findall(r'Title:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if not titles:
        titles = re.findall(r'^([A-Z][^.]{20,100})$', text, re.MULTILINE)

    # Build paper entries from available data
    for i, arxiv_id_year in enumerate(arxiv_ids[:20]):
        paper = {
            "title": titles[i] if i < len(titles) else f"Paper {arxiv_id_year}",
            "date": dates[i] if i < len(dates) else f"{arxiv_id_year}-01-01",
            "abstract_snippet": "",
            "url": f"https://arxiv.org/abs/{arxiv_id_year}",
        }
        papers.append(paper)

    # Deduplicate by date
    seen = set()
    unique_papers = []
    for p in papers:
        if p["date"] not in seen:
            seen.add(p["date"])
            unique_papers.append(p)

    return unique_papers


def tool_node(state: ARIAState) -> dict:
    """
    For up to 3 sub-queries, call arXiv and web search tools.
    Skips gracefully if route is 'retrieve'.
    Adds 0.5s sleep between DuckDuckGo calls to avoid rate-limiting.
    """
    if state.get("route", "both") == "retrieve":
        return {
            "tool_artifact": {},
            "tool_result": "",
            "arxiv_papers": []
        }

    results: list = []
    all_arxiv_text = ""

    for query in state.get("sub_queries", [])[:3]:   # limit API calls to first 3 sub-queries
        # ── arXiv ────────────────────────────────────────────
        try:
            arxiv_result = arxiv_search.invoke({"query": query})
            results.append(f"[arXiv] {query}:\n{arxiv_result}")
            all_arxiv_text += f"\n{arxiv_result}"
        except Exception as e:
            results.append(f"[arXiv] {query}: unavailable ({e})")

        # ── Web search ───────────────────────────────────────
        try:
            web_result = web_search.invoke({"query": query})
            results.append(f"[Web] {query}:\n{web_result}")
        except Exception as e:
            results.append(f"[Web] {query}: unavailable ({e})")

        time.sleep(0.5)   # avoid DuckDuckGo rate-limiting

    return {
        "tool_artifact": {
            "queries_executed": state.get("sub_queries", [])[:3],
            "results": results
        },
        "tool_result": "\n\n".join(results),
        "arxiv_papers": _extract_arxiv_papers(all_arxiv_text)
    }
