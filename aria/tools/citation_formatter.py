# aria/tools/citation_formatter.py
# Formats a list of source strings into a numbered citation list.

from langchain_core.tools import tool


@tool
def citation_formatter(sources: list[str]) -> str:
    """
    Format a list of source strings into a clean numbered citation list.
    Input:  ["Author et al. 2024", "LangChain docs", ...]
    Output: "[1] Author et al. 2024\n[2] LangChain docs\n..."
    """
    if not sources:
        return "No sources available."
    return "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sources))
