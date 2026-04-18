# tests/test_tools.py
# Unit tests for the three ARIA tools (citation_formatter is pure — no I/O).

import pytest
from aria.tools.citation_formatter import citation_formatter


def test_citation_formatter_single():
    result = citation_formatter.invoke({"sources": ["LangChain RAG docs"]})
    assert result == "[1] LangChain RAG docs"


def test_citation_formatter_multiple():
    sources = ["Source A", "Source B", "Source C"]
    result = citation_formatter.invoke({"sources": sources})
    lines = result.strip().split("\n")
    assert len(lines) == 3
    assert lines[0] == "[1] Source A"
    assert lines[1] == "[2] Source B"
    assert lines[2] == "[3] Source C"


def test_citation_formatter_empty():
    result = citation_formatter.invoke({"sources": []})
    assert result == "No sources available."


def test_arxiv_tool_is_callable():
    """Verify the tool is a LangChain @tool (has .invoke method)."""
    from aria.tools.arxiv_tool import arxiv_search
    assert callable(arxiv_search.invoke)


def test_web_search_tool_is_callable():
    """Verify the tool is a LangChain @tool (has .invoke method)."""
    from aria.tools.web_search_tool import web_search
    assert callable(web_search.invoke)
