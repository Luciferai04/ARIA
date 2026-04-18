# aria/tools/web_search_tool.py
# LangChain @tool wrapper around DuckDuckGo search.

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information about a topic.
    Use this for recent events, news, or information that may not be in the KB.
    Returns a summary of the top web results.
    """
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Web search unavailable: {e}"
