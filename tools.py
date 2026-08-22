"""
Tool(s) available to agents. Currently just web search via Tavily,
but structured so you can drop in more tools later (e.g. a brand-doc
retriever, a trends API, etc.) without touching agent code.
"""

import os
from tavily import TavilyClient

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY not set. Get a free key at https://tavily.com "
                "and add it to your .env file."
            )
        _client = TavilyClient(api_key=api_key)
    return _client


def web_search(query: str, max_results: int = 5) -> dict:
    """
    Runs a web search and returns a compact result the Researcher agent
    can reason over: a list of {title, url, content_snippet}.
    """
    client = _get_client()
    response = client.search(query=query, max_results=max_results, search_depth="basic")

    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", "")[:500],
        })
    return {"query": query, "results": results}
