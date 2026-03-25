from __future__ import annotations

import logging

from ddgs import DDGS

LOGGER = logging.getLogger(__name__)


def search_the_web_for_claim(claim_text: str) -> str:
    """Search the web for evidence related to a claim.

    This uses DuckDuckGo to fetch a small number of result snippets (fast/cheap),
    then formats them into a single string suitable for inclusion in an LLM prompt.

    Args:
        claim_text: The claim text to search for.

    Returns:
        A formatted string containing up to 5 search results, ready for an LLM prompt.
    """
    # Steps:
    # - Validate the input claim text
    # - Query DuckDuckGo (limit to 5 snippets)
    # - Normalize fields (snippet + URL where available)
    # - Return a single clean string for LLM prompting
    # - On errors, log and return a safe fallback string
    query = (claim_text or "").strip()
    if not query:
        return "Search Results: No search executed (empty claim text)."

    try:
        with DDGS() as ddgs:
            raw_results = results = ddgs.text(claim_text, region='wt-wt', max_results=5)
            results_list = list(raw_results)[:5]
    except Exception:
        LOGGER.exception("DuckDuckGo search failed for claim query: %s", query)
        return "Search Results: Search unavailable due to an error."

    if not results_list:
        return "Search Results: No results found."

    formatted_parts: list[str] = []
    for idx, item in enumerate(results_list, start=1):
        if not isinstance(item, dict):
            continue

        snippet = (
            item.get("body")
            or item.get("snippet")
            or item.get("text")
            or item.get("description")
            or ""
        )
        snippet_str = str(snippet).strip()

        url = item.get("href") or item.get("url") or ""
        url_str = str(url).strip()

        if not snippet_str and not url_str:
            continue

        if url_str:
            formatted_parts.append(f"Search Result {idx}: {snippet_str} (URL: {url_str})".strip())
        else:
            formatted_parts.append(f"Search Result {idx}: {snippet_str}".strip())

    if not formatted_parts:
        return "Search Results: No usable snippets found."

    return " ".join(formatted_parts)

