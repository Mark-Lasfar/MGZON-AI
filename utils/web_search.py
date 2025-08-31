import os
import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def web_search(query: str) -> str:
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")
        if not google_api_key or not google_cse_id:
            return "Web search requires GOOGLE_API_KEY and GOOGLE_CSE_ID to be set."
        url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={google_cse_id}&q={query}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        results = response.json().get("items", [])
        if not results:
            return "No web results found."
        search_results = []
        for i, item in enumerate(results[:3]):  # قللنا العدد لتسريع البحث
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            search_results.append(f"Result {i+1}:\nTitle: {title}\nLink: {link}\nContent: {snippet}\n")
        return "\n".join(search_results)
    except Exception as e:
        logger.exception(f"Web search failed: {e}")
        return f"Web search error: {e}"
