import os
import requests
from bs4 import BeautifulSoup
import logging
import time

logger = logging.getLogger(__name__)

def web_search(query: str) -> str:
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")
        if not google_api_key or not google_cse_id:
            return "Web search requires GOOGLE_API_KEY and GOOGLE_CSE_ID to be set."
        url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={google_cse_id}&q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        results = response.json().get("items", [])
        if not results:
            return "No web results found."
        search_results = []
        for i, item in enumerate(results[:5]):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            try:
                time.sleep(2)
                page_response = requests.get(link, headers=headers, timeout=10)
                page_response.raise_for_status()
                soup = BeautifulSoup(page_response.text, "html.parser")
                paragraphs = soup.find_all("p")
                page_content = " ".join([p.get_text() for p in paragraphs][:1000])
            except Exception as e:
                logger.warning(f"Failed to fetch page content for {link}: {e}")
                page_content = snippet
            search_results.append(f"Result {i+1}:\nTitle: {title}\nLink: {link}\nContent: {page_content}\n")
        return "\n".join(search_results)
    except Exception as e:
        logger.exception("Web search failed")
        return f"Web search error: {e}"
