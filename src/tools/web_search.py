"""Web search tool using DuckDuckGo"""
import logging
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Search the web for information"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    async def search_web(
        self,
        query: str,
        max_results: int = 3
    ) -> Dict[str, Any]:
        """
        Search the web using DuckDuckGo Instant Answer API

        Args:
            query: Search query
            max_results: Max number of results

        Returns:
            Search results
        """
        try:
            logger.info(f"Searching web: '{query}'")

            # DuckDuckGo Instant Answer API (free, no key)
            response = requests.get(
                'https://api.duckduckgo.com/',
                params={
                    'q': query,
                    'format': 'json',
                    'no_html': 1,
                    'skip_disambig': 1
                },
                headers=self.headers,
                timeout=10
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "message": "Не удалось выполнить поиск"
                }

            data = response.json()

            # Extract info
            results = []

            # Abstract (main answer)
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', 'Результат'),
                    'snippet': data['Abstract'][:300],
                    'url': data.get('AbstractURL', '')
                })

            # Related topics
            for topic in data.get('RelatedTopics', [])[:max_results-1]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '')[:100],
                        'snippet': topic.get('Text', '')[:200],
                        'url': topic.get('FirstURL', '')
                    })

            # If no instant answer, try HTML search
            if not results:
                results = self._html_search(query, max_results)

            if not results:
                return {
                    "success": True,
                    "results": [],
                    "message": f"Ничего не найдено по запросу '{query}'"
                }

            return {
                "success": True,
                "results": results,
                "count": len(results),
                "query": query,
                "message": f"Найдено {len(results)} результатов"
            }

        except Exception as e:
            logger.error(f"Web search error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Ошибка поиска в интернете"
            }

    def _html_search(self, query: str, max_results: int) -> list:
        """Fallback: scrape DuckDuckGo HTML"""
        try:
            response = requests.get(
                'https://html.duckduckgo.com/html/',
                params={'q': query},
                headers=self.headers,
                timeout=10
            )

            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            for result in soup.select('.result')[:max_results]:
                title_elem = result.select_one('.result__title')
                snippet_elem = result.select_one('.result__snippet')

                if title_elem and snippet_elem:
                    results.append({
                        'title': title_elem.get_text(strip=True)[:100],
                        'snippet': snippet_elem.get_text(strip=True)[:200],
                        'url': ''
                    })

            return results

        except Exception as e:
            logger.error(f"HTML search error: {e}")
            return []

# Tool schema
SEARCH_WEB_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Поисковый запрос (на русском или английском). Примеры: 'погода в Красноярске', 'новости Python', 'как приготовить борщ'"
        },
        "max_results": {
            "type": "integer",
            "description": "Максимальное количество результатов (по умолчанию 3)",
            "default": 3
        }
    },
    "required": ["query"]
}
