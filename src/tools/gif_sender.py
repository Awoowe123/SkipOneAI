"""GIF sender tool using Tenor API"""
import logging
import requests
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)

class GifSenderTool:
    """Send GIFs using Tenor API"""

    def __init__(self, telegram_client):
        self.client = telegram_client
        # Tenor API key (get free at https://developers.google.com/tenor/guides/quickstart)
        self.api_key = os.getenv('TENOR_API_KEY', 'AIzaSyAyimkuYQYF_FXVALexPuGQctUWRURdCYQ')  # Demo key
        self.base_url = "https://tenor.googleapis.com/v2"

    async def send_gif(
        self,
        chat_id: int,
        query: str,
        limit: int = 1
    ) -> Dict[str, Any]:
        """
        Search and send GIF

        Args:
            chat_id: Chat ID to send to
            query: Search query (e.g., "happy cat")
            limit: Number of results (default 1)

        Returns:
            Result status
        """
        try:
            logger.info(f"Searching GIF: '{query}' for chat {chat_id}")

            # Search GIFs
            response = requests.get(
                f"{self.base_url}/search",
                params={
                    "q": query,
                    "key": self.api_key,
                    "client_key": "telegram_bot",
                    "limit": limit,
                    "media_filter": "gif"
                },
                timeout=5
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "message": "Не удалось найти GIF"
                }

            data = response.json()
            results = data.get('results', [])

            if not results:
                return {
                    "success": False,
                    "message": f"GIF по запросу '{query}' не найдено"
                }

            # Get first GIF URL
            gif_url = results[0]['media_formats']['gif']['url']

            # Send to Telegram (use actual client, not wrapper)
            actual_client = getattr(self.client, 'client', self.client)
            await actual_client.send_file(chat_id, gif_url)

            logger.info(f"Sent GIF to chat {chat_id}: {gif_url}")

            return {
                "success": True,
                "message": f"Отправлена GIF: {query}",
                "url": gif_url
            }

        except Exception as e:
            logger.error(f"GIF send error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Ошибка отправки GIF"
            }

# Tool schema
SEND_GIF_SCHEMA = {
    "type": "object",
    "properties": {
        "chat_id": {
            "type": "integer",
            "description": "ID чата для отправки"
        },
        "query": {
            "type": "string",
            "description": "Поисковый запрос для GIF (на английском, например: 'happy cat', 'laugh', 'thumbs up')"
        },
        "limit": {
            "type": "integer",
            "description": "Количество результатов (по умолчанию 1)",
            "default": 1
        }
    },
    "required": ["chat_id", "query"]
}
