"""Message search tool"""
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MessageSearchTool:
    """Search in message history"""

    def __init__(self, personality_db):
        self.db = personality_db

    async def search_messages(
        self,
        query: str,
        limit: int = 5,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search for messages in history

        Args:
            query: Search query
            limit: Max results
            user_id: Filter by user ID

        Returns:
            Search results
        """
        try:
            logger.info(f"Searching messages: '{query}' (user_id={user_id}, limit={limit})")

            # Use RAG to find similar messages
            results = self.db.find_similar(
                query=query,
                user_id=user_id,
                top_k=limit
            )

            if not results:
                return {
                    "success": True,
                    "results": [],
                    "message": "Ничего не найдено"
                }

            # Format results
            formatted = []
            for score, message, context in results:
                formatted.append({
                    "message": message,
                    "similarity": round(float(score), 2),
                    "context": context[:100] if context else ""
                })

            return {
                "success": True,
                "results": formatted,
                "count": len(formatted),
                "message": f"Найдено {len(formatted)} сообщений"
            }

        except Exception as e:
            logger.error(f"Message search error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Ошибка поиска"
            }

# Tool schema for registration
SEARCH_MESSAGES_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Поисковый запрос (ключевые слова)"
        },
        "limit": {
            "type": "integer",
            "description": "Максимальное количество результатов (по умолчанию 5)",
            "default": 5
        },
        "user_id": {
            "type": "integer",
            "description": "ID пользователя для фильтрации (опционально)",
            "nullable": True
        }
    },
    "required": ["query"]
}
