"""Tool for summarizing chat history"""
import logging
import asyncio
from typing import Dict, Any, List
from src.core.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class SummarizerTool:
    """Tool to summarize chat history"""

    def __init__(self, client, llm: LLMInterface):
        self.client = client
        self.llm = llm

    async def summarize_chat(
        self,
        chat_id: int,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Summarize the last N messages in the chat

        Args:
            chat_id: Chat ID to summarize
            limit: Number of messages to analyze (default 100, max 300)

        Returns:
            Summary of the conversation
        """
        try:
            limit = min(limit, 300) # Safety cap
            logger.info(f"Summarizing last {limit} messages for chat {chat_id}")

            # Use actual client to fetch history
            actual_client = getattr(self.client, 'client', self.client)

            messages_text = []

            # Fetch messages
            # We use reverse=True to get oldest first for reading flow?
            # Telethon get_messages returns newest first.
            # We want chronological order for summary.
            history = await actual_client.get_messages(chat_id, limit=limit)

            if not history:
                 return {"result": "В чате нет сообщений для анализа."}

            for msg in reversed(history):
                if msg.text:
                    sender = "Я" if msg.out else getattr(msg.sender, 'first_name', 'Unknown')
                    # Improve sender name resolution if possible (requires entity cache)
                    # For now "Unknown" or simple name is fine.
                    messages_text.append(f"{sender}: {msg.text}")

            if not messages_text:
                return {"result": "Нет текстовых сообщений для анализа."}

            # Prepare text block
            full_text = "\n".join(messages_text)

            # Send to LLM for summarization
            # We use a specific prompt for this
            prompt = f"""Проанализируй переписку и сделай краткую выжимку (TL;DR).

Переписка:
{full_text}

Задание:
1. Кратко опиши основные темы обсуждения.
2. Кто были главные участники и их позиции.
3. Были ли конфликты, шутки или важные решения.
4. Итог разговора.

Ответ должен быть кратким (1-2 абзаца), вежливым и информативным."""

            # Generate summary
            logger.info(f"Sending {len(full_text)} chars to LLM for summary")
            summary = await asyncio.to_thread(
                self.llm.generate,
                prompt,
                {'temperature': 0.5, 'max_tokens': 400}
            )

            if isinstance(summary, dict):
                 summary = summary.get('choices', [{}])[0].get('message', {}).get('content', '')

            return {
                "success": True,
                "summary": summary.strip(),
                "message": f"Проанализировано {len(messages_text)} сообщений."
            }

        except Exception as e:
            logger.error(f"Summarizer error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Ошибка при создании саммари"
            }

# Schema
SUMMARIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "limit": {
            "type": "integer",
            "description": "Количество сообщений для анализа (обычно 50-200). Если пользователь пишет 'последние сообщения', используй 50. Если 'за сегодня' или 'много', используй 100-200.",
            "default": 100
        }
    }
}
