"""Proactive agent for monitoring and suggesting topics"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict
from src.core.llm_interface import LLMInterface
from src.config import Config

logger = logging.getLogger(__name__)

class ProactiveAgent:
    """Proactive behavior handler"""

    def __init__(self, telegram_client):
        self.client = telegram_client
        self.llm = LLMInterface()
        self.last_activity: Dict[int, datetime] = {}
        self.running = False
        logger.info("Proactive agent initialized")

    async def start(self):
        """Start background tasks"""
        self.running = True
        asyncio.create_task(self._activity_monitor())
        logger.info("Proactive agent started")

    async def stop(self):
        """Stop the agent"""
        self.running = False
        logger.info("Proactive agent stopped")

    def update_activity(self, chat_id: int):
        """Update last activity timestamp for a chat"""
        self.last_activity[chat_id] = datetime.now()
        logger.debug(f"Updated activity for chat {chat_id}")

    async def _activity_monitor(self):
        """Monitor chat activity in the background"""
        while self.running:
            await asyncio.sleep(Config.behavior.check_activity_interval_minutes * 60)
            await self._check_inactive_chats()

    def set_model_manager(self, model_manager):
        """Set model manager for dynamic loading"""
        self.model_manager = model_manager

    async def _check_inactive_chats(self):
        """Check for inactive chats and suggest topics"""
        if not Config.behavior.proactive_enabled:
            return

        threshold = timedelta(hours=Config.behavior.inactivity_threshold_hours)
        now = datetime.now()

        for chat_id, last_time in list(self.last_activity.items()):
            if now - last_time > threshold:
                logger.info(f"Chat {chat_id} inactive for {Config.behavior.inactivity_threshold_hours}h, suggesting topic")
                await self._suggest_topic(chat_id)

    async def _suggest_topic(self, chat_id: int):
        """Suggest a topic for discussion"""
        try:
            # Ensure model is loaded (if manager available)
            model_name = Config.llm.model # Use default model (likely qwen or GLM)
            # If default model is hot (qwen), no need to load/unload
            # But if user has configured a cold model as main, we need to handle it.
            # Assuming 'chat' usage.

            # Better approach: Use the "chat" model, or just use what LLMInterface uses.
            # But we want to UNLOAD it.

            # Using model_manager if set
            if hasattr(self, 'model_manager') and self.model_manager:
                # Decide which model to use. Let's use the one configured in LLMInterface or just the defaults.
                # Actually, let's use the 'chat' model from router config if possible, but here we just have LLMInterface.
                # Let's assume Config.llm.model is the target.

                # Check if it needs loading
                await self.model_manager.ensure_loaded(model_name)

            # Get context from previous discussions
            messages = await self.client.get_messages(chat_id, limit=20)
            recent_topics = [m.text for m in messages if m.text][:5]

            prompt = f"""На основе предыдущих тем разговора предложи интересную новую тему.

Предыдущие темы:
{chr(10).join(recent_topics)}

Предложи тему КРАТКО, одним предложением, неформально и естественно.
Примеры стиля:
- "Кстати, видел новость про..."
- "Чёт подумал тут..."
- "Слушай, а ты не в курсе про..."

Твоё предложение:"""

            topic = self.llm.generate(prompt, options={'max_tokens': 100})

            if topic and len(topic) > 10:
                await self.client.send_message(chat_id, topic)
                self.last_activity[chat_id] = datetime.now()
                logger.info(f"Suggested topic to chat {chat_id}: {topic[:50]}...")

            # Auto-unload if manager available
            if hasattr(self, 'model_manager') and self.model_manager:
                # Only unload if it's NOT the hot model
                if model_name != self.model_manager.hot_model:
                     logger.info(f"[PROACTIVE] Auto-unloading model: {model_name}")
                     await self.model_manager.unload_model(model_name)

        except Exception as e:
            logger.error(f"Error suggesting topic for {chat_id}: {e}")
