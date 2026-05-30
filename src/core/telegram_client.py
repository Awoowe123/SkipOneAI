import asyncio
import logging
from telethon import TelegramClient, events
from src.config import Config
from src.core.event_processor import EventProcessor
from src.behavior.proactive_agent import ProactiveAgent

logger = logging.getLogger(__name__)

class SmartTelegramClient:
    """Wrapper over Telethon with AI integration"""

    def __init__(self):
        proxy_config = self._get_proxy_config()

        self.client = TelegramClient(
            Config.telegram.session_path,
            Config.telegram.api_id,
            Config.telegram.api_hash,
            proxy=proxy_config
        )

        try:
            import cryptg
            logger.info("🚀 Cryptg detected! High-speed AES encryption enabled.")
        except ImportError:
            logger.warning("🐌 Cryptg NOT found. Downloads will be slow. Install it with: pip install cryptg")

        self.processor = EventProcessor(self.client)
        self.proactive_agent = ProactiveAgent(self.client)

        # Link ProactiveAgent with ModelManager for loading/unloading
        if hasattr(self.processor, 'model_manager'):
            self.proactive_agent.set_model_manager(self.processor.model_manager)

        self._setup_handlers()
        logger.info("Smart Telegram client initialized")

    def _get_proxy_config(self):
        """
        Resolves proxy configuration from Config or System.
        """
        # 1. Manual Proxy
        if Config.telegram.proxy_string:
            try:
                # Format: type:host:port or type:host:port:user:pass
                parts = Config.telegram.proxy_string.split(':')
                if len(parts) >= 3:
                     proxy_type = parts[0].lower() # 'socks5', 'http'
                     addr = parts[1]
                     port = int(parts[2])

                     proxy = {
                         'proxy_type': proxy_type,
                         'addr': addr,
                         'port': port,
                         'rdns': True
                     }

                     if len(parts) >= 5:
                         proxy['username'] = parts[3]
                         proxy['password'] = parts[4]

                     logger.info(f"Using manual proxy: {proxy_type}://{addr}:{port}")
                     return proxy
            except Exception as e:
                logger.error(f"Failed to parse proxy string: {e}")

        # 2. System Proxy
        if Config.telegram.system_proxy_enabled:
            try:
                import urllib.request
                from urllib.parse import urlparse

                proxies = urllib.request.getproxies()
                # Prefer https, then http for Telegram
                proxy_url = proxies.get('https') or proxies.get('http')

                if proxy_url:
                    parsed = urlparse(proxy_url)
                    # Scheme for python-socks/pysocks usually 'http'
                    scheme = parsed.scheme
                    if 'socks' not in scheme and 'http' not in scheme:
                        scheme = 'http'

                    proxy = {
                        'proxy_type': scheme,
                        'addr': parsed.hostname,
                        'port': parsed.port,
                        'rdns': True
                    }

                    if parsed.username:
                        proxy['username'] = parsed.username
                    if parsed.password:
                        proxy['password'] = parsed.password

                    logger.info(f"Using system proxy: {scheme}://{parsed.hostname}:{parsed.port}")
                    return proxy

            except Exception as e:
                logger.error(f"Failed to resolve system proxy: {e}")

        return None

    def _setup_handlers(self):
        """Register event handlers"""

        @self.client.on(events.NewMessage(incoming=True, func=lambda e: e.is_private and not (e.voice or e.audio or e.video_note)))
        async def handle_private_message(event):
            """Handle private messages"""
            try:
                # Check if CONTACTS_ONLY is enabled
                if Config.telegram.contacts_only:
                    # Get sender entity
                    sender = await event.get_sender()

                    # Check if sender is a contact
                    if not sender.mutual_contact and not sender.contact:
                        logger.debug(f"Ignored message from non-contact: {event.chat_id} (@{sender.username if sender.username else 'N/A'})")
                        return  # Ignore non-contacts

                logger.info(f"Received private message from chat {event.chat_id}")
                response = await self.processor.process_message(event)

                if response:
                    await self._safe_reply(event, response)
                    logger.info(f"[SENT] {response[:100]}...")

                # Update activity for proactive agent
                self.proactive_agent.update_activity(event.chat_id)

            except Exception as e:
                logger.error(f"[ERROR] Handling message: {e}", exc_info=True)

        @self.client.on(events.UserUpdate)
        async def handle_user_typing(event):
            """Detect typing events for sticker debounce"""
            try:
                if hasattr(event, 'typing') and event.typing:
                    chat_id = event.chat_id or event.user_id
                    await self.processor.on_typing_detected(chat_id)
            except Exception as e:
                logger.error(f"[ERROR] Handling typing event: {e}", exc_info=True)

        @self.client.on(events.NewMessage(incoming=True, func=lambda e: e.is_private and (e.voice or e.audio or e.video_note)))
        async def handle_private_audio(event):
            """Handle private audio/voice/video_note"""
            try:
                await self.processor.handle_audio_event(event)
            except Exception as e:
                logger.error(f"[ERROR] Handling private audio: {e}", exc_info=True)

        # Callback handler removed (Userbots cannot use inline buttons)

        @self.client.on(events.NewMessage(incoming=True, func=lambda e: e.is_group))
        async def handle_group_message(event):
            """Handle group messages - respond only to mentions/replies"""
            try:
                me = await self.client.get_me()

                # Check if message is a reply to our message
                is_reply_to_us = False
                if event.is_reply:
                    replied_msg = await event.get_reply_message()
                    if replied_msg and replied_msg.sender_id == me.id:
                        is_reply_to_us = True

                # Check if we are mentioned
                is_mentioned = False
                if event.message.text and f"@{me.username}" in event.message.text:
                    is_mentioned = True

                # Respond only if mentioned or replied to
                if is_reply_to_us or is_mentioned:
                    logger.info(f"Received group message (reply/mention) from chat {event.chat_id}")

                    # SPECIAL CASE: Reply to Audio + Mention
                    if event.is_reply:
                        replied_msg = await event.get_reply_message()
                        if replied_msg and (replied_msg.voice or replied_msg.audio or replied_msg.video_note):
                            logger.info("[GROUP] Detected reply to audio with mention")
                            await self.processor.handle_audio_reply(event)
                            return

                    response = await self.processor.process_message(event)

                    if response:
                        await self._safe_reply(event, response)
                        logger.info(f"[SENT GROUP] {response[:100]}...")

                    self.proactive_agent.update_activity(event.chat_id)
                else:
                    logger.debug(f"Group message (ignored) in {event.chat_id}: {event.text[:50] if event.text else 'N/A'}...")

            except Exception as e:
                logger.error(f"[ERROR] Handling group message: {e}", exc_info=True)

        logger.info("Event handlers registered")

    async def _safe_reply(self, event, message: str, retries: int = 3):
        """
        Reply with retry logic for unstable connections

        Args:
            event: Telegram event
            message: Message text
            retries: Number of retries
        """

        for attempt in range(retries):
            try:
                # Parse sticker tag first
                if attempt == 0 and "<sticker>" in message:
                    import re
                    sticker_match = re.search(r'<sticker>(.*?)</sticker>', message)
                    if sticker_match:
                        emoji = sticker_match.group(1).strip()
                        message = message.replace(sticker_match.group(0), '').strip()
                        try:
                            if hasattr(self.processor, 'sticker_manager'):
                                await self.processor.sticker_manager.send_sticker(event.chat_id, emoji)
                                await asyncio.sleep(0.5)
                        except Exception as e:
                            logger.error(f"[SAFE REPLY] Failed to send sticker: {e}")

                # Attempt to send
                await event.reply(message)
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to send reply (attempt {attempt+1}/{retries}): {e}")

                # Check for disconnection
                if "Server closed the connection" in str(e) or "Connection" in str(e):
                    logger.warning("♻️ Connection lost. Waiting before retry...")

                if attempt < retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))  # Exponential backoff

        logger.error(f"❌ Failed to send message after {retries} attempts.")

    async def start(self):
        """Start the client"""
        await self.client.start(phone=Config.telegram.phone)
        logger.info("✅ Telegram client started")

        me = await self.client.get_me()
        logger.info(f"Logged in as: {me.first_name} (@{me.username})")

        # Load contacts for Access Control
        await self.processor.permissions.load_contacts(self.client)

        # Load user-configured stickers
        try:
            await self.processor.sticker_manager.load_custom_config()
        except Exception as e:
            logger.error(f"Failed to load stickers: {e}")

        # Start proactive agent
        await self.proactive_agent.start()
        logger.info("✅ Proactive agent started")

        print(f"🤖 Bot is running 24/7...")
        print(f"📱 Account: {me.first_name} (@{me.username})")
        print(f"💬 Waiting for messages...")

        await self.client.run_until_disconnected()

    async def stop(self):
        """Stop the client"""
        await self.proactive_agent.stop()
        await self.client.disconnect()
        self.processor.cleanup()
        logger.info("Client stopped")
