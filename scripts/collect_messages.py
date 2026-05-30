"""Script to collect your messages from Telegram for training"""
import sys
import os
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import logging
from telethon import TelegramClient
from src.config import Config
from src.personality.database import PersonalityDatabase
from src.personality.analyzer import MessageAnalyzer
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

async def collect_and_categorize(limit_per_chat: int = 100, total_limit: int = 500):
    """
    Collect your messages and categorize them

    Args:
        limit_per_chat: How many messages to collect from each chat
        total_limit: Overall message limit
    """
    setup_logging()

    client = TelegramClient(
        Config.telegram.session_path,
        Config.telegram.api_id,
        Config.telegram.api_hash
    )

    await client.start(phone=Config.telegram.phone)

    db = PersonalityDatabase()
    analyzer = MessageAnalyzer()

    print("📥 Collecting your messages...")
    logger.info("Starting message collection...")

    # Get all dialogs
    dialogs = await client.get_dialogs(limit=100)

    collected = 0
    for dialog in dialogs:
        if collected >= total_limit:
            break

        try:
            # Get dialog partner info
            chat_id = dialog.id
            chat_name = dialog.name or "Unknown"

            # For private chats, get user ID
            user_id = None
            if dialog.is_user:
                user_id = dialog.entity.id

            # Collect YOUR messages (as before)
            your_messages = await client.get_messages(
                dialog.id,
                limit=limit_per_chat,
                from_user='me'
            )

            for msg in your_messages:
                if not msg.text or len(msg.text) < 10:
                    continue

                category = analyzer.categorize_message(msg.text)
                db.add_example(
                    category=category,
                    message=msg.text,
                    metadata={'chat': chat_name, 'date': str(msg.date), 'source': 'me'},
                    chat_id=chat_id,
                    user_id=user_id
                )

                collected += 1
                if collected % 10 == 0:
                    print(f"Processed: {collected}/{total_limit}")
                    logger.info(f"Processed: {collected}/{total_limit}")

                if collected >= total_limit:
                    break

            # Also collect from TARGET USER (BOT_OWNER_ID from .env)
            target_user_id = Config.telegram.bot_owner_id
            target_messages = await client.get_messages(
                dialog.id,
                limit=limit_per_chat,
                from_user=target_user_id
            )

            for msg in target_messages:
                if not msg.text or len(msg.text) < 10:
                    continue

                category = analyzer.categorize_message(msg.text)
                db.add_example(
                    category=category,
                    message=msg.text,
                    metadata={'chat': chat_name, 'date': str(msg.date), 'source': 'target_user'},
                    chat_id=chat_id,
                    user_id=target_user_id  # Mark as coming from this specific user
                )

                collected += 1
                if collected % 10 == 0:
                    print(f"Processed: {collected}/{total_limit} (including target user)")
                    logger.info(f"Processed: {collected}/{total_limit}")

                if collected >= total_limit:
                    break

        except Exception as e:
            print(f"Error processing {dialog.name}: {e}")
            logger.error(f"Error processing {dialog.name}: {e}")
            continue

    print(f"\n✅ Collected {collected} messages")
    print("\n📊 Statistics:")
    stats = db.get_category_stats()
    for category, count in stats.items():
        print(f"  {category}: {count}")

    logger.info(f"Collection complete: {collected} messages")
    logger.info(f"Stats: {stats}")

    db.close()
    await client.disconnect()

if __name__ == '__main__':
    asyncio.run(collect_and_categorize(limit_per_chat=100, total_limit=500))
