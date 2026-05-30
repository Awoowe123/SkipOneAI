"""Main entry point for Telegram AI Bot"""
import asyncio
import signal
import sys
import logging
from src.core.telegram_client import SmartTelegramClient
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    """Graceful shutdown handler"""
    logger.warning("Shutdown signal received...")
    print("\n⚠️  Shutdown signal received...")
    sys.exit(0)

async def main():
    # Setup logging
    setup_logging()

    logger.info("=" * 50)
    logger.info("Starting Telegram AI Bot")
    logger.info("=" * 50)

    print("🚀 Starting Telegram AI Bot...")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and start client
    client = SmartTelegramClient()

    try:
        await client.start()
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt detected")
        logger.info("Keyboard interrupt detected")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await client.stop()
        print("👋 Bot stopped")
        logger.info("Bot stopped")

if __name__ == '__main__':
    asyncio.run(main())
