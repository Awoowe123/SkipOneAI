"""
Sticker Management System

Manages indexed sticker packs and provides emoji-based sticker selection.
"""
import logging
import json
import random
from typing import Optional, List, Dict
from pathlib import Path
from collections import defaultdict

from telethon import functions, types

logger = logging.getLogger(__name__)


class StickerManager:
    """Manages sticker packs and emoji-to-sticker mappings"""

    def __init__(self, telegram_client, index_path: str = "data/sticker_index.json"):
        """
        Initialize Sticker Manager

        Args:
            telegram_client: Telethon client instance
            index_path: Path to sticker index JSON
        """
        self.client = telegram_client
        self.index_path = Path(index_path)
        self.emoji_map = defaultdict(list)  # {emoji: [file_id, ...]}
        self.loaded = False

        logger.info("[STICKERS] StickerManager initialized")

    async def load_packs(self, pack_names: List[str]):
        """
        Load sticker packs and build emoji index

        Args:
            pack_names: List of sticker pack short names
        """
        logger.info(f"[STICKERS] Loading {len(pack_names)} sticker packs...")

        for pack_name in pack_names:
            await self._load_pack_generic(
                types.InputStickerSetShortName(short_name=pack_name)
            )

        self.loaded = True
        logger.info(f"[STICKERS] Index ready: {len(self.emoji_map)} emojis, {sum(len(v) for v in self.emoji_map.values())} stickers")

        # Save index to disk
        self._save_index()

    async def load_custom_config(self):
        """Load specific configuration requested by user"""
        logger.info("[STICKERS] Loading custom configuration...")
        self.emoji_map.clear()

        # 1. Load TheMoomintroll (exclude magnifying glass)
        await self._load_pack_generic(
            types.InputStickerSetShortName(short_name="TheMoomintroll"),
            exclude_emojis=['🔎', '🔍']
        )

        # 2. Load second pack by ID (only magnifying glass)
        # Note: Access hash is usually required. Trying 0, but might fail.
        try:
            pack_id = 4889949716810301445
            # We try to find it in user's installed stickers first to get access_hash
            access_hash = 0

            # Try to fetch
            await self._load_pack_generic(
                types.InputStickerSetID(id=pack_id, access_hash=access_hash),
                only_emojis=['🔎', '🔍']
            )

        except Exception as e:
            logger.error(f"[STICKERS] Failed to load second pack (ID {pack_id}): {e}")
            logger.info("Tip: Send a sticker from this pack to the bot to capture the access_hash.")

        self.loaded = True
        self._save_index()

    async def _load_pack_generic(self, input_stickerset, exclude_emojis=None, only_emojis=None):
        """Generic loader with filters"""
        try:
            stickerset = await self.client(
                functions.messages.GetStickerSetRequest(
                    stickerset=input_stickerset,
                    hash=0
                )
            )

            count = 0
            for document in stickerset.documents:
                for attr in document.attributes:
                    if isinstance(attr, types.DocumentAttributeSticker):
                        emoji = attr.alt
                        if not emoji: continue

                        # Filters
                        if exclude_emojis and emoji in exclude_emojis:
                            continue
                        if only_emojis and emoji not in only_emojis:
                            continue

                        self.emoji_map[emoji].append(document.id)
                        count += 1

            logger.info(f"[STICKERS] Loaded {count} stickers from set")

        except Exception as e:
            logger.error(f"[STICKERS] Load error: {e}")
            # Don't raise, just log, so other packs can load
            pass

    def _save_index(self):
        """Save emoji map to disk"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.emoji_map), f, ensure_ascii=False, indent=2)
        logger.info(f"[STICKERS] Index saved to {self.index_path}")

    def _load_index(self):
        """Load emoji map from disk"""
        if self.index_path.exists():
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.emoji_map = defaultdict(list, data)
                self.loaded = True
                logger.info(f"[STICKERS] Loaded index from disk: {len(self.emoji_map)} emojis")
                return True
        return False

    def get_sticker_by_emoji(self, emoji: str) -> Optional[int]:
        """
        Get random sticker file_id for given emoji

        Args:
            emoji: Emoji character (e.g., '😂')

        Returns:
            Sticker file_id or None
        """
        if not self.loaded:
            self._load_index()

        candidates = self.emoji_map.get(emoji, [])
        if candidates:
            selected = random.choice(candidates)
            logger.debug(f"[STICKERS] Selected {emoji} -> {selected}")
            return selected

        logger.warning(f"[STICKERS] No sticker found for emoji '{emoji}'")
        return None

    async def send_sticker(self, chat_id: int, emoji: str) -> bool:
        """
        Send sticker to chat based on emoji

        Args:
            chat_id: Chat ID
            emoji: Emoji to match

        Returns:
            True if sent successfully
        """
        sticker_id = self.get_sticker_by_emoji(emoji)
        if not sticker_id:
            return False

        try:
            await self.client.send_file(
                chat_id,
                file=sticker_id
            )
            logger.info(f"[STICKERS] Sent {emoji} sticker to {chat_id}")
            return True
        except Exception as e:
            logger.error(f"[STICKERS] Failed to send sticker: {e}")
            return False
