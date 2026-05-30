import os
import logging
from typing import Set
from telethon.tl.functions.contacts import GetContactsRequest

logger = logging.getLogger(__name__)

class PermissionsManager:
    """Manages access control for the bot"""

    # Creator ID loaded from environment variable BOT_OWNER_ID
    CREATOR_ID = int(os.getenv('BOT_OWNER_ID', 0))

    def __init__(self):
        self.allowed_users: Set[int] = {self.CREATOR_ID}
        self.is_initialized = False

    async def load_contacts(self, client):
        """
        Load contacts from Telegram and add them to allowed users

        Args:
            client: Telethon client instance
        """
        try:
            logger.info("Loading contacts for whitelist...")
            # Use the actual client (unwrap if it's SmartTelegramClient)
            actual_client = getattr(client, 'client', client)

            # Get contacts using raw API request
            # hash=0 means get all contacts
            result = await actual_client(GetContactsRequest(hash=0))

            # Result contains 'users' list
            # In some versions result is just User object list, or Contacts object
            # Contacts object has .users
            if hasattr(result, 'users'):
                contacts = result.users
            else:
                # Direct list
                contacts = result

            count = 0
            for contact in contacts:
                # User object has .id
                if hasattr(contact, 'id'):
                    self.allowed_users.add(contact.id)
                    count += 1

            self.is_initialized = True
            logger.info(f"✅ Access Control: Whitelisted {count} contacts + Creator")

        except Exception as e:
            logger.error(f"Failed to load contacts: {e}", exc_info=True)
            # Fallback: only creator allowed if load fails

    def is_allowed(self, user_id: int) -> bool:
        """Check if user is allowed to interact"""
        if not self.is_initialized:
            # If not initialized yet, only allow creator (safety first)
            return user_id == self.CREATOR_ID

        return user_id in self.allowed_users

    def add_user(self, user_id: int):
        """Manually add user to whitelist (runtime)"""
        self.allowed_users.add(user_id)
        logger.info(f"Manually whitelisted user {user_id}")
