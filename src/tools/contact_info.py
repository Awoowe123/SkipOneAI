"""Contact info tool"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ContactInfoTool:
    """Get contact information"""

    def __init__(self, contacts_manager):
        self.contacts = contacts_manager

    async def get_contact_info(
        self,
        name: str
    ) -> Dict[str, Any]:
        """
        Get information about a contact

        Args:
            name: Contact name or username

        Returns:
            Contact information
        """
        try:
            logger.info(f"Getting contact info for: {name}")

            contact = self.contacts.find_by_name(name)

            if not contact:
                return {
                    "success": False,
                    "message": f"Контакт '{name}' не найден"
                }

            result = {
                "success": True,
                "contact": {
                    "name": contact['first_name'],
                    "username": contact['username'],
                    "user_id": contact['user_id'],
                    "bio": contact.get('bio', 'Не задан'),
                    "message_count": contact.get('message_count', 0),
                    "last_message": contact.get('last_message', 'Нет сообщений')[:100]
                }
            }

            return result

        except Exception as e:
            logger.error(f"Contact info error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Ошибка получения информации"
            }

# Tool schema
GET_CONTACT_INFO_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Имя или username контакта"
        }
    },
    "required": ["name"]
}
