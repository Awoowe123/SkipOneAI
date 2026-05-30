"""
XML Context Manager - Serializes conversation context to compact XML format

Replaces KV Cache with static context injection, enabling model swaps without losing context.
"""
import logging
from typing import List, Dict, Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages conversation context serialization to XML"""

    def __init__(self, contacts_manager, personality_db):
        """
        Initialize Context Manager

        Args:
            contacts_manager: ContactsManager instance for bio data
            personality_db: PersonalityDatabase instance for message history
        """
        self.contacts = contacts_manager
        self.personality_db = personality_db

    def serialize_context(self, chat_id: int, user_id: int, recent_limit: int = 10) -> str:
        """
        Serialize full conversation context to XML

        Args:
            chat_id: Chat ID
            user_id: User ID
            recent_limit: Number of recent messages to include

        Returns:
            XML string with compressed context
        """
        try:
            # Root element
            conversation = ET.Element("conversation")

            # Metadata
            metadata = ET.SubElement(conversation, "metadata")
            metadata.set("user_id", str(user_id))
            metadata.set("chat_id", str(chat_id))

            # Context (bio, style, examples)
            context = ET.SubElement(conversation, "context")

            # User bio
            contact = self.contacts.get_contact(user_id)
            if contact and contact.get('bio'):
                bio_elem = ET.SubElement(context, "bio")
                bio_elem.text = contact['bio']

            # Owner style (if available)
            # Note: This would need access to owner style data
            # For now, we'll skip it or add later

            # Recent message history (compressed)
            history = ET.SubElement(conversation, "history")
            history.set("recent", str(recent_limit))

            messages = self.personality_db.get_user_messages(user_id, limit=recent_limit)
            for msg in messages:
                msg_elem = ET.SubElement(history, "msg")
                msg_elem.set("role", msg.get('type', 'user'))  # 'user' or 'assistant'
                msg_elem.text = msg.get('text', '')

            # Convert to pretty XML string
            xml_str = self._prettify_xml(conversation)

            logger.debug(f"[CONTEXT] Serialized {len(messages)} messages to XML ({len(xml_str)} chars)")
            return xml_str

        except Exception as e:
            logger.error(f"[CONTEXT] Serialization error: {e}", exc_info=True)
            return ""

    def _prettify_xml(self, elem: ET.Element) -> str:
        """
        Return a pretty-printed XML string

        Args:
            elem: XML Element tree

        Returns:
            Formatted XML string
        """
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def compress_history(self, messages: List[Dict], limit: int = 10) -> str:
        """
        Compress message history to compact format

        Args:
            messages: List of message dicts
            limit: Max messages to include

        Returns:
            Compressed XML fragment
        """
        history = ET.Element("history")
        history.set("count", str(min(len(messages), limit)))

        for msg in messages[-limit:]:  # Recent N messages
            msg_elem = ET.SubElement(history, "msg")
            msg_elem.set("role", msg.get('type', 'user'))

            # Truncate very long messages
            text = msg.get('text', '')
            if len(text) > 500:
                text = text[:497] + "..."
            msg_elem.text = text

        return ET.tostring(history, encoding='unicode')

    def inject_xml_context(self, prompt: str, chat_id: int, user_id: int) -> str:
        """
        Inject XML context into prompt

        Args:
            prompt: Original prompt
            chat_id: Chat ID
            user_id: User ID

        Returns:
            Augmented prompt with XML context prepended
        """
        xml_context = self.serialize_context(chat_id, user_id, recent_limit=10)

        if not xml_context:
            return prompt

        augmented = f"""<context_memory>
{xml_context}
</context_memory>

---

{prompt}"""

        logger.debug(f"[CONTEXT] Injected XML context ({len(xml_context)} chars)")
        return augmented

    async def build_full_xml_context(
        self,
        messages: List[object],
        owner_style: Optional[str] = None,
        user_bio: Optional[str] = None,
        sender_name: Optional[str] = None
    ) -> str:
        """
        Build a comprehensive XML context block for the LLM.

        Structure:
        <memory_context>
            <user_profile>
                <name>...</name>
                <bio>...</bio>
            </user_profile>
            <bot_persona>
                <style_guide>...</style_guide>
            </bot_persona>
            <conversation_history>
                <msg role="user" time="HH:MM">...</msg>
                <msg role="assistant" time="HH:MM">...</msg>
                <reply_context>...</reply_context>
            </conversation_history>
        </memory_context>

        Args:
            messages: List of Telethon message objects (reversed, newest last)
            owner_style: Text description of owner's style
            user_bio: Text description of user (relationship)
            sender_name: Name of the user

        Returns:
            Formatted XML string
        """
        try:
            root = ET.Element("memory_context")

            # 1. User Profile
            user_profile = ET.SubElement(root, "user_profile")
            if sender_name:
                ET.SubElement(user_profile, "name").text = sender_name
            if user_bio:
                ET.SubElement(user_profile, "bio").text = user_bio

            # 2. Bot Persona (Style)
            bot_profile = ET.SubElement(root, "bot_persona")
            if owner_style:
                style_elem = ET.SubElement(bot_profile, "style_guide")
                style_elem.text = owner_style

            # 3. Conversation History
            history_elem = ET.SubElement(root, "conversation_history")

            for msg in messages:
                # Determine role
                role = "assistant" if msg.out else "user"

                # Create message element
                msg_node = ET.SubElement(history_elem, "msg")
                msg_node.set("role", role)

                # Add timestamp if available
                if hasattr(msg, 'date'):
                     msg_node.set("time", msg.date.strftime("%H:%M"))

                # Add text content
                if msg.text:
                    msg_node.text = msg.text

                # Add reply context if present
                if msg.reply_to:
                    try:
                        reply_msg = await msg.get_reply_message()
                        if reply_msg and reply_msg.text:
                             reply_node = ET.SubElement(msg_node, "reply_context")
                             reply_sender = "assistant" if reply_msg.out else "user"
                             reply_node.set("from_role", reply_sender)
                             reply_node.text = reply_msg.text[:100] + ("..." if len(reply_msg.text) > 100 else "")
                    except Exception:
                        pass

            # Prettify
            return self._prettify_xml(root)

        except Exception as e:
            logger.error(f"[CONTEXT] XML Build failed: {e}", exc_info=True)
            return ""
