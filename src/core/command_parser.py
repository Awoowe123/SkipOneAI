"""Command parser for bot commands"""
import re
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

class CommandParser:
    """Parse and execute bot commands"""

    # Command patterns
    PATTERNS = {
        'bio_set': r'^/bio\s+(.+?):\s*(.+)$',
        'bio_show': r'^/bio\s+show\s+(.+)$',
        'bio_analyze': r'^/bio\s+analyze\s+(.+)$',
        'bio_list': r'^/bio\s+list$',
        'style': r'^/style(?:\s+(update|\d+))?(?:\s+(-?\d+))?$',  # /style, /style update, /style 500 -100XXXXXXXXXX
        'reply_to': r'^(?:@\w+[,\s]+)?(?:ответь|напиши|скажи)\s+(.+?)(?:\s+(?:что|на|про)\s+(.+))?$',
        'load_stickers': r'^/load_stickers$',
        'help': r'^/help$'
    }

    def parse(self, message: str) -> Optional[Dict]:
        """
        Parse message for commands

        Args:
            message: Message text

        Returns:
            Command dict or None if not a command
        """
        message = message.strip()

        # Bio commands
        if message.startswith('/bio'):
            return self._parse_bio_command(message)

        # Help
        if message.startswith('/help'):
            return {'type': 'help'}

        # Reset
        if message.startswith('/reset'):
            return {'type': 'reset'}

        # Stickers
        if message.startswith('/load_stickers'):
            return {'type': 'load_stickers'}

        # Style command
        if message.startswith('/style'):
            match = re.match(self.PATTERNS['style'], message, re.IGNORECASE)
            if match:
                arg1 = match.group(1)  # 'update' or limit number
                arg2 = match.group(2)  # chat_id (optional)

                args = []
                limit = None
                chat_id = None

                # Parse arguments
                if arg1:
                    if arg1.lower() == 'update':
                        args.append('update')
                    elif arg1.isdigit():
                        limit = int(arg1)
                        args.append(str(limit))

                if arg2:
                    chat_id = int(arg2)
                    args.append(str(chat_id))

                return {
                    'type': 'style',
                    'args': args,
                    'limit': limit,
                    'chat_id': chat_id
                }

        # Reply-to commands (natural language)
        reply_cmd = self._parse_reply_command(message)
        if reply_cmd:
            return reply_cmd

        return None

    def _parse_bio_command(self, message: str) -> Optional[Dict]:
        """Parse /bio commands"""

        # /bio Name: bio text
        match = re.match(self.PATTERNS['bio_set'], message, re.IGNORECASE | re.DOTALL)
        if match:
            name, bio = match.groups()
            return {
                'type': 'bio_set',
                'name': name.strip(),
                'bio': bio.strip()
            }

        # /bio analyze Name
        match = re.match(self.PATTERNS['bio_analyze'], message, re.IGNORECASE)
        if match:
            return {
                'type': 'bio_analyze',
                'name': match.group(1).strip()
            }

        # /bio show Name
        match = re.match(self.PATTERNS['bio_show'], message, re.IGNORECASE)
        if match:
            return {
                'type': 'bio_show',
                'name': match.group(1).strip()
            }

        # /bio list
        if re.match(self.PATTERNS['bio_list'], message, re.IGNORECASE):
            return {'type': 'bio_list'}

        return None

    def _parse_reply_command(self, message: str) -> Optional[Dict]:
        """
        Parse natural language reply commands

        Examples:
        - "ответь Ане"
        - "@bot, напиши Васе про ремонт"
        - "скажи Петру что буду позже"
        """
        match = re.match(self.PATTERNS['reply_to'], message, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            topic = match.group(2).strip() if match.group(2) else None

            return {
                'type': 'reply_to',
                'name': name,
                'topic': topic
            }

        return None

    @staticmethod
    def format_help() -> str:
        """Return help message"""
        return """**Доступные команды:**

**Управление контактами (Bio):**
• `/bio Имя: описание` — добавить/обновить bio контакта вручную
• `/bio analyze @username` — автоматический анализ отношений через LLM
• `/bio show @username` — показать bio контакта
• `/bio list` — список всех контактов
• `/reset` — СБРОСИТЬ память бота обо мне (забыть контекст)

**Примеры bio:**
• `/bio @l_SkipOne_l: Друг детства, троллю за рост`
• `/bio analyze @vasya` — LLM сам проанализирует переписку
• `/bio show Коля` — можно и по имени

**Стиль владельца (только для owner):**
• `/style` — показать текущий анализ твоего стиля
• `/style update` — обновить анализ на основе всех примеров
• `/style <лимит> <chat_id>` — быстрый анализ из конкретного чата
  Пример: `/style 1000 -100XXXXXXXXXX` (соберёт 1000 твоих сообщений)

**Команды для ответов:**
• `ответь Ане` — сгенерировать ответ для Ани
• `напиши Васе про встречу` — ответить на тему "встреча"

**Auto-Bio:**
После 50 сообщений с человеком бот автоматически анализирует отношения и создаёт bio.

**Разница между Bio и Style:**
• **Bio** = досье на КОНТАКТОВ (кто они, о чём говорим)
• **Style** = КАК пишет ВЛАДЕЛЕЦ (бот копирует твой стиль)

*Примечание: бот НЕ отправляет сообщения автоматически, только генерирует текст для вас.*"""

    @staticmethod
    def is_command(message: str) -> bool:
        """Check if message is a command"""
        message = message.strip().lower()
        return (
            message.startswith('/bio') or
            message.startswith('/reset') or
            message.startswith('/load_stickers') or
            message.startswith('/help') or
            message.startswith('/style') or
            bool(re.match(r'^(?:@\w+[,\s]+)?(?:ответь|напиши|скажи)\s+', message, re.IGNORECASE))
        )
