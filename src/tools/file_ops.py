"""File operations tool"""
import logging
import asyncio
import os
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FileOpsTool:
    """Tool for reading, writing, downloading and sending files"""

    def __init__(self, client):
        self.client = client

    async def read_file(self, path: str) -> Dict[str, Any]:
        """Read text file content"""
        try:
            if not os.path.exists(path):
                return {"success": False, "message": f"Файл не найден: {path}"}

            content = await asyncio.to_thread(self._read_file_sync, path)
            return {
                "success": True,
                "content": content,
                "message": f"Прочитан файл {path} ({len(content)} символов)"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _read_file_sync(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    async def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Create or overwrite file"""
        try:
            # Ensure directory exists
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            await asyncio.to_thread(self._write_file_sync, path, content)
            return {
                "success": True,
                "path": path,
                "message": f"Файл успешно записан: {path}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _write_file_sync(self, path: str, content: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    async def download_file(self, url: str, path: str) -> Dict[str, Any]:
        """Download file from URL"""
        try:
            logger.info(f"Downloading {url} to {path}")

            # Use requests in thread
            def _download():
                response = requests.get(url, stream=True, timeout=30)
                if response.status_code == 200:
                    directory = os.path.dirname(path)
                    if directory:
                        os.makedirs(directory, exist_ok=True)
                    with open(path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    return True
                return False

            success = await asyncio.to_thread(_download)

            if success:
                return {"success": True, "message": f"Файл скачан: {path}"}
            else:
                return {"success": False, "message": "Ошибка скачивания (не 200 OK)"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def send_file(self, path: str, caption: str = "") -> Dict[str, Any]:
        """Send local file to current chat"""
        try:
            if not os.path.exists(path):
                return {"success": False, "message": f"Файл не найден: {path}"}

            # We need chat_id from context.
            # Current implementation of execute_tool maps arguments.
            # But we don't have chat_id in arguments unless LLM provides it or we inject it.
            # Usually tool call arguments come from LLM.
            # So send_file wrapper in EventProcessor needs to inject chat_id?
            # Or we strictly require chat_id in schema?
            # LLM doesn't always know chat_id effectively.
            # Better: EventProcessor handles 'send_file' special case or passes context?

            # For simplicity, let's ask LLM to NOT provide chat_id, and we rely on 'current chat' concept?
            # But ToolsManager is generic.

            # Solution: We will inject chat_id via kwargs if we modify ToolsManager or EventProcessor wrapper.
            # For now, let's assume 'send_file' is called with chat_id argument by generic wrapper?
            # No, LLM generates args.

            # Let's add 'chat_id' to schema and hope logic allows injection?
            # Or, we make 'send_file' return a special signal to EventProcessor "Please upload this file"?
            # Return {"action": "upload", "path": path}.
            # EventProcessor sees this and uploads.

            # WAIT. I have client reference!
            # But I need to know DESTINATION chat_id.
            # If I can't trust LLM to know chat_id (it's hidden).

            # I will assume that send_file is used in context where chat_id is injected into arguments
            # OR I modify EventProcessor to inject `current_chat_id` into tool calls if tool accepts it.

            # Let's verify implementation of execute_tool in tools_manager.
            # It takes `arguments`.

            # I will define schema with optional chat_id, but logically I need it.
            # Hack: I will store `current_chat_id` in the tool instance? No, concurrency issue.

            # OK, I will modify EventProcessor later to inject `chat_id` into arguments for tools that need it.
            pass

            # For now, implement method assuming chat_id is passed
            return {"success": False, "message": "Требуется chat_id для отправки (будет реализовано в EventProcessor)"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def send_file_to_chat(self, chat_id: int, path: str, caption: str = "") -> Dict[str, Any]:
        """Actual send implementation"""
        try:
             actual_client = getattr(self.client, 'client', self.client)
             await actual_client.send_file(chat_id, path, caption=caption)
             return {"success": True, "message": f"Файл отправлен в чат {chat_id}"}
        except Exception as e:
             return {"success": False, "error": str(e)}

# Schemas
FILE_READ_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"]
}

FILE_WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"}
    },
    "required": ["path", "content"]
}

FILE_DOWNLOAD_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "path": {"type": "string"}
    },
    "required": ["url", "path"]
}

FILE_SEND_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "caption": {"type": "string"}
    },
    "required": ["path"]
}
