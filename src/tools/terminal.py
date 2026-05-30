"""Terminal tool for executing safe commands"""
import logging
import subprocess
import asyncio
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class TerminalTool:
    """Safe terminal command execution tool"""

    # Whitelist of allowed commands for safety
    ALLOWED_COMMANDS = {
        'ping', 'echo', 'dir', 'ls', 'ipconfig', 'ifconfig',
        'whoami', 'netstat', 'nslookup', 'hostname', 'date', 'time',
        'type', 'cat', 'more', 'tree', 'ver', 'systeminfo'
    }

    async def run_command(self, command: str) -> Dict[str, Any]:
        """
        Run a safe shell command
        Supports chaining with ';' (e.g. 'ping ya.ru; ipconfig')
        """
        try:
            # Handle chained commands
            # Replace && with ; for compatibility if user tries unix style
            sub_commands = [cmd.strip() for cmd in command.replace('&&', ';').split(';') if cmd.strip()]

            if not sub_commands:
                return {"success": False, "message": "Пустая команда"}

            # Validate ALL commands in the chain
            for sub_cmd in sub_commands:
                parts = sub_cmd.split()
                if not parts:
                    continue

                base_cmd = parts[0].lower()
                if base_cmd not in self.ALLOWED_COMMANDS:
                    return {
                        "success": False,
                        "message": f"Команда '{base_cmd}' (в '{sub_cmd}') запрещена. Разрешены: {', '.join(sorted(self.ALLOWED_COMMANDS))}"
                    }

            logger.info(f"Executing shell command sequence: {command}")

            full_output = []

            # Execute commands sequentially to avoid shell separator issues
            for i, sub_cmd in enumerate(sub_commands):
                logger.info(f"Executing sub-command {i+1}/{len(sub_commands)}: {sub_cmd}")

                process = await asyncio.create_subprocess_shell(
                    sub_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await process.communicate()

                cmd_output = ""
                if stdout:
                    cmd_output += stdout.decode('cp866', errors='replace')
                if stderr:
                    cmd_output += "\nErrors:\n" + stderr.decode('cp866', errors='replace')

                # Format output for each command
                header = f"--- Command: {sub_cmd} ---"
                full_output.append(f"{header}\n{cmd_output.strip()}")

            return {
                "success": True,
                "output": "\n\n".join(full_output),
                "command": command
            }

        except Exception as e:
            logger.error(f"Terminal error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Ошибка выполнения команды"
            }

# Schema
TERMINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Консольная команда (PowerShell). МОЖНО выполнять несколько команд, разделяя их ';'. Разрешено: ping, dir, ipconfig, netstat и другие безопасные команды."
        }
    },
    "required": ["command"]
}
