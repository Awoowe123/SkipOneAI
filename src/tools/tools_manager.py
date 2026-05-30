"""Tools manager for LLM function calling"""
import logging
from typing import List, Dict, Any, Optional, Callable
import json

logger = logging.getLogger(__name__)

class ToolsManager:
    """Manage available tools for LLM function calling"""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_schemas: List[Dict] = []
        logger.info("Tools manager initialized")

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Dict[str, Any]
    ):
        """
        Register a new tool

        Args:
            name: Tool name
            func: Function to execute
            description: Tool description for LLM
            parameters: JSON schema for parameters
        """
        self.tools[name] = func

        # OpenAI function calling schema
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }

        self.tool_schemas.append(schema)
        logger.info(f"Registered tool: {name}")

    def get_schemas(self) -> List[Dict]:
        """Get all tool schemas for LLM"""
        return self.tool_schemas

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool by name

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if name not in self.tools:
            logger.error(f"Unknown tool: {name}")
            return {"error": f"Tool '{name}' not found"}

        try:
            logger.info(f"Executing tool: {name} with args: {arguments}")
            result = await self.tools[name](**arguments)
            logger.info(f"Tool {name} returned: {str(result)[:100]}...")
            return result
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}", exc_info=True)
            return {"error": str(e)}

    def parse_tool_calls(self, llm_response: Dict) -> Optional[List[Dict]]:
        """
        Parse tool calls from LLM response

        Args:
            llm_response: LLM API response

        Returns:
            List of tool calls or None
        """
        try:
            message = llm_response.get('choices', [{}])[0].get('message', {})
            tool_calls = message.get('tool_calls')

            if tool_calls:
                logger.info(f"LLM requested {len(tool_calls)} tool call(s)")
                return tool_calls

            return None

        except Exception as e:
            logger.error(f"Error parsing tool calls: {e}")
            return None
