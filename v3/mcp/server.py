"""
MCP Tool Server - Exposes L1-L5 tools as MCP endpoints.
"""
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio

from .schemas import (
    ToolSchema, ToolResult, ToolLevel,
    ALL_SCHEMAS, CODE_EXECUTION_SCHEMA, VISION_SCHEMA,
    RAG_SCHEMA, WEB_SEARCH_SCHEMA, ORCHESTRATOR_SCHEMA
)


class MCPToolServer:
    """
    MCP-compatible tool server that exposes L1-L5 tools.

    Tools are lazily initialized when first accessed.
    """

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the MCP tool server.

        Args:
            api_keys: Dict of API keys (GEMINI_API_KEY, TAVILY_API_KEY, etc.)
        """
        self.api_keys = api_keys or self._load_api_keys()
        self._tools: Dict[str, Any] = {}
        self._initialized: Dict[str, bool] = {}

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment."""
        from dotenv import load_dotenv
        load_dotenv()

        return {
            'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY', ''),
            'TAVILY_API_KEY': os.getenv('TAVILY_API_KEY', ''),
            'E2B_API_KEY': os.getenv('E2B_API_KEY', ''),
        }

    def list_tools(self) -> List[Dict]:
        """List available tools in MCP format."""
        return [schema.to_dict() for schema in ALL_SCHEMAS.values()]

    def get_tool_schema(self, level: str) -> Optional[ToolSchema]:
        """Get schema for a specific tool level."""
        return ALL_SCHEMAS.get(level)

    async def initialize_tool(self, level: str) -> bool:
        """
        Initialize a specific tool.

        Args:
            level: Tool level (L1-L5)

        Returns:
            True if initialization successful
        """
        if level in self._initialized and self._initialized[level]:
            return True

        try:
            if level == 'L1':
                from ..tools.code import CodeExecutionTool
                self._tools['L1'] = CodeExecutionTool(
                    api_key=self.api_keys.get('GEMINI_API_KEY')
                )
            elif level == 'L2':
                from ..tools.vision import VisionTool
                self._tools['L2'] = VisionTool(
                    api_key=self.api_keys.get('GEMINI_API_KEY')
                )
            elif level == 'L3':
                from ..tools.rag import RAGTool
                self._tools['L3'] = RAGTool()
            elif level == 'L4':
                from ..tools.web import WebSearchTool
                self._tools['L4'] = WebSearchTool(
                    api_key=self.api_keys.get('TAVILY_API_KEY')
                )
            elif level == 'L5':
                from ..tools.orchestrator import OrchestratorTool
                self._tools['L5'] = OrchestratorTool(
                    api_key=self.api_keys.get('GEMINI_API_KEY')
                )
            else:
                return False

            self._initialized[level] = True
            return True

        except Exception as e:
            print(f"Failed to initialize {level}: {e}")
            return False

    async def call_tool(
        self,
        level: str,
        arguments: Dict[str, Any]
    ) -> ToolResult:
        """
        Call a tool with the given arguments.

        Args:
            level: Tool level (L1-L5)
            arguments: Tool-specific arguments

        Returns:
            ToolResult with success/failure and content
        """
        # Ensure tool is initialized
        if not await self.initialize_tool(level):
            return ToolResult(
                success=False,
                content=None,
                error=f"Failed to initialize tool {level}"
            )

        tool = self._tools.get(level)
        if not tool:
            return ToolResult(
                success=False,
                content=None,
                error=f"Tool {level} not found"
            )

        try:
            import time
            start_time = time.time()

            result = await tool.execute(**arguments)

            execution_time = (time.time() - start_time) * 1000

            return ToolResult(
                success=True,
                content=result,
                metadata={'level': level},
                execution_time_ms=execution_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=None,
                error=str(e)
            )

    async def execute(self, level: str, prompt: str) -> str:
        """
        Simplified execution interface.

        Args:
            level: Tool level
            prompt: Full prompt to execute

        Returns:
            Result as string
        """
        if level == 'L0':
            # L0 is base LLM - no tool
            return prompt

        # Convert prompt to appropriate arguments
        if level == 'L1':
            arguments = {'code': prompt}
        elif level == 'L2':
            arguments = {'question': prompt}
        elif level == 'L3':
            arguments = {'query': prompt}
        elif level == 'L4':
            arguments = {'query': prompt}
        elif level == 'L5':
            arguments = {'task': prompt}
        else:
            return f"Unknown tool level: {level}"

        result = await self.call_tool(level, arguments)

        if result.success:
            return str(result.content)
        else:
            return f"Tool error: {result.error}"

    def get_available_tools(self) -> List[str]:
        """Get list of available tool levels."""
        available = ['L0']  # L0 always available

        for level in ['L1', 'L2', 'L3', 'L4', 'L5']:
            schema = ALL_SCHEMAS.get(level)
            if schema:
                # Check if required API keys are present
                has_keys = all(
                    self.api_keys.get(key)
                    for key in schema.required_api_keys
                )
                if has_keys or not schema.required_api_keys:
                    available.append(level)

        return available


# Singleton instance
_server_instance: Optional[MCPToolServer] = None


def get_mcp_server(api_keys: Optional[Dict[str, str]] = None) -> MCPToolServer:
    """Get or create the MCP server singleton."""
    global _server_instance

    if _server_instance is None:
        _server_instance = MCPToolServer(api_keys)

    return _server_instance
