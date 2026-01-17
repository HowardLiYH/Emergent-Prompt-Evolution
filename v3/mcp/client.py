"""
MCP-Enabled Agent - Agent that can discover and call MCP tools.
"""
from typing import Dict, List, Optional, Any
import asyncio

from .server import MCPToolServer, get_mcp_server
from .schemas import ToolResult


class MCPEnabledAgent:
    """
    An agent wrapper that integrates with MCP tools.

    This wraps a base agent (ModernSpecialist) and provides
    MCP tool calling capabilities.
    """

    def __init__(
        self,
        base_agent: Any,
        mcp_server: Optional[MCPToolServer] = None
    ):
        """
        Initialize MCP-enabled agent.

        Args:
            base_agent: The underlying ModernSpecialist agent
            mcp_server: MCP server instance (uses singleton if None)
        """
        self.agent = base_agent
        self.server = mcp_server or get_mcp_server()

        # Cache available tools
        self._available_tools: Optional[List[str]] = None

    @property
    def id(self) -> int:
        """Get agent ID."""
        return self.agent.id

    @property
    def specialty(self) -> Optional[str]:
        """Get agent specialty."""
        return self.agent.specialty

    @property
    def wins(self) -> Dict[str, int]:
        """Get win counts."""
        return self.agent.wins

    def get_available_tools(self) -> List[str]:
        """Get list of available MCP tools."""
        if self._available_tools is None:
            self._available_tools = self.server.get_available_tools()
        return self._available_tools

    def select_tool(self, regime: str) -> str:
        """
        Select tool using Thompson Sampling.

        Args:
            regime: Current task regime

        Returns:
            Selected tool level
        """
        available = self.get_available_tools()
        return self.agent.beliefs.select(regime, available)

    async def solve_with_tool(
        self,
        task: Dict,
        tool: str
    ) -> Dict:
        """
        Solve a task using an MCP tool.

        Args:
            task: Task dict with 'question', 'regime', etc.
            tool: Tool level to use

        Returns:
            Result dict with 'answer', 'confidence', 'reasoning'
        """
        question = task.get('question', '')
        regime = task.get('regime', 'unknown')

        # Use MCP server to execute
        result = await self.server.execute(tool, question)

        # Parse result
        answer = self._extract_answer(result)
        confidence = self._estimate_confidence(result, tool, regime)

        self.agent.last_tool = tool
        self.agent.last_regime = regime
        self.agent.last_confidence = confidence

        return {
            'answer': answer,
            'confidence': confidence,
            'reasoning': result,
            'tool': tool
        }

    def _extract_answer(self, result: str) -> str:
        """Extract answer from tool result."""
        if not result:
            return ""

        # Look for answer patterns
        result_lower = result.lower()

        # Try to find explicit answer
        for prefix in ['answer:', 'result:', 'the answer is', 'output:']:
            if prefix in result_lower:
                idx = result_lower.index(prefix) + len(prefix)
                return result[idx:].strip()[:500]

        return result[:500]

    def _estimate_confidence(self, result: str, tool: str, regime: str) -> float:
        """Estimate confidence based on result and context."""
        if 'error' in str(result).lower():
            return 0.2

        # Higher confidence if using optimal tool for regime
        from ..core.regimes import DEFAULT_REGIME_CONFIG
        if regime in DEFAULT_REGIME_CONFIG:
            optimal = DEFAULT_REGIME_CONFIG[regime].optimal_tool
            if tool == optimal:
                return 0.8

        # Base confidence
        return 0.6

    def update_on_win(self, task: Dict, regime: str):
        """Update agent after winning."""
        self.agent.update_on_win(task, regime)

    def update_on_loss(self, task: Dict, regime: str):
        """Update agent after losing."""
        self.agent.update_on_loss(task, regime)

    def increment_generation(self):
        """Increment generation counter."""
        self.agent.increment_generation()

    def export_state(self) -> Dict:
        """Export agent state."""
        return self.agent.export_state()

    def import_state(self, state: Dict):
        """Import agent state."""
        self.agent.import_state(state)

    def __repr__(self) -> str:
        return f"MCPEnabledAgent({self.agent})"


def wrap_population_with_mcp(
    population: List[Any],
    mcp_server: Optional[MCPToolServer] = None
) -> List[MCPEnabledAgent]:
    """
    Wrap a population of agents with MCP capabilities.

    Args:
        population: List of ModernSpecialist agents
        mcp_server: MCP server instance

    Returns:
        List of MCPEnabledAgent wrappers
    """
    server = mcp_server or get_mcp_server()
    return [MCPEnabledAgent(agent, server) for agent in population]
