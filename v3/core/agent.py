"""
ModernSpecialist: Agent with MCP tools, Thompson Sampling, and 4-layer memory.

Implements the OBSERVE-RETRIEVE-REASON-ACT-REFLECT loop.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from .thompson import ToolBeliefs
from .regimes import get_regime_list


@dataclass
class AgentState:
    """Serializable agent state."""
    id: int
    specialty: Optional[str] = None
    wins: Dict[str, int] = field(default_factory=dict)
    total_competitions: int = 0
    generation: int = 0


class ModernSpecialist:
    """
    A modern LLM agent with:
    - MCP tool integration
    - Thompson Sampling for tool selection
    - 4-layer memory system
    - Reflective learning loop
    """

    AVAILABLE_TOOLS = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']

    def __init__(
        self,
        agent_id: int,
        regimes: Optional[List[str]] = None,
        llm_client: Optional[Any] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize a modern specialist agent.

        Args:
            agent_id: Unique identifier
            regimes: List of possible regimes
            llm_client: LLM client for generation
            seed: Random seed for reproducibility
        """
        self.id = agent_id
        self.regimes = regimes or get_regime_list()
        self.llm = llm_client
        self.seed = seed

        # Thompson Sampling beliefs for tool selection
        self.beliefs = ToolBeliefs(self.regimes, seed=seed)

        # Agent state
        self.specialty: Optional[str] = None
        self.wins: Dict[str, int] = {r: 0 for r in self.regimes}
        self.total_competitions = 0
        self.generation = 0

        # Last action tracking
        self.last_tool: Optional[str] = None
        self.last_regime: Optional[str] = None
        self.last_confidence: float = 0.5

        # Memory system (initialized lazily)
        self._episodic_memory: List[Dict] = []
        self._semantic_patterns: List[str] = []
        self._changelog: List[Dict] = []

    def select_tool(self, regime: str) -> str:
        """
        Select tool using Thompson Sampling.

        Args:
            regime: Current task regime

        Returns:
            Selected tool name
        """
        tool = self.beliefs.select(regime, self.AVAILABLE_TOOLS)
        self.last_tool = tool
        self.last_regime = regime
        return tool

    async def solve_with_tool(
        self,
        task: Dict,
        tool: str,
        mcp_tools: Optional[Dict] = None
    ) -> Dict:
        """
        Solve a task using the specified tool.

        Implements OBSERVE-RETRIEVE-REASON-ACT-REFLECT loop.

        Args:
            task: Task dict with 'question', 'regime', etc.
            tool: Tool to use (L0-L5)
            mcp_tools: Dict of MCP tool instances

        Returns:
            Dict with 'answer', 'confidence', 'reasoning'
        """
        # OBSERVE: Identify task characteristics
        regime = task.get('regime', self.last_regime or 'unknown')
        question = task.get('question', '')

        # RETRIEVE: Get relevant memories
        relevant_memories = self._retrieve_memories(regime)

        # REASON: Build prompt with context
        prompt = self._build_prompt(question, regime, tool, relevant_memories)

        # ACT: Execute with tool
        if mcp_tools and tool in mcp_tools and tool != 'L0':
            # Use MCP tool
            result = await self._execute_with_mcp_tool(prompt, mcp_tools[tool])
        else:
            # L0: Direct LLM call
            result = await self._execute_direct(prompt)

        # Extract answer and confidence
        answer = self._extract_answer(result)
        confidence = self._extract_confidence(result)
        reasoning = result

        self.last_confidence = confidence

        return {
            'answer': answer,
            'confidence': confidence,
            'reasoning': reasoning,
            'tool': tool
        }

    def update_on_win(self, task: Dict, regime: str):
        """
        Update agent state after winning a competition.

        Only winners update memory (anti-leakage by design).

        Args:
            task: The task that was won
            regime: The regime of the task
        """
        # Update beliefs (success)
        self.beliefs.update(regime, self.last_tool, success=True)

        # Update win counts
        self.wins[regime] = self.wins.get(regime, 0) + 1

        # Update specialty based on win distribution
        self._update_specialty()

        # Store in episodic memory (wins only!)
        self._add_to_episodic_memory(task, regime, won=True)

        # Update changelog
        self._update_changelog(regime, self.last_tool, 'win')

    def update_on_loss(self, task: Dict, regime: str):
        """
        Update agent state after losing a competition.

        Losers update beliefs but NOT memory.

        Args:
            task: The task that was lost
            regime: The regime of the task
        """
        # Update beliefs (failure)
        self.beliefs.update(regime, self.last_tool, success=False)

        # Losers do NOT update memory (anti-leakage)
        # This is intentional - memory is EARNED through competition

    def _update_specialty(self):
        """Update specialty based on win distribution."""
        if not any(self.wins.values()):
            self.specialty = None
            return

        # Find regime with most wins
        best_regime = max(self.wins, key=self.wins.get)
        best_wins = self.wins[best_regime]

        # Need at least 3 wins to claim specialty
        if best_wins >= 3:
            total_wins = sum(self.wins.values())
            # And must have >40% of total wins in this regime
            if best_wins / max(total_wins, 1) > 0.4:
                self.specialty = best_regime

    def _retrieve_memories(self, regime: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant memories for current regime."""
        # Filter episodic memories by regime
        relevant = [
            m for m in self._episodic_memory
            if m.get('regime') == regime
        ]

        # Sort by recency
        relevant = sorted(
            relevant,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )[:top_k]

        # Also include semantic patterns
        patterns = [p for p in self._semantic_patterns if regime in p.lower()]

        memories = []
        for m in relevant:
            memories.append(f"Previous win in {regime}: {m.get('strategy', 'N/A')}")
        for p in patterns[:2]:
            memories.append(f"Pattern: {p}")

        return memories

    def _build_prompt(
        self,
        question: str,
        regime: str,
        tool: str,
        memories: List[str]
    ) -> str:
        """Build prompt with context and memories."""
        prompt_parts = [
            f"You are a specialist agent (ID: {self.id}).",
            f"Current regime: {regime}",
            f"Selected tool: {tool}",
        ]

        if memories:
            prompt_parts.append("\nRelevant experience:")
            for mem in memories:
                prompt_parts.append(f"- {mem}")

        prompt_parts.append(f"\nTask: {question}")
        prompt_parts.append("\nProvide your answer and confidence (0-1).")

        return "\n".join(prompt_parts)

    async def _execute_with_mcp_tool(self, prompt: str, tool: Any) -> str:
        """Execute prompt with MCP tool."""
        try:
            result = await tool.execute(prompt)
            return result
        except Exception as e:
            return f"Tool error: {e}"

    async def _execute_direct(self, prompt: str) -> str:
        """Execute prompt directly with LLM."""
        if self.llm is None:
            return "No LLM client configured"

        try:
            response = await self.llm.generate(prompt)
            return response
        except Exception as e:
            return f"LLM error: {e}"

    def _extract_answer(self, result: str) -> str:
        """Extract answer from LLM response."""
        # Simple extraction - can be made more sophisticated
        if isinstance(result, dict):
            return result.get('answer', str(result))
        return str(result)[:500]  # Truncate long responses

    def _extract_confidence(self, result: str) -> float:
        """Extract confidence from LLM response."""
        # Look for confidence pattern in response
        import re

        text = str(result).lower()

        # Try to find explicit confidence
        patterns = [
            r'confidence[:\s]+(\d*\.?\d+)',
            r'(\d*\.?\d+)\s*confidence',
            r'certainty[:\s]+(\d*\.?\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    conf = float(match.group(1))
                    if conf <= 1:
                        return conf
                    elif conf <= 100:
                        return conf / 100
                except ValueError:
                    pass

        # Default confidence based on specialty match
        if self.specialty == self.last_regime:
            return 0.75  # Higher confidence in specialty
        return 0.5

    def _add_to_episodic_memory(self, task: Dict, regime: str, won: bool):
        """Add episode to memory."""
        episode = {
            'regime': regime,
            'tool': self.last_tool,
            'won': won,
            'confidence': self.last_confidence,
            'strategy': f"Used {self.last_tool} for {regime}",
            'timestamp': datetime.now().isoformat(),
            'generation': self.generation,
        }

        self._episodic_memory.append(episode)

        # Keep only last 100 episodes
        if len(self._episodic_memory) > 100:
            self._episodic_memory = self._episodic_memory[-100:]

    def _update_changelog(self, regime: str, tool: str, outcome: str):
        """Update behavioral changelog."""
        entry = {
            'generation': self.generation,
            'regime': regime,
            'tool': tool,
            'outcome': outcome,
            'timestamp': datetime.now().isoformat(),
        }
        self._changelog.append(entry)

    def increment_generation(self):
        """Increment the generation counter."""
        self.generation += 1
        self.total_competitions += 1

    def get_state(self) -> AgentState:
        """Get serializable agent state."""
        return AgentState(
            id=self.id,
            specialty=self.specialty,
            wins=self.wins.copy(),
            total_competitions=self.total_competitions,
            generation=self.generation,
        )

    def export_state(self) -> Dict:
        """Export full state for serialization."""
        return {
            'id': self.id,
            'specialty': self.specialty,
            'wins': self.wins,
            'total_competitions': self.total_competitions,
            'generation': self.generation,
            'beliefs': self.beliefs.export_state(),
            'episodic_memory': self._episodic_memory,
            'semantic_patterns': self._semantic_patterns,
            'changelog': self._changelog,
        }

    def import_state(self, state: Dict):
        """Import state from serialized dict."""
        self.specialty = state.get('specialty')
        self.wins = state.get('wins', {})
        self.total_competitions = state.get('total_competitions', 0)
        self.generation = state.get('generation', 0)

        if 'beliefs' in state:
            self.beliefs.import_state(state['beliefs'])

        self._episodic_memory = state.get('episodic_memory', [])
        self._semantic_patterns = state.get('semantic_patterns', [])
        self._changelog = state.get('changelog', [])

    def __repr__(self) -> str:
        return f"ModernSpecialist(id={self.id}, specialty={self.specialty}, wins={sum(self.wins.values())})"
