"""
Memory Consolidator - Compresses episodes into semantic patterns.

Implements the consolidation operation (like sleep consolidation in humans).
"""
from typing import List, Dict, Optional, Any
from collections import defaultdict
import asyncio

from .episodic import EpisodicMemory, Episode
from .semantic import SemanticMemory


class MemoryConsolidator:
    """
    Consolidates episodic memories into semantic patterns.

    This is called periodically (e.g., every 10 generations) to:
    1. Cluster similar episodes
    2. Extract patterns via LLM summarization
    3. Store patterns in semantic memory
    4. Evict old episodes
    """

    CONSOLIDATION_FREQUENCY = 10  # Every N generations
    MIN_CLUSTER_SIZE = 3  # Minimum episodes to form a pattern

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize consolidator.

        Args:
            llm_client: LLM client for summarization
        """
        self.llm = llm_client
        self._model = None

    def _get_model(self):
        """Get LLM model for summarization."""
        if self._model is None and self.llm is None:
            try:
                import google.generativeai as genai
                import os
                genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
                self._model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception:
                pass
        return self._model or self.llm

    def should_consolidate(self, generation: int) -> bool:
        """Check if consolidation should run."""
        return generation % self.CONSOLIDATION_FREQUENCY == 0

    async def consolidate(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        keep_recent: int = 50
    ) -> Dict:
        """
        Perform consolidation: episode → patterns.

        Args:
            episodic: Episodic memory to consolidate from
            semantic: Semantic memory to consolidate into
            keep_recent: Number of recent episodes to keep

        Returns:
            Stats about the consolidation
        """
        stats = {
            'episodes_processed': 0,
            'patterns_created': 0,
            'regimes_processed': 0,
        }

        for regime, episodes in episodic.episodes.items():
            if len(episodes) < self.MIN_CLUSTER_SIZE:
                continue

            # Cluster episodes by tool
            clusters = self._cluster_by_tool(episodes)

            for tool, cluster_episodes in clusters.items():
                if len(cluster_episodes) < self.MIN_CLUSTER_SIZE:
                    continue

                # Extract pattern
                pattern = await self._extract_pattern(
                    cluster_episodes, regime, tool
                )

                if pattern:
                    semantic.add_pattern(
                        pattern=pattern,
                        regime=regime,
                        confidence=self._compute_confidence(cluster_episodes),
                        support=len(cluster_episodes)
                    )
                    stats['patterns_created'] += 1

                stats['episodes_processed'] += len(cluster_episodes)

            # Evict old episodes
            if len(episodes) > keep_recent:
                episodic.episodes[regime] = episodes[-keep_recent:]

            stats['regimes_processed'] += 1

        # Apply confidence decay to existing patterns
        semantic.decay_confidence()

        return stats

    def _cluster_by_tool(
        self,
        episodes: List[Episode]
    ) -> Dict[str, List[Episode]]:
        """Cluster episodes by tool used."""
        clusters = defaultdict(list)
        for ep in episodes:
            clusters[ep.tool].append(ep)
        return dict(clusters)

    async def _extract_pattern(
        self,
        episodes: List[Episode],
        regime: str,
        tool: str
    ) -> Optional[str]:
        """
        Extract a pattern from a cluster of episodes.

        Uses LLM to summarize what made these episodes successful.
        """
        model = self._get_model()

        if model is None:
            # Fallback: simple pattern
            return f"Use {tool} for {regime} tasks. Success rate based on {len(episodes)} wins."

        # Build prompt
        episode_descriptions = [
            f"- Task: {ep.task[:100]}... Strategy: {ep.strategy}"
            for ep in episodes[:5]  # Limit to 5 for prompt length
        ]

        prompt = f"""You are analyzing winning strategies from a competition.

Regime: {regime}
Tool used: {tool}
Number of wins: {len(episodes)}

Sample winning episodes:
{chr(10).join(episode_descriptions)}

Extract a generalizable pattern or strategy that explains why these approaches succeeded.
Provide a concise pattern (1-2 sentences) that can be applied to similar future tasks.
Focus on the strategy, not specific answers."""

        try:
            loop = asyncio.get_event_loop()

            if hasattr(model, 'generate_content'):
                response = await loop.run_in_executor(
                    None,
                    lambda: model.generate_content(prompt)
                )
                return response.text[:300]
            elif hasattr(model, 'generate'):
                response = await loop.run_in_executor(
                    None,
                    lambda: model.generate(prompt)
                )
                return response[:300]
            else:
                return f"Use {tool} for {regime}: consistent success pattern."

        except Exception as e:
            return f"Use {tool} for {regime} tasks. (Summarization failed: {e})"

    def _compute_confidence(self, episodes: List[Episode]) -> float:
        """Compute confidence for a pattern."""
        if not episodes:
            return 0.5

        # Average confidence of constituent episodes
        avg_conf = sum(ep.confidence for ep in episodes) / len(episodes)

        # Boost for more supporting episodes
        support_boost = min(0.2, len(episodes) * 0.02)

        return min(1.0, avg_conf + support_boost)
