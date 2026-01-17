"""
Episodic Memory - Stores raw episodes from competition wins.

Key property: Only WINS are stored (anti-leakage by design).
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict


@dataclass
class Episode:
    """A single episode from a competition win."""
    task: str
    regime: str
    tool: str
    strategy: str  # What approach worked
    outcome: str   # What was successful
    confidence: float
    generation: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """
    Layer 2: Episodic Memory

    Stores raw episodes from competition wins.
    - Only wins are stored (anti-leakage by design)
    - Indexed by regime for O(1) lookup
    - Sliding window retention (last 100 per regime)

    This is the "experience log" that captures what worked.
    """

    MAX_PER_REGIME = 100

    def __init__(self):
        """Initialize episodic memory."""
        self.episodes: Dict[str, List[Episode]] = defaultdict(list)
        self.total_episodes = 0

    def add_win(
        self,
        task: str,
        regime: str,
        tool: str,
        strategy: str,
        confidence: float,
        generation: int,
        metadata: Optional[Dict] = None
    ):
        """
        Add a winning episode to memory.

        Only called after competition wins.

        Args:
            task: The task that was won
            regime: Task regime
            tool: Tool that was used
            strategy: Description of the approach
            confidence: Confidence of the winning answer
            generation: Training generation
            metadata: Additional metadata
        """
        episode = Episode(
            task=task[:500],  # Truncate
            regime=regime,
            tool=tool,
            strategy=strategy,
            outcome="win",
            confidence=confidence,
            generation=generation,
            metadata=metadata or {}
        )

        self.episodes[regime].append(episode)
        self.total_episodes += 1

        # Enforce sliding window
        if len(self.episodes[regime]) > self.MAX_PER_REGIME:
            self.episodes[regime] = self.episodes[regime][-self.MAX_PER_REGIME:]

    def retrieve(
        self,
        regime: str,
        top_k: int = 5,
        recency_weight: float = 0.3
    ) -> List[Episode]:
        """
        Retrieve relevant episodes for a regime.

        Uses hybrid scoring: recency + confidence.

        Args:
            regime: Regime to retrieve for
            top_k: Number of episodes to retrieve
            recency_weight: Weight for recency vs confidence

        Returns:
            List of most relevant episodes
        """
        if regime not in self.episodes or not self.episodes[regime]:
            return []

        episodes = self.episodes[regime]

        # Score each episode
        scored = []
        max_gen = max(e.generation for e in episodes) if episodes else 1

        for ep in episodes:
            # Recency score (0-1, higher for more recent)
            recency = ep.generation / max(max_gen, 1)

            # Combined score
            score = (
                recency_weight * recency +
                (1 - recency_weight) * ep.confidence
            )

            scored.append((ep, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        return [ep for ep, _ in scored[:top_k]]

    def get_regime_patterns(self, regime: str) -> Dict[str, int]:
        """
        Get patterns (tool usage counts) for a regime.

        Args:
            regime: Regime to analyze

        Returns:
            Dict of tool -> usage count
        """
        if regime not in self.episodes:
            return {}

        patterns = defaultdict(int)
        for ep in self.episodes[regime]:
            patterns[ep.tool] += 1

        return dict(patterns)

    def get_best_strategy(self, regime: str) -> Optional[str]:
        """
        Get the most successful strategy for a regime.

        Args:
            regime: Regime to check

        Returns:
            Strategy string or None
        """
        if regime not in self.episodes or not self.episodes[regime]:
            return None

        # Get episode with highest confidence
        best = max(self.episodes[regime], key=lambda e: e.confidence)
        return best.strategy

    def get_stats(self) -> Dict:
        """Get episodic memory statistics."""
        return {
            'total_episodes': self.total_episodes,
            'regimes': list(self.episodes.keys()),
            'episodes_per_regime': {
                r: len(eps) for r, eps in self.episodes.items()
            }
        }

    def export(self) -> Dict:
        """Export to serializable dict."""
        return {
            regime: [
                {
                    'task': ep.task,
                    'regime': ep.regime,
                    'tool': ep.tool,
                    'strategy': ep.strategy,
                    'confidence': ep.confidence,
                    'generation': ep.generation,
                    'timestamp': ep.timestamp,
                }
                for ep in episodes
            ]
            for regime, episodes in self.episodes.items()
        }

    def import_from(self, data: Dict):
        """Import from dict."""
        for regime, episodes in data.items():
            for ep_data in episodes:
                self.episodes[regime].append(Episode(
                    task=ep_data['task'],
                    regime=ep_data['regime'],
                    tool=ep_data['tool'],
                    strategy=ep_data['strategy'],
                    outcome='win',
                    confidence=ep_data['confidence'],
                    generation=ep_data['generation'],
                    timestamp=ep_data.get('timestamp', '')
                ))

        self.total_episodes = sum(
            len(eps) for eps in self.episodes.values()
        )

    def clear(self):
        """Clear all episodes."""
        self.episodes = defaultdict(list)
        self.total_episodes = 0
