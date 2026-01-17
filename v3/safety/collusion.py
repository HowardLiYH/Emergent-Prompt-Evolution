"""
Collusion Detection - Identifies suspicious win patterns among agents.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import Counter
import numpy as np


@dataclass
class CollusionAlert:
    """Alert for potential collusion."""
    regime: str
    pattern_type: str  # 'alternating', 'round_robin', 'perfect_split'
    agents_involved: List[int]
    confidence: float
    window_size: int
    evidence: str


def get_recent_winners(
    history: List[Dict],
    regime: str,
    n: int = 20
) -> List[int]:
    """
    Get recent winners for a regime.

    Args:
        history: Competition history
        regime: Regime to filter by
        n: Number of recent rounds to consider

    Returns:
        List of winner IDs
    """
    regime_rounds = [
        h for h in history
        if h.get('regime') == regime and h.get('winner_id') is not None
    ]

    return [h['winner_id'] for h in regime_rounds[-n:]]


def is_alternating(winners: List[int], tolerance: float = 0.1) -> bool:
    """
    Check if winners alternate suspiciously.

    E.g., [1, 2, 1, 2, 1, 2] is alternating.
    """
    if len(winners) < 6:
        return False

    # Check pairs
    pairs = list(zip(winners[:-1], winners[1:]))
    same_pair = sum(1 for a, b in pairs if a == b)

    # If almost no consecutive wins, might be alternating
    if same_pair / len(pairs) < tolerance:
        # Check if only 2-3 agents involved
        unique = len(set(winners))
        if unique <= 3:
            return True

    return False


def is_round_robin(winners: List[int], tolerance: float = 0.15) -> bool:
    """
    Check if winners follow a round-robin pattern.

    E.g., [1, 2, 3, 1, 2, 3, 1, 2, 3]
    """
    if len(winners) < 9:
        return False

    unique = list(set(winners))
    if len(unique) < 3 or len(unique) > 5:
        return False

    # Count occurrences
    counts = Counter(winners)

    # Check if distribution is suspiciously even
    values = list(counts.values())
    mean = np.mean(values)
    std = np.std(values)

    # Very low variance = suspicious
    if mean > 0 and std / mean < tolerance:
        return True

    return False


def is_perfect_split(
    wins_per_agent: Dict[str, Dict[int, int]],
    n_regimes: int,
    n_agents: int
) -> bool:
    """
    Check if agents have perfectly split regimes.

    Suspicious if each agent wins ONLY in their assigned regime.
    """
    # Get agents who have won
    agents = set()
    for regime, wins in wins_per_agent.items():
        agents.update(wins.keys())

    if len(agents) < 2:
        return False

    # Check if each agent only wins in one regime
    agent_regimes = {}
    for regime, wins in wins_per_agent.items():
        for agent_id, count in wins.items():
            if agent_id not in agent_regimes:
                agent_regimes[agent_id] = set()
            agent_regimes[agent_id].add(regime)

    # If all agents win in exactly one regime, suspicious
    if all(len(regimes) == 1 for regimes in agent_regimes.values()):
        if len(agent_regimes) == n_regimes:
            return True

    return False


def detect_collusion(
    history: List[Dict],
    regime: str,
    window: int = 20
) -> Optional[CollusionAlert]:
    """
    Detect potential collusion in a regime.

    Args:
        history: Competition history
        regime: Regime to analyze
        window: Window size for analysis

    Returns:
        CollusionAlert if suspicious pattern detected, else None
    """
    winners = get_recent_winners(history, regime, n=window)

    if len(winners) < window // 2:
        return None  # Not enough data

    # Check for alternating pattern
    if is_alternating(winners):
        return CollusionAlert(
            regime=regime,
            pattern_type='alternating',
            agents_involved=list(set(winners)),
            confidence=0.8,
            window_size=len(winners),
            evidence=f"Winners alternate suspiciously: {winners[-10:]}"
        )

    # Check for round-robin pattern
    if is_round_robin(winners):
        return CollusionAlert(
            regime=regime,
            pattern_type='round_robin',
            agents_involved=list(set(winners)),
            confidence=0.7,
            window_size=len(winners),
            evidence=f"Winners follow round-robin: {Counter(winners)}"
        )

    return None


def detect_population_collusion(
    wins_per_regime: Dict[str, Dict[int, int]],
    n_regimes: int,
    n_agents: int
) -> Optional[CollusionAlert]:
    """
    Detect collusion at the population level.

    Args:
        wins_per_regime: Dict of regime -> {agent_id -> win_count}
        n_regimes: Total number of regimes
        n_agents: Total number of agents

    Returns:
        CollusionAlert if suspicious pattern detected
    """
    if is_perfect_split(wins_per_regime, n_regimes, n_agents):
        agents = set()
        for wins in wins_per_regime.values():
            agents.update(wins.keys())

        return CollusionAlert(
            regime='global',
            pattern_type='perfect_split',
            agents_involved=list(agents),
            confidence=0.6,
            window_size=0,
            evidence="Each agent wins exclusively in one regime"
        )

    return None
