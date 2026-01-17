"""
Fitness sharing mechanism for emergent specialization.

Implements 1/sqrt(n) penalty to prevent niche crowding and guarantee coverage.
"""
import math
from typing import List, Dict, Any
from collections import Counter


def fitness_penalty(regime: str, population: List[Any], specialty_attr: str = 'specialty') -> float:
    """
    Compute fitness sharing penalty for a regime.

    The penalty is 1/sqrt(n) where n is the number of specialists in the regime.
    This discourages crowding and encourages agents to spread across niches.

    Args:
        regime: The regime to compute penalty for
        population: List of agents
        specialty_attr: Attribute name that holds agent's specialty

    Returns:
        Fitness multiplier (0 < penalty <= 1)
    """
    # Count specialists in this regime
    n_specialists = sum(
        1 for agent in population
        if getattr(agent, specialty_attr, None) == regime
    )

    # 1/sqrt(n) penalty, minimum of 1 to avoid division by zero
    return 1.0 / math.sqrt(max(n_specialists, 1))


def compute_fitness_scores(
    results: List[tuple],
    population: List[Any],
    regime: str
) -> List[tuple]:
    """
    Compute fitness-adjusted scores for competition results.

    Args:
        results: List of (agent, tool, correct, confidence) tuples
        population: Full agent population for computing crowding
        regime: Current task regime

    Returns:
        List of (agent, adjusted_score) tuples, sorted by score descending
    """
    penalty = fitness_penalty(regime, population)

    scored_results = []
    for agent, tool, correct, confidence in results:
        if correct:
            # Score = confidence × fitness_penalty
            score = confidence * penalty
        else:
            score = 0.0
        scored_results.append((agent, score))

    # Sort by score descending
    return sorted(scored_results, key=lambda x: x[1], reverse=True)


def find_winner_with_fitness(
    results: List[tuple],
    population: List[Any],
    regime: str
) -> Any:
    """
    Find the competition winner with fitness sharing applied.

    Args:
        results: List of (agent, tool, correct, confidence) tuples
        population: Full agent population
        regime: Current task regime

    Returns:
        Winning agent or None if no correct answers
    """
    scored = compute_fitness_scores(results, population, regime)

    if not scored or scored[0][1] == 0:
        return None

    return scored[0][0]


def compute_specialist_distribution(population: List[Any]) -> Dict[str, int]:
    """
    Count how many agents specialize in each regime.

    Args:
        population: List of agents

    Returns:
        Dict mapping regime -> count
    """
    specialties = [
        agent.specialty for agent in population
        if hasattr(agent, 'specialty') and agent.specialty is not None
    ]
    return dict(Counter(specialties))


def compute_equilibrium_distribution(
    regime_config: Dict[str, Dict],
    n_agents: int
) -> Dict[str, float]:
    """
    Compute theoretical equilibrium distribution based on Theorem 4.

    n_r ∝ (f_r × R_r × D_r)^(2/3)

    Args:
        regime_config: Dict with 'frequency', 'reward', 'difficulty' per regime
        n_agents: Total number of agents

    Returns:
        Dict mapping regime -> expected number of specialists
    """
    # Compute raw scores
    scores = {}
    for regime, config in regime_config.items():
        f = config.get('frequency', 0.2)
        r = config.get('reward', 1.0)
        d = config.get('difficulty', 0.5)
        scores[regime] = (f * r * d) ** (2/3)

    # Normalize to sum to n_agents
    total = sum(scores.values())
    if total == 0:
        return {r: n_agents / len(regime_config) for r in regime_config}

    return {
        regime: (score / total) * n_agents
        for regime, score in scores.items()
    }


def compute_equilibrium_error(
    observed: Dict[str, int],
    expected: Dict[str, float]
) -> float:
    """
    Compute mean absolute error between observed and expected distribution.

    Args:
        observed: Actual specialist counts
        expected: Theoretical equilibrium counts

    Returns:
        Mean absolute error (0 = perfect match)
    """
    errors = []
    for regime in expected:
        obs = observed.get(regime, 0)
        exp = expected[regime]
        if exp > 0:
            errors.append(abs(obs - exp) / exp)

    return sum(errors) / len(errors) if errors else 0.0
