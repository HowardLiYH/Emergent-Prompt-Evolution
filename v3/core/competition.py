"""
Competition Engine for the Competitive Specialist Ecosystem.

Implements subset selection (K=3), epsilon exploration, and winner-only updates.
"""
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import asyncio
import json

from .agent import ModernSpecialist
from .regimes import RegimeSampler, DEFAULT_REGIME_CONFIG
from .fitness import find_winner_with_fitness, compute_specialist_distribution


@dataclass
class CompetitionResult:
    """Result of a single competition round."""
    generation: int
    regime: str
    task: Dict
    winner: Optional[ModernSpecialist]
    participants: List[int]  # Agent IDs
    scores: Dict[int, float]  # Agent ID -> score
    timestamp: str


class CompetitionEngine:
    """
    Engine for running competition-based training.

    Features:
    - Subset selection (K=3 competitors per round)
    - Epsilon exploration (10% random selection)
    - Fitness sharing for diversity
    - Winner-only memory updates (anti-leakage)
    """

    def __init__(
        self,
        population: List[ModernSpecialist],
        regime_config: Optional[Dict] = None,
        k: int = 3,
        epsilon: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize competition engine.

        Args:
            population: List of agents
            regime_config: Regime configuration (uses default if None)
            k: Number of competitors per round
            epsilon: Exploration rate for random selection
            seed: Random seed
        """
        self.population = population
        self.regime_sampler = RegimeSampler(regime_config, seed=seed)
        self.k = min(k, len(population))
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # History tracking
        self.history: List[CompetitionResult] = []
        self.generation = 0

        # Metrics
        self.wins_per_regime: Dict[str, Dict[int, int]] = {}
        self.routing_data: List[Tuple[Dict, int]] = []  # (task_embedding, winner_id)

    def select_competitors(self) -> List[ModernSpecialist]:
        """
        Select K competitors with epsilon-exploration.

        With probability epsilon, select randomly.
        Otherwise, select top performers with some randomness.
        """
        if self.rng.random() < self.epsilon or len(self.history) < 10:
            # Exploration: random selection
            indices = self.rng.choice(
                len(self.population),
                size=min(self.k, len(self.population)),
                replace=False
            )
            return [self.population[i] for i in indices]

        # Exploitation: bias toward recent winners
        recent_winners = [
            r.winner.id for r in self.history[-20:]
            if r.winner is not None
        ]

        # Include recent winners with higher probability
        weights = []
        for agent in self.population:
            if agent.id in recent_winners:
                weights.append(2.0)  # Double weight for recent winners
            else:
                weights.append(1.0)

        weights = np.array(weights) / sum(weights)
        indices = self.rng.choice(
            len(self.population),
            size=min(self.k, len(self.population)),
            replace=False,
            p=weights
        )
        return [self.population[i] for i in indices]

    async def run_competition_round(
        self,
        task: Dict,
        mcp_tools: Optional[Dict] = None
    ) -> CompetitionResult:
        """
        Run a single competition round.

        Args:
            task: Task dict with 'question', 'regime', 'answer'
            mcp_tools: Dict of MCP tool instances

        Returns:
            CompetitionResult with winner and scores
        """
        regime = task.get('regime', self.regime_sampler.sample())

        # Select competitors
        competitors = self.select_competitors()

        # Each competitor solves the task
        results = []
        for agent in competitors:
            tool = agent.select_tool(regime)
            response = await agent.solve_with_tool(task, tool, mcp_tools)

            # Evaluate correctness
            correct = self._evaluate_answer(
                response.get('answer', ''),
                task.get('answer', '')
            )

            results.append((
                agent,
                tool,
                correct,
                response.get('confidence', 0.5)
            ))

        # Find winner with fitness sharing
        winner = find_winner_with_fitness(results, self.population, regime)

        # Update agents
        for agent, tool, correct, confidence in results:
            if agent == winner:
                agent.update_on_win(task, regime)
            else:
                agent.update_on_loss(task, regime)
            agent.increment_generation()

        # Record result
        scores = {
            agent.id: confidence if correct else 0.0
            for agent, tool, correct, confidence in results
        }

        result = CompetitionResult(
            generation=self.generation,
            regime=regime,
            task=task,
            winner=winner,
            participants=[a.id for a in competitors],
            scores=scores,
            timestamp=datetime.now().isoformat()
        )

        self.history.append(result)
        self.generation += 1

        # Update tracking
        self._update_tracking(result)

        return result

    async def run_training(
        self,
        n_generations: int,
        task_generator: Any,
        mcp_tools: Optional[Dict] = None,
        log_frequency: int = 10
    ) -> Dict:
        """
        Run full competition training.

        Args:
            n_generations: Number of competition rounds
            task_generator: Callable that generates tasks
            mcp_tools: Dict of MCP tool instances
            log_frequency: How often to print progress

        Returns:
            Training summary dict
        """
        print(f"Starting training: {n_generations} generations, {len(self.population)} agents")

        for gen in range(n_generations):
            # Sample regime and get task
            regime = self.regime_sampler.sample()
            task = task_generator(regime)
            task['regime'] = regime

            # Run competition
            result = await self.run_competition_round(task, mcp_tools)

            # Log progress
            if (gen + 1) % log_frequency == 0:
                metrics = self.compute_metrics()
                print(f"Gen {gen + 1}/{n_generations}: "
                      f"SCI={metrics['sci']:.3f}, "
                      f"Coverage={metrics['coverage']:.1%}, "
                      f"Winner={result.winner.id if result.winner else 'None'}")

        return self.compute_metrics()

    def _evaluate_answer(self, predicted: str, expected: str) -> bool:
        """Evaluate if predicted answer matches expected."""
        if not expected:
            return True  # No ground truth

        # Normalize answers
        pred_norm = str(predicted).strip().lower()
        exp_norm = str(expected).strip().lower()

        # Exact match
        if pred_norm == exp_norm:
            return True

        # Contains match (for multiple choice)
        if exp_norm in pred_norm or pred_norm in exp_norm:
            return True

        return False

    def _update_tracking(self, result: CompetitionResult):
        """Update internal tracking metrics."""
        regime = result.regime

        if regime not in self.wins_per_regime:
            self.wins_per_regime[regime] = {}

        if result.winner:
            winner_id = result.winner.id
            self.wins_per_regime[regime][winner_id] = (
                self.wins_per_regime[regime].get(winner_id, 0) + 1
            )

            # Store routing data
            self.routing_data.append((result.task, winner_id))

    def compute_metrics(self) -> Dict:
        """Compute current training metrics."""
        # Specialist distribution
        distribution = compute_specialist_distribution(self.population)

        # Coverage: fraction of regimes with at least one specialist
        regimes = list(self.regime_sampler.config.keys())
        covered = sum(1 for r in regimes if r in distribution)
        coverage = covered / len(regimes)

        # Specialization Concentration Index (SCI)
        # 1.0 = perfect specialization, 0.0 = no specialization
        specialties = [a.specialty for a in self.population if a.specialty]
        if not specialties:
            sci = 0.0
        else:
            from collections import Counter
            counts = Counter(specialties)
            # Gini coefficient as SCI proxy
            n = len(counts)
            if n <= 1:
                sci = 1.0
            else:
                values = sorted(counts.values())
                cumsum = np.cumsum(values)
                sci = (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n
                sci = max(0, min(1, 1 - sci))  # Invert so higher = more specialized

        # Average wins
        total_wins = sum(sum(a.wins.values()) for a in self.population)
        avg_wins = total_wins / len(self.population)

        return {
            'generation': self.generation,
            'coverage': coverage,
            'sci': sci,
            'distribution': distribution,
            'total_wins': total_wins,
            'avg_wins': avg_wins,
            'n_specialists': len([a for a in self.population if a.specialty]),
        }

    def get_specialist_for_regime(self, regime: str) -> Optional[ModernSpecialist]:
        """Get the specialist for a specific regime."""
        # Find agent with most wins in this regime
        best_agent = None
        best_wins = 0

        for agent in self.population:
            wins = agent.wins.get(regime, 0)
            if wins > best_wins:
                best_wins = wins
                best_agent = agent

        return best_agent

    def export_routing_data(self) -> List[Dict]:
        """Export routing data for training a router."""
        return [
            {'task': task, 'specialist_id': winner_id}
            for task, winner_id in self.routing_data
        ]

    def export_history(self) -> List[Dict]:
        """Export competition history."""
        return [
            {
                'generation': r.generation,
                'regime': r.regime,
                'winner_id': r.winner.id if r.winner else None,
                'participants': r.participants,
                'scores': r.scores,
                'timestamp': r.timestamp,
            }
            for r in self.history
        ]
