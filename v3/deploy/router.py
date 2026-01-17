"""
Task Router - Routes tasks to appropriate specialists.

Trained from competition outcomes.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from collections import defaultdict


@dataclass
class RoutingDecision:
    """A routing decision."""
    task_id: str
    selected_specialist: int
    confidence: float
    regime_prediction: str


class TaskRouter:
    """
    Routes incoming tasks to the most appropriate specialist.

    Trained from competition history - learns which specialist
    performs best on which types of tasks.
    """

    def __init__(self, n_regimes: int):
        """
        Initialize task router.

        Args:
            n_regimes: Number of task regimes
        """
        self.n_regimes = n_regimes
        self.regime_to_specialist: Dict[str, int] = {}
        self.specialist_win_rates: Dict[int, Dict[str, float]] = defaultdict(dict)
        self.trained = False

    def train(
        self,
        competition_history: List[Dict],
        agents: List[Any]
    ):
        """
        Train router from competition history.

        Args:
            competition_history: List of competition results
            agents: List of agents
        """
        # Count wins per (agent, regime)
        wins = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(lambda: defaultdict(int))

        for round in competition_history:
            regime = round.get('regime')
            winner_id = round.get('winner_id')
            participants = round.get('participants', [])

            if not regime:
                continue

            for agent_id in participants:
                totals[agent_id][regime] += 1
                if agent_id == winner_id:
                    wins[agent_id][regime] += 1

        # Compute win rates
        for agent_id in wins:
            for regime in wins[agent_id]:
                if totals[agent_id][regime] > 0:
                    rate = wins[agent_id][regime] / totals[agent_id][regime]
                    self.specialist_win_rates[agent_id][regime] = rate

        # Assign best specialist per regime
        for regime in set(r for t in totals.values() for r in t.keys()):
            best_agent = None
            best_rate = 0

            for agent_id in self.specialist_win_rates:
                rate = self.specialist_win_rates[agent_id].get(regime, 0)
                if rate > best_rate:
                    best_rate = rate
                    best_agent = agent_id

            if best_agent is not None:
                self.regime_to_specialist[regime] = best_agent

        self.trained = True
        print(f"Router trained: {len(self.regime_to_specialist)} regime-specialist mappings")

    def route(
        self,
        task: Dict,
        regime_hint: Optional[str] = None
    ) -> Tuple[int, float]:
        """
        Route a task to a specialist.

        Args:
            task: Task to route
            regime_hint: Optional regime hint

        Returns:
            (specialist_id, confidence)
        """
        if not self.trained:
            # Random routing if not trained
            return 0, 0.5

        regime = regime_hint or task.get('regime')

        if regime and regime in self.regime_to_specialist:
            specialist = self.regime_to_specialist[regime]
            confidence = self.specialist_win_rates.get(specialist, {}).get(regime, 0.5)
            return specialist, confidence

        # Fallback: return specialist with highest overall win rate
        if self.specialist_win_rates:
            best_agent = max(
                self.specialist_win_rates.keys(),
                key=lambda a: np.mean(list(self.specialist_win_rates[a].values()))
            )
            return best_agent, 0.5

        return 0, 0.5

    def get_routing_table(self) -> Dict[str, Dict]:
        """Get the routing table for inspection."""
        return {
            regime: {
                'specialist_id': specialist_id,
                'win_rate': self.specialist_win_rates.get(specialist_id, {}).get(regime, 0)
            }
            for regime, specialist_id in self.regime_to_specialist.items()
        }

    def accuracy(
        self,
        test_history: List[Dict]
    ) -> float:
        """
        Compute routing accuracy on test data.

        Args:
            test_history: Test competition results

        Returns:
            Routing accuracy
        """
        correct = 0
        total = 0

        for round in test_history:
            regime = round.get('regime')
            winner_id = round.get('winner_id')
            task = round.get('task', {})

            if not regime or winner_id is None:
                continue

            predicted, _ = self.route(task, regime_hint=regime)

            if predicted == winner_id:
                correct += 1
            total += 1

        return correct / max(total, 1)

    def export(self) -> Dict:
        """Export router state."""
        return {
            'regime_to_specialist': self.regime_to_specialist,
            'specialist_win_rates': dict(self.specialist_win_rates),
            'trained': self.trained
        }

    def import_from(self, data: Dict):
        """Import router state."""
        self.regime_to_specialist = data.get('regime_to_specialist', {})
        self.specialist_win_rates = defaultdict(dict, data.get('specialist_win_rates', {}))
        self.trained = data.get('trained', False)
