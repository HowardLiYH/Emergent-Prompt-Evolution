"""
Thompson Sampling for tool selection.

Each agent maintains Beta distribution beliefs over tool effectiveness per regime.
Selection samples from beliefs; updates are wins-only.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random


@dataclass
class BetaDistribution:
    """Beta distribution for Thompson Sampling."""
    alpha: float = 1.0  # Successes + prior
    beta: float = 1.0   # Failures + prior

    def sample(self, rng: Optional[np.random.Generator] = None) -> float:
        """Sample from the Beta distribution."""
        if rng is None:
            return np.random.beta(self.alpha, self.beta)
        return rng.beta(self.alpha, self.beta)

    def mean(self) -> float:
        """Expected value of the distribution."""
        return self.alpha / (self.alpha + self.beta)

    def update_success(self):
        """Update after a successful outcome."""
        self.alpha += 1

    def update_failure(self):
        """Update after a failed outcome."""
        self.beta += 1

    def confidence(self) -> float:
        """Return confidence based on sample size."""
        total = self.alpha + self.beta - 2  # Subtract prior
        return min(1.0, total / 20)  # Max confidence after 20 samples


class ToolBeliefs:
    """
    Thompson Sampling beliefs for tool selection.

    Maintains Beta(alpha, beta) distributions for each (regime, tool) pair.
    Agents use this to select tools via Thompson Sampling.
    """

    TOOLS = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']

    def __init__(self, regimes: List[str], seed: Optional[int] = None):
        """
        Initialize beliefs for all regime-tool pairs.

        Args:
            regimes: List of regime names
            seed: Random seed for reproducibility
        """
        self.regimes = regimes
        self.rng = np.random.default_rng(seed)

        # Initialize Beta(1,1) uniform prior for each (regime, tool) pair
        self.beliefs: Dict[str, Dict[str, BetaDistribution]] = {
            regime: {tool: BetaDistribution() for tool in self.TOOLS}
            for regime in regimes
        }

    def select(self, regime: str, available_tools: Optional[List[str]] = None) -> str:
        """
        Select tool using Thompson Sampling.

        Args:
            regime: Current task regime
            available_tools: Tools available to this agent (defaults to all)

        Returns:
            Selected tool name
        """
        if available_tools is None:
            available_tools = self.TOOLS

        if regime not in self.beliefs:
            # Unknown regime - return random tool
            return random.choice(available_tools)

        # Sample from each tool's belief distribution
        samples = {
            tool: self.beliefs[regime][tool].sample(self.rng)
            for tool in available_tools
            if tool in self.beliefs[regime]
        }

        if not samples:
            return random.choice(available_tools)

        # Select tool with highest sample
        return max(samples, key=samples.get)

    def update(self, regime: str, tool: str, success: bool):
        """
        Update beliefs based on outcome.

        Args:
            regime: Task regime
            tool: Tool that was used
            success: Whether the outcome was successful
        """
        if regime not in self.beliefs or tool not in self.beliefs[regime]:
            return

        if success:
            self.beliefs[regime][tool].update_success()
        else:
            self.beliefs[regime][tool].update_failure()

    def get_best_tool(self, regime: str) -> str:
        """Get tool with highest expected value for regime."""
        if regime not in self.beliefs:
            return 'L0'

        means = {
            tool: self.beliefs[regime][tool].mean()
            for tool in self.TOOLS
        }
        return max(means, key=means.get)

    def get_specialty(self) -> Optional[str]:
        """
        Determine agent's specialty based on accumulated beliefs.

        Returns the regime where the agent has the highest confidence.
        """
        best_regime = None
        best_confidence = 0

        for regime in self.regimes:
            # Find the tool with highest confidence in this regime
            max_conf = max(
                self.beliefs[regime][tool].confidence()
                for tool in self.TOOLS
            )
            if max_conf > best_confidence:
                best_confidence = max_conf
                best_regime = regime

        return best_regime if best_confidence > 0.3 else None

    def export_state(self) -> Dict:
        """Export beliefs for serialization."""
        return {
            regime: {
                tool: {'alpha': b.alpha, 'beta': b.beta}
                for tool, b in tools.items()
            }
            for regime, tools in self.beliefs.items()
        }

    def import_state(self, state: Dict):
        """Import beliefs from serialized state."""
        for regime, tools in state.items():
            if regime not in self.beliefs:
                self.beliefs[regime] = {}
            for tool, params in tools.items():
                self.beliefs[regime][tool] = BetaDistribution(
                    alpha=params['alpha'],
                    beta=params['beta']
                )
