"""
Non-uniform regime configuration and sampling.

Regimes have different frequencies, rewards, and difficulties.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RegimeConfig:
    """Configuration for a single regime."""
    name: str
    frequency: float      # How often this regime appears (0-1, sums to 1)
    reward: float         # Reward multiplier for winning
    difficulty: float     # Base success probability (0-1)
    optimal_tool: str     # Best tool for this regime
    description: str = "" # Human-readable description

    def expected_value(self, n_specialists: int = 1) -> float:
        """
        Compute expected value of specializing in this regime.

        EV = frequency × reward × difficulty / n^1.5
        """
        return (self.frequency * self.reward * self.difficulty) / (n_specialists ** 1.5)


# Default regime configuration for 5 regimes
DEFAULT_REGIME_CONFIG: Dict[str, RegimeConfig] = {
    'code_math': RegimeConfig(
        name='code_math',
        frequency=0.30,
        reward=2.0,
        difficulty=0.70,
        optimal_tool='L1',
        description='Mathematical and coding tasks requiring Python execution'
    ),
    'vision': RegimeConfig(
        name='vision',
        frequency=0.15,
        reward=3.0,
        difficulty=0.50,
        optimal_tool='L2',
        description='Visual understanding tasks requiring image processing'
    ),
    'rag': RegimeConfig(
        name='rag',
        frequency=0.25,
        reward=2.5,
        difficulty=0.60,
        optimal_tool='L3',
        description='Document QA requiring retrieval augmented generation'
    ),
    'web': RegimeConfig(
        name='web',
        frequency=0.20,
        reward=3.0,
        difficulty=0.40,
        optimal_tool='L4',
        description='Real-time information tasks requiring web access'
    ),
    'pure_qa': RegimeConfig(
        name='pure_qa',
        frequency=0.10,
        reward=1.0,
        difficulty=0.90,
        optimal_tool='L0',
        description='General knowledge QA solvable without tools'
    ),
}


class RegimeSampler:
    """
    Sample regimes according to non-uniform frequency distribution.
    """

    def __init__(
        self,
        config: Optional[Dict[str, RegimeConfig]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize regime sampler.

        Args:
            config: Regime configuration dict (uses DEFAULT if None)
            seed: Random seed for reproducibility
        """
        self.config = config or DEFAULT_REGIME_CONFIG
        self.rng = np.random.default_rng(seed)

        # Extract names and frequencies
        self.regime_names = list(self.config.keys())
        self.frequencies = np.array([
            self.config[name].frequency for name in self.regime_names
        ])

        # Normalize frequencies to sum to 1
        self.frequencies = self.frequencies / self.frequencies.sum()

    def sample(self) -> str:
        """Sample a regime according to frequency distribution."""
        return self.rng.choice(self.regime_names, p=self.frequencies)

    def get_config(self, regime: str) -> RegimeConfig:
        """Get configuration for a specific regime."""
        return self.config[regime]

    def get_optimal_tool(self, regime: str) -> str:
        """Get the optimal tool for a regime."""
        return self.config[regime].optimal_tool

    def get_reward(self, regime: str) -> float:
        """Get reward multiplier for a regime."""
        return self.config[regime].reward

    def get_difficulty(self, regime: str) -> float:
        """Get base difficulty for a regime."""
        return self.config[regime].difficulty


def sample_regime(
    config: Optional[Dict[str, RegimeConfig]] = None,
    seed: Optional[int] = None
) -> str:
    """
    Convenience function to sample a single regime.

    Args:
        config: Regime configuration (uses DEFAULT if None)
        seed: Random seed

    Returns:
        Sampled regime name
    """
    sampler = RegimeSampler(config, seed)
    return sampler.sample()


def get_regime_list() -> List[str]:
    """Get list of all regime names."""
    return list(DEFAULT_REGIME_CONFIG.keys())


def get_regime_config_dict() -> Dict[str, Dict]:
    """Get regime config as plain dict for serialization."""
    return {
        name: {
            'frequency': cfg.frequency,
            'reward': cfg.reward,
            'difficulty': cfg.difficulty,
            'optimal_tool': cfg.optimal_tool,
        }
        for name, cfg in DEFAULT_REGIME_CONFIG.items()
    }
