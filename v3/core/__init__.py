"""
Core CSE (Competitive Specialist Ecosystem) algorithm components.
"""
from .thompson import ToolBeliefs
from .fitness import fitness_penalty, compute_fitness_scores
from .competition import CompetitionEngine
from .regimes import RegimeConfig, sample_regime
from .agent import ModernSpecialist

__all__ = [
    'ToolBeliefs',
    'fitness_penalty',
    'compute_fitness_scores',
    'CompetitionEngine',
    'RegimeConfig',
    'sample_regime',
    'ModernSpecialist',
]
