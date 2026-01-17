"""
Baseline Implementations for CSE Comparison

Two baselines per Prof. Abbeel's requirement:
A. RandomBaseline: No learning, random tool selection
B. IndividualBaseline: Learning without competition (uniform task sampling)

Both are compared against CSE (Competition-based Specialization via Evolution).

Run with: python -m experiments.baselines.run_baselines
"""
import os
import sys
import json
import asyncio
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import numpy as np

# Add v3 to path
V3_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(V3_ROOT))

from dotenv import load_dotenv
load_dotenv(V3_ROOT / '.env')


# =============================================================================
# BASELINE A: RANDOM (No Learning)
# =============================================================================

class RandomBaseline:
    """
    Baseline A: Random tool selection with NO learning.
    
    This measures pure chance performance - what happens when
    agents randomly select tools without any belief updates.
    
    Key properties:
    - No Thompson Sampling (beliefs never updated)
    - Random tool selection each time
    - No memory
    - No specialization possible
    
    Per Prof. Abbeel: "Truly random - no belief updates at all"
    """
    
    def __init__(self, agent_id: int, tools: List[str], seed: int):
        self.id = agent_id
        self.tools = tools
        self.rng = np.random.default_rng(seed + agent_id)
        
        # Track performance (for analysis only, not used for learning)
        self.correct_by_regime: Dict[str, int] = defaultdict(int)
        self.total_by_regime: Dict[str, int] = defaultdict(int)
        self.specialty = None  # Never specializes
    
    def select_tool(self, regime: str) -> str:
        """Randomly select a tool - NO learning."""
        return self.rng.choice(self.tools)
    
    def update(self, regime: str, tool: str, success: bool, task: str = ""):
        """Record result but DO NOT update beliefs."""
        self.total_by_regime[regime] += 1
        if success:
            self.correct_by_regime[regime] += 1
        # NO belief update - this is the key difference from CSE
    
    def get_accuracy(self, regime: Optional[str] = None) -> float:
        if regime:
            total = self.total_by_regime.get(regime, 0)
            correct = self.correct_by_regime.get(regime, 0)
        else:
            total = sum(self.total_by_regime.values())
            correct = sum(self.correct_by_regime.values())
        return correct / max(total, 1)


# =============================================================================
# BASELINE B: INDIVIDUAL LEARNING (No Competition)
# =============================================================================

class IndividualBaseline:
    """
    Baseline B: Individual learning WITHOUT competition.
    
    Each agent learns independently via Thompson Sampling,
    but there's no competition - all agents train on all regimes
    with uniform sampling (not weighted by performance).
    
    Key properties:
    - Thompson Sampling for tool selection
    - Belief updates on success/failure
    - NO fitness sharing (no competition pressure)
    - NO specialization pressure
    - Uniform regime sampling (not weighted)
    
    Per Prof. Abbeel: "Learning without competition - uniform task sampling"
    """
    
    def __init__(self, agent_id: int, regimes: List[str], tools: List[str], 
                 seed: int, memory_enabled: bool = True):
        self.id = agent_id
        self.regimes = regimes
        self.tools = tools
        self.rng = np.random.default_rng(seed + agent_id)
        
        # Thompson Sampling beliefs (same as CSE)
        self.beliefs = {
            regime: {tool: {'alpha': 1.0, 'beta': 1.0} for tool in tools}
            for regime in regimes
        }
        
        # Performance tracking
        self.wins: Dict[str, int] = {r: 0 for r in regimes}
        self.total_wins = 0
        self.specialty = None  # May or may not specialize
    
    def select_tool(self, regime: str) -> str:
        """Select tool using Thompson Sampling."""
        samples = {}
        for tool in self.tools:
            b = self.beliefs[regime][tool]
            samples[tool] = self.rng.beta(b['alpha'], b['beta'])
        return max(samples, key=samples.get)
    
    def update(self, regime: str, tool: str, success: bool, task: str = ""):
        """Update beliefs based on outcome - BUT no competition pressure."""
        if regime in self.beliefs and tool in self.beliefs[regime]:
            if success:
                self.beliefs[regime][tool]['alpha'] += 1
                self.wins[regime] = self.wins.get(regime, 0) + 1
                self.total_wins += 1
            else:
                self.beliefs[regime][tool]['beta'] += 1
        
        # Update specialty (may emerge naturally, but no competition pressure)
        self._update_specialty()
    
    def _update_specialty(self):
        """Check if agent has naturally specialized."""
        if not any(self.wins.values()):
            return
        
        total = sum(self.wins.values())
        for regime, wins in self.wins.items():
            if wins >= 5 and wins / max(total, 1) >= 0.35:
                self.specialty = regime
                return
    
    def get_best_tool(self, regime: str) -> str:
        """Get the tool with highest expected success rate."""
        means = {}
        for tool in self.tools:
            b = self.beliefs[regime][tool]
            means[tool] = b['alpha'] / (b['alpha'] + b['beta'])
        return max(means, key=means.get)


# =============================================================================
# BASELINE RUNNER
# =============================================================================

async def run_random_baseline(
    n_agents: int = 8,
    n_generations: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run Baseline A: Random (No Learning).
    
    All agents randomly select tools - measures chance performance.
    """
    print("\n" + "=" * 60)
    print("BASELINE A: RANDOM (No Learning)")
    print("=" * 60)
    print(f"Agents: {n_agents}, Generations: {n_generations}, Seed: {seed}")
    
    from experiments.training.run_training_v2 import (
        GroundTruthTaskBank,
        CompleteToolExecutor
    )
    
    np.random.seed(seed)
    random.seed(seed)
    
    task_bank = GroundTruthTaskBank(use_huggingface=False)
    tool_executor = CompleteToolExecutor(use_real_rag=True)
    
    regimes = task_bank.get_regimes()
    tools = ['L0', 'L1', 'L2', 'L3', 'L4']
    
    # Create random agents
    population = [RandomBaseline(i, tools, seed) for i in range(n_agents)]
    
    print(f"\nRunning {n_generations} generations with random tool selection...")
    
    for gen in range(n_generations):
        regime = random.choice(regimes)
        task = task_bank.sample(regime)
        
        # All agents try the task (no subset selection)
        for agent in population:
            tool = agent.select_tool(regime)  # Random selection
            result = await tool_executor.execute(task, tool)
            agent.update(regime, tool, result['correct'], task['question'])
        
        if (gen + 1) % 20 == 0:
            avg_acc = np.mean([a.get_accuracy() for a in population])
            print(f"Gen {gen+1:3d}: Avg accuracy = {avg_acc:.1%}")
    
    # Compute final metrics
    n_specialists = sum(1 for a in population if a.specialty is not None)
    coverage = len(set(a.specialty for a in population if a.specialty)) / len(regimes)
    avg_accuracy = np.mean([a.get_accuracy() for a in population])
    
    print(f"\n--- RANDOM BASELINE RESULTS ---")
    print(f"Specialists: {n_specialists}/{n_agents} (expected: 0)")
    print(f"Coverage: {coverage:.0%}")
    print(f"Avg accuracy: {avg_accuracy:.1%}")
    print(f"API calls: {tool_executor.call_count}")
    
    return {
        'baseline': 'random',
        'n_specialists': n_specialists,
        'coverage': coverage,
        'avg_accuracy': avg_accuracy,
        'api_calls': tool_executor.call_count
    }


async def run_individual_baseline(
    n_agents: int = 8,
    n_generations: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run Baseline B: Individual Learning (No Competition).
    
    Each agent learns independently with uniform task sampling.
    No competition pressure, no fitness sharing.
    """
    print("\n" + "=" * 60)
    print("BASELINE B: INDIVIDUAL LEARNING (No Competition)")
    print("=" * 60)
    print(f"Agents: {n_agents}, Generations: {n_generations}, Seed: {seed}")
    
    from experiments.training.run_training_v2 import (
        GroundTruthTaskBank,
        CompleteToolExecutor
    )
    
    np.random.seed(seed)
    random.seed(seed)
    
    task_bank = GroundTruthTaskBank(use_huggingface=False)
    tool_executor = CompleteToolExecutor(use_real_rag=True)
    
    regimes = task_bank.get_regimes()
    tools = ['L0', 'L1', 'L2', 'L3', 'L4']
    
    # Create learning agents (no competition)
    population = [
        IndividualBaseline(i, regimes, tools, seed)
        for i in range(n_agents)
    ]
    
    print(f"\nRunning {n_generations} generations with individual learning...")
    print("(Uniform regime sampling, no competition pressure)")
    
    for gen in range(n_generations):
        # UNIFORM sampling (not weighted by performance)
        regime = random.choice(regimes)  # Equal probability for all regimes
        task = task_bank.sample(regime)
        
        # Each agent learns independently
        for agent in population:
            tool = agent.select_tool(regime)  # Thompson Sampling
            result = await tool_executor.execute(task, tool)
            agent.update(regime, tool, result['correct'], task['question'])
            # NO competition - all agents update regardless of who was "best"
        
        if (gen + 1) % 20 == 0:
            n_spec = sum(1 for a in population if a.specialty)
            print(f"Gen {gen+1:3d}: Specialists = {n_spec}/{n_agents}")
    
    # Compute final metrics
    n_specialists = sum(1 for a in population if a.specialty is not None)
    specialties = [a.specialty for a in population if a.specialty]
    coverage = len(set(specialties)) / len(regimes)
    
    # Compute per-regime accuracy
    regime_accuracies = {}
    for regime in regimes:
        # Find best agent for this regime based on beliefs
        best_tools = [a.get_best_tool(regime) for a in population]
        regime_accuracies[regime] = Counter(best_tools)
    
    print(f"\n--- INDIVIDUAL LEARNING RESULTS ---")
    print(f"Specialists: {n_specialists}/{n_agents}")
    print(f"Coverage: {coverage:.0%}")
    print(f"Distribution: {Counter(specialties)}")
    print(f"API calls: {tool_executor.call_count}")
    
    return {
        'baseline': 'individual',
        'n_specialists': n_specialists,
        'coverage': coverage,
        'distribution': dict(Counter(specialties)),
        'regime_tool_preferences': {r: dict(c) for r, c in regime_accuracies.items()},
        'api_calls': tool_executor.call_count
    }


async def run_all_baselines(
    n_agents: int = 8,
    n_generations: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """Run both baselines and compare."""
    print("=" * 70)
    print("RUNNING ALL BASELINES")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    
    results = {}
    
    # Baseline A: Random
    results['random'] = await run_random_baseline(n_agents, n_generations, seed)
    
    # Baseline B: Individual
    results['individual'] = await run_individual_baseline(n_agents, n_generations, seed)
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20} {'Random':<15} {'Individual':<15}")
    print("-" * 50)
    print(f"{'Specialists':<20} {results['random']['n_specialists']:<15} {results['individual']['n_specialists']:<15}")
    print(f"{'Coverage':<20} {results['random']['coverage']:.0%:<15} {results['individual']['coverage']:.0%:<15}")
    print(f"{'API Calls':<20} {results['random']['api_calls']:<15} {results['individual']['api_calls']:<15}")
    
    # Save results
    output_dir = V3_ROOT / 'results' / 'baselines'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results['timestamp'] = datetime.now().isoformat()
    results['config'] = {
        'n_agents': n_agents,
        'n_generations': n_generations,
        'seed': seed
    }
    
    with open(output_dir / 'baseline_comparison.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir / 'baseline_comparison.json'}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument('--agents', type=int, default=8)
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--baseline', type=str, default='all',
                       choices=['random', 'individual', 'all'])
    
    args = parser.parse_args()
    
    if args.baseline == 'random':
        asyncio.run(run_random_baseline(args.agents, args.generations, args.seed))
    elif args.baseline == 'individual':
        asyncio.run(run_individual_baseline(args.agents, args.generations, args.seed))
    else:
        asyncio.run(run_all_baselines(args.agents, args.generations, args.seed))


if __name__ == '__main__':
    main()
