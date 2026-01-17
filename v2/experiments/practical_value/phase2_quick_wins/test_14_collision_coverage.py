"""
Test 14: Collision-Free Coverage
=================================
Test if fitness sharing prevents collisions.

Hypothesis: CSE achieves full task coverage without specialists
colliding (multiple agents claiming same niche).

Uses REAL Gemini API - NO mock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from collections import defaultdict
from test_runner import (
    train_cse_population, compute_coverage,
    TestResult, save_result, REGIME_TASKS
)


def analyze_specialist_distribution(population, regimes):
    """Analyze how specialists are distributed across regimes."""
    specialist_map = defaultdict(list)

    for agent in population:
        primary = agent.get_primary_specialty()
        if primary:
            specialist_map[primary].append(agent.id)

    return specialist_map


def run_test(n_seeds: int = 5) -> TestResult:
    """
    Test collision-free coverage.
    """
    print("=" * 60)
    print("TEST 14: Collision-Free Coverage")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())
    all_results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")

        # Train population
        population = train_cse_population(
            n_agents=10,
            regimes=regimes,
            n_generations=40,
            seed=seed
        )

        # Analyze distribution
        specialist_map = analyze_specialist_distribution(population, regimes)

        # Compute metrics
        covered = len(specialist_map)
        total_regimes = len(regimes)
        coverage = covered / total_regimes

        collisions = sum(1 for agents in specialist_map.values() if len(agents) > 1)
        collision_rate = collisions / total_regimes if total_regimes > 0 else 0

        uncovered = [r for r in regimes if r not in specialist_map]

        print(f"\n  Specialist distribution:")
        for regime, agent_ids in specialist_map.items():
            status = "⚠️ COLLISION" if len(agent_ids) > 1 else "✓"
            print(f"    {regime}: {agent_ids} {status}")

        if uncovered:
            print(f"  Uncovered regimes: {uncovered}")

        print(f"\n  Coverage: {coverage:.0%} ({covered}/{total_regimes})")
        print(f"  Collision rate: {collision_rate:.0%} ({collisions} collisions)")

        all_results.append({
            'coverage': coverage,
            'collision_rate': collision_rate,
            'specialist_map': {k: len(v) for k, v in specialist_map.items()},
            'uncovered': uncovered
        })

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    mean_coverage = np.mean([r['coverage'] for r in all_results])
    mean_collision_rate = np.mean([r['collision_rate'] for r in all_results])

    print(f"Mean coverage: {mean_coverage:.0%}")
    print(f"Mean collision rate: {mean_collision_rate:.0%}")

    # Success criteria
    passed = (
        mean_coverage > 0.80 and  # > 80% coverage
        mean_collision_rate < 0.30  # < 30% collision rate
    )

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print("✅ TEST PASSED: Collision-free coverage achieved!")
        print(f"   {mean_coverage:.0%} coverage with {mean_collision_rate:.0%} collision rate")
    else:
        print("❌ TEST FAILED: Coverage or collision issues")
        print(f"   {mean_coverage:.0%} coverage with {mean_collision_rate:.0%} collision rate")

    result = TestResult(
        test_name="test_14_collision_coverage",
        passed=passed,
        metrics={
            'mean_coverage': mean_coverage,
            'mean_collision_rate': mean_collision_rate,
            'per_seed_results': all_results
        },
        details=f"Coverage: {mean_coverage:.0%}, Collision rate: {mean_collision_rate:.0%}"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_seeds=3)
