"""
Test 15: Scaling to Many Regimes
================================
Does CSE work with many task types?

Uses REAL Gemini API - NO mock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from test_runner import (
    train_cse_population, compute_coverage,
    TestResult, save_result, REGIME_TASKS
)


def run_test() -> TestResult:
    """Test scaling to many regimes."""
    print("=" * 60)
    print("TEST 15: Scaling to Many Regimes")
    print("=" * 60)

    # Test with different numbers of regimes
    all_regimes = list(REGIME_TASKS.keys())

    regime_configs = [
        all_regimes[:2],  # 2 regimes
        all_regimes[:3],  # 3 regimes
        all_regimes[:4],  # 4 regimes
        all_regimes,      # 5 regimes
    ]

    results = {}

    for regimes in regime_configs:
        n_regimes = len(regimes)
        print(f"\n--- Testing with {n_regimes} regimes ---")

        # Train population (more agents for more regimes)
        n_agents = max(6, n_regimes * 2)

        population = train_cse_population(
            n_agents=n_agents,
            regimes=regimes,
            n_generations=n_regimes * 8,  # Scale generations
            seed=42
        )

        coverage = compute_coverage(population, regimes)

        results[n_regimes] = {
            'coverage': coverage,
            'n_agents': n_agents,
            'n_generations': n_regimes * 8
        }

        print(f"  Coverage: {coverage:.0%}")

    # Analyze scaling
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)

    for n_regimes, data in results.items():
        print(f"{n_regimes} regimes: {data['coverage']:.0%} coverage")

    # Check if coverage stays high
    coverages = [data['coverage'] for data in results.values()]
    min_coverage = min(coverages)
    mean_coverage = np.mean(coverages)

    print(f"\nMin coverage: {min_coverage:.0%}")
    print(f"Mean coverage: {mean_coverage:.0%}")

    # Success: maintain good coverage at scale
    passed = min_coverage >= 0.60 and mean_coverage >= 0.80

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print("✅ TEST PASSED: CSE scales to many regimes!")
    else:
        print("❌ TEST FAILED: CSE doesn't scale well")

    result = TestResult(
        test_name="test_15_scaling_regimes",
        passed=passed,
        metrics={
            'per_regime_results': results,
            'min_coverage': min_coverage,
            'mean_coverage': mean_coverage
        },
        details=f"Min coverage: {min_coverage:.0%}, Mean: {mean_coverage:.0%}"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test()
