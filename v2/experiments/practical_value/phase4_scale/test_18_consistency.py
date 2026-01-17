"""
Test 18: Consistency / Reproducibility
======================================
Are results consistent across runs?

Uses REAL Gemini API - NO mock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from test_runner import (
    train_cse_population, compute_coverage, get_specialist_for_regime,
    TestResult, save_result, REGIME_TASKS
)


def run_test(n_seeds: int = 5) -> TestResult:
    """Test consistency across multiple runs."""
    print("=" * 60)
    print("TEST 18: Consistency / Reproducibility")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())

    all_coverages = []
    all_specialist_distributions = []

    for seed in range(n_seeds):
        print(f"\n--- Run {seed + 1}/{n_seeds} ---")

        population = train_cse_population(
            n_agents=10,
            regimes=regimes,
            n_generations=30,
            seed=seed
        )

        coverage = compute_coverage(population, regimes)
        all_coverages.append(coverage)

        # Record specialist distribution
        specialist_dist = {}
        for regime in regimes:
            specialist = get_specialist_for_regime(population, regime)
            specialist_dist[regime] = specialist.id if specialist else None

        all_specialist_distributions.append(specialist_dist)

        print(f"  Coverage: {coverage:.0%}")
        print(f"  Specialists: {specialist_dist}")

    # Analyze consistency
    print("\n" + "=" * 60)
    print("CONSISTENCY ANALYSIS")
    print("=" * 60)

    coverage_mean = np.mean(all_coverages)
    coverage_std = np.std(all_coverages)
    coverage_cv = coverage_std / coverage_mean if coverage_mean > 0 else 1.0

    print(f"Coverage: {coverage_mean:.0%} ± {coverage_std:.0%}")
    print(f"Coefficient of variation: {coverage_cv:.2f}")

    # Check if same regimes get covered
    regime_coverage_rate = {}
    for regime in regimes:
        covered_count = sum(
            1 for dist in all_specialist_distributions
            if dist.get(regime) is not None
        )
        regime_coverage_rate[regime] = covered_count / n_seeds
        print(f"  {regime} coverage rate: {regime_coverage_rate[regime]:.0%}")

    min_regime_rate = min(regime_coverage_rate.values())

    # Success: low variance and consistent coverage
    passed = coverage_cv < 0.20 and min_regime_rate >= 0.60

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print("✅ TEST PASSED: Results are consistent!")
    else:
        print("❌ TEST FAILED: Results vary too much")

    result = TestResult(
        test_name="test_18_consistency",
        passed=passed,
        metrics={
            'coverage_mean': coverage_mean,
            'coverage_std': coverage_std,
            'coverage_cv': coverage_cv,
            'regime_coverage_rates': regime_coverage_rate,
            'min_regime_rate': min_regime_rate
        },
        details=f"CV: {coverage_cv:.2f}, Min regime rate: {min_regime_rate:.0%}"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_seeds=3)
