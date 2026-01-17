"""
Test 16: Low-Resource Regimes Handling
======================================
Does CSE handle rare task types?

Uses REAL Gemini API - NO mock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from test_runner import (
    generate_tasks, call_gemini, evaluate_response,
    get_specialist_for_regime, Agent, compute_coverage,
    TestResult, save_result, REGIME_TASKS
)


def train_with_imbalanced_distribution(
    n_agents: int,
    regimes: list,
    regime_weights: dict,
    n_generations: int,
    seed: int = 42
):
    """Train CSE with imbalanced task distribution."""
    random.seed(seed)
    np.random.seed(seed)

    agents = [Agent(id=i, wins_per_regime={r: 0 for r in regimes}) for i in range(n_agents)]

    for gen in range(n_generations):
        # Sample regime according to weights
        regime = random.choices(regimes, weights=[regime_weights[r] for r in regimes])[0]
        task = generate_tasks(regime, n=1, seed=gen)[0]

        k = min(3, n_agents)
        competitors = random.sample(agents, k)

        responses = []
        for agent in competitors:
            response, confidence = call_gemini(task.question)
            correct = evaluate_response(response, task.correct_answer)
            responses.append((agent, response, confidence, correct))

        correct_responses = [(a, c) for a, r, c, is_correct in responses if is_correct]
        if correct_responses:
            winner, _ = max(correct_responses, key=lambda x: x[1])
            winner.wins_per_regime[regime] = winner.wins_per_regime.get(regime, 0) + 1
            winner.total_wins += 1

    return agents


def run_test(n_seeds: int = 2) -> TestResult:
    """Test handling of low-resource regimes."""
    print("=" * 60)
    print("TEST 16: Low-Resource Regimes Handling")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())

    # Imbalanced distribution: some regimes are rare
    weights = {
        'math': 0.40,      # Common
        'code': 0.30,      # Common
        'reasoning': 0.15, # Less common
        'knowledge': 0.10, # Rare
        'creative': 0.05,  # Very rare
    }

    all_results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")

        print("\n  Training with imbalanced distribution:")
        for r, w in weights.items():
            print(f"    {r}: {w:.0%}")

        population = train_with_imbalanced_distribution(
            n_agents=10,
            regimes=regimes,
            regime_weights=weights,
            n_generations=50,
            seed=seed
        )

        # Check coverage of rare regimes
        rare_regimes = [r for r, w in weights.items() if w <= 0.10]

        print(f"\n  Rare regime coverage:")
        rare_covered = 0

        for regime in rare_regimes:
            specialist = get_specialist_for_regime(population, regime)
            has_specialist = specialist is not None and specialist.wins_per_regime.get(regime, 0) > 0

            if has_specialist:
                rare_covered += 1

            status = "✓ Covered" if has_specialist else "✗ Not covered"
            print(f"    {regime} (weight={weights[regime]:.0%}): {status}")

        rare_coverage_rate = rare_covered / len(rare_regimes) if rare_regimes else 1.0

        # Overall coverage
        overall_coverage = compute_coverage(population, regimes)

        print(f"\n  Rare regime coverage: {rare_coverage_rate:.0%}")
        print(f"  Overall coverage: {overall_coverage:.0%}")

        all_results.append({
            'rare_coverage': rare_coverage_rate,
            'overall_coverage': overall_coverage,
            'rare_regimes': rare_regimes
        })

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    mean_rare_coverage = np.mean([r['rare_coverage'] for r in all_results])
    mean_overall_coverage = np.mean([r['overall_coverage'] for r in all_results])

    print(f"Mean rare regime coverage: {mean_rare_coverage:.0%}")
    print(f"Mean overall coverage: {mean_overall_coverage:.0%}")

    # Success: rare regimes still get coverage
    passed = mean_rare_coverage >= 0.50 and mean_overall_coverage >= 0.80

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print("✅ TEST PASSED: CSE handles low-resource regimes!")
    else:
        print("❌ TEST FAILED: Low-resource regimes not well covered")

    result = TestResult(
        test_name="test_16_low_resource",
        passed=passed,
        metrics={
            'mean_rare_coverage': mean_rare_coverage,
            'mean_overall_coverage': mean_overall_coverage,
            'per_seed_results': all_results
        },
        details=f"Rare coverage: {mean_rare_coverage:.0%}, Overall: {mean_overall_coverage:.0%}"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_seeds=2)
