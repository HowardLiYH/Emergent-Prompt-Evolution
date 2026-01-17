"""
Test 6: Distribution Shift Robustness
=====================================
Does CSE handle changing task distributions?

Uses REAL Gemini API - NO mock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from test_runner import (
    train_cse_population, generate_tasks, call_gemini,
    evaluate_response, get_specialist_for_regime,
    TestResult, save_result, REGIME_TASKS
)


def evaluate_on_distribution(population, regime_weights, n_tasks=30):
    """Evaluate population on a given distribution of tasks."""
    regimes = list(regime_weights.keys())
    weights = list(regime_weights.values())

    correct = 0
    total = 0

    for _ in range(n_tasks):
        regime = random.choices(regimes, weights=weights)[0]
        task = generate_tasks(regime, n=1, seed=random.randint(0, 10000))[0]

        specialist = get_specialist_for_regime(population, regime)

        response, _ = call_gemini(task.question)
        if evaluate_response(response, task.correct_answer):
            correct += 1
        total += 1

    return correct / total if total > 0 else 0


def run_test(n_seeds: int = 3) -> TestResult:
    """Test distribution shift robustness."""
    print("=" * 60)
    print("TEST 6: Distribution Shift Robustness")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())

    # Training distribution: uniform
    train_dist = {r: 1/len(regimes) for r in regimes}

    # Test distributions: skewed
    test_distributions = {
        'uniform': {r: 0.2 for r in regimes},
        'math_heavy': {'math': 0.6, 'code': 0.15, 'reasoning': 0.1, 'knowledge': 0.1, 'creative': 0.05},
        'code_heavy': {'math': 0.15, 'code': 0.6, 'reasoning': 0.1, 'knowledge': 0.1, 'creative': 0.05},
    }

    all_results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        random.seed(seed)
        np.random.seed(seed)

        # Train population
        population = train_cse_population(
            n_agents=10,
            regimes=regimes,
            n_generations=30,
            seed=seed
        )

        seed_results = {}

        for dist_name, dist in test_distributions.items():
            accuracy = evaluate_on_distribution(population, dist, n_tasks=20)
            seed_results[dist_name] = accuracy
            print(f"  {dist_name}: {accuracy:.1%}")

        all_results.append(seed_results)

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    mean_results = {}
    for dist_name in test_distributions:
        accs = [r[dist_name] for r in all_results]
        mean_results[dist_name] = np.mean(accs)
        print(f"{dist_name}: {np.mean(accs):.1%} ± {np.std(accs):.1%}")

    # Check degradation under shift
    uniform_acc = mean_results['uniform']
    worst_shift_acc = min(mean_results['math_heavy'], mean_results['code_heavy'])
    degradation = uniform_acc - worst_shift_acc

    print(f"\nWorst-case degradation under shift: {degradation:.1%}")

    passed = degradation < 0.15  # Less than 15% degradation

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print("✅ TEST PASSED: CSE handles distribution shift well!")
    else:
        print("❌ TEST FAILED: CSE degrades under distribution shift")

    result = TestResult(
        test_name="test_06_distribution_shift",
        passed=passed,
        metrics={
            'mean_results': mean_results,
            'degradation': degradation,
            'per_seed_results': all_results
        },
        details=f"Worst degradation: {degradation:.1%}"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_seeds=2)
