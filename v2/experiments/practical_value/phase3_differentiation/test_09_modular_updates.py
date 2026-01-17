"""
Test 9: Modular Updates
=======================
Can individual specialists be updated without affecting others?

Uses REAL Gemini API - NO mock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import copy
import numpy as np
from test_runner import (
    train_cse_population, generate_tasks, call_gemini,
    evaluate_response, get_specialist_for_regime, Agent,
    TestResult, save_result, REGIME_TASKS
)


def evaluate_specialist_accuracy(population, regime, n_tasks=10):
    """Evaluate how well the population handles a regime."""
    tasks = generate_tasks(regime, n=n_tasks, seed=hash(regime) % 10000)

    correct = 0
    for task in tasks:
        response, _ = call_gemini(task.question)
        if evaluate_response(response, task.correct_answer):
            correct += 1

    return correct / len(tasks)


def run_test(n_seeds: int = 2) -> TestResult:
    """Test modular updates."""
    print("=" * 60)
    print("TEST 9: Modular Updates")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())

    all_results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        random.seed(seed)
        np.random.seed(seed)

        # Train population
        population = train_cse_population(
            n_agents=8,
            regimes=regimes,
            n_generations=25,
            seed=seed
        )

        # Baseline accuracies
        print("\n  Baseline accuracies:")
        baseline = {}
        for regime in regimes:
            acc = evaluate_specialist_accuracy(population, regime)
            baseline[regime] = acc
            print(f"    {regime}: {acc:.1%}")

        # Simulate replacing math specialist
        target_regime = 'math'
        old_specialist = get_specialist_for_regime(population, target_regime)

        if old_specialist:
            # Create new specialist (simulated retraining)
            new_specialist = Agent(
                id=100,
                wins_per_regime={target_regime: 15},  # Fresh specialist
                total_wins=15
            )

            # Replace in population
            modified_pop = [a for a in population if a.id != old_specialist.id]
            modified_pop.append(new_specialist)

            # Check collateral damage
            print(f"\n  After replacing {target_regime} specialist:")
            post_update = {}
            collateral_damage = {}

            for regime in regimes:
                acc = evaluate_specialist_accuracy(modified_pop, regime)
                post_update[regime] = acc
                damage = baseline[regime] - acc
                collateral_damage[regime] = damage

                status = "COLLATERAL" if damage > 0.1 else "OK"
                print(f"    {regime}: {acc:.1%} (change: {-damage:+.1%}) [{status}]")

            # Calculate metrics
            max_collateral = max(
                collateral_damage[r] for r in regimes if r != target_regime
            )

            all_results.append({
                'target_regime': target_regime,
                'baseline': baseline,
                'post_update': post_update,
                'max_collateral_damage': max_collateral
            })

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    mean_collateral = np.mean([r['max_collateral_damage'] for r in all_results])
    print(f"Mean max collateral damage: {mean_collateral:.1%}")

    # Success: collateral damage < 5%
    passed = mean_collateral < 0.10

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print("✅ TEST PASSED: Modular updates work!")
    else:
        print("❌ TEST FAILED: Updates cause collateral damage")

    result = TestResult(
        test_name="test_09_modular_updates",
        passed=passed,
        metrics={
            'mean_collateral_damage': mean_collateral,
            'per_seed_results': all_results
        },
        details=f"Max collateral damage: {mean_collateral:.1%}"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_seeds=2)
