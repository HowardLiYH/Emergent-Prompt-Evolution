"""
Test 1: Specialist Accuracy Advantage
=====================================
CRITICAL TEST - If this fails, CSE has no practical value.

Hypothesis: Agents that specialize in a task type achieve higher accuracy
on that task type than generalist agents.

Uses REAL Gemini API - NO mock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from scipy import stats
from test_runner import (
    train_cse_population, generate_tasks, call_gemini,
    evaluate_response, get_specialist_for_regime,
    TestResult, save_result, REGIME_TASKS
)


def run_test(n_seeds: int = 5, n_held_out_tasks: int = 20) -> TestResult:
    """
    Test if specialists outperform generalists on their specialty.
    """
    print("=" * 60)
    print("TEST 1: Specialist Accuracy Advantage")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())
    all_results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")

        # Train population
        population = train_cse_population(
            n_agents=10,
            regimes=regimes,
            n_generations=30,
            seed=seed
        )

        seed_results = {}

        for regime in regimes:
            print(f"\n  Testing regime: {regime}")

            # Get specialist for this regime
            specialist = get_specialist_for_regime(population, regime)
            if specialist is None:
                print(f"    No specialist found for {regime}")
                continue

            # Get a random non-specialist
            non_specialists = [a for a in population if a.id != specialist.id]
            generalist = random.choice(non_specialists)

            # Generate held-out tasks
            held_out = generate_tasks(regime, n=n_held_out_tasks, seed=1000 + seed)

            # Test specialist
            specialist_correct = 0
            for task in held_out:
                response, _ = call_gemini(task.question)
                if evaluate_response(response, task.correct_answer):
                    specialist_correct += 1
            specialist_acc = specialist_correct / len(held_out)

            # Test generalist
            generalist_correct = 0
            for task in held_out:
                response, _ = call_gemini(task.question)
                if evaluate_response(response, task.correct_answer):
                    generalist_correct += 1
            generalist_acc = generalist_correct / len(held_out)

            advantage = specialist_acc - generalist_acc

            print(f"    Specialist (Agent {specialist.id}): {specialist_acc:.1%}")
            print(f"    Generalist (Agent {generalist.id}): {generalist_acc:.1%}")
            print(f"    Advantage: {advantage:+.1%}")

            seed_results[regime] = {
                'specialist_acc': specialist_acc,
                'generalist_acc': generalist_acc,
                'advantage': advantage,
                'specialist_id': specialist.id,
                'specialist_wins': specialist.wins_per_regime.get(regime, 0)
            }

        all_results.append(seed_results)

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    regime_advantages = {r: [] for r in regimes}
    for seed_result in all_results:
        for regime, data in seed_result.items():
            regime_advantages[regime].append(data['advantage'])

    summary = {}
    wins = 0

    for regime in regimes:
        advs = regime_advantages[regime]
        if not advs:
            continue

        mean_adv = np.mean(advs)
        std_adv = np.std(advs)

        # One-sample t-test: is advantage > 0?
        if len(advs) > 1:
            t_stat, p_value = stats.ttest_1samp(advs, 0)
            p_value = p_value / 2  # One-tailed
        else:
            t_stat, p_value = 0, 1.0

        specialist_wins = mean_adv > 0
        if specialist_wins:
            wins += 1

        summary[regime] = {
            'mean_advantage': mean_adv,
            'std_advantage': std_adv,
            'p_value': p_value,
            'specialist_wins': specialist_wins
        }

        print(f"\n{regime}:")
        print(f"  Mean advantage: {mean_adv:+.1%} ± {std_adv:.1%}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Specialist wins: {'YES' if specialist_wins else 'NO'}")

    # Overall assessment
    overall_advantages = [adv for advs in regime_advantages.values() for adv in advs]
    mean_overall = np.mean(overall_advantages) if overall_advantages else 0

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"Overall mean advantage: {mean_overall:+.1%}")
    print(f"Regimes where specialist wins: {wins}/{len(regimes)}")

    # Success criteria
    passed = (
        mean_overall > 0.05 and  # > 5% average advantage
        wins >= len(regimes) * 0.6  # Win in at least 60% of regimes
    )

    if passed:
        print("\n✅ TEST PASSED: Specialists outperform generalists!")
    else:
        print("\n❌ TEST FAILED: Specialists do NOT consistently outperform generalists")
        print("   This is a CRITICAL failure - CSE may not have practical value")

    result = TestResult(
        test_name="test_01_specialist_accuracy",
        passed=passed,
        metrics={
            'mean_overall_advantage': mean_overall,
            'regimes_where_specialist_wins': wins,
            'total_regimes': len(regimes),
            'per_regime_summary': summary
        },
        details=f"Mean advantage: {mean_overall:+.1%}, Wins: {wins}/{len(regimes)}"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_seeds=3, n_held_out_tasks=10)
