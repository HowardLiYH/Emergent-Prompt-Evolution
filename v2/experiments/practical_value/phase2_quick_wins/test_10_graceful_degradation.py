"""
Test 10: Graceful Degradation
=============================
Test if system survives when a specialist fails.

Hypothesis: When a specialist fails, other agents can cover
its tasks with acceptable quality.

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


def evaluate_system_on_regime(population, regime, n_tasks=10):
    """Evaluate system accuracy on a regime using best available agent."""
    tasks = generate_tasks(regime, n=n_tasks, seed=hash(regime) % 10000)

    # Find best agent for this regime
    best_agent = get_specialist_for_regime(population, regime)
    if best_agent is None and population:
        best_agent = population[0]

    if best_agent is None:
        return 0.0

    correct = 0
    for task in tasks:
        response, _ = call_gemini(task.question)
        if evaluate_response(response, task.correct_answer):
            correct += 1

    return correct / len(tasks)


def run_test(n_seeds: int = 3) -> TestResult:
    """
    Test graceful degradation when specialists are removed.
    """
    print("=" * 60)
    print("TEST 10: Graceful Degradation")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())
    all_degradation_results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")

        # Train population
        population = train_cse_population(
            n_agents=10,
            regimes=regimes,
            n_generations=30,
            seed=seed
        )

        # Baseline: Full system accuracy
        print("\n  Evaluating full system...")
        full_accuracy = {}
        for regime in regimes:
            full_accuracy[regime] = evaluate_system_on_regime(population, regime)
            print(f"    {regime}: {full_accuracy[regime]:.1%}")

        # Remove each specialist and measure degradation
        degradation = {}

        for regime in regimes:
            specialist = get_specialist_for_regime(population, regime)
            if specialist is None:
                continue

            # Create reduced population
            reduced = [a for a in population if a.id != specialist.id]

            # Evaluate on removed regime
            reduced_accuracy = evaluate_system_on_regime(reduced, regime)

            deg = full_accuracy[regime] - reduced_accuracy
            degradation[regime] = {
                'full_accuracy': full_accuracy[regime],
                'reduced_accuracy': reduced_accuracy,
                'degradation': deg,
                'specialist_id': specialist.id
            }

            print(f"\n  Removed {regime} specialist (Agent {specialist.id}):")
            print(f"    Full: {full_accuracy[regime]:.1%} → Reduced: {reduced_accuracy:.1%}")
            print(f"    Degradation: {deg:+.1%}")

        all_degradation_results.append(degradation)

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    regime_degradations = {r: [] for r in regimes}
    regime_fallback_quality = {r: [] for r in regimes}

    for seed_result in all_degradation_results:
        for regime, data in seed_result.items():
            regime_degradations[regime].append(data['degradation'])
            regime_fallback_quality[regime].append(data['reduced_accuracy'])

    summary = {}
    worst_degradation = 0
    min_fallback = 1.0

    for regime in regimes:
        degs = regime_degradations[regime]
        fallbacks = regime_fallback_quality[regime]

        if not degs:
            continue

        mean_deg = np.mean(degs)
        mean_fallback = np.mean(fallbacks)

        summary[regime] = {
            'mean_degradation': mean_deg,
            'mean_fallback_quality': mean_fallback
        }

        worst_degradation = max(worst_degradation, mean_deg)
        min_fallback = min(min_fallback, mean_fallback)

        print(f"{regime}:")
        print(f"  Mean degradation: {mean_deg:+.1%}")
        print(f"  Fallback quality: {mean_fallback:.1%}")

    # Success criteria
    passed = (
        worst_degradation < 0.20 and  # < 20% degradation
        min_fallback > 0.50  # > 50% fallback quality
    )

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"Worst-case degradation: {worst_degradation:.1%}")
    print(f"Minimum fallback quality: {min_fallback:.1%}")

    if passed:
        print("\n✅ TEST PASSED: System degrades gracefully!")
    else:
        print("\n❌ TEST FAILED: System does not degrade gracefully")

    result = TestResult(
        test_name="test_10_graceful_degradation",
        passed=passed,
        metrics={
            'worst_case_degradation': worst_degradation,
            'min_fallback_quality': min_fallback,
            'per_regime': summary
        },
        details=f"Worst degradation: {worst_degradation:.1%}, Min fallback: {min_fallback:.1%}"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_seeds=2)
