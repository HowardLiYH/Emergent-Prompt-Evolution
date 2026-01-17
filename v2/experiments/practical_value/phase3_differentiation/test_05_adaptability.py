"""
Test 5: Adaptability to New Task Types
======================================
Can CSE adapt to new regimes without full retraining?

Uses REAL Gemini API - NO mock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from test_runner import (
    train_cse_population, generate_tasks, call_gemini,
    evaluate_response, get_specialist_for_regime, Agent,
    TestResult, save_result, REGIME_TASKS
)


def run_test(n_seeds: int = 3) -> TestResult:
    """Test adaptability to new task types."""
    print("=" * 60)
    print("TEST 5: Adaptability to New Regimes")
    print("=" * 60)

    # Original regimes (train on subset)
    original_regimes = ['math', 'code', 'reasoning']
    new_regime = 'knowledge'  # Introduce later

    all_results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        random.seed(seed)
        np.random.seed(seed)

        # Phase 1: Train on original regimes
        print("\n  Phase 1: Training on original regimes...")
        population = train_cse_population(
            n_agents=8,
            regimes=original_regimes,
            n_generations=20,
            seed=seed
        )

        # Phase 2: Introduce new regime and continue training
        print(f"\n  Phase 2: Introducing '{new_regime}' regime...")
        all_regimes = original_regimes + [new_regime]

        # Continue training with new regime
        for agent in population:
            agent.wins_per_regime[new_regime] = 0

        for gen in range(20):  # 20 more generations
            regime = random.choice(all_regimes)
            task = generate_tasks(regime, n=1, seed=1000 + gen)[0]

            k = min(3, len(population))
            competitors = random.sample(population, k)

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

        # Check adaptation
        new_specialist = get_specialist_for_regime(population, new_regime)
        new_specialist_wins = new_specialist.wins_per_regime.get(new_regime, 0) if new_specialist else 0

        # Check if original specialists degraded
        original_degraded = False
        for regime in original_regimes:
            specialist = get_specialist_for_regime(population, regime)
            if specialist is None or specialist.wins_per_regime.get(regime, 0) < 3:
                original_degraded = True

        print(f"\n  New regime specialist: Agent {new_specialist.id if new_specialist else 'None'}")
        print(f"  New regime wins: {new_specialist_wins}")
        print(f"  Original specialists degraded: {original_degraded}")

        all_results.append({
            'new_specialist_found': new_specialist is not None,
            'new_specialist_wins': new_specialist_wins,
            'original_degraded': original_degraded,
            'adapted_in_20_gens': new_specialist_wins >= 3
        })

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    adaptation_rate = np.mean([r['adapted_in_20_gens'] for r in all_results])
    degradation_rate = np.mean([r['original_degraded'] for r in all_results])

    print(f"Adaptation success rate: {adaptation_rate:.0%}")
    print(f"Original degradation rate: {degradation_rate:.0%}")

    passed = adaptation_rate >= 0.5 and degradation_rate < 0.5

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print("✅ TEST PASSED: CSE adapts to new regimes!")
    else:
        print("❌ TEST FAILED: CSE does not adapt well")

    result = TestResult(
        test_name="test_05_adaptability",
        passed=passed,
        metrics={
            'adaptation_rate': adaptation_rate,
            'degradation_rate': degradation_rate,
            'per_seed_results': all_results
        },
        details=f"Adaptation: {adaptation_rate:.0%}, Degradation: {degradation_rate:.0%}"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_seeds=2)
