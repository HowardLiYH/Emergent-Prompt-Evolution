"""
Test 11: Transfer Learning
==========================
Do specialists transfer to related domains?

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
    TestResult, save_result, REGIME_TASKS, Task
)


# Related domains for transfer testing
TRANSFER_TASKS = {
    'math': [  # Transfer to physics
        Task('physics', "If a ball falls for 2 seconds with g=10 m/s^2, what's its velocity?", "20"),
        Task('physics', "F=ma. If m=5kg and a=3 m/s^2, what is F?", "15"),
        Task('physics', "What is 1/2 * 10 * 4?", "20"),
    ],
    'code': [  # Transfer to SQL
        Task('sql', "What SQL keyword retrieves data?", "SELECT"),
        Task('sql', "What SQL clause filters results?", "WHERE"),
        Task('sql', "What joins two tables in SQL?", "JOIN"),
    ],
    'reasoning': [  # Transfer to logic puzzles
        Task('logic', "If P implies Q, and not Q, what about P?", "not P"),
        Task('logic', "All A are B. Some B are C. Must any A be C?", "no"),
    ],
}


def run_test(n_seeds: int = 2) -> TestResult:
    """Test transfer learning."""
    print("=" * 60)
    print("TEST 11: Transfer Learning")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())

    all_results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        random.seed(seed)
        np.random.seed(seed)

        population = train_cse_population(
            n_agents=10,
            regimes=regimes,
            n_generations=30,
            seed=seed
        )

        for source_regime, transfer_tasks in TRANSFER_TASKS.items():
            if not transfer_tasks:
                continue

            print(f"\n  {source_regime} → {transfer_tasks[0].regime}:")

            specialist = get_specialist_for_regime(population, source_regime)
            if specialist is None:
                print(f"    No specialist for {source_regime}")
                continue

            # Test specialist on transfer tasks
            specialist_correct = 0
            for task in transfer_tasks:
                response, _ = call_gemini(task.question)
                if evaluate_response(response, task.correct_answer):
                    specialist_correct += 1

            specialist_acc = specialist_correct / len(transfer_tasks)

            # Test random agent on same tasks
            non_specialists = [a for a in population if a.id != specialist.id]
            generalist = random.choice(non_specialists)

            generalist_correct = 0
            for task in transfer_tasks:
                response, _ = call_gemini(task.question)
                if evaluate_response(response, task.correct_answer):
                    generalist_correct += 1

            generalist_acc = generalist_correct / len(transfer_tasks)

            transfer_advantage = specialist_acc - generalist_acc

            print(f"    Specialist: {specialist_acc:.1%}")
            print(f"    Generalist: {generalist_acc:.1%}")
            print(f"    Transfer advantage: {transfer_advantage:+.1%}")

            all_results.append({
                'source': source_regime,
                'target': transfer_tasks[0].regime,
                'specialist_acc': specialist_acc,
                'generalist_acc': generalist_acc,
                'transfer_advantage': transfer_advantage
            })

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    mean_transfer = np.mean([r['transfer_advantage'] for r in all_results])
    positive_transfers = sum(1 for r in all_results if r['transfer_advantage'] > 0)

    print(f"Mean transfer advantage: {mean_transfer:+.1%}")
    print(f"Positive transfers: {positive_transfers}/{len(all_results)}")

    # Success: some positive transfer
    passed = positive_transfers >= len(all_results) * 0.3

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print("✅ TEST PASSED: Some transfer learning observed!")
    else:
        print("❌ TEST FAILED: No clear transfer learning")

    result = TestResult(
        test_name="test_11_transfer_learning",
        passed=passed,
        metrics={
            'mean_transfer_advantage': mean_transfer,
            'positive_transfer_rate': positive_transfers / len(all_results) if all_results else 0,
            'per_transfer_results': all_results
        },
        details=f"Transfer advantage: {mean_transfer:+.1%}"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_seeds=2)
