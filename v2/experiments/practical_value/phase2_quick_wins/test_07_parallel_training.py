"""
Test 7: Parallelizable Training
===============================
Test if CSE training scales with more compute.

Hypothesis: CSE training can be parallelized across agents,
achieving near-linear speedup with more compute.

Uses REAL Gemini API - NO mock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from test_runner import (
    generate_tasks, call_gemini, evaluate_response,
    Agent, TestResult, save_result, REGIME_TASKS
)


def train_single_round_parallel(agents, task, n_workers):
    """Run one competition round with parallel agent evaluation."""

    def evaluate_agent(agent):
        response, confidence = call_gemini(task.question)
        correct = evaluate_response(response, task.correct_answer)
        return (agent, response, confidence, correct)

    if n_workers == 1:
        # Sequential
        results = [evaluate_agent(a) for a in agents]
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(evaluate_agent, a) for a in agents]
            results = [f.result() for f in as_completed(futures)]

    return results


def run_test(n_generations: int = 20, n_agents: int = 8) -> TestResult:
    """
    Test parallel training speedup.
    """
    print("=" * 60)
    print("TEST 7: Parallelizable Training")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())
    worker_configs = [1, 2, 4, 8]
    results = {}

    for n_workers in worker_configs:
        print(f"\n--- Testing with {n_workers} workers ---")

        random.seed(42)
        np.random.seed(42)

        agents = [Agent(id=i, wins_per_regime={r: 0 for r in regimes}) for i in range(n_agents)]

        start_time = time.time()

        for gen in range(n_generations):
            regime = random.choice(regimes)
            task = generate_tasks(regime, n=1, seed=gen)[0]

            # Subset competition
            k = min(3, n_agents)
            competitors = random.sample(agents, k)

            # Parallel evaluation
            eval_results = train_single_round_parallel(competitors, task, n_workers)

            # Find winner
            correct_responses = [(a, c) for a, r, c, is_correct in eval_results if is_correct]
            if correct_responses:
                winner, _ = max(correct_responses, key=lambda x: x[1])
                winner.wins_per_regime[regime] = winner.wins_per_regime.get(regime, 0) + 1
                winner.total_wins += 1

        elapsed = time.time() - start_time

        results[n_workers] = {
            'wall_clock': elapsed,
            'generations': n_generations,
            'agents': n_agents
        }

        print(f"  Wall clock: {elapsed:.2f}s")
        print(f"  Time per generation: {elapsed/n_generations:.3f}s")

    # Compute speedups
    print("\n" + "=" * 60)
    print("SPEEDUP ANALYSIS")
    print("=" * 60)

    baseline = results[1]['wall_clock']

    for n_workers in worker_configs:
        elapsed = results[n_workers]['wall_clock']
        speedup = baseline / elapsed if elapsed > 0 else 0
        efficiency = speedup / n_workers if n_workers > 0 else 0

        results[n_workers]['speedup'] = speedup
        results[n_workers]['efficiency'] = efficiency

        print(f"{n_workers} workers: {elapsed:.2f}s, speedup={speedup:.2f}x, efficiency={efficiency:.0%}")

    # Success criteria
    speedup_8 = results.get(8, {}).get('speedup', 1)
    efficiency_8 = results.get(8, {}).get('efficiency', 0)

    passed = (
        speedup_8 > 2.0 and  # At least 2x speedup with 8 workers
        efficiency_8 > 0.25  # At least 25% efficiency
    )

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print(f"✅ TEST PASSED: Parallel training works!")
        print(f"   8 workers: {speedup_8:.1f}x speedup, {efficiency_8:.0%} efficiency")
    else:
        print(f"❌ TEST FAILED: Parallel training doesn't scale well")
        print(f"   8 workers: {speedup_8:.1f}x speedup, {efficiency_8:.0%} efficiency")

    result = TestResult(
        test_name="test_07_parallel_training",
        passed=passed,
        metrics={
            'per_worker_results': results,
            'speedup_8_workers': speedup_8,
            'efficiency_8_workers': efficiency_8
        },
        details=f"8 workers: {speedup_8:.1f}x speedup, {efficiency_8:.0%} efficiency"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_generations=15, n_agents=6)
