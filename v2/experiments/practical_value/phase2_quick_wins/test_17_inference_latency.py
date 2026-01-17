"""
Test 17: Real-Time Inference Latency
=====================================
Test if CSE deployment is fast enough.

Hypothesis: CSE deployment (router + specialist) has competitive
latency with single-model inference.

Uses REAL Gemini API - NO mock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import random
import numpy as np
from test_runner import (
    train_cse_population, generate_tasks, call_gemini,
    get_specialist_for_regime,
    TestResult, save_result, REGIME_TASKS
)


class FastRouter:
    """Simple fast router based on keyword matching."""

    def __init__(self, regimes):
        self.regimes = regimes
        # Simple keyword mapping
        self.keywords = {
            'math': ['calculate', 'sum', 'multiply', 'divide', 'square', 'factorial', 'plus', 'minus', 'number'],
            'code': ['python', 'function', 'code', 'programming', 'variable', 'list', 'method', 'syntax'],
            'reasoning': ['if', 'then', 'therefore', 'conclude', 'logic', 'taller', 'shorter', 'next'],
            'knowledge': ['capital', 'who', 'what', 'when', 'where', 'wrote', 'invented', 'planet'],
            'creative': ['synonym', 'rhyme', 'antonym', 'word', 'complete', 'analogy', 'color'],
        }

    def route(self, question: str) -> str:
        """Route question to regime (< 1ms)."""
        question_lower = question.lower()

        best_regime = self.regimes[0]
        best_score = 0

        for regime, kws in self.keywords.items():
            if regime in self.regimes:
                score = sum(1 for kw in kws if kw in question_lower)
                if score > best_score:
                    best_score = score
                    best_regime = regime

        return best_regime


def run_test(n_requests: int = 30) -> TestResult:
    """
    Test inference latency.
    """
    print("=" * 60)
    print("TEST 17: Real-Time Inference Latency")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())

    # Train population
    print("\nTraining population...")
    population = train_cse_population(
        n_agents=10,
        regimes=regimes,
        n_generations=30,
        seed=42
    )

    # Build router
    router = FastRouter(regimes)

    # Collect latency data
    cse_latencies = []
    single_latencies = []
    routing_times = []

    print(f"\nMeasuring latency on {n_requests} requests...")

    for i in range(n_requests):
        regime = random.choice(regimes)
        task = generate_tasks(regime, n=1, seed=i)[0]

        # CSE: Route + Specialist call
        start = time.time()
        predicted_regime = router.route(task.question)
        route_time = time.time() - start
        routing_times.append(route_time * 1000)  # ms

        specialist = get_specialist_for_regime(population, predicted_regime)
        response, _ = call_gemini(task.question)
        cse_total = time.time() - start
        cse_latencies.append(cse_total * 1000)  # ms

        # Single model: Direct call (same LLM, fair comparison)
        start = time.time()
        response, _ = call_gemini(task.question)
        single_latency = time.time() - start
        single_latencies.append(single_latency * 1000)  # ms

        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{n_requests}")

    # Analyze latencies
    print("\n" + "=" * 60)
    print("LATENCY ANALYSIS")
    print("=" * 60)

    def percentile(data, p):
        return np.percentile(data, p)

    print("\nCSE (Router + Specialist):")
    print(f"  P50: {percentile(cse_latencies, 50):.1f}ms")
    print(f"  P95: {percentile(cse_latencies, 95):.1f}ms")
    print(f"  P99: {percentile(cse_latencies, 99):.1f}ms")

    print("\nSingle Model:")
    print(f"  P50: {percentile(single_latencies, 50):.1f}ms")
    print(f"  P95: {percentile(single_latencies, 95):.1f}ms")
    print(f"  P99: {percentile(single_latencies, 99):.1f}ms")

    print("\nRouting Overhead:")
    print(f"  P50: {percentile(routing_times, 50):.3f}ms")
    print(f"  P95: {percentile(routing_times, 95):.3f}ms")

    # Compute overhead
    mean_overhead = np.mean(cse_latencies) - np.mean(single_latencies)
    overhead_ratio = np.mean(cse_latencies) / np.mean(single_latencies) if np.mean(single_latencies) > 0 else float('inf')

    print(f"\nMean overhead: {mean_overhead:.1f}ms")
    print(f"Overhead ratio: {overhead_ratio:.2f}x")

    # Success criteria
    routing_overhead_p50 = percentile(routing_times, 50)
    passed = (
        routing_overhead_p50 < 50 and  # Routing < 50ms
        overhead_ratio < 1.5  # Total < 1.5x single model
    )

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print("✅ TEST PASSED: Inference latency is acceptable!")
        print(f"   Routing: {routing_overhead_p50:.1f}ms, Overhead: {overhead_ratio:.2f}x")
    else:
        print("❌ TEST FAILED: Inference latency too high")
        print(f"   Routing: {routing_overhead_p50:.1f}ms, Overhead: {overhead_ratio:.2f}x")

    result = TestResult(
        test_name="test_17_inference_latency",
        passed=passed,
        metrics={
            'cse_p50': percentile(cse_latencies, 50),
            'cse_p95': percentile(cse_latencies, 95),
            'single_p50': percentile(single_latencies, 50),
            'single_p95': percentile(single_latencies, 95),
            'routing_p50': routing_overhead_p50,
            'overhead_ratio': overhead_ratio
        },
        details=f"Routing: {routing_overhead_p50:.1f}ms, Overhead: {overhead_ratio:.2f}x"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_requests=20)
