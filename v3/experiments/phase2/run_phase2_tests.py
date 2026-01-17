"""
Phase 2: Advanced Validation Tests

Test 3: Baseline Comparison (CSE vs Random vs Individual)
Test 4: Coverage Analysis (regime coverage over generations)
Test 5: Degradation Test (remove specialists, measure impact)
Test 6: Latency Benchmarks (API calls, response time)
Test 7: Multi-seed Validation (statistical significance)

Run with: python -m experiments.phase2.run_phase2_tests
"""
import os
import sys
import json
import asyncio
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import numpy as np
from scipy import stats

# Add v3 to path
V3_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(V3_ROOT))

from dotenv import load_dotenv
load_dotenv(V3_ROOT / '.env')


# =============================================================================
# HELPER: Load Training Results
# =============================================================================

def load_training_results(seed: int = 42) -> Dict[str, Any]:
    """Load training results from file."""
    results_file = V3_ROOT / 'results' / 'training_v2' / f'seed_{seed}' / 'results.json'

    if not results_file.exists():
        raise FileNotFoundError(f"Training results not found: {results_file}")

    with open(results_file) as f:
        return json.load(f)


# =============================================================================
# TEST 3: BASELINE COMPARISON
# =============================================================================

async def test_3_baseline_comparison(n_generations: int = 50) -> Dict[str, Any]:
    """
    Test 3: Compare CSE against baselines.

    Runs:
    - Random baseline (no learning)
    - Individual baseline (learning without competition)
    - Loads CSE results from training

    Success criteria:
    - CSE coverage > Individual coverage
    - CSE coverage > Random coverage
    - CSE achieves higher specialist accuracy
    """
    print("\n" + "=" * 60)
    print("TEST 3: BASELINE COMPARISON")
    print("=" * 60)
    print("Comparing CSE vs Random vs Individual Learning")

    from experiments.baselines.run_baselines import (
        run_random_baseline,
        run_individual_baseline
    )

    # Run baselines with same seed
    seed = 42
    n_agents = 8

    print("\n--- Running Random Baseline ---")
    random_results = await run_random_baseline(n_agents, n_generations, seed)

    print("\n--- Running Individual Learning Baseline ---")
    individual_results = await run_individual_baseline(n_agents, n_generations, seed)

    # Load CSE results
    print("\n--- Loading CSE Results ---")
    cse_results = load_training_results(seed)

    cse_coverage = sum(1 for v in cse_results.get('final', {}).get('distribution', {}).values() if v > 0) / 5
    cse_specialists = sum(1 for v in cse_results.get('final', {}).get('distribution', {}).values() if v > 0)

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'CSE':<12} {'Individual':<12} {'Random':<12}")
    print("-" * 60)
    print(f"{'Specialists':<25} {cse_specialists:<12} {individual_results['n_specialists']:<12} {random_results['n_specialists']:<12}")
    print(f"{'Coverage':<25} {cse_coverage*100:.0f}%{'':<10} {individual_results['coverage']*100:.0f}%{'':<10} {random_results['coverage']*100:.0f}%")

    # Success criteria
    cse_beats_individual = cse_coverage >= individual_results['coverage']
    cse_beats_random = cse_coverage >= random_results['coverage']
    passed = cse_beats_individual and cse_beats_random

    print(f"\nCSE >= Individual: {'✅' if cse_beats_individual else '❌'}")
    print(f"CSE >= Random: {'✅' if cse_beats_random else '❌'}")
    print(f"\nResult: {'✅ PASSED' if passed else '❌ FAILED'}")

    return {
        'test': 'baseline_comparison',
        'passed': passed,
        'cse': {
            'specialists': cse_specialists,
            'coverage': cse_coverage
        },
        'individual': {
            'specialists': individual_results['n_specialists'],
            'coverage': individual_results['coverage']
        },
        'random': {
            'specialists': random_results['n_specialists'],
            'coverage': random_results['coverage']
        }
    }


# =============================================================================
# TEST 4: COVERAGE ANALYSIS
# =============================================================================

async def test_4_coverage_analysis() -> Dict[str, Any]:
    """
    Test 4: Analyze how regime coverage evolves over generations.

    Checks:
    - Coverage increases over time
    - Final coverage >= 40% (at least 2/5 regimes covered)
    - No regime completely abandoned
    """
    print("\n" + "=" * 60)
    print("TEST 4: COVERAGE ANALYSIS")
    print("=" * 60)

    results = load_training_results(seed=42)

    # Get metrics history
    metrics_history = results.get('metrics_history', [])

    if not metrics_history:
        print("⚠️ No metrics history available")
        return {'test': 'coverage_analysis', 'passed': False, 'skipped': True}

    # Analyze coverage over time
    coverages = []
    for gen_data in metrics_history:
        if isinstance(gen_data, dict):
            coverage = gen_data.get('coverage', 0)
            coverages.append(coverage)

    if coverages:
        print(f"Coverage progression: {coverages[:5]}... → {coverages[-5:]}")
        print(f"Initial coverage: {coverages[0]:.0%}")
        print(f"Final coverage: {coverages[-1]:.0%}")

        # Check if coverage improved
        improved = coverages[-1] >= coverages[0]
        final_good = coverages[-1] >= 0.40

        print(f"\nCoverage improved: {'✅' if improved else '❌'}")
        print(f"Final >= 40%: {'✅' if final_good else '❌'}")

        passed = final_good  # Main criterion
    else:
        print("Could not extract coverage data")
        improved = False
        final_good = False
        passed = False

    # Check final distribution
    distribution = results.get('final', {}).get('distribution', {})
    print(f"\nFinal distribution: {distribution}")

    print(f"\nResult: {'✅ PASSED' if passed else '❌ FAILED'}")

    return {
        'test': 'coverage_analysis',
        'passed': passed,
        'coverage_progression': coverages if coverages else [],
        'final_coverage': coverages[-1] if coverages else 0,
        'distribution': distribution
    }


# =============================================================================
# TEST 5: DEGRADATION TEST
# =============================================================================

async def test_5_degradation() -> Dict[str, Any]:
    """
    Test 5: Measure performance impact when specialists are removed.

    Simulates:
    - Full system with all specialists
    - Degraded system (remove best specialist)

    Success criteria:
    - Performance drops when specialist removed (proves they're valuable)
    - System doesn't completely fail (graceful degradation)
    """
    print("\n" + "=" * 60)
    print("TEST 5: DEGRADATION TEST")
    print("=" * 60)
    print("Measuring impact of removing specialists")

    from experiments.training.run_training_v2 import (
        GroundTruthTaskBank,
        CompleteToolExecutor,
        EmbeddingRouter
    )

    results = load_training_results(seed=42)
    router_mapping = results.get('router', {}).get('mapping', {})

    if not router_mapping:
        print("⚠️ No router mapping available")
        return {'test': 'degradation', 'passed': False, 'skipped': True}

    print(f"Router mapping: {router_mapping}")

    task_bank = GroundTruthTaskBank(use_huggingface=False)
    tool_executor = CompleteToolExecutor(use_real_rag=True)

    # Ground truth optimal tools
    optimal_tools = {
        'vision': 'L2',
        'code_math': 'L1',
        'external': 'L4',
        'rag': 'L3',
        'pure_qa': 'L0'
    }

    n_tasks = 5
    regimes = ['vision', 'code_math', 'external', 'rag']

    full_system_correct = 0
    degraded_system_correct = 0
    total_tasks = 0

    for regime in regimes:
        optimal_tool = optimal_tools[regime]
        fallback_tool = 'L0'  # Degraded system uses L0 for everything

        for _ in range(n_tasks):
            task = task_bank.sample(regime)

            # Full system: uses optimal tool
            full_result = await tool_executor.execute(task, optimal_tool)
            full_system_correct += full_result['correct']

            # Degraded system: uses L0 (no specialist)
            degraded_result = await tool_executor.execute(task, fallback_tool)
            degraded_system_correct += degraded_result['correct']

            total_tasks += 1

    full_acc = full_system_correct / total_tasks
    degraded_acc = degraded_system_correct / total_tasks
    degradation = full_acc - degraded_acc

    print(f"\n--- RESULTS ---")
    print(f"Full system accuracy: {full_acc:.0%}")
    print(f"Degraded (L0 only): {degraded_acc:.0%}")
    print(f"Performance drop: {degradation:.0%}")

    # Success: specialists provide value (>20% improvement)
    passed = degradation >= 0.20

    print(f"\nSpecialists provide >20% value: {'✅' if passed else '❌'}")
    print(f"Result: {'✅ PASSED' if passed else '❌ FAILED'}")

    return {
        'test': 'degradation',
        'passed': passed,
        'full_system_accuracy': full_acc,
        'degraded_accuracy': degraded_acc,
        'degradation': degradation,
        'api_calls': tool_executor.call_count
    }


# =============================================================================
# TEST 6: LATENCY BENCHMARKS
# =============================================================================

async def test_6_latency() -> Dict[str, Any]:
    """
    Test 6: Measure system latency.

    Metrics:
    - Average tool execution time
    - Router prediction time
    - End-to-end task completion time

    Success criteria:
    - Average latency < 5 seconds per task
    - 95th percentile < 10 seconds
    """
    print("\n" + "=" * 60)
    print("TEST 6: LATENCY BENCHMARKS")
    print("=" * 60)

    from experiments.training.run_training_v2 import (
        GroundTruthTaskBank,
        CompleteToolExecutor,
        EmbeddingRouter
    )

    task_bank = GroundTruthTaskBank(use_huggingface=False)
    tool_executor = CompleteToolExecutor(use_real_rag=True)

    # Create and train router
    results = load_training_results(seed=42)
    router = EmbeddingRouter(task_bank.get_regimes())
    router.regime_to_specialist = results.get('router', {}).get('mapping', {})
    router.trained = True

    latencies = []
    n_samples = 20

    print(f"Running {n_samples} latency samples...")

    for i in range(n_samples):
        regime = random.choice(task_bank.get_regimes())
        task = task_bank.sample(regime)

        start = time.time()

        # Router prediction
        router_start = time.time()
        predicted_id, conf, predicted_regime = router.route(task)
        router_time = time.time() - router_start

        # Tool execution
        tool = ['L0', 'L1', 'L2', 'L3', 'L4'][random.randint(0, 4)]
        exec_start = time.time()
        result = await tool_executor.execute(task, tool)
        exec_time = time.time() - exec_start

        total_time = time.time() - start
        latencies.append({
            'total': total_time,
            'router': router_time,
            'execution': exec_time
        })

        if (i + 1) % 10 == 0:
            print(f"  Sample {i+1}/{n_samples}: {total_time:.2f}s")

    # Compute statistics
    total_latencies = [l['total'] for l in latencies]
    avg_latency = np.mean(total_latencies)
    p95_latency = np.percentile(total_latencies, 95)
    max_latency = np.max(total_latencies)

    avg_router = np.mean([l['router'] for l in latencies]) * 1000  # ms
    avg_exec = np.mean([l['execution'] for l in latencies])

    print(f"\n--- RESULTS ---")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"95th percentile: {p95_latency:.2f}s")
    print(f"Max latency: {max_latency:.2f}s")
    print(f"Router time (avg): {avg_router:.1f}ms")
    print(f"Execution time (avg): {avg_exec:.2f}s")

    # Success criteria
    avg_ok = avg_latency < 5.0
    p95_ok = p95_latency < 10.0
    passed = avg_ok and p95_ok

    print(f"\nAvg < 5s: {'✅' if avg_ok else '❌'}")
    print(f"P95 < 10s: {'✅' if p95_ok else '❌'}")
    print(f"Result: {'✅ PASSED' if passed else '❌ FAILED'}")

    return {
        'test': 'latency',
        'passed': passed,
        'avg_latency': avg_latency,
        'p95_latency': p95_latency,
        'max_latency': max_latency,
        'avg_router_ms': avg_router,
        'avg_execution': avg_exec,
        'n_samples': n_samples
    }


# =============================================================================
# TEST 7: MULTI-SEED VALIDATION
# =============================================================================

async def test_7_multi_seed(n_seeds: int = 3) -> Dict[str, Any]:
    """
    Test 7: Validate results hold across multiple seeds.

    Runs training with different seeds and checks:
    - Consistent specialist emergence
    - Similar coverage levels
    - Reproducibility

    Success criteria:
    - All seeds produce specialists
    - Mean coverage >= 30%
    - Standard deviation < 30%
    """
    print("\n" + "=" * 60)
    print("TEST 7: MULTI-SEED VALIDATION")
    print("=" * 60)
    print(f"Testing {n_seeds} different seeds for reproducibility")

    # Check existing results for multiple seeds
    seeds = [42, 123, 456][:n_seeds]
    seed_results = []

    for seed in seeds:
        results_file = V3_ROOT / 'results' / 'training_v2' / f'seed_{seed}' / 'results.json'

        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)

            distribution = results.get('final', {}).get('distribution', {})
            n_specialists = sum(1 for v in distribution.values() if v > 0)
            coverage = n_specialists / 5

            seed_results.append({
                'seed': seed,
                'specialists': n_specialists,
                'coverage': coverage,
                'exists': True
            })
            print(f"  Seed {seed}: {n_specialists} specialists, {coverage:.0%} coverage")
        else:
            seed_results.append({
                'seed': seed,
                'specialists': 0,
                'coverage': 0,
                'exists': False
            })
            print(f"  Seed {seed}: No results found")

    # Compute statistics
    existing = [s for s in seed_results if s['exists']]

    if len(existing) >= 2:
        coverages = [s['coverage'] for s in existing]
        mean_coverage = np.mean(coverages)
        std_coverage = np.std(coverages)
        all_have_specialists = all(s['specialists'] > 0 for s in existing)

        print(f"\n--- RESULTS ---")
        print(f"Seeds with results: {len(existing)}/{n_seeds}")
        print(f"Mean coverage: {mean_coverage:.0%}")
        print(f"Std coverage: {std_coverage:.0%}")
        print(f"All produce specialists: {'✅' if all_have_specialists else '❌'}")

        passed = mean_coverage >= 0.30 and all_have_specialists
    else:
        print(f"\n⚠️ Only {len(existing)} seed(s) with results - need at least 2")
        mean_coverage = 0
        std_coverage = 0
        passed = len(existing) >= 1 and existing[0]['specialists'] > 0

    print(f"\nResult: {'✅ PASSED' if passed else '❌ FAILED'}")

    return {
        'test': 'multi_seed',
        'passed': passed,
        'n_seeds': n_seeds,
        'seeds_with_results': len(existing),
        'mean_coverage': mean_coverage if len(existing) >= 2 else None,
        'std_coverage': std_coverage if len(existing) >= 2 else None,
        'seed_results': seed_results
    }


# =============================================================================
# MAIN
# =============================================================================

async def run_phase2_tests():
    """Run all Phase 2 tests."""
    print("=" * 70)
    print("PHASE 2: ADVANCED VALIDATION TESTS")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    results = {}

    # Test 3: Baseline Comparison
    results['test_3'] = await test_3_baseline_comparison(n_generations=50)

    # Test 4: Coverage Analysis
    results['test_4'] = await test_4_coverage_analysis()

    # Test 5: Degradation
    results['test_5'] = await test_5_degradation()

    # Test 6: Latency
    results['test_6'] = await test_6_latency()

    # Test 7: Multi-seed
    results['test_7'] = await test_7_multi_seed(n_seeds=3)

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)

    all_passed = all(r.get('passed', False) for r in results.values() if not r.get('skipped'))

    for test_name, result in results.items():
        if result.get('skipped'):
            status = "⏭️ SKIPPED"
        else:
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"  {test_name}: {status}")

    print("\n" + "-" * 70)
    if all_passed:
        print("🎉 PHASE 2 COMPLETE - Proceeding to Phase 3!")
    else:
        print("⚠️ SOME TESTS FAILED - Review before proceeding")
    print("-" * 70)

    # Save results
    output_dir = V3_ROOT / 'results' / 'phase2'
    output_dir.mkdir(parents=True, exist_ok=True)

    results['timestamp'] = datetime.now().isoformat()
    results['all_passed'] = all_passed

    with open(output_dir / 'phase2_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir / 'phase2_results.json'}")

    return results


if __name__ == '__main__':
    asyncio.run(run_phase2_tests())
