#!/usr/bin/env python3
"""
Master Test Orchestrator - All 19 Practical Value Tests
========================================================
Uses REAL Gemini API - NO mock/simulated data.

WARNING: Never upload API keys to public repositories!
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# Load API key from .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not found in .env")
    print("Please add your API key to .env file")
    sys.exit(1)

print("API key found (not displayed for security)")

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_phase(phase_name: str, tests: list, phase_results: dict):
    """Run all tests in a phase."""
    print(f"\n{'='*80}")
    print(f"PHASE: {phase_name}")
    print(f"{'='*80}")

    for test_name, test_func, test_args in tests:
        print(f"\n>>> Running {test_name}...")
        start = time.time()

        try:
            result = test_func(**test_args)
            elapsed = time.time() - start

            phase_results[test_name] = {
                'passed': bool(result.passed),
                'metrics': result.metrics,
                'details': result.details,
                'elapsed_seconds': elapsed
            }

            status = "✅ PASSED" if result.passed else "❌ FAILED"
            print(f"\n<<< {test_name}: {status} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start
            print(f"\n<<< {test_name}: ⚠️ ERROR: {e}")
            import traceback
            traceback.print_exc()
            phase_results[test_name] = {
                'passed': False,
                'error': str(e),
                'elapsed_seconds': elapsed
            }

    return phase_results


def generate_summary_report(all_results: dict):
    """Generate markdown summary report."""

    passed = sum(1 for r in all_results.values() if r.get('passed', False))
    total = len(all_results)

    report = f"""# Practical Value Test Results

**Generated:** {datetime.now().isoformat()}
**Tests Run:** {total}
**Tests Passed:** {passed}/{total} ({100*passed/total:.0f}%)

## Summary

| Test | Status | Key Metric |
|------|--------|------------|
"""

    for test_name, result in all_results.items():
        status = "✅ PASS" if result.get('passed') else "❌ FAIL"
        details = result.get('details', result.get('error', 'N/A'))[:50]
        report += f"| {test_name} | {status} | {details} |\n"

    report += f"""

## Detailed Results

"""

    for test_name, result in all_results.items():
        report += f"### {test_name}\n\n"
        report += f"**Status:** {'PASSED' if result.get('passed') else 'FAILED'}\n\n"
        report += f"**Details:** {result.get('details', 'N/A')}\n\n"

        if 'metrics' in result:
            report += "**Metrics:**\n```json\n"
            try:
                report += json.dumps(result['metrics'], indent=2, default=str)
            except:
                report += str(result['metrics'])
            report += "\n```\n\n"

    report += """
## Commercial Recommendation

Based on test results, the following value propositions are validated:

"""

    validated = [name for name, r in all_results.items() if r.get('passed')]
    failed = [name for name, r in all_results.items() if not r.get('passed')]

    if validated:
        report += "**Validated:**\n"
        for name in validated:
            report += f"- {name}\n"

    if failed:
        report += "\n**Not Validated (Areas for Improvement):**\n"
        for name in failed:
            report += f"- {name}\n"

    return report


def main():
    print("=" * 80)
    print("PRACTICAL VALUE TEST SUITE - ALL 19 TESTS")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print("Using REAL Gemini API - All data is genuine")

    all_results = {}

    # Phase 1: Critical (MUST run first)
    from phase1_critical.test_01_specialist_accuracy import run_test as test_01
    from phase1_critical.test_02_automatic_routing import run_test as test_02

    phase1_tests = [
        ("test_01_specialist_accuracy", test_01, {'n_seeds': 3, 'n_held_out_tasks': 10}),
        ("test_02_automatic_routing", test_02, {'n_seeds': 3, 'n_test_tasks': 15}),
    ]
    run_phase("Phase 1: Critical Foundation", phase1_tests, all_results)

    # Phase 2: Quick Wins
    from phase2_quick_wins.test_07_parallel_training import run_test as test_07
    from phase2_quick_wins.test_10_graceful_degradation import run_test as test_10
    from phase2_quick_wins.test_14_collision_coverage import run_test as test_14
    from phase2_quick_wins.test_17_inference_latency import run_test as test_17

    phase2_tests = [
        ("test_07_parallel_training", test_07, {'n_generations': 15, 'n_agents': 6}),
        ("test_10_graceful_degradation", test_10, {'n_seeds': 2}),
        ("test_14_collision_coverage", test_14, {'n_seeds': 3}),
        ("test_17_inference_latency", test_17, {'n_requests': 20}),
    ]
    run_phase("Phase 2: Quick Wins", phase2_tests, all_results)

    # Phase 3: Differentiation
    from phase3_differentiation.test_05_adaptability import run_test as test_05
    from phase3_differentiation.test_06_distribution_shift import run_test as test_06
    from phase3_differentiation.test_08_interpretability import run_test as test_08
    from phase3_differentiation.test_09_modular_updates import run_test as test_09
    from phase3_differentiation.test_12_memory_value import run_test as test_12
    from phase3_differentiation.test_13_confidence_calibration import run_test as test_13

    phase3_tests = [
        ("test_05_adaptability", test_05, {'n_seeds': 2}),
        ("test_06_distribution_shift", test_06, {'n_seeds': 2}),
        ("test_08_interpretability", test_08, {'n_seeds': 2}),
        ("test_09_modular_updates", test_09, {'n_seeds': 2}),
        ("test_12_memory_value", test_12, {'n_seeds': 2}),
        ("test_13_confidence_calibration", test_13, {'n_seeds': 2}),
    ]
    run_phase("Phase 3: Differentiation", phase3_tests, all_results)

    # Phase 4: Scale and Polish
    from phase4_scale.test_03_cost_vs_finetuning import run_test as test_03
    from phase4_scale.test_04_engineering_savings import run_test as test_04
    from phase4_scale.test_11_transfer_learning import run_test as test_11
    from phase4_scale.test_15_scaling_regimes import run_test as test_15
    from phase4_scale.test_16_low_resource import run_test as test_16
    from phase4_scale.test_18_consistency import run_test as test_18
    from phase4_scale.test_19_human_alignment import run_test as test_19

    phase4_tests = [
        ("test_03_cost_vs_finetuning", test_03, {}),
        ("test_04_engineering_savings", test_04, {}),
        ("test_11_transfer_learning", test_11, {'n_seeds': 2}),
        ("test_15_scaling_regimes", test_15, {}),
        ("test_16_low_resource", test_16, {'n_seeds': 2}),
        ("test_18_consistency", test_18, {'n_seeds': 3}),
        ("test_19_human_alignment", test_19, {'n_seeds': 2}),
    ]
    run_phase("Phase 4: Scale and Polish", phase4_tests, all_results)

    # Generate and save summary
    report = generate_summary_report(all_results)

    report_file = RESULTS_DIR / 'PRACTICAL_VALUE_REPORT.md'
    with open(report_file, 'w') as f:
        f.write(report)

    # Save raw results
    results_file = RESULTS_DIR / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Final summary
    passed = sum(1 for r in all_results.values() if r.get('passed', False))
    total = len(all_results)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {passed}/{total} ({100*passed/total:.0f}%)")

    print("\nPassed Tests:")
    for name, r in all_results.items():
        if r.get('passed'):
            print(f"  ✅ {name}")

    print("\nFailed Tests:")
    for name, r in all_results.items():
        if not r.get('passed'):
            print(f"  ❌ {name}")

    print(f"\nReport saved to: {report_file}")
    print(f"Raw results saved to: {results_file}")

    if passed >= total * 0.7:
        print("\n🎉 STRONG PRACTICAL VALUE DEMONSTRATED!")
    elif passed >= total * 0.5:
        print("\n⚠️ MODERATE PRACTICAL VALUE - Some propositions validated")
    else:
        print("\n❌ LIMITED PRACTICAL VALUE - Need to pivot strategy")

    return all_results


if __name__ == "__main__":
    main()
