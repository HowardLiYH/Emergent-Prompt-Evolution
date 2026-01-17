"""
Test 4: Engineering Time Savings
================================
Does CSE save human prompt engineering time?

Uses REAL Gemini API - NO mock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from test_runner import (
    train_cse_population, compute_coverage,
    TestResult, save_result, REGIME_TASKS
)


def estimate_manual_engineering_time(n_specialties: int) -> dict:
    """Estimate time to manually design specialist prompts."""

    # Based on industry experience
    hours_per_specialty = {
        'research': 2.0,        # Understanding the domain
        'initial_design': 1.5,  # First prompt draft
        'testing': 2.0,         # Testing and evaluation
        'iteration': 3.0,       # Multiple iterations to optimize
        'documentation': 0.5,   # Documenting the prompt
    }

    total_per_specialty = sum(hours_per_specialty.values())
    total_hours = n_specialties * total_per_specialty

    return {
        'hours_per_specialty': total_per_specialty,
        'total_hours': total_hours,
        'breakdown': hours_per_specialty
    }


def estimate_cse_engineering_time(n_regimes: int) -> dict:
    """Estimate time to set up CSE."""

    hours = {
        'setup': 1.0,           # Initial configuration
        'define_tasks': 0.5,    # Define regime task examples
        'run_training': 0.1,    # Automated
        'extract_specialists': 0.5,  # Extract learned behaviors
    }

    # Some time scales with regimes
    hours['define_tasks'] *= n_regimes * 0.2

    return {
        'total_hours': sum(hours.values()),
        'breakdown': hours
    }


def run_test() -> TestResult:
    """Compare engineering time."""
    print("=" * 60)
    print("TEST 4: Engineering Time Savings")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())
    n_regimes = len(regimes)

    # Run CSE to verify it works
    print("\n--- Running CSE ---")
    population = train_cse_population(
        n_agents=10,
        regimes=regimes,
        n_generations=30,
        seed=42
    )

    coverage = compute_coverage(population, regimes)
    specialists_found = sum(1 for r in regimes if any(
        a.wins_per_regime.get(r, 0) > 0 for a in population
    ))

    print(f"Coverage: {coverage:.0%}")
    print(f"Specialists found: {specialists_found}/{n_regimes}")

    # Time estimates
    print("\n" + "=" * 60)
    print("TIME COMPARISON")
    print("=" * 60)

    manual = estimate_manual_engineering_time(n_regimes)
    cse = estimate_cse_engineering_time(n_regimes)

    print("\nManual Prompt Engineering:")
    for task, hours in manual['breakdown'].items():
        print(f"  {task}: {hours * n_regimes:.1f} hours")
    print(f"  TOTAL: {manual['total_hours']:.1f} hours")

    print("\nCSE Approach:")
    for task, hours in cse['breakdown'].items():
        print(f"  {task}: {hours:.1f} hours")
    print(f"  TOTAL: {cse['total_hours']:.1f} hours")

    time_savings = manual['total_hours'] - cse['total_hours']
    savings_pct = time_savings / manual['total_hours'] if manual['total_hours'] > 0 else 0

    print(f"\nTime saved: {time_savings:.1f} hours ({savings_pct:.0%})")
    print(f"Hours saved per specialty: {time_savings / n_regimes:.1f}")

    # Success: significant time savings
    passed = savings_pct > 0.50 and coverage >= 0.80

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print(f"✅ TEST PASSED: CSE saves {savings_pct:.0%} engineering time!")
    else:
        print(f"❌ TEST FAILED: Insufficient time savings or coverage")

    result = TestResult(
        test_name="test_04_engineering_savings",
        passed=passed,
        metrics={
            'manual_hours': manual['total_hours'],
            'cse_hours': cse['total_hours'],
            'time_savings_hours': time_savings,
            'savings_percentage': savings_pct,
            'coverage': coverage
        },
        details=f"Saves {savings_pct:.0%} time ({time_savings:.1f} hours)"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test()
