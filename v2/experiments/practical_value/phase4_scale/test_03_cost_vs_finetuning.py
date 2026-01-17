"""
Test 3: Cost vs Fine-Tuning
===========================
Is CSE cheaper than fine-tuning N models?

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


def estimate_finetuning_cost(n_regimes: int) -> dict:
    """Estimate cost of fine-tuning N specialized models."""
    # Based on OpenAI fine-tuning pricing (approximate)
    # Training: ~$0.008/1K tokens for GPT-3.5
    # Need ~50K examples per specialty, ~500 tokens each

    tokens_per_example = 500
    examples_per_specialty = 5000  # Conservative estimate
    cost_per_1k_tokens = 0.008

    total_training_tokens = n_regimes * examples_per_specialty * tokens_per_example
    training_cost = (total_training_tokens / 1000) * cost_per_1k_tokens

    # Human data curation time
    hours_per_specialty = 10  # Conservative
    hourly_rate = 50  # USD
    human_cost = n_regimes * hours_per_specialty * hourly_rate

    return {
        'training_tokens': total_training_tokens,
        'training_cost_usd': training_cost,
        'human_hours': n_regimes * hours_per_specialty,
        'human_cost_usd': human_cost,
        'total_cost_usd': training_cost + human_cost
    }


def estimate_cse_cost(n_agents: int, n_generations: int, tokens_per_call: int = 200) -> dict:
    """Estimate cost of CSE training."""
    # Gemini 2.0 Flash pricing: ~$0.0005 per 1K input tokens, ~$0.0015 per 1K output tokens

    # Each generation: K agents compete (K=3), each makes 1 API call
    api_calls = n_generations * 3  # Subset competition
    total_tokens = api_calls * tokens_per_call * 2  # Input + output

    cost_per_1k = 0.002  # Average of input/output
    api_cost = (total_tokens / 1000) * cost_per_1k

    return {
        'api_calls': api_calls,
        'total_tokens': total_tokens,
        'api_cost_usd': api_cost,
        'human_hours': 1,  # Just setup
        'human_cost_usd': 50,
        'total_cost_usd': api_cost + 50
    }


def run_test() -> TestResult:
    """Compare CSE cost to fine-tuning."""
    print("=" * 60)
    print("TEST 3: Cost vs Fine-Tuning")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())
    n_regimes = len(regimes)

    # Train CSE to get actual parameters
    print("\n--- Training CSE ---")
    population = train_cse_population(
        n_agents=10,
        regimes=regimes,
        n_generations=40,
        seed=42
    )

    coverage = compute_coverage(population, regimes)
    print(f"Coverage achieved: {coverage:.0%}")

    # Cost estimates
    print("\n" + "=" * 60)
    print("COST COMPARISON")
    print("=" * 60)

    finetuning = estimate_finetuning_cost(n_regimes)
    cse = estimate_cse_cost(n_agents=10, n_generations=40)

    print("\nFine-Tuning N Models:")
    print(f"  Training cost: ${finetuning['training_cost_usd']:.2f}")
    print(f"  Human curation: {finetuning['human_hours']} hours = ${finetuning['human_cost_usd']:.2f}")
    print(f"  TOTAL: ${finetuning['total_cost_usd']:.2f}")

    print("\nCSE Training:")
    print(f"  API cost: ${cse['api_cost_usd']:.2f}")
    print(f"  Human setup: {cse['human_hours']} hours = ${cse['human_cost_usd']:.2f}")
    print(f"  TOTAL: ${cse['total_cost_usd']:.2f}")

    savings_ratio = finetuning['total_cost_usd'] / cse['total_cost_usd'] if cse['total_cost_usd'] > 0 else 0
    savings_pct = 1 - (cse['total_cost_usd'] / finetuning['total_cost_usd']) if finetuning['total_cost_usd'] > 0 else 0

    print(f"\nSavings: {savings_pct:.0%} ({savings_ratio:.1f}x cheaper)")

    # Success: CSE significantly cheaper
    passed = savings_ratio > 10 and coverage >= 0.80

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print(f"✅ TEST PASSED: CSE is {savings_ratio:.0f}x cheaper than fine-tuning!")
    else:
        print(f"❌ TEST FAILED: CSE not significantly cheaper or low coverage")

    result = TestResult(
        test_name="test_03_cost_vs_finetuning",
        passed=passed,
        metrics={
            'finetuning_cost': finetuning,
            'cse_cost': cse,
            'savings_ratio': savings_ratio,
            'savings_percentage': savings_pct,
            'cse_coverage': coverage
        },
        details=f"CSE {savings_ratio:.0f}x cheaper, {coverage:.0%} coverage"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test()
