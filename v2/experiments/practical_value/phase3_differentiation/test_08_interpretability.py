"""
Test 8: Interpretability / Auditability
=======================================
Can humans understand what specialists do?

Uses REAL Gemini API - NO mock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from test_runner import (
    train_cse_population, get_specialist_for_regime, call_gemini,
    TestResult, save_result, REGIME_TASKS
)


def generate_specialist_profile(agent, regime):
    """Generate a human-readable profile of a specialist."""
    wins = agent.wins_per_regime.get(regime, 0)
    total_wins = agent.total_wins
    specialty_strength = wins / total_wins if total_wins > 0 else 0

    profile = f"""
Specialist Profile:
- Agent ID: {agent.id}
- Primary specialty: {agent.get_primary_specialty()}
- Wins in {regime}: {wins}
- Total wins: {total_wins}
- Specialty strength: {specialty_strength:.1%}
- Win distribution: {dict(agent.wins_per_regime)}
"""
    return profile


def judge_specialist_alignment(profile: str, expected_regime: str) -> float:
    """Use LLM to judge if specialist profile matches expected specialty."""
    prompt = f"""You are evaluating an AI specialist agent.

{profile}

Question: Based on this profile, what regime/task type does this agent specialize in?

Answer with ONE word from this list: math, code, reasoning, knowledge, creative

Your answer:"""

    response, _ = call_gemini(prompt, temperature=0.1)
    response = response.lower().strip()

    # Check if response matches expected
    if expected_regime.lower() in response:
        return 1.0
    return 0.0


def run_test(n_seeds: int = 3) -> TestResult:
    """Test interpretability of specialists."""
    print("=" * 60)
    print("TEST 8: Interpretability / Auditability")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())
    all_identification_scores = []

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

        for regime in regimes:
            specialist = get_specialist_for_regime(population, regime)
            if specialist is None:
                continue

            profile = generate_specialist_profile(specialist, regime)
            score = judge_specialist_alignment(profile, regime)

            print(f"  {regime}: Agent {specialist.id} - Identified correctly: {score == 1.0}")
            all_identification_scores.append(score)

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    mean_identification = np.mean(all_identification_scores)
    print(f"Identification accuracy: {mean_identification:.0%}")

    passed = mean_identification >= 0.8  # 80% identification

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print("✅ TEST PASSED: Specialists are interpretable!")
    else:
        print("❌ TEST FAILED: Specialists hard to identify")

    result = TestResult(
        test_name="test_08_interpretability",
        passed=passed,
        metrics={
            'identification_accuracy': mean_identification,
            'n_evaluations': len(all_identification_scores)
        },
        details=f"Identification accuracy: {mean_identification:.0%}"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_seeds=2)
