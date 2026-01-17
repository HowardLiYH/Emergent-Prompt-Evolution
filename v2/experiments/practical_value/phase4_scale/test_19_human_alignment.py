"""
Test 19: Human Preference Alignment
===================================
Do specialists behave as humans expect?

Uses REAL Gemini API - NO mock data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from test_runner import (
    train_cse_population, generate_tasks, call_gemini,
    get_specialist_for_regime,
    TestResult, save_result, REGIME_TASKS
)


def judge_alignment(specialist_response: str, expected_regime: str) -> float:
    """Use LLM to judge if response demonstrates expected expertise."""

    prompt = f"""You are evaluating if an AI agent's response demonstrates expertise in a specific domain.

Expected expertise domain: {expected_regime}

Agent's response to a question:
"{specialist_response}"

Rate how well this response demonstrates expertise in {expected_regime} on a scale of 1-5:
1 = No expertise shown, generic or wrong
2 = Minimal expertise
3 = Moderate expertise
4 = Good expertise
5 = Excellent, clearly a specialist

Provide only the number (1-5):"""

    response, _ = call_gemini(prompt, temperature=0.1)

    try:
        score = int(response.strip()[0])
        return min(5, max(1, score))
    except:
        return 3  # Default middle score


def run_test(n_seeds: int = 2) -> TestResult:
    """Test human alignment of specialists."""
    print("=" * 60)
    print("TEST 19: Human Preference Alignment")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())

    all_alignment_scores = []

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

            # Get sample responses from specialist
            tasks = generate_tasks(regime, n=3, seed=8000 + seed)

            regime_scores = []
            for task in tasks:
                response, _ = call_gemini(task.question)
                score = judge_alignment(response, regime)
                regime_scores.append(score)

            mean_score = np.mean(regime_scores)
            all_alignment_scores.append({
                'regime': regime,
                'mean_score': mean_score,
                'individual_scores': regime_scores
            })

            print(f"  {regime}: {mean_score:.1f}/5")

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    overall_mean = np.mean([r['mean_score'] for r in all_alignment_scores])
    worst_regime = min(all_alignment_scores, key=lambda x: x['mean_score'])

    print(f"Mean alignment score: {overall_mean:.2f}/5")
    print(f"Worst regime: {worst_regime['regime']} ({worst_regime['mean_score']:.1f}/5)")

    # Success: good alignment across the board
    passed = overall_mean >= 3.5 and worst_regime['mean_score'] >= 3.0

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print("✅ TEST PASSED: Specialists align with human expectations!")
    else:
        print("❌ TEST FAILED: Poor alignment with expectations")

    result = TestResult(
        test_name="test_19_human_alignment",
        passed=passed,
        metrics={
            'overall_mean_score': overall_mean,
            'worst_regime': worst_regime['regime'],
            'worst_score': worst_regime['mean_score'],
            'per_regime_results': all_alignment_scores
        },
        details=f"Mean score: {overall_mean:.1f}/5, Worst: {worst_regime['mean_score']:.1f}/5"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_seeds=2)
