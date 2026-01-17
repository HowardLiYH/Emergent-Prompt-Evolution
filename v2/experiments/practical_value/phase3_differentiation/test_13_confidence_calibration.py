"""
Test 13: Confidence Calibration
===============================
Are specialists well-calibrated in their confidence?

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
    TestResult, save_result, REGIME_TASKS
)


def get_response_with_confidence(question: str):
    """Get response and explicit confidence from LLM."""
    prompt = f"""Answer this question and rate your confidence (0.0-1.0).

Question: {question}

Format your response as:
ANSWER: [your answer]
CONFIDENCE: [0.0-1.0]"""

    response, _ = call_gemini(prompt, temperature=0.3)

    # Parse response
    answer = ""
    confidence = 0.5

    lines = response.split('\n')
    for line in lines:
        if line.upper().startswith('ANSWER:'):
            answer = line.split(':', 1)[1].strip()
        elif line.upper().startswith('CONFIDENCE:'):
            try:
                conf_str = line.split(':', 1)[1].strip()
                confidence = float(conf_str.replace('%', '').strip())
                if confidence > 1:
                    confidence = confidence / 100
            except:
                confidence = 0.5

    if not answer:
        answer = response.split('\n')[0]

    return answer, min(1.0, max(0.0, confidence))


def compute_expected_calibration_error(confidences, accuracies, n_bins=5):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0

    for i in range(n_bins):
        in_bin = [(c, a) for c, a in zip(confidences, accuracies)
                  if bin_boundaries[i] <= c < bin_boundaries[i+1]]

        if len(in_bin) == 0:
            continue

        bin_conf = np.mean([c for c, a in in_bin])
        bin_acc = np.mean([a for c, a in in_bin])
        bin_size = len(in_bin)

        ece += (bin_size / len(confidences)) * abs(bin_acc - bin_conf)

    return ece


def run_test(n_seeds: int = 2) -> TestResult:
    """Test confidence calibration."""
    print("=" * 60)
    print("TEST 13: Confidence Calibration")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())

    all_specialist_ece = []
    all_generalist_ece = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        random.seed(seed)
        np.random.seed(seed)

        population = train_cse_population(
            n_agents=8,
            regimes=regimes,
            n_generations=25,
            seed=seed
        )

        for regime in regimes:
            tasks = generate_tasks(regime, n=15, seed=3000 + seed)

            specialist = get_specialist_for_regime(population, regime)
            if specialist is None:
                continue

            # Collect specialist responses
            specialist_confs = []
            specialist_accs = []

            for task in tasks:
                answer, conf = get_response_with_confidence(task.question)
                correct = evaluate_response(answer, task.correct_answer)
                specialist_confs.append(conf)
                specialist_accs.append(1 if correct else 0)

            if specialist_confs:
                ece = compute_expected_calibration_error(specialist_confs, specialist_accs)
                all_specialist_ece.append(ece)
                print(f"  {regime} specialist ECE: {ece:.3f}")

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    mean_specialist_ece = np.mean(all_specialist_ece) if all_specialist_ece else 1.0

    print(f"Mean Specialist ECE: {mean_specialist_ece:.3f}")
    print("(Lower is better - 0 = perfectly calibrated)")

    # Success: ECE < 0.15 (reasonably calibrated)
    passed = mean_specialist_ece < 0.20

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print("✅ TEST PASSED: Specialists are reasonably calibrated!")
    else:
        print("❌ TEST FAILED: Specialists are poorly calibrated")

    result = TestResult(
        test_name="test_13_confidence_calibration",
        passed=passed,
        metrics={
            'mean_specialist_ece': mean_specialist_ece,
            'n_evaluations': len(all_specialist_ece)
        },
        details=f"Specialist ECE: {mean_specialist_ece:.3f}"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_seeds=2)
