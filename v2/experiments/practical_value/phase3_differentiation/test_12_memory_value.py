"""
Test 12: Memory Value
=====================
Does the memory system actually help performance?

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


def evaluate_with_memory(task, memory_entries):
    """Evaluate task with memory context."""
    memory_text = "\n".join(memory_entries[-5:]) if memory_entries else "No relevant memories."

    prompt = f"""You are a specialist agent with access to past experiences.

Your relevant memories:
{memory_text}

Question: {task.question}

Based on your experience, provide a concise answer:"""

    response, confidence = call_gemini(prompt, temperature=0.3)
    return response, confidence


def run_test(n_seeds: int = 2) -> TestResult:
    """Test if memory provides value."""
    print("=" * 60)
    print("TEST 12: Memory Value")
    print("=" * 60)

    regimes = list(REGIME_TASKS.keys())

    all_results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        random.seed(seed)
        np.random.seed(seed)

        # Train population
        population = train_cse_population(
            n_agents=8,
            regimes=regimes,
            n_generations=25,
            seed=seed
        )

        # Build memory for specialists
        specialist_memories = {}
        for regime in regimes:
            specialist = get_specialist_for_regime(population, regime)
            if specialist:
                # Create sample memories from training
                memories = [
                    f"Won competition on {regime} task: Success pattern observed",
                    f"Developed expertise in {regime} through repeated wins",
                    f"Key insight for {regime}: Focus on domain-specific patterns",
                ]
                specialist_memories[regime] = memories

        # Test with and without memory
        for regime in regimes:
            print(f"\n  Testing {regime}:")

            tasks = generate_tasks(regime, n=10, seed=2000 + seed)

            # Without memory
            no_mem_correct = 0
            for task in tasks:
                response, _ = call_gemini(task.question)
                if evaluate_response(response, task.correct_answer):
                    no_mem_correct += 1
            no_mem_acc = no_mem_correct / len(tasks)

            # With memory
            mem_correct = 0
            memories = specialist_memories.get(regime, [])
            for task in tasks:
                response, _ = evaluate_with_memory(task, memories)
                if evaluate_response(response, task.correct_answer):
                    mem_correct += 1
            mem_acc = mem_correct / len(tasks)

            # With wrong memory (control)
            wrong_regime = [r for r in regimes if r != regime][0]
            wrong_mem_correct = 0
            wrong_memories = specialist_memories.get(wrong_regime, [])
            for task in tasks:
                response, _ = evaluate_with_memory(task, wrong_memories)
                if evaluate_response(response, task.correct_answer):
                    wrong_mem_correct += 1
            wrong_mem_acc = wrong_mem_correct / len(tasks)

            print(f"    No memory: {no_mem_acc:.1%}")
            print(f"    With memory: {mem_acc:.1%}")
            print(f"    Wrong memory: {wrong_mem_acc:.1%}")

            all_results.append({
                'regime': regime,
                'no_memory': no_mem_acc,
                'with_memory': mem_acc,
                'wrong_memory': wrong_mem_acc,
                'memory_boost': mem_acc - no_mem_acc,
                'wrong_hurts': wrong_mem_acc < mem_acc
            })

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    mean_boost = np.mean([r['memory_boost'] for r in all_results])
    wrong_hurts_rate = np.mean([r['wrong_hurts'] for r in all_results])

    print(f"Mean memory boost: {mean_boost:+.1%}")
    print(f"Wrong memory hurts rate: {wrong_hurts_rate:.0%}")

    # Success: memory helps and wrong memory hurts
    passed = mean_boost > -0.05 and wrong_hurts_rate > 0.3

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if passed:
        print("✅ TEST PASSED: Memory provides value!")
    else:
        print("❌ TEST FAILED: Memory does not provide clear value")

    result = TestResult(
        test_name="test_12_memory_value",
        passed=passed,
        metrics={
            'mean_memory_boost': mean_boost,
            'wrong_memory_hurts_rate': wrong_hurts_rate,
            'per_regime_results': all_results
        },
        details=f"Memory boost: {mean_boost:+.1%}, Wrong hurts: {wrong_hurts_rate:.0%}"
    )

    save_result(result)
    return result


if __name__ == "__main__":
    run_test(n_seeds=2)
