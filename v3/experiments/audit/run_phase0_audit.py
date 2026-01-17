"""
Phase 0: Pre-Execution Audit for CSE Training

This script runs all audit checks before main training:
1. Held-out test set (80/20 split)
2. Gap verification (tool > L0 for each regime)
3. Gen 50 checkpoint implementation
4. Image determinism + vision sanity
5. RAG retrieval test (free, no LLM)
6. Baseline implementations (random + individual)

Run with: python -m experiments.audit.run_phase0_audit
"""
import os
import sys
import json
import asyncio
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import numpy as np

# Add v3 to path
V3_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(V3_ROOT))

from dotenv import load_dotenv
load_dotenv(V3_ROOT / '.env')

from PIL import Image

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CHECK 1: HELD-OUT TEST SET (80/20 SPLIT)
# =============================================================================

def check_1_held_out_split() -> Dict[str, Any]:
    """
    Verify that the training script properly splits competition history
    into 80% train / 20% test for router evaluation.

    Expected: Lines 1244-1250 in run_training_v2.py implement this split.
    """
    print("\n" + "=" * 60)
    print("CHECK 1: Held-Out Test Set (80/20 Split)")
    print("=" * 60)

    training_file = V3_ROOT / 'experiments' / 'training' / 'run_training_v2.py'

    if not training_file.exists():
        return {'passed': False, 'error': 'Training file not found'}

    with open(training_file) as f:
        content = f.read()

    # Check for 80/20 split implementation
    has_split = 'split_idx = int(len(competition_history) * 0.8)' in content
    has_train = 'train_history = competition_history[:split_idx]' in content
    has_test = 'test_history = competition_history[split_idx:]' in content

    if has_split and has_train and has_test:
        print("✅ 80/20 split FOUND in training script")
        print("   - split_idx calculation: ✓")
        print("   - train_history slicing: ✓")
        print("   - test_history slicing: ✓")
        return {'passed': True, 'details': 'Split at line ~1244-1250'}
    else:
        print("❌ 80/20 split NOT FOUND - needs implementation")
        return {'passed': False, 'missing': [
            'split_idx' if not has_split else None,
            'train_history' if not has_train else None,
            'test_history' if not has_test else None
        ]}


# =============================================================================
# CHECK 2: GAP VERIFICATION (Tool vs L0)
# =============================================================================

async def check_2_gap_verification(n_tasks_per_regime: int = 10) -> Dict[str, Any]:
    """
    Verify that each tool provides significant advantage over L0.

    Expected gaps:
    - Vision (L2): L2 > L0 by 50%+ on chart tasks
    - Code (L1): L1 > L0 by 20%+ on hash/computation
    - Web (L4): L4 > L0 by 20%+ on current events
    - RAG (L3): L3 > L0 by 15%+ on document QA
    """
    print("\n" + "=" * 60)
    print("CHECK 2: Gap Verification (Tool Advantage over L0)")
    print("=" * 60)

    from experiments.training.run_training_v2 import (
        GroundTruthTaskBank,
        CompleteToolExecutor
    )

    random.seed(42)  # Reproducibility

    task_bank = GroundTruthTaskBank(use_huggingface=False)
    tool_executor = CompleteToolExecutor(use_real_rag=True)

    optimal_tools = {
        'vision': ('L2', 0.50),     # 50% gap expected
        'code_math': ('L1', 0.20),  # 20% gap expected
        'external': ('L4', 0.50),   # 50% gap expected (synthetic corpus - guaranteed high gap)
        'rag': ('L3', 0.15),        # 15% gap expected (after fix)
    }

    results = {}
    all_passed = True

    for regime, (opt_tool, expected_gap) in optimal_tools.items():
        print(f"\n--- {regime.upper()} ({opt_tool}) ---")

        opt_correct = 0
        l0_correct = 0

        for i in range(n_tasks_per_regime):
            task = task_bank.sample(regime)

            # Test with optimal tool
            res_opt = await tool_executor.execute(task, opt_tool)
            if res_opt.get('correct', False):
                opt_correct += 1

            # Test with L0 baseline
            res_l0 = await tool_executor.execute(task, 'L0')
            if res_l0.get('correct', False):
                l0_correct += 1

        opt_acc = opt_correct / n_tasks_per_regime
        l0_acc = l0_correct / n_tasks_per_regime
        gap = opt_acc - l0_acc

        # Use tolerance for floating point comparison
        passed = gap >= expected_gap - 0.001  # Small tolerance for floating point
        status = "✅" if passed else "❌"

        print(f"  {opt_tool}: {opt_correct}/{n_tasks_per_regime} = {opt_acc:.0%}")
        print(f"  L0:  {l0_correct}/{n_tasks_per_regime} = {l0_acc:.0%}")
        print(f"  Gap: {gap:+.0%} (expected >= {expected_gap:+.0%}) {status}")

        results[regime] = {
            'tool': opt_tool,
            'tool_acc': opt_acc,
            'l0_acc': l0_acc,
            'gap': gap,
            'expected_gap': expected_gap,
            'passed': passed
        }

        if not passed:
            all_passed = False

    print(f"\nTotal API calls: {tool_executor.call_count}")

    return {
        'passed': all_passed,
        'regime_results': results,
        'api_calls': tool_executor.call_count
    }


# =============================================================================
# CHECK 3: GEN 50 CHECKPOINT
# =============================================================================

def check_3_gen50_checkpoint() -> Dict[str, Any]:
    """
    Verify Gen 50 checkpoint is implemented with refined criterion:
    - Stop if n_specialists == 0 AND max_concentration < 40%

    Per Prof. Levine's feedback.
    """
    print("\n" + "=" * 60)
    print("CHECK 3: Gen 50 Checkpoint (Stopping Criterion)")
    print("=" * 60)

    training_file = V3_ROOT / 'experiments' / 'training' / 'run_training_v2.py'

    with open(training_file) as f:
        content = f.read()

    # Check for Gen 50 checkpoint
    has_gen50_check = 'gen == 49' in content or 'gen == 50' in content or 'gen + 1) == 50' in content
    has_concentration_check = 'concentration' in content.lower() or 'max_wins' in content

    if has_gen50_check:
        print("✅ Gen 50 checkpoint EXISTS")
        return {'passed': True, 'needs_refinement': not has_concentration_check}
    else:
        print("⚠️ Gen 50 checkpoint NOT FOUND - needs implementation")
        print("   Recommended implementation:")
        print("""
        if (gen + 1) == 50:
            n_specialists = sum(1 for a in population if a.specialty)
            max_conc = max(sum(a.wins.values()) / max(a.total_wins, 1) for a in population)

            if n_specialists == 0 and max_conc < 0.40:
                print("STOP: No specialization emerging after 50 generations")
                print(f"Specialists: {n_specialists}, Max concentration: {max_conc:.1%}")
                # Save checkpoint and exit early
                break
        """)
        return {'passed': False, 'needs_implementation': True}


# =============================================================================
# CHECK 4: IMAGE DETERMINISM + VISION SANITY
# =============================================================================

async def check_4_vision_sanity() -> Dict[str, Any]:
    """
    Verify:
    1. Image paths are deterministic (same seed → same images)
    2. Vision model actually sees image content (describe 3 diverse images)

    Per Prof. Fei-Fei Li's feedback.
    """
    print("\n" + "=" * 60)
    print("CHECK 4: Vision Sanity (Image Determinism + Content)")
    print("=" * 60)

    chartqa_dir = V3_ROOT / 'data' / 'images' / 'chartqa'
    tasks_file = chartqa_dir / 'tasks.json'

    if not tasks_file.exists():
        return {'passed': False, 'error': 'ChartQA tasks.json not found'}

    with open(tasks_file) as f:
        tasks = json.load(f)

    # Test 1: Determinism - same seed produces same order
    random.seed(42)
    sample_a = [random.choice(tasks) for _ in range(3)]

    random.seed(42)
    sample_b = [random.choice(tasks) for _ in range(3)]

    deterministic = sample_a == sample_b
    print(f"1. Determinism (seed 42): {'✅ PASSED' if deterministic else '❌ FAILED'}")

    # Test 2: Image content - have model describe 3 diverse images
    print("\n2. Vision Content Test (3 diverse images):")

    from google import genai
    from dotenv import load_dotenv
    load_dotenv(V3_ROOT / '.env')

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return {'passed': False, 'error': 'GEMINI_API_KEY not found'}

    client = genai.Client(api_key=api_key)

    # Select 3 diverse images
    test_images = [
        chartqa_dir / 'chart_0.png',   # Bar chart
        chartqa_dir / 'chart_16.png',  # Line chart
        chartqa_dir / 'chart_25.png',  # Pie chart
    ]

    vision_results = []
    for img_path in test_images:
        if not img_path.exists():
            print(f"   ❌ Image not found: {img_path.name}")
            vision_results.append({'image': img_path.name, 'passed': False})
            continue

        try:
            img = Image.open(img_path)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[img, "Briefly describe what type of chart/graph this is."]
            )
            description = response.text.strip()[:100]

            # Check if description mentions chart-related terms
            chart_terms = ['chart', 'graph', 'bar', 'line', 'pie', 'data', 'axis', 'value', 'percent']
            has_chart_content = any(term in description.lower() for term in chart_terms)

            status = "✅" if has_chart_content else "❌"
            print(f"   {status} {img_path.name}: {description}")

            vision_results.append({
                'image': img_path.name,
                'description': description,
                'passed': has_chart_content
            })
        except Exception as e:
            print(f"   ❌ {img_path.name}: Error - {e}")
            vision_results.append({'image': img_path.name, 'passed': False, 'error': str(e)})

    all_vision_passed = all(r['passed'] for r in vision_results)

    return {
        'passed': deterministic and all_vision_passed,
        'determinism': deterministic,
        'vision_results': vision_results
    }


# =============================================================================
# CHECK 5: RAG RETRIEVAL TEST
# =============================================================================

async def check_5_rag_retrieval() -> Dict[str, Any]:
    """
    Test RAG retrieval quality:
    5a. Index all 15 RAG contexts (free, no LLM)
    5b. Run retrieval test and compute recall@5

    Per Prof. Manning's feedback: free retrieval test before LLM calls.
    """
    print("\n" + "=" * 60)
    print("CHECK 5: RAG Retrieval Test")
    print("=" * 60)

    try:
        from tools.rigorous_rag import RigorousRAGSystem, create_natural_questions_corpus
    except ImportError as e:
        return {'passed': False, 'error': f'RAG imports failed: {e}'}

    # 5a: Index documents
    print("\n5a. Indexing documents (no LLM cost)...")
    rag = RigorousRAGSystem(corpus_name="audit_test")
    rag.initialize()

    corpus = create_natural_questions_corpus()
    n_indexed = rag.index_documents(corpus)
    print(f"   Indexed {n_indexed} documents")

    # 5b: Retrieval-only test (no generation)
    print("\n5b. Retrieval-only test...")

    test_queries = [
        ("What is the capital of France?", "Paris"),
        ("Who developed relativity?", "Einstein"),
        ("When was Python created?", "1991"),
        ("Who first climbed Mount Everest?", "Hillary"),
        ("What is the largest ocean?", "Pacific"),
    ]

    hits = 0
    for question, expected in test_queries:
        # Use retriever directly (no LLM generation)
        retriever = rag._index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(question)

        retrieved_texts = [node.text.lower() for node in nodes]
        hit = any(expected.lower() in text for text in retrieved_texts)

        status = "✅" if hit else "❌"
        print(f"   {status} Q: {question[:40]}... -> {expected}")

        if hit:
            hits += 1

    recall = hits / len(test_queries)
    passed = recall >= 0.8  # Expect at least 80% recall

    print(f"\n   Recall@5: {recall:.0%} (expected >= 80%)")
    print(f"   Status: {'✅ PASSED' if passed else '❌ FAILED'}")

    # Cleanup
    rag.clear()

    return {
        'passed': passed,
        'n_indexed': n_indexed,
        'recall_at_5': recall,
        'hits': hits,
        'total': len(test_queries)
    }


# =============================================================================
# CHECK 6: BASELINES IMPLEMENTATION
# =============================================================================

def check_6_baselines() -> Dict[str, Any]:
    """
    Verify baselines are implemented:
    A. Random baseline: No learning, random tool per task
    B. Individual baseline: Learning without competition

    Per Prof. Abbeel's feedback.
    """
    print("\n" + "=" * 60)
    print("CHECK 6: Baseline Implementations")
    print("=" * 60)

    baselines_file = V3_ROOT / 'experiments' / 'baselines' / 'run_baselines.py'

    if baselines_file.exists():
        print("✅ Baselines file EXISTS")

        with open(baselines_file) as f:
            content = f.read()

        has_random = 'RandomBaseline' in content or 'random_baseline' in content
        has_individual = 'IndividualBaseline' in content or 'individual_baseline' in content

        print(f"   Random baseline: {'✅' if has_random else '❌ MISSING'}")
        print(f"   Individual baseline: {'✅' if has_individual else '❌ MISSING'}")

        return {
            'passed': has_random and has_individual,
            'random_exists': has_random,
            'individual_exists': has_individual
        }
    else:
        print("⚠️ Baselines file NOT FOUND - needs implementation")
        print(f"   Expected path: {baselines_file}")
        return {'passed': False, 'needs_implementation': True}


# =============================================================================
# MAIN AUDIT RUNNER
# =============================================================================

async def run_full_audit():
    """Run all Phase 0 audit checks."""
    print("=" * 70)
    print("PHASE 0: PRE-EXECUTION AUDIT")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"V3 Root: {V3_ROOT}")

    results = {}

    # Check 1: Held-out split (no API)
    results['check_1_held_out'] = check_1_held_out_split()

    # Check 3: Gen 50 checkpoint (no API)
    results['check_3_gen50'] = check_3_gen50_checkpoint()

    # Check 4: Vision sanity (~3 API calls)
    results['check_4_vision'] = await check_4_vision_sanity()

    # Check 5: RAG retrieval (no LLM, just embedding)
    results['check_5_rag'] = await check_5_rag_retrieval()

    # Check 6: Baselines (no API)
    results['check_6_baselines'] = check_6_baselines()

    # Check 2: Gap verification (~80 API calls) - run last (most expensive)
    print("\n⚠️ Check 2 requires ~80 API calls. Run? [Y/n]: ", end="")
    # For automated runs, default to yes
    run_check_2 = True
    if run_check_2:
        results['check_2_gaps'] = await check_2_gap_verification(n_tasks_per_regime=10)
    else:
        results['check_2_gaps'] = {'skipped': True}

    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    all_passed = True
    for check_name, result in results.items():
        passed = result.get('passed', False)
        skipped = result.get('skipped', False)

        if skipped:
            status = "⏭️ SKIPPED"
        elif passed:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
            all_passed = False

        print(f"  {check_name}: {status}")

    print("\n" + "-" * 70)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Ready for training!")
    else:
        print("⚠️ SOME CHECKS FAILED - Review issues above before training")
    print("-" * 70)

    # Save results
    output_dir = V3_ROOT / 'results' / 'audit'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'phase0_audit_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir / 'phase0_audit_results.json'}")

    return results


if __name__ == '__main__':
    asyncio.run(run_full_audit())
