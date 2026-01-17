"""
Multi-seed training for statistical validity.

Runs training with multiple seeds and aggregates results.
For publication, we need 5+ seeds to report mean ± std.

Run with: python -m experiments.run_multi_seed --seeds 123 456 789 1000
"""
import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

V3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V3_ROOT))


def run_single_seed(seed: int, generations: int = 100, agents: int = 8) -> Dict[str, Any]:
    """Run training for a single seed."""
    print(f"\n{'='*60}")
    print(f"RUNNING SEED {seed}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "experiments.training.run_training_v2",
        "--agents", str(agents),
        "--generations", str(generations),
        "--seed", str(seed)
    ]

    result = subprocess.run(
        cmd,
        cwd=str(V3_ROOT),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Seed {seed} failed:")
        print(result.stderr[-500:] if result.stderr else "No error output")
        return {'seed': seed, 'success': False, 'error': result.stderr[-200:]}

    # Load results
    results_file = V3_ROOT / 'results' / 'training_v2' / f'seed_{seed}' / 'results.json'
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)

        final = data.get('final', {})
        dist = final.get('distribution', {})
        n_specialists = sum(1 for v in dist.values() if v > 0)
        coverage = n_specialists / 5

        print(f"✅ Seed {seed} complete: {n_specialists} specialists, {coverage:.0%} coverage")

        return {
            'seed': seed,
            'success': True,
            'n_specialists': n_specialists,
            'coverage': coverage,
            'distribution': dist,
            'api_calls': data.get('total_api_calls', 0)
        }
    else:
        print(f"⚠️ Seed {seed}: Results file not found")
        return {'seed': seed, 'success': False, 'error': 'Results file not found'}


def aggregate_results(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate results across seeds."""
    successful = [r for r in results if r.get('success', False)]

    if not successful:
        return {'error': 'No successful runs'}

    coverages = [r['coverage'] for r in successful]
    specialists = [r['n_specialists'] for r in successful]

    # Count regime coverage across seeds
    regime_counts = {}
    for r in successful:
        for regime in r.get('distribution', {}).keys():
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

    return {
        'n_seeds': len(successful),
        'coverage': {
            'mean': np.mean(coverages),
            'std': np.std(coverages),
            'min': np.min(coverages),
            'max': np.max(coverages)
        },
        'specialists': {
            'mean': np.mean(specialists),
            'std': np.std(specialists),
            'min': np.min(specialists),
            'max': np.max(specialists)
        },
        'regime_frequency': regime_counts,
        'per_seed': successful
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-seed training")
    parser.add_argument('--seeds', type=int, nargs='+', default=[123, 456, 789, 1000],
                        help='Seeds to run (default: 123 456 789 1000)')
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--agents', type=int, default=8)

    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-SEED TRAINING FOR STATISTICAL VALIDITY")
    print("=" * 70)
    print(f"Seeds: {args.seeds}")
    print(f"Generations: {args.generations}")
    print(f"Agents: {args.agents}")
    print(f"Started: {datetime.now().isoformat()}")

    results = []

    for seed in args.seeds:
        result = run_single_seed(seed, args.generations, args.agents)
        results.append(result)

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)

    # Include existing seed 42
    seed_42_file = V3_ROOT / 'results' / 'training_v2' / 'seed_42' / 'results.json'
    if seed_42_file.exists():
        with open(seed_42_file) as f:
            data = json.load(f)
        final = data.get('final', {})
        dist = final.get('distribution', {})
        n_specialists = sum(1 for v in dist.values() if v > 0)
        results.insert(0, {
            'seed': 42,
            'success': True,
            'n_specialists': n_specialists,
            'coverage': n_specialists / 5,
            'distribution': dist,
            'api_calls': data.get('total_api_calls', 0)
        })

    summary = aggregate_results(results)

    print(f"\n--- SUMMARY ({summary['n_seeds']} seeds) ---")
    print(f"Coverage: {summary['coverage']['mean']:.1%} ± {summary['coverage']['std']:.1%}")
    print(f"Specialists: {summary['specialists']['mean']:.1f} ± {summary['specialists']['std']:.1f}")
    print(f"Regime frequency: {summary['regime_frequency']}")

    # Save summary
    output_dir = V3_ROOT / 'results' / 'multi_seed'
    output_dir.mkdir(parents=True, exist_ok=True)

    summary['timestamp'] = datetime.now().isoformat()
    summary['config'] = {
        'generations': args.generations,
        'agents': args.agents,
        'seeds': [42] + args.seeds
    }

    with open(output_dir / 'multi_seed_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir / 'multi_seed_summary.json'}")

    # Publication-ready stats
    print("\n" + "=" * 70)
    print("PUBLICATION-READY STATISTICS")
    print("=" * 70)
    print(f"N = {summary['n_seeds']} independent runs")
    print(f"Coverage: {summary['coverage']['mean']*100:.1f}% ± {summary['coverage']['std']*100:.1f}%")
    print(f"Range: [{summary['coverage']['min']*100:.0f}%, {summary['coverage']['max']*100:.0f}%]")


if __name__ == '__main__':
    main()
