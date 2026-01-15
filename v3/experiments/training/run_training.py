"""
Main training script for V3 CSE.

Runs the complete competition-based training pipeline.
"""
import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path

# Add v3 to path
V3_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(V3_ROOT))

from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv(V3_ROOT / '.env')


async def run_training(
    n_agents: int = 8,
    n_generations: int = 100,
    seed: int = 42,
    output_dir: str = None
):
    """
    Run the complete CSE training pipeline.
    
    Args:
        n_agents: Number of agents in population
        n_generations: Number of training generations
        seed: Random seed
        output_dir: Directory for output files
    """
    print("=" * 60)
    print("V3 COMPETITIVE SPECIALIST ECOSYSTEM (CSE) TRAINING")
    print("=" * 60)
    print(f"Agents: {n_agents}, Generations: {n_generations}, Seed: {seed}")
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    # Set random seed
    np.random.seed(seed)
    
    # Import core components
    from core.agent import ModernSpecialist
    from core.competition import CompetitionEngine
    from core.regimes import DEFAULT_REGIME_CONFIG, get_regime_list
    from core.fitness import compute_equilibrium_distribution, compute_equilibrium_error, find_winner_with_fitness
    from mcp.server import get_mcp_server
    from safety.monitor import EmergenceMonitor
    from deploy.router import TaskRouter
    from deploy.profiles import extract_profiles
    from deploy.cache import SpecialistCache
    
    # Setup output directory
    if output_dir is None:
        output_dir = V3_ROOT / 'results' / 'training' / f'seed_{seed}'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize MCP server
    print("Initializing MCP tools...")
    mcp_server = get_mcp_server()
    available_tools = mcp_server.get_available_tools()
    print(f"  Available tools: {available_tools}")
    
    # Create agent population
    print("\nCreating agent population...")
    regimes = get_regime_list()
    population = [
        ModernSpecialist(
            agent_id=i,
            regimes=regimes,
            seed=seed + i
        )
        for i in range(n_agents)
    ]
    
    # Verify agents start identical
    print("\nVerifying agents start identical...")
    first_state = population[0].export_state()
    all_identical = True
    for agent in population[1:]:
        agent_state = agent.export_state()
        if agent_state['specialty'] != first_state['specialty']:
            all_identical = False
            break
        if agent_state['wins'] != first_state['wins']:
            all_identical = False
            break
    
    if all_identical:
        print("  ✓ All agents start identical (no pre-assigned specialties)")
    else:
        print("  ✗ WARNING: Agents are not identical at start!")
    
    # Initialize competition engine
    print("\nInitializing competition engine...")
    engine = CompetitionEngine(
        population=population,
        regime_config=DEFAULT_REGIME_CONFIG,
        k=3,  # Subset competition
        epsilon=0.1,  # Exploration rate
        seed=seed
    )
    
    # Initialize monitor
    monitor = EmergenceMonitor(n_agents=n_agents, n_regimes=len(regimes))
    
    # Task generator with varying difficulty
    def task_generator(regime: str):
        """Generate a task for a regime with appropriate difficulty."""
        import random
        
        # Each regime has different difficulty (affects base accuracy)
        regime_configs = {
            'code_math': {
                'questions': [
                    ('What is 17 * 23 + 42?', '433'),
                    ('What is 99 * 101?', '9999'),
                    ('What is 256 / 16?', '16'),
                ],
                'optimal_tool': 'L1',
                'difficulty': 0.70
            },
            'vision': {
                'questions': [
                    ('Describe the main object in this image', 'object'),
                    ('What colors are present?', 'color'),
                ],
                'optimal_tool': 'L2',
                'difficulty': 0.50
            },
            'rag': {
                'questions': [
                    ('What is machine learning?', 'subset of AI'),
                    ('Explain neural networks', 'neural network'),
                ],
                'optimal_tool': 'L3',
                'difficulty': 0.60
            },
            'web': {
                'questions': [
                    ('What is the latest news?', 'news'),
                    ('Current weather?', 'weather'),
                ],
                'optimal_tool': 'L4',
                'difficulty': 0.40
            },
            'pure_qa': {
                'questions': [
                    ('What is the capital of France?', 'Paris'),
                    ('Who wrote Romeo and Juliet?', 'Shakespeare'),
                ],
                'optimal_tool': 'L0',
                'difficulty': 0.90
            }
        }
        
        config = regime_configs.get(regime, regime_configs['pure_qa'])
        q, a = random.choice(config['questions'])
        
        return {
            'question': q,
            'answer': a,
            'regime': regime,
            'optimal_tool': config['optimal_tool'],
            'difficulty': config['difficulty']
        }
    
    # Custom competition loop with simulated tool effectiveness
    print(f"\nRunning competition training with simulated tool success...")
    print("(Tool selection via Thompson Sampling, fitness sharing enabled)")
    print("-" * 60)
    
    import random
    metrics_history = []
    
    for gen in range(n_generations):
        # Sample regime
        regime = engine.regime_sampler.sample()
        task = task_generator(regime)
        optimal_tool = task['optimal_tool']
        difficulty = task['difficulty']
        
        # Select competitors (subset K=3 with exploration)
        competitors = engine.select_competitors()
        
        # Each competitor selects tool and attempts task
        results = []
        for agent in competitors:
            # Thompson Sampling for tool selection
            tool = agent.beliefs.select(regime, available_tools)
            
            # Tool bonus: if using optimal tool, higher success probability
            if tool == optimal_tool:
                tool_bonus = 0.30
            elif tool == 'L0':
                tool_bonus = 0.0  # Base LLM only
            else:
                tool_bonus = 0.10  # Wrong tool, small bonus
            
            # Success probability
            success_prob = difficulty + tool_bonus + random.gauss(0, 0.05)
            success_prob = max(0, min(1, success_prob))
            
            # Simulate success
            correct = random.random() < success_prob
            
            # Confidence based on tool match
            if tool == optimal_tool:
                confidence = 0.7 + random.gauss(0, 0.1)
            else:
                confidence = 0.5 + random.gauss(0, 0.1)
            confidence = max(0.1, min(0.99, confidence))
            
            agent.last_tool = tool
            agent.last_regime = regime
            agent.last_confidence = confidence
            
            results.append((agent, tool, correct, confidence))
        
        # Find winner with fitness sharing
        winner = find_winner_with_fitness(results, population, regime)
        
        # Update agents
        for agent, tool, correct, confidence in results:
            if agent == winner:
                agent.update_on_win(task, regime)
            else:
                agent.update_on_loss(task, regime)
            agent.increment_generation()
        
        engine.generation = gen + 1
        
        # Log competition
        monitor.log_competition(
            generation=gen,
            regime=regime,
            winner_id=winner.id if winner else None,
            participants=[a.id for a in competitors]
        )
        
        # Periodic metrics
        if (gen + 1) % 10 == 0:
            metrics = engine.compute_metrics()
            metrics_history.append({
                'generation': gen + 1,
                **metrics
            })
            
            monitor.log_specialization_snapshot(
                generation=gen + 1,
                distribution=metrics['distribution'],
                sci=metrics['sci']
            )
            
            print(f"Gen {gen+1:4d}: SCI={metrics['sci']:.3f}, "
                  f"Coverage={metrics['coverage']:.1%}, "
                  f"Specialists={metrics['n_specialists']}, "
                  f"Total Wins={metrics['total_wins']}")
    
    print("-" * 60)
    print("Training complete!")
    
    # Final metrics
    final_metrics = engine.compute_metrics()
    print(f"\nFinal Results:")
    print(f"  SCI: {final_metrics['sci']:.3f}")
    print(f"  Coverage: {final_metrics['coverage']:.1%}")
    print(f"  Specialists: {final_metrics['n_specialists']}")
    print(f"  Distribution: {final_metrics['distribution']}")
    
    # Verify specialization emerged
    print("\nVerifying specialization emergence...")
    if final_metrics['sci'] > 0.75:
        print(f"  ✓ Strong specialization emerged (SCI={final_metrics['sci']:.3f} > 0.75)")
    elif final_metrics['sci'] > 0.5:
        print(f"  ~ Moderate specialization emerged (SCI={final_metrics['sci']:.3f})")
    else:
        print(f"  ✗ Weak specialization (SCI={final_metrics['sci']:.3f})")
    
    # Validate against Theorem 4
    print("\nValidating against Theorem 4...")
    regime_config_dict = {
        name: {
            'frequency': cfg.frequency,
            'reward': cfg.reward,
            'difficulty': cfg.difficulty
        }
        for name, cfg in DEFAULT_REGIME_CONFIG.items()
    }
    expected_dist = compute_equilibrium_distribution(regime_config_dict, n_agents)
    observed_dist = final_metrics['distribution']
    
    eq_error = compute_equilibrium_error(observed_dist, expected_dist)
    print(f"  Expected: {expected_dist}")
    print(f"  Observed: {observed_dist}")
    print(f"  Equilibrium Error: {eq_error:.1%}")
    
    if eq_error < 0.20:
        print(f"  ✓ Matches Theorem 4 predictions (<20% error)")
    else:
        print(f"  ✗ Does not match Theorem 4 (>20% error)")
    
    # Train router
    print("\nTraining deployment router...")
    router = TaskRouter(n_regimes=len(regimes))
    router.train(engine.export_history(), population)
    routing_table = router.get_routing_table()
    print(f"  Routing table: {routing_table}")
    
    # Extract profiles
    print("\nExtracting specialist profiles...")
    profiles = extract_profiles(population, engine.export_history())
    for profile in profiles:
        if profile.specialty != 'generalist':
            print(f"  Agent {profile.agent_id}: {profile.specialty} ({profile.best_tool}), "
                  f"Win rate: {profile.win_rate:.1%}")
    
    # Cache specialists
    print("\nCaching specialists...")
    cache = SpecialistCache(cache_dir=str(output_dir))
    cache.cache_population(population)
    latency = cache.benchmark_latency(n_lookups=100)
    print(f"  Cache latency: {latency.get('mean_ms', 0):.4f}ms mean")
    
    # Check for emergent patterns
    print("\nChecking for emergent patterns...")
    events = monitor.check_for_emergent_patterns(n_generations)
    summary = monitor.get_summary()
    print(f"  Events: {summary['total_events']}, "
          f"Warnings: {summary['warnings']}, "
          f"Health: {summary['health']}")
    
    # Save results
    print("\nSaving results...")
    
    results = {
        'config': {
            'n_agents': n_agents,
            'n_generations': n_generations,
            'seed': seed,
        },
        'final_metrics': final_metrics,
        'equilibrium_error': eq_error,
        'agents_identical_at_start': all_identical,
        'metrics_history': metrics_history,
        'profiles': [p.to_dict() for p in profiles],
        'routing_table': routing_table,
        'monitor_summary': summary,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save agent states
    agent_states = [a.export_state() for a in population]
    with open(output_dir / 'agent_states.json', 'w') as f:
        json.dump(agent_states, f, indent=2, default=str)
    
    # Save competition history
    with open(output_dir / 'competition_history.json', 'w') as f:
        json.dump(engine.export_history(), f, indent=2, default=str)
    
    # Save cache
    cache.save(str(output_dir / 'specialist_cache.json'))
    
    print(f"\nResults saved to: {output_dir}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run V3 CSE Training')
    parser.add_argument('--agents', type=int, default=8, help='Number of agents')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Run training
    asyncio.run(run_training(
        n_agents=args.agents,
        n_generations=args.generations,
        seed=args.seed,
        output_dir=args.output
    ))


if __name__ == '__main__':
    main()
