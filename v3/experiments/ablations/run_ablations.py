"""
V3 Ablation Studies - Professor Panel Recommendations

Tests:
1. No Fitness Sharing - Does specialization still emerge without 1/sqrt(n)?
2. No Competition - Does specialization emerge with random winner selection?
"""
import os
import sys
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add v3 to path
V3_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(V3_ROOT))

# Load environment
env_path = V3_ROOT / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key] = val

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found")
    sys.exit(1)

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)


class AblationToolExecutor:
    """Simplified tool executor for ablations."""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.call_count = 0
        self.total_tokens = 0
    
    def execute(self, tool: str, question: str) -> Dict[str, Any]:
        """Execute tool and return result."""
        import time
        start = time.time()
        
        try:
            if tool == 'L1':  # Code
                model = genai.GenerativeModel('gemini-2.5-flash', tools='code_execution')
                response = model.generate_content(f"Solve using Python code: {question}")
            elif tool == 'L4' and TAVILY_API_KEY:  # Web
                from tavily import TavilyClient
                tavily = TavilyClient(api_key=TAVILY_API_KEY)
                search_result = tavily.search(question, max_results=1)
                answer = search_result.get('results', [{}])[0].get('content', '')[:200]
                response = type('obj', (object,), {'text': answer})()
            else:  # L0, L2, L3 - use base model
                response = self.model.generate_content(question)
            
            latency = (time.time() - start) * 1000
            self.call_count += 1
            
            return {
                'answer': response.text[:500] if hasattr(response, 'text') else str(response)[:500],
                'latency_ms': latency,
                'tool': tool
            }
        except Exception as e:
            return {'answer': f'ERROR: {e}', 'latency_ms': 0, 'tool': tool}


class AblationAgent:
    """Agent for ablation studies."""
    
    TOOLS = ['L0', 'L1', 'L2', 'L4']
    
    def __init__(self, agent_id: int, seed: int = None):
        self.id = agent_id
        self.rng = np.random.default_rng(seed)
        self.specialty = None
        self.wins = {'code_math': 0, 'vision': 0, 'web': 0, 'pure_qa': 0}
        self.total_wins = 0
        
        # Thompson Sampling beliefs
        self.beliefs = {tool: {'alpha': 1.0, 'beta': 1.0} for tool in self.TOOLS}
    
    def select_tool(self, regime: str) -> str:
        """Thompson Sampling for tool selection."""
        samples = {}
        for tool, params in self.beliefs.items():
            samples[tool] = self.rng.beta(params['alpha'], params['beta'])
        return max(samples, key=samples.get)
    
    def update_beliefs(self, tool: str, success: bool):
        """Update Beta distribution."""
        if success:
            self.beliefs[tool]['alpha'] += 1
        else:
            self.beliefs[tool]['beta'] += 1
    
    def check_specialty(self):
        """Check if agent has specialized."""
        total = sum(self.wins.values())
        if total < 3:
            return
        for regime, wins in self.wins.items():
            if wins >= 3 and wins / total > 0.4:
                self.specialty = regime
                break


def generate_task(regime: str, rng) -> Dict[str, Any]:
    """Generate a task for the given regime."""
    tasks = {
        'code_math': [
            {"question": f"Calculate {rng.integers(100, 999)} * {rng.integers(100, 999)}", "type": "code"},
            {"question": f"What is the factorial of {rng.integers(5, 12)}?", "type": "code"},
            {"question": f"Find the sum of first {rng.integers(50, 200)} prime numbers", "type": "code"},
        ],
        'vision': [
            {"question": "Describe what you see in a bar chart showing sales data", "type": "vision"},
            {"question": "Analyze a pie chart showing market share distribution", "type": "vision"},
        ],
        'web': [
            {"question": "What is the current price of Bitcoin in USD?", "type": "web"},
            {"question": "What is the latest news about AI regulation?", "type": "web"},
        ],
        'pure_qa': [
            {"question": "What is the capital of France?", "type": "qa"},
            {"question": "Who wrote Romeo and Juliet?", "type": "qa"},
            {"question": "What is photosynthesis?", "type": "qa"},
        ]
    }
    
    task_list = tasks.get(regime, tasks['pure_qa'])
    task = rng.choice(task_list)
    task['regime'] = regime
    task['answer'] = ''  # Will be evaluated by response quality
    return task


def run_ablation(
    condition: str,  # 'baseline', 'no_fitness', 'no_competition'
    n_agents: int = 8,
    n_generations: int = 30,
    seed: int = 42
) -> Dict[str, Any]:
    """Run a single ablation experiment."""
    
    print(f"\n{'='*60}")
    print(f"ABLATION: {condition.upper()}")
    print(f"{'='*60}")
    print(f"Agents: {n_agents}, Generations: {n_generations}, Seed: {seed}")
    
    rng = np.random.default_rng(seed)
    executor = AblationToolExecutor()
    
    # Create agents
    agents = [AblationAgent(i, seed=seed+i) for i in range(n_agents)]
    
    regimes = ['code_math', 'vision', 'web', 'pure_qa']
    regime_weights = [0.30, 0.20, 0.25, 0.25]
    
    metrics_history = []
    tool_traces = []
    
    for gen in range(1, n_generations + 1):
        # Sample regime
        regime = rng.choice(regimes, p=regime_weights)
        task = generate_task(regime, rng)
        
        # Each agent selects tool and answers
        results = []
        for agent in agents:
            tool = agent.select_tool(regime)
            result = executor.execute(tool, task['question'])
            
            # Evaluate correctness (simplified)
            correct = len(result['answer']) > 20 and 'ERROR' not in result['answer']
            confidence = agent.beliefs[tool]['alpha'] / (agent.beliefs[tool]['alpha'] + agent.beliefs[tool]['beta'])
            
            results.append({
                'agent': agent,
                'tool': tool,
                'correct': correct,
                'confidence': confidence,
                'latency': result['latency_ms']
            })
            
            tool_traces.append({
                'gen': gen,
                'tool': tool,
                'latency_ms': result['latency_ms'],
                'correct': correct
            })
        
        # Select winner based on condition
        correct_results = [r for r in results if r['correct']]
        
        if condition == 'no_competition':
            # Random winner - no competitive pressure
            winner = rng.choice(agents) if agents else None
        elif condition == 'no_fitness':
            # Best performer wins, no fitness sharing penalty
            if correct_results:
                winner = max(correct_results, key=lambda x: x['confidence'])['agent']
            else:
                winner = None
        else:  # baseline - full CSE with fitness sharing
            if correct_results:
                # Apply fitness sharing: penalize crowded niches
                for r in correct_results:
                    n_in_niche = sum(1 for a in agents if a.specialty == regime)
                    penalty = 1.0 / np.sqrt(max(n_in_niche, 1))
                    r['adjusted_score'] = r['confidence'] * penalty
                winner = max(correct_results, key=lambda x: x['adjusted_score'])['agent']
            else:
                winner = None
        
        # Update agents
        for r in results:
            agent = r['agent']
            agent.update_beliefs(r['tool'], r['correct'])
            
            if agent == winner and r['correct']:
                agent.wins[regime] += 1
                agent.total_wins += 1
                agent.check_specialty()
        
        # Log progress
        if gen % 10 == 0:
            n_spec = sum(1 for a in agents if a.specialty)
            coverage = len(set(a.specialty for a in agents if a.specialty)) / len(regimes)
            print(f"Gen {gen:3d}: Specialists={n_spec}, Coverage={coverage*100:.0f}%")
            
            metrics_history.append({
                'generation': gen,
                'n_specialists': n_spec,
                'coverage': coverage
            })
    
    # Final results
    specialists = [a for a in agents if a.specialty]
    distribution = {}
    for a in specialists:
        distribution[a.specialty] = distribution.get(a.specialty, 0) + 1
    
    n_spec = len(specialists)
    coverage = len(set(a.specialty for a in specialists)) / len(regimes) if specialists else 0
    
    # SCI calculation
    if n_spec > 0:
        shares = [distribution.get(r, 0) / n_spec for r in regimes]
        sci = 1 - sum((s - 1/len(regimes))**2 for s in shares) * len(regimes)
    else:
        sci = 0
    
    result = {
        'condition': condition,
        'config': {
            'n_agents': n_agents,
            'n_generations': n_generations,
            'seed': seed
        },
        'final': {
            'n_specialists': n_spec,
            'coverage': coverage,
            'sci': sci,
            'distribution': distribution
        },
        'metrics_history': metrics_history,
        'api_calls': executor.call_count
    }
    
    print(f"\nFinal: Specialists={n_spec}/{n_agents}, Coverage={coverage*100:.0f}%, SCI={sci:.3f}")
    print(f"Distribution: {distribution}")
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, default='all', 
                        choices=['baseline', 'no_fitness', 'no_competition', 'all'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--agents', type=int, default=8)
    parser.add_argument('--generations', type=int, default=30)
    args = parser.parse_args()
    
    results_dir = V3_ROOT / 'results' / 'ablations'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    conditions = ['baseline', 'no_fitness', 'no_competition'] if args.condition == 'all' else [args.condition]
    
    all_results = {}
    
    for condition in conditions:
        condition_results = []
        for seed in args.seeds:
            result = run_ablation(
                condition=condition,
                n_agents=args.agents,
                n_generations=args.generations,
                seed=seed
            )
            condition_results.append(result)
        
        all_results[condition] = condition_results
        
        # Save per-condition results
        with open(results_dir / f'{condition}_results.json', 'w') as f:
            json.dump(condition_results, f, indent=2)
    
    # Summary comparison
    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    print(f"{'Condition':<20} {'Specialists':>12} {'Coverage':>10} {'SCI':>8}")
    print("-"*60)
    
    summary = {}
    for condition, results in all_results.items():
        specs = [r['final']['n_specialists'] for r in results]
        covs = [r['final']['coverage'] for r in results]
        scis = [r['final']['sci'] for r in results]
        
        summary[condition] = {
            'specialists_mean': np.mean(specs),
            'specialists_std': np.std(specs),
            'coverage_mean': np.mean(covs),
            'coverage_std': np.std(covs),
            'sci_mean': np.mean(scis),
            'sci_std': np.std(scis)
        }
        
        print(f"{condition:<20} {np.mean(specs):>8.1f}±{np.std(specs):.1f} {np.mean(covs)*100:>8.0f}% {np.mean(scis):>8.3f}")
    
    # Save summary
    with open(results_dir / 'ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()
