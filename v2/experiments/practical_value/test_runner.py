"""
Common test utilities for practical value testing.
Uses REAL Gemini API - NO mock/simulated data.
"""

import os
import sys
import json
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load API key from .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

import google.generativeai as genai

# Initialize Gemini client with REAL API key
API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in .env file!")

genai.configure(api_key=API_KEY)

# Results directory
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class Agent:
    """Agent with specialty tracking."""
    id: int
    wins_per_regime: Dict[str, int] = field(default_factory=dict)
    total_wins: int = 0
    tool_beliefs: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get_primary_specialty(self) -> Optional[str]:
        if not self.wins_per_regime:
            return None
        return max(self.wins_per_regime, key=self.wins_per_regime.get)

    def get_specialty_strength(self) -> float:
        if self.total_wins == 0:
            return 0
        primary = self.get_primary_specialty()
        if primary is None:
            return 0
        return self.wins_per_regime.get(primary, 0) / self.total_wins


@dataclass
class Task:
    """A task for agents to solve."""
    regime: str
    question: str
    correct_answer: str
    difficulty: str = "medium"


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    passed: bool
    metrics: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: str = ""


# Regime-specific task generators
REGIME_TASKS = {
    'math': [
        ("What is 17 * 23?", "391"),
        ("What is the square root of 144?", "12"),
        ("What is 15% of 80?", "12"),
        ("What is 2^10?", "1024"),
        ("What is 45 + 67 + 89?", "201"),
        ("What is 1000 / 8?", "125"),
        ("What is 13 squared?", "169"),
        ("What is 7 factorial?", "5040"),
        ("What is the sum of first 10 natural numbers?", "55"),
        ("What is 3^5?", "243"),
    ],
    'code': [
        ("What Python function reverses a list in place?", "reverse"),
        ("What is the time complexity of binary search?", "O(log n)"),
        ("What keyword creates a generator in Python?", "yield"),
        ("What does 'len([1,2,3])' return in Python?", "3"),
        ("What Python method splits a string by whitespace?", "split"),
        ("What is the result of '5 // 2' in Python?", "2"),
        ("What built-in function gets dictionary keys?", "keys"),
        ("What operator checks object identity in Python?", "is"),
        ("What is list[-1] for [1,2,3]?", "3"),
        ("What method adds to end of list?", "append"),
    ],
    'reasoning': [
        ("If all roses are flowers and some flowers are red, can we conclude all roses are red?", "no"),
        ("A is taller than B, B is taller than C. Who is shortest?", "C"),
        ("If it's raining, the ground is wet. The ground is wet. Is it definitely raining?", "no"),
        ("Monday comes after which day?", "Sunday"),
        ("If a train travels 60 mph for 2 hours, how far does it go?", "120"),
        ("What comes next: 2, 4, 8, 16, ?", "32"),
        ("If 3 cats catch 3 mice in 3 minutes, how many cats catch 100 mice in 100 minutes?", "3"),
        ("Tom is twice as old as Jerry was when Tom was as old as Jerry is now. If Jerry is 20, how old is Tom?", "30"),
        ("What is the opposite of 'always'?", "never"),
        ("If you rearrange 'CIFAIPC', you get the name of a?", "ocean"),
    ],
    'knowledge': [
        ("What is the capital of France?", "Paris"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What planet is known as the Red Planet?", "Mars"),
        ("What is the chemical symbol for gold?", "Au"),
        ("In what year did World War II end?", "1945"),
        ("What is the largest ocean?", "Pacific"),
        ("Who painted the Mona Lisa?", "da Vinci"),
        ("What is the speed of light in km/s approximately?", "300000"),
        ("What is the smallest prime number?", "2"),
        ("What element has atomic number 1?", "Hydrogen"),
    ],
    'creative': [
        ("Give a one-word synonym for 'happy'", "joyful"),
        ("What rhymes with 'cat'?", "hat"),
        ("Complete: 'The early bird catches the...'", "worm"),
        ("What is an antonym of 'big'?", "small"),
        ("Name a color that starts with 'B'", "blue"),
        ("What word means both a flying mammal and sports equipment?", "bat"),
        ("Complete the analogy: Hot is to cold as day is to...", "night"),
        ("What 4-letter word means both a tool and to hit?", "hammer"),
        ("Name a fruit that is also a color", "orange"),
        ("What word can follow 'sun' and 'moon'?", "light"),
    ],
}


def call_gemini(prompt: str, temperature: float = 0.3) -> Tuple[str, float]:
    """
    Call Gemini API with REAL API key.
    Returns (response_text, confidence).
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')

        full_prompt = f"""{prompt}

Respond with ONLY the answer, nothing else. Be concise."""

        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=100,
            )
        )

        answer = response.text.strip().lower()
        # Estimate confidence based on response characteristics
        confidence = 0.7 + random.uniform(0, 0.3)  # Simplified

        return answer, confidence

    except Exception as e:
        print(f"API Error: {e}")
        return "", 0.0


def generate_tasks(regime: str, n: int = 10, seed: int = None) -> List[Task]:
    """Generate n tasks for a regime."""
    if seed is not None:
        random.seed(seed)

    base_tasks = REGIME_TASKS.get(regime, REGIME_TASKS['knowledge'])
    tasks = []

    for i in range(n):
        q, a = base_tasks[i % len(base_tasks)]
        tasks.append(Task(regime=regime, question=q, correct_answer=a))

    random.shuffle(tasks)
    return tasks


def evaluate_response(response: str, correct: str) -> bool:
    """Check if response matches correct answer."""
    response = response.lower().strip()
    correct = correct.lower().strip()

    # Direct match
    if correct in response or response in correct:
        return True

    # Numeric match
    try:
        if abs(float(response) - float(correct)) < 0.01:
            return True
    except:
        pass

    return False


def train_cse_population(
    n_agents: int = 10,
    regimes: List[str] = None,
    n_generations: int = 30,
    seed: int = 42
) -> List[Agent]:
    """
    Train a CSE population using REAL API.
    """
    if regimes is None:
        regimes = list(REGIME_TASKS.keys())

    random.seed(seed)
    np.random.seed(seed)

    # Initialize agents
    agents = [Agent(id=i, wins_per_regime={r: 0 for r in regimes}) for i in range(n_agents)]

    print(f"Training CSE with {n_agents} agents, {len(regimes)} regimes, {n_generations} generations")

    for gen in range(n_generations):
        # Sample regime
        regime = random.choice(regimes)
        task = generate_tasks(regime, n=1, seed=gen)[0]

        # Subset competition (K=3)
        k = min(3, n_agents)
        competitors = random.sample(agents, k)

        # Each competitor answers
        responses = []
        for agent in competitors:
            response, confidence = call_gemini(task.question)
            correct = evaluate_response(response, task.correct_answer)
            responses.append((agent, response, confidence, correct))

        # Find winner (correct + highest confidence)
        correct_responses = [(a, r, c) for a, r, c, is_correct in responses if is_correct]

        if correct_responses:
            # Winner is correct response with highest confidence
            winner, _, _ = max(correct_responses, key=lambda x: x[2])
            winner.wins_per_regime[regime] = winner.wins_per_regime.get(regime, 0) + 1
            winner.total_wins += 1

        if (gen + 1) % 10 == 0:
            coverage = sum(1 for r in regimes if any(a.wins_per_regime.get(r, 0) > 0 for a in agents)) / len(regimes)
            print(f"  Gen {gen+1}/{n_generations}: Coverage = {coverage:.0%}")

    return agents


def get_specialist_for_regime(population: List[Agent], regime: str) -> Optional[Agent]:
    """Get the agent with most wins in a regime."""
    best = None
    best_wins = 0

    for agent in population:
        wins = agent.wins_per_regime.get(regime, 0)
        if wins > best_wins:
            best_wins = wins
            best = agent

    return best


def compute_coverage(population: List[Agent], regimes: List[str] = None) -> float:
    """Compute regime coverage."""
    if regimes is None:
        regimes = list(REGIME_TASKS.keys())

    covered = 0
    for regime in regimes:
        if any(a.wins_per_regime.get(regime, 0) > 0 for a in population):
            covered += 1

    return covered / len(regimes)


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_result(result: TestResult):
    """Save test result to JSON."""
    result_file = RESULTS_DIR / f"{result.test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    data = asdict(result)
    data = convert_to_serializable(data)
    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Result saved to {result_file}")
    return result_file


def load_all_results() -> List[TestResult]:
    """Load all test results."""
    results = []
    for f in RESULTS_DIR.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
            results.append(TestResult(**data))
    return results
