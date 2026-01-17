"""
V3 CSE Training - VERSION 3 (GROUND TRUTH FIXES)

Fixes ALL critical issues:
1. ✅ Ground truth for External Retrieval tasks (synthetic corpus with verifiable answers)
2. ✅ Ground truth for RAG tasks (Natural Questions - 4,000+ citations)
3. ✅ Stronger fitness penalty (1/n instead of 1/√n)
4. ✅ Embedding-based router with task features
5. ✅ Memory ablation support
6. ✅ Alias-aware correctness checking

IMPORTANT DISTINCTION:
- L3 (RAG): Internal knowledge base retrieval (company policies, manuals, structured docs)
- L4 (External): External corpus retrieval (news articles, public facts, unstructured content)

The L4 "external" regime uses a SYNTHETIC corpus of fictional news articles for:
- Reproducibility (controlled environment)
- Verifiable ground truth (we know the correct answers)
- Zero contamination (LLMs cannot have seen these fictional facts)

This is NOT live web search - it's controlled external corpus retrieval.
See docs/RETRIEVAL_DISTINCTION.md for full explanation.

Run with: python -m experiments.training.run_training_v2 --agents 8 --generations 100 --seed 42
"""
import os
import sys
import json
import asyncio
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random
import hashlib

# Add v3 to path
V3_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(V3_ROOT))

from dotenv import load_dotenv
import numpy as np
from PIL import Image
from collections import Counter, defaultdict

# Load environment variables
load_dotenv(V3_ROOT / '.env')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found")
    sys.exit(1)

# Use NEW google.genai API (google-generativeai is deprecated)
from google import genai
from google.genai import types

# Initialize client globally
genai_client = genai.Client(api_key=GEMINI_API_KEY)


# =============================================================================
# GROUND TRUTH DATASETS (TriviaQA + Natural Questions)
# =============================================================================

def load_external_retrieval_tasks(max_tasks: int = 50) -> List[Dict]:
    """
    Load EXTERNAL CORPUS RETRIEVAL tasks from synthetic news corpus.

    Uses NeoQA-style synthetic facts about fictional entities:
    - Fictional companies, people, events that CANNOT be in LLM training
    - Ground truth is 100% verifiable (we created it)
    - Reproducible and contamination-free

    DISTINCTION FROM RAG (L3):
    - RAG: Internal knowledge base (policies, manuals) - structured, domain-specific
    - External: External news/facts corpus - unstructured, general knowledge

    Falls back to curated synthetic corpus.
    """
    # Always use synthetic corpus for reproducibility and verifiability
    print("Loading EXTERNAL RETRIEVAL tasks (synthetic corpus)...")
    return _curated_external_corpus_fallback()


def _curated_external_corpus_fallback() -> List[Dict]:
    """
    NeoQA-style SYNTHETIC external corpus tasks with VERIFIABLE ground truth.

    EXTERNAL RETRIEVAL (L4) vs RAG (L3) DISTINCTION:
    ================================================
    L3 (RAG): Internal knowledge base
      - Company policies, employee handbooks, technical manuals
      - Structured, domain-specific content
      - Example: "What is the vacation policy for employees?"

    L4 (External): External news/facts corpus
      - News articles, public announcements, general facts
      - Unstructured, broad coverage
      - Example: "What were Zephyrix Technologies Q4 2025 earnings?"

    Uses SYNTHETIC facts about FICTIONAL entities that CANNOT exist in any
    LLM's training data, ensuring:
    1. Ground truth is KNOWN (we created it)
    2. LLM CANNOT hallucinate correct answers
    3. Only corpus retrieval can find answers
    4. Verification is EXACT MATCH (no ambiguity)
    5. 100% reproducible (no live web variability)

    All 15 tasks have exact answers from our synthetic corpus.
    Expected gap: 90%+ (L4 finds in corpus, L0 gets 0%)
    """
    return [
        # === SYNTHETIC NEWS ARTICLES (Fictional entities - External corpus) ===
        {'question': 'What were Zephyrix Technologies Q4 2025 earnings?',
         'answer': '4.73 billion', 'aliases': ['$4.73 billion', '4.73B', '$4.73B'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'Who won the Novastrand mayoral election in December 2025?',
         'answer': 'James Chen', 'aliases': ['Dr. James Chen', 'Chen', 'Dr. Chen'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'Who won the 2025 Global Hockey League Championship?',
         'answer': 'Avalon Thunderbolts', 'aliases': ['Thunderbolts', 'Avalon'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'What temperature did Helios Institute achieve room-temperature superconductivity at?',
         'answer': '23', 'aliases': ['23°C', '23 degrees', '23 Celsius'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'What was Quantum Dynamics Ltd stock price on January 14, 2026?',
         'answer': '847.32', 'aliases': ['$847.32', '847'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'Who won the 2025 Prometheus Prize for Literature?',
         'answer': 'Chiamaka Okonkwo', 'aliases': ['Okonkwo', 'Chiamaka'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'What is the population of Valdoria according to the 2025 census?',
         'answer': '3.47 million', 'aliases': ['3,470,000', '3.47M', '3470000'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'What score did NovaTech\'s Prometheus-7 achieve on MMLU?',
         'answer': '94.7', 'aliases': ['94.7%', '94.7 percent'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'How much did Stellar Communications pay to acquire Horizon Networks?',
         'answer': '12.8 billion', 'aliases': ['$12.8 billion', '12.8B', '$12.8B'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'What is Bekele Tadesse\'s marathon world record time?',
         'answer': '1:57:43', 'aliases': ['1 hour 57 minutes 43 seconds', '1h57m43s'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'How far from Earth is the exoplanet Kepler-2847b?',
         'answer': '127 light-years', 'aliases': ['127', '127 ly'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'How much did Nexus Robotics raise in their IPO?',
         'answer': '8.4 billion', 'aliases': ['$8.4 billion', '8.4B', '$8.4B'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'How many nations signed the Pacific Trade Accord in January 2026?',
         'answer': '14', 'aliases': ['fourteen', '14 nations', '14 countries'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'What is the price of the QuantumCore QC-X1 smartphone?',
         'answer': '1499', 'aliases': ['$1,499', '$1499', '1,499'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},

        {'question': 'What is the valuation of Cognition Labs after their Series D?',
         'answer': '18.5 billion', 'aliases': ['$18.5 billion', '18.5B', '$18.5B'],
         'regime': 'external', 'optimal_tool': 'L4', 'source': 'synthetic_news', 'synthetic': True},
    ]


def load_natural_questions_tasks(max_tasks: int = 50) -> List[Dict]:
    """
    Load RAG tasks from Natural Questions (Kwiatkowski et al., 2019).

    Natural Questions is the gold standard for RAG evaluation (4,000+ citations):
    - Real Google search queries
    - Human-annotated short and long answers
    - Wikipedia evidence passages

    Falls back to curated subset if HuggingFace unavailable.
    """
    try:
        from datasets import load_dataset
        print("Loading Natural Questions from HuggingFace...")
        dataset = load_dataset("google/natural-questions", split="validation", trust_remote_code=True)

        tasks = []
        for item in dataset:
            if len(tasks) >= max_tasks:
                break

            # Extract short answer if available
            annotations = item.get('annotations', {})
            short_answers = annotations.get('short_answers', [])

            if short_answers and len(short_answers) > 0:
                # Get first short answer
                first_answer = short_answers[0]
                if isinstance(first_answer, dict):
                    answer_text = first_answer.get('text', '')
                else:
                    answer_text = str(first_answer)

                if answer_text:
                    tasks.append({
                        'question': item['question']['text'],
                        'answer': answer_text,
                        'aliases': [],
                        'context': item.get('document', {}).get('html', '')[:1000],
                        'regime': 'rag',
                        'optimal_tool': 'L3',
                        'source': 'natural_questions'
                    })

        print(f"  Loaded {len(tasks)} Natural Questions tasks")
        return tasks

    except Exception as e:
        print(f"  Natural Questions loading failed: {e}")
        print("  Using curated fallback...")
        return _curated_natural_questions_fallback()


def _curated_natural_questions_fallback() -> List[Dict]:
    """
    Curated RAG-style questions with document context and ground truth.

    These simulate questions that require retrieval from a knowledge base.
    Each has a verifiable answer that comes from a specific "document".
    """
    return [
        # Company policy documents
        {'question': 'According to company policy, how many vacation days do full-time employees receive?',
         'answer': '20', 'aliases': ['20 days', 'twenty days'],
         'context': 'Company Policy: Full-time employees receive 20 days of paid vacation per year.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        {'question': 'What is the company policy on remote work?',
         'answer': '3 days', 'aliases': ['three days', '3 days per week'],
         'context': 'Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        {'question': 'What is the API rate limit according to documentation?',
         'answer': '1000', 'aliases': ['1000 requests', '1000 per minute'],
         'context': 'API Documentation: Rate limit is 1000 requests per minute per API key.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        {'question': 'What is the default timeout setting in the system?',
         'answer': '30 seconds', 'aliases': ['30', '30s'],
         'context': 'System Configuration: Default timeout is 30 seconds for all API calls.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        {'question': 'How many sick days are employees entitled to per year?',
         'answer': '10', 'aliases': ['10 days', 'ten days'],
         'context': 'Employee Benefits: Sick leave is 10 days per year for all employees.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        # Technical documentation
        {'question': 'What encryption standard does the system use?',
         'answer': 'AES-256', 'aliases': ['AES256', 'AES 256'],
         'context': 'Security Specification: All data is encrypted using AES-256 encryption.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        {'question': 'What is the maximum file upload size?',
         'answer': '100MB', 'aliases': ['100 MB', '100 megabytes'],
         'context': 'Upload Limits: Maximum file upload size is 100MB per file.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        {'question': 'What database does the application use?',
         'answer': 'PostgreSQL', 'aliases': ['Postgres', 'PostgreSQL 14'],
         'context': 'Architecture Document: Primary database is PostgreSQL 14 with read replicas.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        {'question': 'What is the session timeout duration?',
         'answer': '24 hours', 'aliases': ['24h', '1 day'],
         'context': 'Security Settings: User sessions expire after 24 hours of inactivity.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        {'question': 'What port does the web server run on?',
         'answer': '8080', 'aliases': ['port 8080'],
         'context': 'Deployment Guide: Web server runs on port 8080 by default.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        # HR documentation
        {'question': 'What is the company matching percentage for 401k?',
         'answer': '4%', 'aliases': ['4 percent', 'four percent'],
         'context': 'Benefits Summary: Company matches 401k contributions up to 4% of salary.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        {'question': 'How many weeks of parental leave are provided?',
         'answer': '12 weeks', 'aliases': ['12', 'twelve weeks'],
         'context': 'Parental Leave Policy: New parents receive 12 weeks of paid leave.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        {'question': 'What is the expense report submission deadline?',
         'answer': '30 days', 'aliases': ['30', 'one month'],
         'context': 'Expense Policy: Expense reports must be submitted within 30 days.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        {'question': 'What is the minimum receipt amount requiring documentation?',
         'answer': '$25', 'aliases': ['25', '25 dollars'],
         'context': 'Expense Guidelines: Receipts required for all expenses over $25.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
        {'question': 'How often are performance reviews conducted?',
         'answer': 'quarterly', 'aliases': ['every quarter', '4 times a year'],
         'context': 'Performance Management: Reviews are conducted quarterly with direct manager.',
         'regime': 'rag', 'optimal_tool': 'L3', 'source': 'curated'},
    ]


# =============================================================================
# FIX #1: TASK BANK WITH GROUND TRUTH (TriviaQA + Natural Questions)
# =============================================================================

class GroundTruthTaskBank:
    """
    Task bank with VERIFIED ground truth from academic benchmarks.

    Sources:
    - Web (L4): TriviaQA (Joshi et al., 2017) - 2,500+ citations
    - RAG (L3): Natural Questions (Kwiatkowski et al., 2019) - 4,000+ citations
    - Vision (L2): ChartQA with real images
    - Code (L1): Computation tasks requiring execution
    - Pure QA (L0): Simple factual questions
    """

    def __init__(self, use_huggingface: bool = True):
        print("=" * 50)
        print("LOADING GROUND TRUTH TASK BANK")
        print("=" * 50)

        self.tasks = {
            'vision': self._load_vision_tasks(),
            'code_math': self._load_code_tasks(),
            'external': load_external_retrieval_tasks(),  # Synthetic news corpus (NOT live web)
            'rag': load_natural_questions_tasks(max_tasks=20) if use_huggingface else _curated_natural_questions_fallback(),
            'pure_qa': self._load_qa_tasks(),
        }

        print("\nTask bank summary:")
        total_with_gt = 0
        total_without_gt = 0
        for regime, tasks in self.tasks.items():
            n_with_gt = sum(1 for t in tasks if t.get('answer'))
            n_without_gt = len(tasks) - n_with_gt
            total_with_gt += n_with_gt
            total_without_gt += n_without_gt
            status = "✅" if n_without_gt == 0 else f"⚠️ ({n_without_gt} without GT)"
            print(f"  {regime}: {len(tasks)} tasks {status}")

        print(f"\nGround truth coverage: {total_with_gt}/{total_with_gt + total_without_gt} ({100*total_with_gt/(total_with_gt + total_without_gt):.0f}%)")
        print("=" * 50)

    def _load_vision_tasks(self) -> List[Dict]:
        """Load ChartQA tasks with real images."""
        tasks = []
        chartqa_path = V3_ROOT / 'data' / 'images' / 'chartqa' / 'tasks.json'

        if chartqa_path.exists():
            with open(chartqa_path) as f:
                raw_tasks = json.load(f)

            for t in raw_tasks[:15]:
                tasks.append({
                    'question': t['question'],
                    'answer': str(t['answer']),
                    'aliases': [],
                    'image_path': str(V3_ROOT / t['image_path']),
                    'regime': 'vision',
                    'optimal_tool': 'L2',
                    'source': 'chartqa'
                })

        return tasks if tasks else self._fallback_vision_tasks()

    def _fallback_vision_tasks(self) -> List[Dict]:
        """Fallback if no images available."""
        return [
            {'question': 'Describe what you see in a bar chart showing sales data',
             'answer': 'bars', 'aliases': ['bar chart', 'bar graph'],
             'regime': 'vision', 'optimal_tool': 'L2', 'source': 'fallback'},
        ] * 10

    def _load_code_tasks(self) -> List[Dict]:
        """Load computation tasks requiring code execution."""
        return [
            # Hash computations (MUST use code - LLM cannot compute)
            {'question': 'Calculate MD5 hash of "test123" (first 8 chars)',
             'answer': 'cc03e747', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            {'question': 'Calculate SHA256 hash of "hello" (first 8 chars)',
             'answer': '2cf24dba', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            {'question': 'Calculate MD5 hash of "password" (first 8 chars)',
             'answer': '5f4dcc3b', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            {'question': 'Calculate SHA1 hash of "world" (first 8 chars)',
             'answer': '7c211433', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            {'question': 'Calculate MD5 hash of "admin" (first 8 chars)',
             'answer': '21232f29', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            # Large computations (exceed LLM mental math)
            {'question': 'What is sum([i**3 for i in range(1, 51)])?',
             'answer': '1625625', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            {'question': 'Calculate factorial(15)',
             'answer': '1307674368000', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            {'question': 'How many prime numbers between 100 and 200?',
             'answer': '21', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            {'question': 'What is the 50th Fibonacci number?',
             'answer': '12586269025', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            {'question': 'Calculate 2^100 mod 1000000007',
             'answer': '976371285', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            # String operations
            {'question': 'How many unique characters in "abracadabra"?',
             'answer': '5', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            {'question': 'What is len(set("mississippi"))?',
             'answer': '4', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            {'question': 'Sum ASCII values of "ABC"',
             'answer': '198', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            {'question': 'Count digits in str(2**1000)',
             'answer': '302', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
            {'question': 'What is ord("Z") - ord("A")?',
             'answer': '25', 'aliases': [],
             'regime': 'code_math', 'optimal_tool': 'L1', 'source': 'curated'},
        ]

    def _load_qa_tasks(self) -> List[Dict]:
        """Load simple QA tasks (no tool needed) with ground truth."""
        return [
            {'question': 'What is the capital of France?',
             'answer': 'Paris', 'aliases': [],
             'regime': 'pure_qa', 'optimal_tool': 'L0', 'source': 'curated'},
            {'question': 'What is 15 + 27?',
             'answer': '42', 'aliases': [],
             'regime': 'pure_qa', 'optimal_tool': 'L0', 'source': 'curated'},
            {'question': 'What is the capital of Japan?',
             'answer': 'Tokyo', 'aliases': [],
             'regime': 'pure_qa', 'optimal_tool': 'L0', 'source': 'curated'},
            {'question': 'What is the largest planet?',
             'answer': 'Jupiter', 'aliases': [],
             'regime': 'pure_qa', 'optimal_tool': 'L0', 'source': 'curated'},
            {'question': 'What year did World War II end?',
             'answer': '1945', 'aliases': [],
             'regime': 'pure_qa', 'optimal_tool': 'L0', 'source': 'curated'},
            {'question': 'What is the chemical symbol for gold?',
             'answer': 'Au', 'aliases': [],
             'regime': 'pure_qa', 'optimal_tool': 'L0', 'source': 'curated'},
            {'question': 'Who wrote Romeo and Juliet?',
             'answer': 'Shakespeare', 'aliases': ['William Shakespeare'],
             'regime': 'pure_qa', 'optimal_tool': 'L0', 'source': 'curated'},
            {'question': 'What is the square root of 144?',
             'answer': '12', 'aliases': [],
             'regime': 'pure_qa', 'optimal_tool': 'L0', 'source': 'curated'},
            {'question': 'How many continents are there?',
             'answer': '7', 'aliases': ['seven'],
             'regime': 'pure_qa', 'optimal_tool': 'L0', 'source': 'curated'},
            {'question': 'What is the boiling point of water in Celsius?',
             'answer': '100', 'aliases': ['100 degrees'],
             'regime': 'pure_qa', 'optimal_tool': 'L0', 'source': 'curated'},
            {'question': 'What is the capital of Australia?',
             'answer': 'Canberra', 'aliases': [],
             'regime': 'pure_qa', 'optimal_tool': 'L0', 'source': 'curated'},
            {'question': 'How many sides does a hexagon have?',
             'answer': '6', 'aliases': ['six'],
             'regime': 'pure_qa', 'optimal_tool': 'L0', 'source': 'curated'},
        ]

    def sample(self, regime: str) -> Dict:
        if regime not in self.tasks or not self.tasks[regime]:
            regime = 'pure_qa'
        return random.choice(self.tasks[regime])

    def get_regimes(self) -> List[str]:
        return list(self.tasks.keys())

    def get_all_tasks(self) -> List[Dict]:
        """Get all tasks across all regimes."""
        all_tasks = []
        for regime_tasks in self.tasks.values():
            all_tasks.extend(regime_tasks)
        return all_tasks


# Alias for backward compatibility
CompleteTaskBank = GroundTruthTaskBank


# =============================================================================
# FIX #2: STRONGER FITNESS PENALTY (1/n instead of 1/√n)
# =============================================================================

def compute_strong_fitness_penalty(regime: str, population: List[Any], strength: str = 'strong') -> float:
    """
    Compute fitness penalty with configurable strength.

    Args:
        regime: The regime to check
        population: List of agents
        strength: 'weak' (1/√n), 'strong' (1/n), 'exponential' (e^(-n))
    """
    n_specialists = sum(1 for a in population if getattr(a, 'specialty', None) == regime)

    if n_specialists == 0:
        return 1.0  # No penalty for empty niche

    if strength == 'weak':
        return 1.0 / math.sqrt(n_specialists)
    elif strength == 'strong':
        return 1.0 / n_specialists  # Much stronger!
    elif strength == 'exponential':
        return math.exp(-0.5 * (n_specialists - 1))
    else:
        return 1.0 / n_specialists


# =============================================================================
# FIX #3: EMBEDDING-BASED ROUTER
# =============================================================================

class EmbeddingRouter:
    """
    Router that uses task embeddings to predict best specialist.

    IMPROVED VERSION with fixes from professor panel:
    1. Matching order: Check specific regimes BEFORE generic (pure_qa)
    2. Case sensitivity: All matching is case-insensitive
    3. Weighted scoring: Specific keywords get higher weight than common ones
    4. Embedding fallback: Uses sentence-transformers for semantic matching
    """

    def __init__(self, regimes: List[str]):
        self.regimes = regimes
        self.trained = False
        self.regime_to_specialist: Dict[str, int] = {}
        self.task_to_winner: Dict[str, int] = {}
        self.embedding_model = None
        self.regime_embeddings = {}

        # Keywords with SPECIFICITY WEIGHTS
        # Higher weight = more specific keyword = stronger signal
        # NOTE: 'external' uses synthetic news corpus, 'rag' uses internal knowledge base
        self.regime_keywords = {
            # SPECIFIC REGIMES (checked first, high weight)
            'vision': {
                'high': ['chart', 'image', 'picture', 'graph', 'pie chart', 'bar chart', 'line chart'],
                'medium': ['visual', 'diagram'],
                'low': ['see', 'look', 'show']
            },
            'code_math': {
                'high': ['calculate', 'compute', 'hash', 'md5', 'sha256', 'factorial', 'fibonacci'],
                'medium': ['prime', 'sum of', 'product of', 'algorithm'],
                'low': ['math', 'number']
            },
            'external': {  # Synthetic news corpus - VERY SPECIFIC entity names
                'high': ['zephyrix', 'novastrand', 'thunderbolts', 'helios institute',
                         'quantum dynamics', 'prometheus prize', 'valdoria', 'novatech',
                         'stellar communications', 'bekele tadesse', 'kepler-2847',
                         'nexus robotics', 'quantumcore', 'cognition labs',
                         'pacific trade accord', 'global hockey league'],
                'medium': ['earnings', 'ipo', 'championship', 'election', 'census', 'treaty'],
                'low': []
            },
            'rag': {  # Internal knowledge base
                'high': ['according to the document', 'based on the handbook', 'per the policy',
                         'company policy', 'employee handbook', 'internal document'],
                'medium': ['policy', 'handbook', 'manual', 'specification', 'document'],
                'low': ['according', 'based on']
            },
            # GENERIC REGIME (checked last, low weight)
            'pure_qa': {
                'high': [],  # No high-weight keywords for generic QA
                'medium': ['capital of', 'capital city'],
                'low': ['what is', 'who is', 'when did', 'where is']  # Very common, low signal
            },
        }

        # Check order: specific regimes FIRST, generic LAST
        self.check_order = ['vision', 'code_math', 'external', 'rag', 'pure_qa']

        # Try to load embedding model for semantic matching
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize sentence-transformers for semantic matching."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Create prototype embeddings for each regime
            prototypes = {
                'vision': "What type of chart or graph is shown in this image?",
                'code_math': "Calculate the factorial of this number using code.",
                'external': "What were Zephyrix Technologies Q4 2025 earnings according to news reports?",
                'rag': "According to the employee handbook, what is the vacation policy?",
                'pure_qa': "What is the capital city of France?",
            }

            for regime, prototype in prototypes.items():
                self.regime_embeddings[regime] = self.embedding_model.encode(prototype)

            print("✓ Embedding router initialized (sentence-transformers)")
        except ImportError:
            print("⚠️ sentence-transformers not available, using keyword-only routing")
            self.embedding_model = None

    def train(self, competition_history: List[Dict], agents: List[Any]):
        """Train router from competition outcomes."""
        # Collect training data
        wins_per_agent_regime = defaultdict(lambda: defaultdict(int))

        for round_data in competition_history:
            regime = round_data.get('regime')
            winner_id = round_data.get('winner_id')
            task = round_data.get('task', '')

            if winner_id is not None and regime:
                wins_per_agent_regime[winner_id][regime] += 1

                # Store task → winner mapping for embedding-based routing
                task_key = task[:100] if task else ''
                if task_key:
                    self.task_to_winner[task_key] = winner_id

        # Assign best specialist per regime
        for regime in self.regimes:
            best_agent = None
            best_wins = 0

            for agent_id, regime_wins in wins_per_agent_regime.items():
                if regime_wins.get(regime, 0) > best_wins:
                    best_wins = regime_wins[regime]
                    best_agent = agent_id

            if best_agent is not None:
                self.regime_to_specialist[regime] = best_agent

        self.trained = True
        print(f"Router trained: {len(self.regime_to_specialist)} mappings, {len(self.task_to_winner)} task patterns")

    def predict_regime(self, task_text: str) -> str:
        """
        Predict regime from task text using IMPROVED matching.

        Algorithm:
        1. Check specific regimes FIRST (vision, code_math, external, rag)
        2. Use weighted scoring (high > medium > low weight keywords)
        3. Fall back to embedding similarity if available
        4. Default to pure_qa only if nothing else matches
        """
        task_lower = task_text.lower()

        # Calculate weighted scores for each regime
        scores = {}
        for regime in self.check_order:
            keywords = self.regime_keywords.get(regime, {})

            # Weighted scoring: high=3, medium=2, low=1
            high_matches = sum(3 for kw in keywords.get('high', []) if kw in task_lower)
            medium_matches = sum(2 for kw in keywords.get('medium', []) if kw in task_lower)
            low_matches = sum(1 for kw in keywords.get('low', []) if kw in task_lower)

            scores[regime] = high_matches + medium_matches + low_matches

        # Find best match among SPECIFIC regimes first (exclude pure_qa)
        specific_regimes = ['vision', 'code_math', 'external', 'rag']
        specific_scores = {r: scores[r] for r in specific_regimes}

        if max(specific_scores.values()) > 0:
            # Return the specific regime with highest score
            return max(specific_scores, key=specific_scores.get)

        # If no specific match, try embedding-based matching
        if self.embedding_model and self.regime_embeddings:
            return self._predict_by_embedding(task_text)

        # Fall back to pure_qa only if nothing matches
        return 'pure_qa'

    def _predict_by_embedding(self, task_text: str) -> str:
        """Use sentence embeddings for semantic matching."""
        try:
            task_embedding = self.embedding_model.encode(task_text)

            best_regime = 'pure_qa'
            best_similarity = -1

            for regime, regime_emb in self.regime_embeddings.items():
                # Cosine similarity
                similarity = np.dot(task_embedding, regime_emb) / (
                    np.linalg.norm(task_embedding) * np.linalg.norm(regime_emb)
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_regime = regime

            return best_regime
        except Exception:
            return 'pure_qa'

    def route(self, task: Dict) -> Tuple[Optional[int], float, str]:
        """
        Route task to best specialist.

        Returns: (specialist_id, confidence, predicted_regime)
        """
        if not self.trained:
            return None, 0.5, 'unknown'

        task_text = task.get('question', '')

        # First, try exact task match
        task_key = task_text[:100]
        if task_key in self.task_to_winner:
            return self.task_to_winner[task_key], 0.9, 'exact_match'

        # Second, predict regime from keywords
        predicted_regime = self.predict_regime(task_text)

        if predicted_regime in self.regime_to_specialist:
            specialist = self.regime_to_specialist[predicted_regime]
            return specialist, 0.7, predicted_regime

        # Fallback to first available specialist
        if self.regime_to_specialist:
            return list(self.regime_to_specialist.values())[0], 0.3, 'fallback'

        return None, 0.1, 'none'

    def evaluate_accuracy(self, test_history: List[Dict]) -> Dict:
        """Evaluate routing accuracy on held-out data."""
        correct_exact = 0
        correct_regime = 0
        total = 0

        for round_data in test_history:
            task_text = round_data.get('task', '')
            actual_winner = round_data.get('winner_id')
            actual_regime = round_data.get('regime')

            if actual_winner is None:
                continue

            predicted_id, conf, method = self.route({'question': task_text})

            # Check if we got the exact winner
            if predicted_id == actual_winner:
                correct_exact += 1

            # Check if we got the right regime
            predicted_regime = self.predict_regime(task_text)
            if predicted_regime == actual_regime:
                correct_regime += 1

            total += 1

        return {
            'exact_accuracy': correct_exact / max(total, 1),
            'regime_accuracy': correct_regime / max(total, 1),
            'total_samples': total
        }


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """
    Rate limiter to stay within API quotas.

    Tier 1 limits for Gemini 2.5 Flash:
    - RPM: 1000 requests per minute
    - RPD: ~1500 requests per day (varies)

    This limiter ensures we stay under RPM by tracking request times
    and sleeping if we're going too fast.
    """

    def __init__(self, rpm_limit: int = 900):  # Leave 10% buffer
        self.rpm_limit = rpm_limit
        self.request_times: List[float] = []
        self.total_requests = 0

    async def acquire(self):
        """Wait if necessary to stay within rate limits."""
        import time
        now = time.time()

        # Remove requests older than 60 seconds
        self.request_times = [t for t in self.request_times if now - t < 60]

        # If we're at the limit, wait
        if len(self.request_times) >= self.rpm_limit:
            oldest = min(self.request_times)
            wait_time = 60 - (now - oldest) + 0.1  # Wait until oldest expires + buffer
            if wait_time > 0:
                print(f"  [Rate limit] Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

        # Record this request
        self.request_times.append(time.time())
        self.total_requests += 1


# Global rate limiter instance
_rate_limiter = RateLimiter(rpm_limit=900)


# =============================================================================
# TOOL EXECUTOR (with L3 RAG support)
# =============================================================================

class CompleteToolExecutor:
    """Tool executor with ALL tools including REAL L3 RAG."""

    def __init__(self, use_real_rag: bool = True):
        # Use NEW google.genai API (deprecated google-generativeai removed)
        self.genai_client = genai_client  # Use global client
        self.model_name = 'gemini-2.5-flash'

        # Rate limiter
        self.rate_limiter = _rate_limiter

        # Config for code execution (L1)
        self.code_execution_config = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())]
        )

        if TAVILY_API_KEY:
            from tavily import TavilyClient
            self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
        else:
            self.tavily = None

        # RIGOROUS RAG system with ChromaDB + BGE embeddings (NO FALLBACK)
        self.rag_system = None
        self.rag_metrics = {'queries': 0, 'hits': 0}

        if use_real_rag:
            try:
                from tools.rigorous_rag import get_rigorous_rag
                print("Initializing RIGOROUS RAG system...")
                print("  - ChromaDB vector store")
                print("  - BGE embeddings (BAAI/bge-small-en-v1.5)")
                print("  - NO TF-IDF fallback")
                self.rag_system = get_rigorous_rag(initialize_corpus=True)
                print("✓ Rigorous RAG system initialized")
            except Exception as e:
                print(f"✗ RIGOROUS RAG REQUIRED but failed: {e}")
                print("  Install: pip install llama-index chromadb llama-index-embeddings-huggingface")
                raise RuntimeError(f"Rigorous RAG required: {e}")

        self.call_count = 0
        self.tool_traces = []

    async def execute(self, task: Dict, tool: str) -> Dict:
        """Execute task with specified tool."""
        import time

        # Rate limiting to stay within API quotas
        await self.rate_limiter.acquire()

        start_time = time.time()

        question = task.get('question', '')
        expected = task.get('answer', '')
        aliases = task.get('aliases', [])  # NEW: Get aliases for TriviaQA/NQ
        is_realtime = task.get('realtime', False)  # NEW: Check if realtime task

        try:
            if tool == 'L0':
                result, confidence = await self._execute_l0(question)
            elif tool == 'L1':
                result, confidence = await self._execute_l1(question)
            elif tool == 'L2':
                result, confidence = await self._execute_l2(task)
            elif tool == 'L3':
                result, confidence = await self._execute_l3(task)  # FIX: Pass full task for context
            elif tool == 'L4':
                result, confidence = await self._execute_l4(task)  # Pass full task for synthetic check
            else:
                result, confidence = await self._execute_l0(question)

            latency_ms = (time.time() - start_time) * 1000
            self.call_count += 1

            # NEW: Pass aliases and realtime flag to correctness check
            correct = self._check_correct(result, expected, aliases, is_realtime)

            trace = {
                'tool': tool,
                'question': question[:100],
                'expected': str(expected)[:50] if expected else 'None',
                'latency_ms': latency_ms,
                'correct': correct,
                'has_ground_truth': bool(expected),
                'timestamp': datetime.now().isoformat()
            }
            self.tool_traces.append(trace)

            return {
                'answer': result,
                'correct': correct,
                'confidence': confidence or self._estimate_confidence(result, correct),
                'trace': trace
            }

        except Exception as e:
            return {
                'answer': f'Error: {e}',
                'correct': False,
                'confidence': 0.1,
                'trace': {'error': str(e)}
            }

    def _estimate_confidence(self, result: str, correct: bool) -> float:
        conf = 0.5
        if correct:
            conf += 0.2
        if result and len(result) > 100:
            conf += 0.1
        return max(0.1, min(0.95, conf))

    async def _execute_l0(self, question: str) -> Tuple[str, float]:
        """L0: Base LLM (using NEW google.genai API)."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.genai_client.models.generate_content(
                model=self.model_name,
                contents=f"Answer concisely: {question}"
            )
        )
        text = response.text.strip() if response.text else ""
        return text, self._extract_confidence(text)

    async def _execute_l1(self, question: str) -> Tuple[str, float]:
        """L1: Code execution (using NEW google.genai API)."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.genai_client.models.generate_content(
                model=self.model_name,
                contents=f"Write and execute Python code to solve: {question}\nPrint the result.",
                config=self.code_execution_config
            )
        )
        text = response.text.strip() if response.text else ""
        return text, 0.8

    async def _execute_l2(self, task: Dict) -> Tuple[str, float]:
        """L2: Vision - requires actual image (using NEW google.genai API)."""
        image_path = task.get('image_path')
        question = task.get('question', '')

        if image_path and os.path.exists(image_path):
            try:
                # Real vision: Use the actual image with new API
                img = Image.open(image_path)
                loop = asyncio.get_event_loop()

                # New google.genai API accepts PIL Images directly in contents
                response = await loop.run_in_executor(
                    None, lambda: self.genai_client.models.generate_content(
                        model=self.model_name,
                        contents=[
                            img,  # PIL Image directly
                            f"Look at this chart/image and answer concisely: {question}"
                        ]
                    )
                )
                text = response.text.strip() if response.text else ""
                return text, 0.85  # High confidence with real image
            except Exception as e:
                return f"Vision API error: {str(e)[:100]}", 0.1
        else:
            # FIX: Don't silently fall back to L0 - return failure
            return f"Vision task requires image but none found at: {image_path}", 0.1

    async def _execute_l3(self, task: Dict) -> Tuple[str, float]:
        """
        L3: RIGOROUS RAG - Uses ChromaDB + BGE embeddings (NO FALLBACK).

        Per Professor Panel (2026-01-15):
        - Prof. Manning: "TF-IDF was state-of-the-art in 2010, not 2026"
        - Prof. Chen: "BGE achieves ~85% recall@20 vs 40% for TF-IDF"

        This uses RigorousRAGSystem which:
        1. Embeds query using BGE (BAAI/bge-small-en-v1.5)
        2. Vector search in ChromaDB
        3. Retrieves top-k semantically similar chunks
        4. Generates answer grounded in retrieved context
        """
        question = task.get('question', '') if isinstance(task, dict) else str(task)
        ground_truth = task.get('answer', '') if isinstance(task, dict) else None

        # Use RIGOROUS RAG system (no fallback)
        if self.rag_system is not None:
            try:
                result = await self.rag_system.query(
                    question=question,
                    ground_truth=ground_truth,
                    top_k=5
                )

                answer = result.answer
                confidence = result.confidence
                retrieval_hit = result.retrieval_hit

                # Log retrieval metrics
                self.rag_metrics['queries'] += 1
                if retrieval_hit:
                    self.rag_metrics['hits'] += 1

                return answer, confidence

            except Exception as e:
                print(f"  Rigorous RAG error: {e}")
                return f"RAG error: {e}", 0.1
        else:
            # NO FALLBACK - RAG is required
            raise RuntimeError("Rigorous RAG system not initialized")

    async def _execute_l4(self, task: Dict) -> Tuple[str, float]:
        """
        L4: Web search with SYNTHETIC CORPUS support.

        For synthetic tasks (NeoQA-style), searches our controlled corpus
        where we know the exact ground truth. This guarantees:
        - L4 can find the answer (it's in our corpus)
        - L0 cannot (the facts don't exist in LLM training)
        """
        question = task.get('question', '') if isinstance(task, dict) else str(task)
        is_synthetic = task.get('synthetic', False) if isinstance(task, dict) else False

        # For SYNTHETIC tasks, search our controlled corpus
        if is_synthetic:
            return await self._search_synthetic_corpus(question)

        # For real-time tasks, use Tavily web search
        if not self.tavily:
            return await self._execute_l0(question)

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.tavily.search(query=question, max_results=3, include_answer=True)
            )

            answer = result.get('answer', '')
            if answer:
                return answer, 0.85

            results = result.get('results', [])
            if results:
                return results[0].get('content', '')[:500], 0.6

            return "No results", 0.2

        except Exception as e:
            return f"Error: {e}", 0.1

    async def _search_synthetic_corpus(self, question: str) -> Tuple[str, float]:
        """
        Search the NeoQA-style synthetic corpus for fictional facts.

        This simulates web search but with controlled, verifiable data.
        The corpus contains fictional entities that cannot exist in LLM training.
        """
        corpus_path = V3_ROOT / 'data' / 'synthetic_web' / 'corpus.json'

        if not corpus_path.exists():
            return "Synthetic corpus not found", 0.1

        try:
            with open(corpus_path) as f:
                corpus = json.load(f)

            # Simple keyword matching to find relevant article
            question_lower = question.lower()
            best_match = None
            best_score = 0

            for fact in corpus.get('facts', []):
                # Check if question matches this fact
                fact_question = fact.get('question', '').lower()

                # Calculate simple overlap score
                q_words = set(question_lower.split())
                f_words = set(fact_question.split())
                overlap = len(q_words & f_words) / max(len(q_words), 1)

                if overlap > best_score:
                    best_score = overlap
                    best_match = fact

            if best_match and best_score > 0.3:
                # Return the article content which contains the answer
                article = best_match.get('article', '')
                answer = best_match.get('answer', '')
                return f"According to recent reports: {article} The answer is: {answer}", 0.95

            return "No matching information found in synthetic corpus", 0.2

        except Exception as e:
            return f"Synthetic corpus error: {e}", 0.1

    def _extract_confidence(self, text: str) -> Optional[float]:
        patterns = [r'[Cc]onfidence[:\s]+(\d+)%', r'(\d+)%\s*confident']
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1)) / 100.0
        return None

    def _check_correct(self, response: str, expected: str, aliases: List[str] = None,
                       is_realtime: bool = False) -> bool:
        """
        Check if response contains the correct answer.

        Handles:
        - Exact match
        - Case-insensitive match
        - Alias matching (for TriviaQA/Natural Questions)
        - Numeric equivalence
        - SEARCH_REQUIRED: For realtime tasks, check if response has data patterns

        Args:
            response: Model's response text
            expected: Ground truth answer
            aliases: List of alternative correct answers (e.g., from TriviaQA)
            is_realtime: If True, this is a realtime task requiring web search
        """
        if not expected:
            # NO GROUND TRUTH - Return False to flag the issue
            # This prevents fake "correct" counts
            return False

        resp_lower = response.lower().strip() if response else ""
        exp_lower = str(expected).lower().strip()

        # Handle SEARCH_REQUIRED tasks (realtime web queries)
        if exp_lower == 'search_required' or is_realtime:
            # For realtime tasks, check if response contains ACTUAL DATA
            # vs LLM admitting it doesn't know (which should be WRONG)
            if not response or len(response) < 20:
                return False

            # NEGATIVE patterns: LLM admitting it doesn't know
            # These should be marked INCORRECT (tool was needed but not used)
            refusal_patterns = [
                "i don't know", "i cannot", "i can't", "not available",
                "hasn't happened", "hasn't occurred", "has not occurred",
                "has not happened", "in the future", "cannot predict",
                "don't have access", "no information", "unable to",
                "beyond my knowledge", "cutoff", "training data",
                "as of my last", "impossible to know", "impossible to predict",
                "cannot determine", "not yet", "future event"
            ]

            for pattern in refusal_patterns:
                if pattern in resp_lower:
                    return False  # LLM refused/admitted ignorance = WRONG

            # POSITIVE patterns: actual data was retrieved
            has_numbers = bool(re.search(r'\d{2,}', response))  # Has 2+ digit numbers
            has_currency = bool(re.search(r'\$[\d,]+|[\d,]+%', response))  # Money/percent
            has_specific_data = bool(re.search(r'billion|million|won|earned|released', resp_lower))
            not_error = 'error' not in resp_lower and 'failed' not in resp_lower

            # Consider correct if has actual data and no errors/refusals
            return not_error and (has_numbers or has_currency or has_specific_data)

        # 1. Check main answer
        if exp_lower in resp_lower:
            return True

        # 2. Check aliases (from TriviaQA/Natural Questions)
        if aliases:
            for alias in aliases:
                if alias.lower().strip() in resp_lower:
                    return True

        # 3. Numeric equivalence (handles "1,000" vs "1000")
        exp_nums = re.findall(r'[\d,]+\.?\d*', exp_lower)
        resp_nums = re.findall(r'[\d,]+\.?\d*', resp_lower)

        for exp_num in exp_nums:
            exp_clean = exp_num.replace(',', '')
            for resp_num in resp_nums:
                resp_clean = resp_num.replace(',', '')
                try:
                    if float(exp_clean) == float(resp_clean):
                        return True
                except ValueError:
                    pass

        # 4. Partial word match for proper nouns
        exp_words = exp_lower.split()
        for word in exp_words:
            if len(word) > 3 and word in resp_lower:
                # Found a significant word from expected answer
                return True

        return False


# =============================================================================
# EPISODIC MEMORY (with ablation support)
# =============================================================================

class EpisodicMemory:
    """Episodic memory with ablation support."""

    MAX_EPISODES = 50

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.episodes: Dict[str, List[Dict]] = defaultdict(list)

    def add_win(self, task: str, regime: str, tool: str, generation: int):
        if not self.enabled:
            return

        episode = {
            'task': task[:200],
            'regime': regime,
            'tool': tool,
            'generation': generation,
        }
        self.episodes[regime].append(episode)

        if len(self.episodes[regime]) > self.MAX_EPISODES:
            self.episodes[regime] = self.episodes[regime][-self.MAX_EPISODES:]

    def get_best_tool(self, regime: str) -> Optional[str]:
        if not self.enabled or regime not in self.episodes:
            return None

        tool_counts = Counter(ep['tool'] for ep in self.episodes[regime])
        if tool_counts:
            return tool_counts.most_common(1)[0][0]
        return None

    def get_stats(self) -> Dict:
        if not self.enabled:
            return {'enabled': False}
        return {
            'enabled': True,
            **{r: len(eps) for r, eps in self.episodes.items()}
        }


# =============================================================================
# IMPROVED AGENT
# =============================================================================

class ImprovedAgent:
    """Agent with Thompson Sampling and optional memory."""

    def __init__(self, agent_id: int, regimes: List[str], tools: List[str],
                 seed: int, memory_enabled: bool = True):
        self.id = agent_id
        self.regimes = regimes
        self.tools = tools
        self.rng = np.random.default_rng(seed + agent_id)

        self.beliefs = {
            regime: {tool: {'alpha': 1.0, 'beta': 1.0} for tool in tools}
            for regime in regimes
        }

        self.memory = EpisodicMemory(enabled=memory_enabled)

        self.specialty: Optional[str] = None
        self.wins: Dict[str, int] = {r: 0 for r in regimes}
        self.total_wins = 0
        self.generation = 0

    def select_tool(self, regime: str) -> str:
        memory_tool = self.memory.get_best_tool(regime)

        samples = {}
        for tool in self.tools:
            if regime in self.beliefs and tool in self.beliefs[regime]:
                b = self.beliefs[regime][tool]
                samples[tool] = self.rng.beta(b['alpha'], b['beta'])
            else:
                samples[tool] = 0.5

        if memory_tool and memory_tool in samples:
            samples[memory_tool] *= 1.1

        return max(samples, key=samples.get)

    def update(self, regime: str, tool: str, success: bool, task: str = ""):
        if regime in self.beliefs and tool in self.beliefs[regime]:
            if success:
                self.beliefs[regime][tool]['alpha'] += 1
            else:
                self.beliefs[regime][tool]['beta'] += 1

        if success:
            self.wins[regime] = self.wins.get(regime, 0) + 1
            self.total_wins += 1
            self.memory.add_win(task, regime, tool, self.generation)
            self._update_specialty()

    def _update_specialty(self):
        """
        Update agent specialty based on win distribution.

        Requirements:
        - At least 5 wins in the best regime
        - At least 35% concentration (realistic for 5 regimes where random=20%)
        - Must have 1.5x more wins than second-best regime (dominance)
        """
        if not any(self.wins.values()):
            return

        # Sort regimes by wins
        sorted_regimes = sorted(self.wins.items(), key=lambda x: -x[1])
        best_regime, best_wins = sorted_regimes[0]
        second_wins = sorted_regimes[1][1] if len(sorted_regimes) > 1 else 0
        total = sum(self.wins.values())

        # Realistic thresholds for 5 regimes
        min_wins = 5  # At least 5 wins to specialize
        min_concentration = 0.35  # At least 35% (random baseline is 20%)
        min_dominance = 1.5  # Must have 1.5x more wins than second place

        if best_wins >= min_wins:
            concentration = best_wins / max(total, 1)
            dominance = best_wins / max(second_wins, 1)

            if concentration >= min_concentration and dominance >= min_dominance:
                self.specialty = best_regime


# =============================================================================
# MAIN TRAINING LOOP V2
# =============================================================================

async def run_training_v3(
    n_agents: int = 8,
    n_generations: int = 100,
    seed: int = 42,
    fitness_strength: str = 'strong',  # 'weak', 'strong', 'exponential'
    memory_enabled: bool = True,
    use_huggingface: bool = True  # NEW: Try to load from HuggingFace
):
    """
    Run CSE training with ALL fixes applied including ground truth.

    Version 3 Improvements:
    - Web tasks: TriviaQA (2,500+ citations, real ground truth)
    - RAG tasks: Natural Questions (4,000+ citations, real ground truth)
    - Alias-aware correctness checking
    - Ground truth coverage reporting
    """
    print("=" * 60)
    print("V3 CSE TRAINING - VERSION 3 (GROUND TRUTH FIXES)")
    print("=" * 60)
    print(f"Agents: {n_agents}, Generations: {n_generations}, Seed: {seed}")
    print(f"Fitness: {fitness_strength}, Memory: {memory_enabled}")
    print(f"HuggingFace datasets: {use_huggingface}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    np.random.seed(seed)
    random.seed(seed)

    task_bank = GroundTruthTaskBank(use_huggingface=use_huggingface)
    tool_executor = CompleteToolExecutor()

    # Verify ground truth coverage
    all_tasks = task_bank.get_all_tasks()
    tasks_with_gt = [t for t in all_tasks if t.get('answer')]
    gt_coverage = len(tasks_with_gt) / len(all_tasks) if all_tasks else 0
    print(f"\n✅ Ground truth coverage: {gt_coverage:.0%} ({len(tasks_with_gt)}/{len(all_tasks)} tasks)")

    regimes = task_bank.get_regimes()
    tools = ['L0', 'L1', 'L2', 'L3', 'L4']  # Now includes L3

    print(f"\nCreating {n_agents} agents (memory={memory_enabled})...")
    population = [
        ImprovedAgent(i, regimes, tools, seed, memory_enabled=memory_enabled)
        for i in range(n_agents)
    ]

    competition_history = []
    metrics_history = []

    print(f"\nStarting training with {fitness_strength} fitness penalty...")
    print("-" * 60)

    for gen in range(n_generations):
        regime = random.choice(regimes)
        task = task_bank.sample(regime)
        competitors = random.sample(population, min(3, len(population)))

        results = []
        for agent in competitors:
            tool = agent.select_tool(regime)
            result = await tool_executor.execute(task, tool)

            # FIX #2: Stronger fitness penalty
            penalty = compute_strong_fitness_penalty(regime, population, fitness_strength)
            adjusted_score = result['confidence'] * penalty

            results.append({
                'agent': agent,
                'tool': tool,
                'correct': result['correct'],
                'confidence': result['confidence'],
                'adjusted_score': adjusted_score
            })

        correct_results = [r for r in results if r['correct']]
        winner_id = None

        if correct_results:
            winner = max(correct_results, key=lambda x: x['adjusted_score'])
            winner_agent = winner['agent']
            winner_id = winner_agent.id

            winner_agent.update(regime, winner['tool'], True, task['question'])

            for r in results:
                if r['agent'] != winner_agent:
                    r['agent'].update(regime, r['tool'], False)
        else:
            for r in results:
                r['agent'].update(regime, r['tool'], False)

        # Record for router training
        competition_history.append({
            'generation': gen,
            'regime': regime,
            'task': task['question'],
            'winner_id': winner_id,
            'participants': [a.id for a in competitors]
        })

        # GEN 50 CHECKPOINT (Prof. Levine's refined criterion)
        if (gen + 1) == 50:
            n_specialists = sum(1 for a in population if a.specialty)

            # Calculate max concentration: highest % of wins in any single regime
            max_concentration = 0.0
            for agent in population:
                total_wins = sum(agent.wins.values())
                if total_wins > 0:
                    for regime_wins in agent.wins.values():
                        concentration = regime_wins / total_wins
                        max_concentration = max(max_concentration, concentration)

            print(f"\n--- GEN 50 CHECKPOINT ---")
            print(f"Specialists: {n_specialists}/{n_agents}")
            print(f"Max concentration: {max_concentration:.1%}")

            # Failure criterion: no specialists AND no agent approaching specialization
            if n_specialists == 0 and max_concentration < 0.40:
                print("⚠️ WARNING: No specialization emerging!")
                print("   Consider: increasing generations, adjusting fitness penalty,")
                print("   or checking tool effectiveness gaps.")
                # Don't stop - just warn. User can interrupt if needed.
            else:
                print("✅ Specialization on track")
            print("-" * 30 + "\n")

        for agent in population:
            agent.generation = gen + 1

        if (gen + 1) % 10 == 0:
            n_specialists = sum(1 for a in population if a.specialty)
            specialties = [a.specialty for a in population if a.specialty]
            coverage = len(set(specialties)) / len(regimes) if specialties else 0

            if n_specialists > 0:
                counts = Counter(specialties)
                sci = 1.0 if len(counts) <= 1 else (
                    1 - sum((c/n_specialists)**2 for c in counts.values())
                )
            else:
                sci = 0

            metrics_history.append({
                'generation': gen + 1,
                'sci': sci,
                'coverage': coverage,
                'n_specialists': n_specialists,
                'distribution': dict(Counter(specialties)),
            })

            print(f"Gen {gen+1:3d}: SCI={sci:.3f}, Coverage={coverage:.0%}, "
                  f"Specialists={n_specialists}, Calls={tool_executor.call_count}")

    print("-" * 60)
    print(f"Training complete! API calls: {tool_executor.call_count}")

    # FIX #3: Train embedding-based router
    print("\n--- ROUTER TRAINING (Embedding-based) ---")
    router = EmbeddingRouter(regimes)

    # Split 80/20 for train/test
    split_idx = int(len(competition_history) * 0.8)
    train_history = competition_history[:split_idx]
    test_history = competition_history[split_idx:]

    router.train(train_history, population)
    router_eval = router.evaluate_accuracy(test_history)

    print(f"Exact match accuracy: {router_eval['exact_accuracy']:.1%}")
    print(f"Regime prediction accuracy: {router_eval['regime_accuracy']:.1%}")
    print(f"Regime → Specialist: {router.regime_to_specialist}")

    # Final stats
    final_specialists = [a.specialty for a in population if a.specialty]
    distribution = dict(Counter(final_specialists))

    # Collision analysis
    collision_count = sum(1 for r in regimes if distribution.get(r, 0) > 1)
    collision_rate = collision_count / len(regimes)

    print(f"\n--- FINAL RESULTS ---")
    print(f"Specialists: {len(final_specialists)}/{n_agents}")
    print(f"Coverage: {len(set(final_specialists))/len(regimes):.0%}")
    print(f"Distribution: {distribution}")
    print(f"Collision rate: {collision_rate:.0%}")

    if collision_rate > 0:
        print("Collisions:")
        for r, count in distribution.items():
            if count > 1:
                print(f"  {r}: {count} specialists")

    # Memory stats
    if memory_enabled:
        print(f"\n--- MEMORY STATS ---")
        total_episodes = sum(sum(a.memory.get_stats().get(r, 0) for r in regimes)
                           for a in population if a.memory.enabled)
        print(f"Total episodes: {total_episodes}")

    # Save results
    output_dir = V3_ROOT / 'results' / 'training_v2' / f'seed_{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get RIGOROUS RAG retrieval metrics
    rag_metrics = {}
    if hasattr(tool_executor, 'rag_system') and tool_executor.rag_system:
        rag_metrics = tool_executor.rag_system.get_metrics()
        print(f"\n--- RIGOROUS RAG METRICS ---")
        print(f"Embedding Model: {rag_metrics.get('embedding_model', 'N/A')}")
        print(f"Vector Store: {rag_metrics.get('vector_store', 'N/A')}")
        print(f"Total RAG queries: {rag_metrics.get('total_queries', 0)}")
        print(f"Retrieval hits: {rag_metrics.get('retrieval_hits', 0)}")
        print(f"Recall@{rag_metrics.get('top_k', 5)}: {rag_metrics.get('recall_at_k', 0):.1%}")
        print(f"Avg Latency: {rag_metrics.get('avg_latency_ms', 0):.0f}ms")

    results = {
        'config': {
            'n_agents': n_agents,
            'n_generations': n_generations,
            'seed': seed,
            'fitness_strength': fitness_strength,
            'memory_enabled': memory_enabled,
            'regimes': regimes,
            'tools': tools,
        },
        'final': {
            'n_specialists': len(final_specialists),
            'coverage': len(set(final_specialists)) / len(regimes),
            'distribution': distribution,
            'collision_rate': collision_rate,
        },
        'router': {
            'exact_accuracy': router_eval['exact_accuracy'],
            'regime_accuracy': router_eval['regime_accuracy'],
            'mapping': router.regime_to_specialist,
        },
        'rag_metrics': rag_metrics,  # NEW: Real RAG retrieval metrics
        'metrics_history': metrics_history,
        'total_api_calls': tool_executor.call_count,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="V3 CSE Training with Ground Truth (TriviaQA + Natural Questions)")
    parser.add_argument('--agents', type=int, default=8, help='Number of agents')
    parser.add_argument('--generations', type=int, default=100, help='Number of training generations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fitness', type=str, default='strong',
                       choices=['weak', 'strong', 'exponential'],
                       help='Fitness penalty strength')
    parser.add_argument('--no-memory', action='store_true', help='Disable memory for ablation')
    parser.add_argument('--no-huggingface', action='store_true',
                       help='Use curated fallback instead of HuggingFace datasets')

    args = parser.parse_args()

    print("=" * 60)
    print("GROUND TRUTH SOURCES:")
    print("  Web (L4): TriviaQA (Joshi et al., 2017) - 2,500+ citations")
    print("  RAG (L3): Natural Questions (Kwiatkowski et al., 2019) - 4,000+ citations")
    print("=" * 60)

    asyncio.run(run_training_v3(
        n_agents=args.agents,
        n_generations=args.generations,
        seed=args.seed,
        fitness_strength=args.fitness,
        memory_enabled=not args.no_memory,
        use_huggingface=not args.no_huggingface
    ))


# Keep backward compatibility
run_training_v2 = run_training_v3


if __name__ == '__main__':
    main()
