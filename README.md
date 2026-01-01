# Competitive Selection Drives Role Specialization in LLM Agent Populations

<p align="center">
  <img src="assets/cover.jpeg" alt="Emergent Specialization" width="600"/>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-innovation">Key Innovation</a> •
  <a href="#10-cognitive-domains">Domains</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Paper-NeurIPS%202026-blue" alt="Paper"/>
  <img src="https://img.shields.io/badge/Python-3.9+-green" alt="Python"/>
  <img src="https://img.shields.io/badge/Strategies-280-orange" alt="Strategies"/>
  <img src="https://img.shields.io/badge/Domains-10-purple" alt="Domains"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

## Overview

**Can LLM agents develop specialized roles through competitive selection from a fixed strategy space?**

We study the conditions under which role specialization emerges in populations of LLM agents. Unlike prior work on prompt evolution that focuses on single-task optimization, we investigate multi-agent dynamics where agents develop **stable specialist identities** through **cumulative strategy acquisition**.

```
Generation 0                         Generation 100
┌────────────────────────┐           ┌────────────────────────┐
│ "I am a general-       │           │ "I am an Arithmetic    │
│  purpose AI..."        │   ───→    │  Specialist. I excel   │
│                        │           │  at calculation..."    │
│ Strategies: []         │           │ Strategies: [12]       │
└────────────────────────┘           └────────────────────────┘
     (20 identical)                      (10 unique specialists)
```

### Key Contribution

We are the **first to study emergent role specialization in LLM agents through competitive dynamics with prompt-based identity persistence and fixed strategy spaces**.

---

## Key Innovation

### Context Engineering v7: Cumulative Evolution

Unlike traditional approaches where LLMs generate new prompts, we use a **fixed strategy library** extracted from established benchmarks:

| Component | Size | Source |
|-----------|------|--------|
| **Strategies** | 280 | GSM8K, HumanEval, BIG-Bench, MMLU |
| **Examples** | 150 | Worked solutions from benchmarks |
| **Domains** | 10 | Cognitive science (Gardner, Bloom, Pearl) |

### The Core Mechanism

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPETITION ROUND                             │
├─────────────────────────────────────────────────────────────────┤
│  1. All 20 agents attempt task from random domain                │
│  2. Score responses (exact match or LLM-as-judge)               │
│  3. Apply fitness sharing (penalize crowded niches)             │
│  4. WINNER acquires strategy from fixed library                 │
│  5. Strategy ADDED to prompt (cumulative, not replacement)      │
│  6. Repeat for 100 generations                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why Fixed Library?

| Approach | Reproducibility | Scientific Rigor | Overclaiming Risk |
|----------|-----------------|------------------|-------------------|
| LLM-generated strategies | ❌ Variable | ❌ Low | ❌ High |
| **Fixed strategy library** | ✅ Perfect | ✅ High | ✅ Low |

---

## 10 Cognitive Domains

Based on Gardner's Multiple Intelligences, Bloom's Taxonomy, and LLM benchmarks:

| # | Domain | Core Operation | Dataset | Strategies |
|---|--------|----------------|---------|------------|
| 1 | **Arithmetic** | Calculate | GSM8K | 30 |
| 2 | **Algorithmic** | Decompose | BIG-Bench | 28 |
| 3 | **Factual** | Retrieve | TriviaQA, MMLU | 25 |
| 4 | **Procedural** | Sequence | wikiHow | 25 |
| 5 | **Creative** | Imagine | WritingPrompts | 30 |
| 6 | **Summarization** | Compress | CNN/DailyMail | 25 |
| 7 | **Causal** | Explain why | BIG-Bench | 28 |
| 8 | **Analogical** | Match pattern | ARC | 28 |
| 9 | **Coding** | Implement | HumanEval | 35 |
| 10 | **Scientific** | Hypothesize | SciQ | 26 |

Each domain has **unique input/output signatures** to maximize distinctiveness.

---

## Quick Start

### Installation

```bash
git clone https://github.com/HowardLiYH/Emergent-Prompt-Evolution.git
cd Emergent-Prompt-Evolution
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
```

### Run Validation Test

```bash
# Test if prompts meaningfully affect behavior
python experiments/exp_validation.py
```

Expected output:
- In-domain accuracy: > 80%
- Out-of-domain accuracy: < 40%
- Performance gap: > 40 percentage points

### Run Baseline Comparisons

```bash
python experiments/exp_baselines.py
```

### Run Full Experiment

```python
from experiments.run_full_experiment import run_main_experiments, ExperimentConfig
import asyncio

config = ExperimentConfig(
    name="main_experiment",
    num_agents=20,
    num_generations=100,
    num_seeds=10,
    use_fitness_sharing=True
)

results = asyncio.run(run_main_experiments(config))
print(f"Final LSI: {results['summary']['lsi']['mean']:.3f}")
```

---

## Experiments

### Experiment Suite

| # | Experiment | Question | Success Criterion |
|---|------------|----------|-------------------|
| 1 | **Validation Pre-Test** | Do prompts affect behavior? | Gap > 40% |
| 2 | **Cumulative Evolution** | Do agents specialize? | LSI > 0.4 |
| 3 | **No Evolution** | Is evolution necessary? | LSI ≈ 0.1 |
| 4 | **Random Strategy** | Is selection necessary? | LSI < cumulative |
| 5 | **Handcrafted** | What's the ceiling? | LSI ≈ 1.0 |
| 6 | **Prompt Swap Test** | Is specialization causal? | Transfer > 0.5 |

### Baseline Comparison

| Condition | Description | Expected LSI |
|-----------|-------------|--------------|
| Handcrafted | Expert-designed specialists | ~1.0 (upper bound) |
| **Cumulative** | Our method | > 0.4 |
| Random Strategy | Random strategy selection | < 0.3 |
| No Evolution | Initial prompt only | ~0.1 (lower bound) |

---

## Project Structure

```
emergent_prompt_evolution/
├── src/genesis/
│   ├── domains.py           # 10 orthogonal cognitive domains
│   ├── strategy_library.py  # 280 fixed strategies
│   ├── example_library.py   # 150 worked examples
│   ├── evolution_v2.py      # Cumulative evolution mechanism
│   ├── fitness_sharing.py   # Diversity preservation
│   ├── cost_tracker.py      # API cost tracking
│   ├── analysis.py          # Statistical analysis & plots
│   ├── agent.py             # GenesisAgent with evolvable prompt
│   ├── tasks.py             # Task generation
│   ├── competition.py       # Winner-take-all engine
│   ├── metrics.py           # LSI, semantic, behavioral metrics
│   ├── counterfactual.py    # Prompt swap test
│   └── simulation.py        # Main orchestrator
├── experiments/
│   ├── exp_validation.py    # Validation pre-test
│   ├── exp_baselines.py     # 5 baseline conditions
│   ├── run_full_experiment.py # Complete experiment suite
│   └── run_experiments.py   # Legacy experiment runner
├── paper/
│   └── draft_outline.md     # NeurIPS paper draft
├── results/
│   ├── baselines/           # Baseline comparison results
│   ├── main/                # Main experiment results
│   └── validation/          # Validation test results
├── CHANGELOG.md             # Project progress log
└── requirements.txt
```

---

## Metrics

### LLM Specialization Index (LSI)

Entropy-based measure of performance concentration:

- **LSI = 0**: Equal performance across all domains (generalist)
- **LSI = 1**: Perfect in one domain, zero in others (specialist)

### Success Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| LSI | > 0.4 | Strong specialization |
| Unique Specs | 8-10 / 10 | Diverse niche occupation |
| Transfer Coefficient | > 0.5 | Prompts cause behavior |

---

## Theoretical Foundation

### Grounded in Established Frameworks

| Source | Contribution |
|--------|--------------|
| **Gardner (1983)** | Multiple intelligences theory |
| **Bloom (1956)** | Taxonomy of cognitive objectives |
| **Pearl (2009)** | Causal reasoning framework |
| **HELM (Stanford)** | LLM evaluation categories |
| **BIG-Bench (Google)** | Cognitive task taxonomy |

### Key Insight

> We study the **conditions** under which specialization emerges from a fixed strategy space, not the emergence of novel capabilities. This is **competitive selection**, not emergence.

---

## Related Projects

| Project | Relationship |
|---------|--------------|
| [Emergent-Specialization](https://github.com/HowardLiYH/Emergent-Specialization-in-Multi-Agent-Systems) | Paper 1: Rule-based agents (foundation) |
| [Emergent-Civilizations](https://github.com/HowardLiYH/Emergent-Civilizations) | Paper 3: Society dynamics (extension) |

---

## Cost Estimation

| Configuration | API Calls | Tokens | Est. Cost |
|---------------|-----------|--------|-----------|
| Validation (10 specialists × 50 tasks) | ~500 | ~250K | ~$0.50 |
| Baselines (4 conditions × 5 seeds) | ~10K | ~5M | ~$5 |
| Main (20 agents × 100 gen × 10 seeds) | ~200K | ~100M | ~$50 |

Using `gpt-4o-mini` for cost efficiency.

---

## Citation

```bibtex
@article{competitive_selection_2026,
  title={Competitive Selection Drives Role Specialization in LLM Agent Populations},
  author={Li, Yuhao and others},
  journal={Advances in Neural Information Processing Systems},
  year={2026}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Part of the Emergent Specialization Research Series</b><br>
  <i>Paper 2 of 3</i>
</p>
