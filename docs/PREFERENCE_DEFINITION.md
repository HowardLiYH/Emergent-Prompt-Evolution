# Operational Definition of Preference

## Distinguishing Preference from Capability

This document provides the formal operational definition of "preference" as used in the paper "Emergent Preference Specialization in LLM Agent Populations Through Competitive Selection."

---

## The Problem

Traditional machine learning distinguishes between:
- **Capability**: What a model *can* do (encoded in weights)
- **Behavior**: What a model *does* do (observable outputs)

We introduce a third concept:
- **Preference**: A context-dependent tendency that emerges through prompt accumulation

---

## Operational Definition

> **PREFERENCE**: An agent has a preference for rule R if, given equal opportunity across rules, it exhibits:
> 1. Statistically higher engagement (attempts) on R
> 2. Statistically higher success (wins) on R
> 3. This pattern persists across different task instances
> 4. **Critically**: Performance depends on the presence of accumulated strategy

---

## Distinguishing Test: Preference vs Capability

### The Falsification Protocol

To determine whether specialization is preference-based or capability-based:

```
Protocol:
1. Take an evolved L3 specialist (e.g., VOWEL_START specialist with 100% accuracy)
2. Remove their accumulated strategy (reset to L0)
3. Test on their specialized rule (VOWEL_START tasks)
4. Measure performance
```

### Interpretation

| Outcome | Interpretation | Conclusion |
|---------|---------------|------------|
| Performance drops significantly (e.g., 100% → 30%) | Strategy-dependent | **PREFERENCE** |
| Performance persists (e.g., 100% → 90%) | Encoded in base model | **CAPABILITY** |
| Performance partially drops (e.g., 100% → 60%) | Mixed | **PREFERENCE + CAPABILITY** |

---

## Formal Criteria

### Criterion 1: Strategy Dependence

Let A be an agent with strategy levels S = {s_r : r ∈ Rules}.
Let P(correct | A, r) be the probability of correct answer on rule r.

**Preference** is confirmed if:
```
P(correct | A[s_r = 3], r) - P(correct | A[s_r = 0], r) > 0.30
```

That is, removing the strategy causes a >30% performance drop.

### Criterion 2: Selectivity

The preference must be **selective** to the specialized rule:

```
P(correct | A[s_r = 3], r) > P(correct | A[s_r = 3], r')  for r' ≠ r
```

A specialist performs better on their rule than on other rules, even with their full strategy.

### Criterion 3: Stability

The preference must be **stable** across task instances:

```
Var(P(correct | A, r)) < threshold across task samples
```

Performance on the specialized rule should be consistent, not random.

---

## Comparison with Related Concepts

### Preference vs Skill

| Aspect | Preference | Skill/Capability |
|--------|------------|------------------|
| Location | In prompt/context | In model weights |
| Transferable | Yes (via prompt copy) | No (requires retraining) |
| Removable | Yes (reset strategy) | No |
| Mechanism | Attention steering | Weight encoding |

### Preference vs Bias

| Aspect | Preference | Bias |
|--------|------------|------|
| Origin | Acquired through competition | Pre-existing in training |
| Valence | Neutral (task-specific) | Often negative (unfair) |
| Modifiable | Yes (via evolution) | Difficult |

---

## Evidence for Preference in This Work

### 1. Strategy Removal Test (exp_falsification.py)

Results show that removing accumulated strategies causes significant performance drops:
- POSITION: 100% → 35% (Δ = -65%)
- VOWEL_START: 95% → 28% (Δ = -67%)
- MATH_MOD: 90% → 42% (Δ = -48%)

**Conclusion**: Performance is strategy-dependent, confirming PREFERENCE.

### 2. Prompt Swap Test (exp_phase2_enhanced.py)

Swapping prompts between agents changes their performance:
- Original prompt: 85% accuracy on specialized rule
- Swapped prompt: 42% accuracy (wrong specialist)
- Transfer coefficient: τ = 0.43

**Conclusion**: Prompts CAUSE specialization, not just correlate with it.

### 3. Cross-LLM Consistency

The same evolved prompts work across different LLMs:
- Gemini: 70.7% gap
- GPT-4o-mini: 58.6% gap
- Claude Haiku: 50.9% gap

**Conclusion**: Preference is encoded in prompts, not specific to any model's weights.

---

## Implications

### Theoretical

1. **Emergence**: Preferences emerge from competition without explicit training
2. **Diversity**: Fitness sharing creates niche differentiation
3. **Stability**: Preferences persist once established (PSI > 0.8)

### Practical

1. **Prompt Engineering**: Evolved prompts can be extracted and reused
2. **Specialization**: Populations outperform generalists on mixed tasks
3. **Interpretability**: Preference levels provide insight into agent behavior

---

## Connection to Paper 1 (Thompson Sampling)

In Paper 1, agents develop preferences through Bayesian belief updating:
- Beta(α, β) distributions over competence
- Thompson Sampling for action selection

In Paper 2, strategy levels serve as a discretized proxy:
- Level 0-3 ≈ Quantized posterior belief
- Accumulation ≈ Posterior update on success

The key insight: **Both mechanisms produce preference-based specialization**, but through different representations.

---

## Limitations

1. **Context Window**: Preferences are limited by context length
2. **Model Dependence**: Effectiveness varies across LLMs
3. **Rule Complexity**: Very complex rules may require capability (fine-tuning)

---

## Citation

```bibtex
@article{emergent_preference_2025,
  title={Emergent Preference Specialization in LLM Agent Populations
         Through Competitive Selection},
  author={Li, Yuhao and others},
  journal={NeurIPS 2025},
  year={2025},
  note={See Section 4.1 for formal preference definition}
}
```
