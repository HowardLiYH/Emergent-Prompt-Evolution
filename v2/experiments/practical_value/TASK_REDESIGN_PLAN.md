# Task Redesign Plan for Practical Value Testing

**Date:** January 14, 2026
**Issue:** Test 1 (Specialist Accuracy) failed due to ceiling effect - tasks too easy
**Root Cause:** All tasks achieved 90-100% accuracy, leaving no room for specialization advantage

---

## Problem Analysis

### Current Task Design Flaws

| Regime | Example Task | Base Accuracy | Problem |
|--------|--------------|---------------|---------|
| Math | "What is 17 × 23?" | ~100% | Trivial arithmetic in training data |
| Code | "What Python function reverses a list?" | ~100% | One-word factual recall |
| Reasoning | "A > B > C, who is shortest?" | ~90% | Simple transitive inference |
| Knowledge | "Capital of France?" | ~100% | Memorized in all LLMs |
| Creative | "What rhymes with cat?" | ~90% | Basic pattern matching |

### Why This Matters

1. **Ceiling Effect:** When accuracy = 100%, winner is random, not skill-based
2. **No Gradient:** System can't learn what makes a specialist different
3. **Statistical Power:** Need variance to detect specialist advantage

---

## Redesign Principles

### Principle 1: Target 40-60% Base Accuracy

Tasks should be hard enough that a generalist LLM struggles, creating room for specialists to excel.

### Principle 2: Require Domain-Specific Reasoning

Not just recall, but application of domain knowledge to novel situations.

### Principle 3: Multi-Step Composition

Simple one-hop questions are too easy. Require chaining multiple reasoning steps.

### Principle 4: Include Adversarial/Tricky Variants

Edge cases and counterintuitive examples expose generalist weaknesses.

### Principle 5: Partial Credit Scoring

Binary correct/wrong loses information. Score quality, not just correctness.

---

## New Task Design by Regime

### 1. MATH Regime (Target: 45% base accuracy)

**Easy (Calibration):**
```
Q: What is 15% of 80?
A: 12
```

**Medium (Multi-step):**
```
Q: A store offers 20% off, then an additional 15% off the discounted price.
   What is the total percentage discount?
A: 32% (not 35% - common mistake)
```

**Hard (Competition-level):**
```
Q: Find the sum of all positive integers n < 100 such that n² + n + 41 is prime.
A: 39 (only n=0,1,2,...,39 work, sum = 780)
```

**Expert (Requires insight):**
```
Q: A ball is dropped from 10 meters. Each bounce reaches 60% of previous height.
   What is the total distance traveled when the ball comes to rest?
A: 40 meters (10 + 2*(10*0.6 + 10*0.36 + ...) = 10 + 2*10*0.6/(1-0.6) = 40)
```

**Adversarial:**
```
Q: What is 0.1 + 0.2 in floating point arithmetic?
A: 0.30000000000000004 (not 0.3 - floating point error)
```

---

### 2. CODE Regime (Target: 40% base accuracy)

**Easy (Syntax):**
```
Q: What does `[1,2,3][-1]` return in Python?
A: 3
```

**Medium (Bug finding):**
```python
Q: What's wrong with this code?
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n)  # Bug here

A: Infinite recursion - should be factorial(n-1)
```

**Hard (Algorithm design):**
```
Q: Write a function to find the longest palindromic substring in O(n) time.
   Describe the algorithm (don't need to code it).
A: Manacher's algorithm using center expansion with boundary tracking
```

**Expert (System design):**
```
Q: Design a rate limiter that allows 100 requests per minute per user,
   with a sliding window. What data structure would you use and why?
A: Sorted set (Redis ZSET) with timestamps, or sliding window counter with two buckets
```

**Adversarial:**
```python
Q: What does this print?
x = [1, 2, 3]
y = x
y.append(4)
print(x)

A: [1, 2, 3, 4] (reference semantics - tricky for those expecting copy)
```

---

### 3. REASONING Regime (Target: 50% base accuracy)

**Easy (Deduction):**
```
Q: All mammals are warm-blooded. All whales are mammals. Are whales warm-blooded?
A: Yes
```

**Medium (Multi-step):**
```
Q: Alice is taller than Bob. Carol is shorter than Diana. Diana is shorter than Bob.
   Rank all four from tallest to shortest.
A: Alice > Bob > Diana > Carol
```

**Hard (Constraint satisfaction):**
```
Q: Five houses in a row. Each has a different color and owner nationality.
   - The British person lives in the red house
   - The green house is immediately to the left of the white house
   - The Norwegian lives in the first house
   - The blue house is next to the Norwegian's house
   What color is the Norwegian's house?
A: Yellow (process of elimination from constraints)
```

**Expert (Paradox resolution):**
```
Q: A barber shaves all those, and only those, who do not shave themselves.
   Does the barber shave himself?
A: This is a paradox - no consistent answer exists. The set is self-referentially inconsistent.
```

**Adversarial:**
```
Q: If some cats are black, and some black things are hats,
   can we conclude that some cats are hats?
A: No - this is an invalid syllogism (undistributed middle)
```

---

### 4. KNOWLEDGE Regime (Target: 35% base accuracy)

**Easy (Common facts):**
```
Q: What year did World War II end?
A: 1945
```

**Medium (Specific details):**
```
Q: What is the half-life of Carbon-14 in years (to the nearest 100)?
A: 5,730 years
```

**Hard (Obscure facts):**
```
Q: What is the name of the theorem that states every continuous function
   from a closed ball to itself has a fixed point?
A: Brouwer fixed-point theorem
```

**Expert (Recent/Specialized):**
```
Q: What was the key architectural innovation in the GPT-4 paper
   regarding mixture of experts?
A: Trick question - GPT-4 paper didn't disclose architecture details
```

**Adversarial (Common misconceptions):**
```
Q: How many planets did the solar system have in 1930?
A: 9 (Pluto was considered a planet then)
```

---

### 5. CREATIVE Regime (Target: Subjective, use quality scoring)

**Easy (Pattern):**
```
Q: Complete the analogy: Hot is to Cold as Light is to ___
A: Dark
```

**Medium (Wordplay):**
```
Q: Create a sentence using the word "buffalo" five times that is grammatically correct.
A: "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo."
```

**Hard (Constraint satisfaction):**
```
Q: Write a grammatically correct sentence that contains all 26 letters
   of the alphabet in under 40 characters.
A: "Pack my box with five dozen liquor jugs." (32 chars)
```

**Expert (Style transfer):**
```
Q: Rewrite "The quick brown fox jumps over the lazy dog"
   in the style of Shakespeare's sonnets.
A: (Evaluate for iambic pentameter, archaic vocabulary, metaphor use)
```

---

## New Scoring System

### Replace Binary with Partial Credit

```python
def score_response(response, gold, task_type):
    scores = {}

    # 1. Correctness (0-1)
    scores['correct'] = exact_match(response, gold) or semantic_match(response, gold)

    # 2. Reasoning Quality (0-1) - LLM-as-judge
    scores['reasoning'] = judge_reasoning_quality(response, task_type)

    # 3. Confidence Calibration (0-1)
    stated_conf = extract_confidence(response)
    scores['calibration'] = 1 - abs(stated_conf - scores['correct'])

    # 4. Efficiency (0-1)
    scores['efficiency'] = min(1.0, 100 / len(response))  # Penalize verbose

    # Weighted total
    return (0.5 * scores['correct'] +
            0.3 * scores['reasoning'] +
            0.1 * scores['calibration'] +
            0.1 * scores['efficiency'])
```

---

## Difficulty Calibration Protocol

Before running main experiment:

1. **Sample 50 tasks per regime**
2. **Run Gemini 2.5 Flash on each** (no specialization)
3. **Measure base accuracy per regime**
4. **Adjust difficulty until:**
   - Easy tasks: 80-90% accuracy
   - Medium tasks: 50-70% accuracy
   - Hard tasks: 30-50% accuracy
   - Expert tasks: 10-30% accuracy

---

## Expected Results After Redesign

| Regime | Generalist Accuracy | Expected Specialist Accuracy | Expected Advantage |
|--------|--------------------|-----------------------------|-------------------|
| Math | 45% | 60% | +15% |
| Code | 40% | 55% | +15% |
| Reasoning | 50% | 65% | +15% |
| Knowledge | 35% | 50% | +15% |
| Creative | N/A (scored) | +0.15 quality | +15% quality |

---

## Implementation Plan

### Phase 1: Task Collection (2 hours)

1. Create 50 tasks per regime across difficulty levels
2. Include 10 adversarial tasks per regime
3. Document gold answers and scoring rubrics

### Phase 2: Calibration (1 hour)

1. Run baseline accuracy measurement
2. Adjust task difficulty distribution
3. Verify 40-60% base accuracy target

### Phase 3: Scoring System (1 hour)

1. Implement partial credit scoring
2. Set up LLM-as-judge for reasoning quality
3. Add confidence extraction

### Phase 4: Re-run Tests (3 hours)

1. Run Test 1 with new tasks
2. Run Test 2 with new routing challenges
3. Compare specialist vs generalist with proper difficulty

---

## Task Sources

| Regime | Source | Why |
|--------|--------|-----|
| Math | MATH dataset, AMC/AIME problems | Calibrated difficulty, known hard |
| Code | LeetCode Medium/Hard, HumanEval+ | Requires real problem solving |
| Reasoning | LogiQA, bAbi tasks, FOLIO | Formal logic, multi-step |
| Knowledge | TriviaQA hard, SciQ | Beyond common knowledge |
| Creative | Custom + BigBench | Requires composition |

---

## Success Criteria

After redesign, Test 1 passes if:

1. **Mean specialist advantage > 10%** (vs current 0%)
2. **Specialist wins in ≥ 4/5 regimes**
3. **Statistical significance: p < 0.05**

If these criteria are met, we have demonstrated that **specialization provides practical value when tasks are appropriately challenging.**

---

## Key Insight

> "The original test didn't fail because CSE doesn't work. It failed because we measured nothing. With properly challenging tasks, specialists should shine."

---

## Next Steps

1. [ ] Fix model to Gemini 2.5 Flash
2. [ ] Implement new task sets
3. [ ] Calibrate difficulty levels
4. [ ] Implement partial credit scoring
5. [ ] Re-run Phase 1 tests
6. [ ] Document results
