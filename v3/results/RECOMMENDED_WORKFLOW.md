# Recommended Workflow: Testing Practical Value

**Date**: 2026-01-15
**System Status**: ✅ All systems solid (0 flaws detected)

---

## Current State Assessment

| Component | Status |
|-----------|--------|
| Tool Execution (L0-L4) | ✅ Working |
| Competition Loop | ✅ Working |
| Thompson Sampling | ✅ Working |
| Fitness Sharing | ✅ Working |
| Router | ✅ Implemented |
| Memory | ✅ Implemented |
| Ground Truth | ✅ 63 verified answers |
| google.genai API | ✅ Migrated |

---

## My Recommendation: Streamlined 3-Phase Approach

Given that the system is solid, I recommend a **focused 3-phase workflow** instead of the full 19 tests:

### Phase 1: Prove Core Thesis (CRITICAL)
**Time: 2-3 hours | Must Pass**

| Test | What it proves | Effort |
|------|----------------|--------|
| **#1: Specialist Accuracy Advantage** | Specialists beat generalists | 1 hr |
| **#2: Automatic Task Routing** | Competition trains a router | 1 hr |

If these fail, the project has no value. If they pass, we have the core thesis.

---

### Phase 2: Quick Wins (Demonstrate Practical Value)
**Time: 2-3 hours | Build Confidence**

| Test | What it proves | Effort |
|------|----------------|--------|
| **#14: Collision-Free Coverage** | Agents self-organize efficiently | 30 min |
| **#10: Graceful Degradation** | System is fault-tolerant | 30 min |
| **#17: Real-Time Inference Latency** | Deployment-ready | 30 min |
| **#7: Parallelizable Training** | Scalable | 30 min |

These are easy metrics that show practical value.

---

### Phase 3: Differentiation (Unique Value)
**Time: 3-4 hours | Set Apart from Baselines**

| Test | What it proves | Effort |
|------|----------------|--------|
| **#5: Adaptability to New Task Types** | Self-updating specialists | 1 hr |
| **#6: Distribution Shift Robustness** | Enterprise-ready | 1 hr |
| **#18: Consistency Across Runs** | Reproducible | 2 hrs |

---

## Why NOT the Full 19 Tests?

| Full 19 Tests | Streamlined 9 Tests |
|---------------|---------------------|
| 30-40 hours | 8-10 hours |
| Many redundant | Core coverage |
| Diminishing returns | Each test adds value |
| Overwhelms paper | Fits in one paper |

The streamlined approach covers:
- Core thesis (2 tests)
- Practical value (4 tests)
- Differentiation (3 tests)

---

## Detailed Execution Plan

### Step 1: Run Training First (30 min)

```bash
cd v3
python -m experiments.training.run_training_v2 \
    --agents 8 \
    --generations 100 \
    --seed 42
```

This produces:
- Trained specialists
- Competition history
- Router training data

---

### Step 2: Phase 1 Tests (2-3 hours)

#### Test #1: Specialist Accuracy Advantage

```python
# After training, compare specialist vs generalist on held-out tasks
for regime in REGIMES:
    specialist = get_specialist(population, regime)
    generalist = random.choice(non_specialists)

    specialist_acc = evaluate(specialist, held_out_tasks[regime])
    generalist_acc = evaluate(generalist, held_out_tasks[regime])

    advantage = specialist_acc - generalist_acc
    print(f"{regime}: {advantage:.1%} advantage")

# Success: Mean advantage > 10%, p < 0.05
```

#### Test #2: Automatic Task Routing

```python
# Train router from competition history
router.train(competition_history)

# Test on held-out tasks
routing_accuracy = 0
for task in held_out_tasks:
    predicted = router.route(task)
    actual_best = find_best_performer(population, task)
    if predicted.specialty == actual_best.specialty:
        routing_accuracy += 1

routing_accuracy /= len(held_out_tasks)
print(f"Routing accuracy: {routing_accuracy:.1%}")

# Success: > 80% accuracy
```

---

### Step 3: Phase 2 Tests (2-3 hours)

#### Test #14: Collision-Free Coverage

```python
# Already computed in training metrics
coverage = metrics['coverage']  # % regimes with specialist
collision_rate = metrics['collision_rate']  # % regimes with >1 specialist

# Success: Coverage > 90%, Collision rate < 20%
```

#### Test #10: Graceful Degradation

```python
# Remove each specialist, measure system accuracy
for regime in REGIMES:
    specialist = get_specialist(population, regime)
    reduced_population = [a for a in population if a != specialist]

    reduced_accuracy = evaluate_system(reduced_population)
    degradation = full_accuracy - reduced_accuracy
    print(f"Remove {regime} specialist: {degradation:.1%} degradation")

# Success: Worst-case degradation < 15%
```

#### Test #17: Real-Time Inference Latency

```python
import time

latencies = []
for task in test_tasks:
    start = time.time()
    specialist = router.route(task)
    response = specialist.solve(task)
    latencies.append(time.time() - start)

print(f"P50 latency: {np.percentile(latencies, 50)*1000:.0f}ms")
print(f"P95 latency: {np.percentile(latencies, 95)*1000:.0f}ms")

# Success: P50 overhead < 50ms vs single model
```

#### Test #7: Parallelizable Training

```python
# Already inherent - each agent evaluation is independent
# Measure speedup with more workers
for n_workers in [1, 2, 4, 8]:
    time_taken = train_cse(n_workers=n_workers)
    speedup = time_1_worker / time_taken
    print(f"{n_workers} workers: {speedup:.1f}x speedup")

# Success: 8-worker speedup > 5x
```

---

### Step 4: Phase 3 Tests (3-4 hours)

#### Test #5: Adaptability to New Task Types

```python
# After training on 5 regimes, add a 6th
population = train_cse(regimes=['code', 'vision', 'rag', 'web', 'qa'])

# Continue training with new regime
new_regime = 'audio_transcription'
continue_training(population, regimes + [new_regime], generations=20)

# Check if new specialist emerged
new_specialist = get_specialist(population, new_regime)
print(f"New specialist emerged: {new_specialist is not None}")

# Success: Adaptation in < 20 generations
```

#### Test #6: Distribution Shift Robustness

```python
# Train on uniform distribution
population = train_cse(distribution='uniform')

# Test on shifted distributions
shifted_distributions = [
    {'math': 0.8, 'rest': 0.2},  # Math-heavy
    {'code': 0.8, 'rest': 0.2},  # Code-heavy
]

for shift in shifted_distributions:
    cse_accuracy = evaluate_with_routing(population, shift)
    single_accuracy = evaluate_single_model(shift)
    advantage = cse_accuracy - single_accuracy
    print(f"CSE advantage under {shift}: {advantage:.1%}")

# Success: CSE advantage > 5% under shift
```

#### Test #18: Consistency Across Runs

```python
# Run 10 seeds
results = []
for seed in range(10):
    population = train_cse(seed=seed)
    metrics = compute_metrics(population)
    results.append(metrics)

print(f"Coverage: {np.mean([r['coverage'] for r in results]):.1%} ± {np.std([r['coverage'] for r in results]):.1%}")
print(f"Accuracy: {np.mean([r['accuracy'] for r in results]):.1%} ± {np.std([r['accuracy'] for r in results]):.1%}")

# Success: Accuracy std < 5%
```

---

## Expected Outcomes

| Phase | Test | Expected Result |
|-------|------|-----------------|
| 1 | Specialist Advantage | 15-25% improvement |
| 1 | Router Accuracy | 80-90% correct |
| 2 | Coverage | 90-100% |
| 2 | Collision Rate | < 20% |
| 2 | Degradation | < 15% |
| 2 | Latency Overhead | < 50ms |
| 2 | Parallel Speedup | 5-7x |
| 3 | Adaptation Speed | 10-20 generations |
| 3 | Shift Advantage | 5-15% |
| 3 | Consistency | < 5% std |

---

## If Tests Fail

| Test | If Fails | Action |
|------|----------|--------|
| #1 Accuracy | Specialists NOT better | Debug: Check if tasks are tool-gated |
| #2 Routing | Router inaccurate | Debug: Check competition history quality |
| #5 Adaptability | No new specialist | Increase fitness penalty |
| #18 Consistency | High variance | Increase generations or agents |

---

## My Verdict

**Run the 9 essential tests in order:**

1. ✅ System is solid - proceed to testing
2. Start with Phase 1 (must pass)
3. If Phase 1 passes, continue to Phase 2+3
4. If Phase 1 fails, debug before proceeding

**Total time: 8-10 hours for complete practical value validation**

---

## Next Step

```bash
# Run training first
cd /Users/yuhaoli/code/MAS_For_Finance/emergent_prompt_evolution/v3
python -m experiments.training.run_training_v2 --agents 8 --generations 100 --seed 42
```

Then run the 9 tests in order.

---

*Workflow designed: 2026-01-15*
