# Model Routing Validation - Evaluation Rubric

## Pre-Defined Success Criteria

**COMMITTED BEFORE ANY EXPERIMENTS RUN**

| Metric | Target | Fail Condition | Measurement |
|--------|--------|----------------|-------------|
| Router Accuracy | >= 75% | < 70% | Correct tier predictions / Total |
| Quality Retention | >= 90% | < 85% | Mean(CSE) / Mean(Pro) |
| Cost Savings | >= 50% | < 40% | (Pro_cost - CSE_cost) / Pro_cost |

**Rule**: If ANY metric falls in "Fail" range, the validation FAILS.

---

## Quality Evaluation Rubric (1-5 Scale)

### Score 5: Perfect
- Complete and accurate response
- Addresses all aspects of the query
- Well-structured and clear
- No errors or omissions

### Score 4: Good
- Minor issues but fully usable
- Addresses the main query correctly
- May have slight formatting or completeness issues
- 90%+ of required information present

### Score 3: Acceptable
- Some gaps but addresses the query
- Missing some details or context
- Partially correct information
- 70-90% of required information present

### Score 2: Poor
- Significant issues
- Only partially addresses the query
- Contains errors or misleading information
- 40-70% of required information present

### Score 1: Fail
- Wrong, unhelpful, or off-topic
- Does not address the query
- Contains major errors
- <40% of required information present

---

## Blind Evaluation Protocol

1. **Shuffle**: Randomize all responses before evaluation
2. **Anonymize**: Remove model identifiers from responses
3. **Evaluate**: Score each response using rubric above
4. **Record**: Log scores with response IDs only
5. **Reveal**: Match scores to conditions after all evaluations complete

---

## Tier Mapping

| Regime | Model Tier | Gemini Model | Cost ($/1M tokens) |
|--------|------------|--------------|-------------------|
| pure_qa | Cheap | gemini-2.0-flash | $0.075 in, $0.30 out |
| code_math | Medium | gemini-2.5-flash | $0.15 in, $0.60 out |
| rag | Medium | gemini-2.5-flash | $0.15 in, $0.60 out |
| external | Medium | gemini-2.5-flash | $0.15 in, $0.60 out |
| vision | Expensive | gemini-2.5-pro | $1.25 in, $5.00 out |

---

## Cost Calculation Formula

```python
# Baseline cost (all to expensive)
baseline_cost = n_queries * AVG_TOKENS * EXPENSIVE_RATE

# CSE routed cost
routed_cost = sum(
    tier_cost[predicted_tier] * AVG_TOKENS
    for query in queries
)

# Savings
savings_percent = (baseline_cost - routed_cost) / baseline_cost * 100
```

---

## Statistical Requirements

- **Paired comparison**: Same queries through all conditions
- **Dual runs**: Two independent samples for variance estimation
- **Stratified sampling**: All query categories represented

---

*Created: 2026-01-16*
*Status: LOCKED - Do not modify after experiments begin*
