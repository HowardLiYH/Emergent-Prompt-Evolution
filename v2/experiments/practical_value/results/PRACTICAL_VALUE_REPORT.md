# Practical Value Test Results

**Generated:** 2026-01-14T19:21:14.818931
**Tests Run:** 6
**Tests Passed:** 1/6 (17%)

## Summary

| Test | Status | Key Metric |
|------|--------|------------|
| test_01_specialist_accuracy | ❌ FAIL | Object of type bool is not JSON serializable |
| test_02_automatic_routing | ❌ FAIL | Object of type bool is not JSON serializable |
| test_07_parallel_training | ✅ PASS | 8 workers: 2.9x speedup, 36% efficiency |
| test_10_graceful_degradation | ❌ FAIL | Object of type bool is not JSON serializable |
| test_14_collision_coverage | ❌ FAIL | Object of type bool is not JSON serializable |
| test_17_inference_latency | ❌ FAIL | Object of type bool is not JSON serializable |


## Detailed Results

### test_01_specialist_accuracy

**Status:** FAILED

**Details:** N/A

### test_02_automatic_routing

**Status:** FAILED

**Details:** N/A

### test_07_parallel_training

**Status:** PASSED

**Details:** 8 workers: 2.9x speedup, 36% efficiency

**Metrics:**
```json
{
  "per_worker_results": {
    "1": {
      "wall_clock": 19.218994140625,
      "generations": 15,
      "agents": 6,
      "speedup": 1.0,
      "efficiency": 1.0
    },
    "2": {
      "wall_clock": 11.844714164733887,
      "generations": 15,
      "agents": 6,
      "speedup": 1.622579816898164,
      "efficiency": 0.811289908449082
    },
    "4": {
      "wall_clock": 6.783898115158081,
      "generations": 15,
      "agents": 6,
      "speedup": 2.8330310706880586,
      "efficiency": 0.7082577676720146
    },
    "8": {
      "wall_clock": 6.7074079513549805,
      "generations": 15,
      "agents": 6,
      "speedup": 2.8653384854491404,
      "efficiency": 0.35816731068114255
    }
  },
  "speedup_8_workers": 2.8653384854491404,
  "efficiency_8_workers": 0.35816731068114255
}
```

### test_10_graceful_degradation

**Status:** FAILED

**Details:** N/A

### test_14_collision_coverage

**Status:** FAILED

**Details:** N/A

### test_17_inference_latency

**Status:** FAILED

**Details:** N/A


## Commercial Recommendation

Based on test results, the following value propositions are validated:

**Validated:**
- test_07_parallel_training

**Not Validated (Areas for Improvement):**
- test_01_specialist_accuracy
- test_02_automatic_routing
- test_10_graceful_degradation
- test_14_collision_coverage
- test_17_inference_latency
