# Analysis: Code/Web Evaluation Ties & Solutions

**Date**: 2026-01-15
**Problem**: Code and Web evaluations show ties (100% vs 100%) - specialists don't outperform generalists

---

## Root Cause Analysis

### Why Ties Occur

| Regime | Current Tasks | Why Generalist Succeeds |
|--------|---------------|------------------------|
| **Code** | `987654321 * 123456789` | Gemini 2.5 Flash can compute large arithmetic mentally |
| **Web** | "Current Bitcoin price" | LLM has price ranges in training data, can approximate |

### The Fundamental Issue

**Tasks are NOT truly tool-gated** - they CAN be solved without the tool.

From research (ToolQA, ToolGate, WTU-Eval):
> "A task is tool-gated when the model's internal knowledge is insufficient to solve it correctly, making external tool invocation necessary."

---

## Research Solutions (2024-2025)

### 1. ToolQA Approach
**Key Insight**: Tasks must require information the LLM demonstrably DOESN'T have.

Examples of truly tool-gated tasks:
- Questions about data AFTER training cutoff
- Queries requiring private/dynamic databases
- Computations with random numbers (non-deterministic)

### 2. ToolGate Framework (Jan 2026)
**Key Insight**: Use formal contracts with preconditions and postconditions.

- Precondition: "This task REQUIRES external data"
- Postcondition: "Answer must include data from tool output"

### 3. ToolBench / API-Bank Approach
**Key Insight**: Use REAL APIs that return dynamic data.

- Stock prices (change every second)
- Weather data (changes hourly)
- Database queries (private data LLM never saw)

---

## Truly Tool-Gated Task Categories

### For CODE (L1) - Tasks LLMs CANNOT Solve Mentally

| Category | Example | Why Tool-Gated |
|----------|---------|----------------|
| **Random Numbers** | "Generate 10 random integers between 1-1000, compute their median" | Non-deterministic |
| **File Operations** | "Read file X, count lines containing 'error'" | Requires file access |
| **API Calls** | "Fetch data from API, parse JSON, sum values" | Requires network |
| **Cryptographic** | "Compute SHA256 hash of string 'hello world'" | Complex deterministic |
| **Floating Point** | "Compute sin(37.5°) * cos(82.3°) to 15 decimal places" | Precision limits |
| **Large Data** | "Sort these 1000 numbers and find 73rd percentile" | Memory/compute limits |
| **Time-Dependent** | "What is the current Unix timestamp?" | Changes every second |
| **System Info** | "List files in current directory" | Requires execution |

### For WEB (L4) - Tasks Requiring Real-Time Data

| Category | Example | Why Tool-Gated |
|----------|---------|----------------|
| **Live Prices** | "What is Bitcoin price RIGHT NOW to the cent?" | Changes every second |
| **Breaking News** | "What happened in the last hour on Reuters?" | After training cutoff |
| **Live Events** | "What is the current score of [ongoing game]?" | Real-time only |
| **Weather Now** | "Current temperature in Tokyo to 0.1°C precision" | Changes constantly |
| **Stock Quotes** | "AAPL stock price at this exact moment" | Millisecond precision |
| **Server Status** | "Is github.com currently responding?" | Requires live check |

---

## Proposed Fix: New Tool-Gated Tasks

### Code Tasks (Truly Impossible Without Execution)

```python
CODE_TASKS = [
    # Random number tasks - non-deterministic
    {
        "q": "Generate 5 random integers between 1-100 and compute their sum",
        "verify": "sum_of_5_random",  # Must verify output is plausible
        "tool_required": True
    },

    # Cryptographic tasks - too complex to compute mentally
    {
        "q": "Compute the MD5 hash of 'emergent_specialization_2026'",
        "a": "must_match_actual_hash",  # Verified against real execution
        "tool_required": True
    },

    # High-precision floating point
    {
        "q": "Calculate math.sin(1.2345) * math.cos(6.7890) to 15 decimal places",
        "a": "verified_against_python",
        "tool_required": True
    },

    # Current time (changes every second)
    {
        "q": "What is the current Unix timestamp?",
        "verify": "within_10_seconds_of_actual",
        "tool_required": True
    },

    # File system operations
    {
        "q": "How many .py files are in the v3 directory?",
        "a": "actual_count",
        "tool_required": True
    }
]
```

### Web Tasks (Truly Impossible Without Live Search)

```python
WEB_TASKS = [
    # Exact current prices (to the cent)
    {
        "q": "What is Bitcoin price in USD right now? Give exact value to cents.",
        "verify": "matches_actual_price_within_1%",
        "tool_required": True
    },

    # Very recent news (after any training cutoff)
    {
        "q": "What is the top headline on CNN.com RIGHT NOW?",
        "verify": "matches_actual_headline",
        "tool_required": True
    },

    # Live server status
    {
        "q": "Is the website https://www.google.com currently responding? What is its response code?",
        "a": "200",  # or actual status
        "tool_required": True
    },

    # Real-time events
    {
        "q": "What was the most recent tweet from @OpenAI?",
        "verify": "matches_actual_tweet",
        "tool_required": True
    }
]
```

---

## Verification Strategy

### For Code Tasks
1. **Execute code ourselves** to get ground truth
2. **Compare LLM answer** to actual execution result
3. **For random tasks**: Verify plausibility (5 random 1-100 sum should be 5-500)
4. **For hash tasks**: Must match exactly

### For Web Tasks
1. **Perform actual web search** to get ground truth
2. **Compare within time window** (price within 1% of actual)
3. **Keyword matching** for news headlines
4. **Timestamp verification** (answer within 10 seconds of query time)

---

## Implementation Plan

### Step 1: Create New Tool-Gated Tasks

```python
# tasks_truly_gated.py

import hashlib
import random
import time

def generate_code_tasks():
    """Generate tasks that REQUIRE code execution."""
    tasks = []

    # Task 1: Random number (non-deterministic)
    nums = [random.randint(1, 100) for _ in range(5)]
    tasks.append({
        "question": "Generate 5 random integers between 1-100 and compute their product",
        "correct_answer_type": "plausibility_check",
        "plausible_range": (1, 100**5)
    })

    # Task 2: Hash (too complex for mental math)
    test_string = f"test_{int(time.time())}"
    actual_hash = hashlib.md5(test_string.encode()).hexdigest()
    tasks.append({
        "question": f"Compute the MD5 hash of '{test_string}'",
        "correct_answer": actual_hash,
        "exact_match": True
    })

    # Task 3: Current timestamp
    tasks.append({
        "question": "What is the current Unix timestamp (seconds since 1970)?",
        "correct_answer_type": "timestamp_check",
        "tolerance_seconds": 10
    })

    return tasks
```

### Step 2: Update Evaluation Logic

```python
def verify_code_answer(response, task):
    """Verify answer against ground truth computed at evaluation time."""
    if task.get("exact_match"):
        return task["correct_answer"].lower() in response.lower()
    elif task.get("correct_answer_type") == "timestamp_check":
        # Extract number from response, compare to current time
        import re
        numbers = re.findall(r'\d{10}', response)
        if numbers:
            return abs(int(numbers[0]) - int(time.time())) < task["tolerance_seconds"]
    elif task.get("correct_answer_type") == "plausibility_check":
        # Check if answer is in plausible range
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            val = int(numbers[0])
            return task["plausible_range"][0] <= val <= task["plausible_range"][1]
    return False
```

---

## Expected Results After Fix

| Regime | Generalist | Specialist | Gap |
|--------|------------|------------|-----|
| **Code (fixed)** | ~10-20% | ~80-90% | **+60-70%** |
| **Web (fixed)** | ~10-20% | ~80-90% | **+60-70%** |
| **Vision** | 10% | 90% | +80% |

### Why Generalist Will Fail
- Cannot generate random numbers (will make up values)
- Cannot compute hashes (will hallucinate)
- Cannot know current timestamp (training cutoff)
- Cannot fetch live prices (no internet access)

### Why Specialist Will Succeed
- Code execution returns actual random numbers
- Hash computation is deterministic and correct
- Timestamp retrieved from system
- Web search fetches live data

---

## References

1. **ToolQA** (2023): Dataset distinguishing tool-needed vs answerable-from-knowledge
2. **ToolGate** (2026): Contract-grounded verified tool execution
3. **WTU-Eval** (2024): Whether-or-Not Tool Usage Evaluation
4. **ToolBench** (2024): Real API benchmarks for tool use

---

*Analysis: 2026-01-15*
