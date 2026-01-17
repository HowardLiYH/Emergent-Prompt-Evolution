# Task Redesign V2: Tool-Dependent Tasks

**Date:** January 14, 2026
**Key Insight:** Tasks must REQUIRE tool usage to solve correctly. Pure LLM calls should FAIL.

---

## The Fundamental Problem with V1 Design

V1 tasks were still answerable by pure LLM reasoning:
- "What is 20% off then 15% off?" → LLM can calculate mentally
- "Find the bug in this code" → LLM can reason about code

**These don't require tools - they just require smarter prompts.**

---

## Tool Capability Hierarchy (L0-L5)

| Level | Capability | What It Enables |
|-------|------------|-----------------|
| L0 | Pure LLM | Text reasoning only |
| L1 | Code Execution | Run Python, precise calculations |
| L2 | Multi-modal | Analyze images, charts, PDFs |
| L3 | RAG/Retrieval | Search documents, databases |
| L4 | Web Access | Real-time information, APIs |
| L5 | Agent Orchestration | Coordinate multiple agents |

---

## Design Principle: Tool-Gated Tasks

A task is "tool-gated" if:
1. **Without tool:** Accuracy < 20% (essentially random guessing)
2. **With tool:** Accuracy > 80%
3. **Gap:** At least 60% accuracy difference

---

## L1 Tasks: Code Execution Required

### Why Pure LLM Fails
LLMs are notoriously bad at:
- Precise arithmetic with many digits
- Floating point operations
- Complex algorithm execution
- String manipulation edge cases

### Task Examples

**L1-Task-1: Large Number Arithmetic**
```
Question: What is 987654321 × 123456789? Give the exact answer.
Gold Answer: 121932631112635269

Without code execution: LLM will hallucinate a wrong number
With code execution: Python returns exact result
```

**L1-Task-2: Floating Point Precision**
```
Question: Calculate the sum: 0.1 + 0.2 + 0.3 + ... + 0.9 + 1.0
          Give the answer to 15 decimal places.
Gold Answer: 5.500000000000001

Without code: LLM says "5.5"
With code: Python reveals floating point accumulation error
```

**L1-Task-3: Algorithm Execution**
```
Question: Apply the Euclidean algorithm to find GCD(1071, 462).
          List every step (quotient and remainder).
Gold Answer:
  1071 = 2 × 462 + 147
  462 = 3 × 147 + 21
  147 = 7 × 21 + 0
  GCD = 21

Without code: LLM often makes arithmetic errors in intermediate steps
With code: Exact step-by-step execution
```

**L1-Task-4: Regex Matching**
```
Question: How many words in this text match the regex pattern
          `\b[A-Z][a-z]*[0-9]+\b`?
          Text: "Alice3 met Bob at Room42 with Charlie99 and David"
Gold Answer: 3 (Alice3, Room42, Charlie99)

Without code: LLM struggles with precise regex interpretation
With code: re.findall() gives exact answer
```

**L1-Task-5: Simulation**
```
Question: Simulate 1000 rolls of two dice.
          What percentage of rolls sum to exactly 7?
          (Use seed=42 for reproducibility)
Gold Answer: ~16.7% (varies with seed, but deterministic)

Without code: LLM can only give theoretical probability, not simulation
With code: numpy.random with seed gives exact reproducible answer
```

---

## L2 Tasks: Multi-modal Required

### Why Pure LLM Fails
LLMs cannot "see" images in text-only mode. They can describe images if provided, but cannot:
- Read text from images (OCR)
- Count objects in images
- Interpret charts/graphs
- Analyze visual patterns

### Task Examples

**L2-Task-1: Chart Reading**
```
Input: [Bar chart image showing Q1-Q4 sales]
Question: What quarter had the highest sales and by how much
          did it exceed Q2?
Gold Answer: Q3 had $45M, exceeded Q2 by $12M

Without vision: LLM cannot see the chart
With vision: Can extract exact values from bars
```

**L2-Task-2: Document OCR**
```
Input: [Scanned receipt image]
Question: What is the total amount on this receipt?
Gold Answer: $127.43

Without vision: Cannot read the image
With vision: OCR extracts text, finds total
```

**L2-Task-3: Diagram Understanding**
```
Input: [Network topology diagram]
Question: How many nodes are directly connected to the central router?
Gold Answer: 7

Without vision: Cannot count visual elements
With vision: Can analyze topology
```

**L2-Task-4: Handwriting Recognition**
```
Input: [Handwritten math equation image]
Question: What equation is written and what is its solution?
Gold Answer: 3x + 7 = 22, x = 5

Without vision: Cannot read handwriting
With vision: OCR + solving
```

**L2-Task-5: Photo Analysis**
```
Input: [Photo of a parking lot]
Question: How many red cars are visible?
Gold Answer: 4

Without vision: Cannot count objects in image
With vision: Object detection + counting
```

---

## L3 Tasks: RAG/Retrieval Required

### Why Pure LLM Fails
LLMs have:
- Training data cutoff (don't know recent events)
- No access to private documents
- No access to specific databases
- Hallucination risk for specific facts

### Task Examples

**L3-Task-1: Private Document QA**
```
Context: [Company internal policy document, 50 pages]
Question: According to the Q3 2025 policy update, what is the
          maximum reimbursement for international travel?
Gold Answer: $350/day (from page 37)

Without RAG: LLM has never seen this document
With RAG: Retrieves relevant section, extracts answer
```

**L3-Task-2: Database Query**
```
Context: [SQL database with employee records]
Question: What is the average salary of employees in the
          Engineering department who joined after 2020?
Gold Answer: $125,432

Without RAG: LLM cannot access database
With RAG: Generates SQL, executes, returns result
```

**L3-Task-3: Multi-Document Synthesis**
```
Context: [10 research papers on climate change]
Question: Which paper disagrees with the IPCC 2023 findings
          on sea level rise projections?
Gold Answer: Paper 7 (Smith et al.) argues for higher estimates

Without RAG: LLM hasn't read these specific papers
With RAG: Searches across documents, finds contradiction
```

**L3-Task-4: Legal Document Analysis**
```
Context: [Contract document, 100 pages]
Question: What is the termination clause penalty in Section 8.3?
Gold Answer: 15% of remaining contract value

Without RAG: LLM doesn't have this contract
With RAG: Retrieves Section 8.3, extracts clause
```

**L3-Task-5: Technical Manual Lookup**
```
Context: [Product technical manual PDF]
Question: What is the maximum operating temperature for Model X-500?
Gold Answer: 85°C

Without RAG: LLM doesn't know this specific product
With RAG: Finds spec table, extracts value
```

---

## L4 Tasks: Web Access Required

### Why Pure LLM Fails
LLMs have:
- No real-time information
- Training data cutoff (typically 1-2 years old)
- No access to live APIs
- Cannot check current prices, weather, news

### Task Examples

**L4-Task-1: Current Stock Price**
```
Question: What is the current stock price of NVIDIA (NVDA)?
          (As of the moment you answer)
Gold Answer: $XXX.XX (real-time value)

Without web: LLM only knows historical prices
With web: Fetches current market data
```

**L4-Task-2: Current Weather**
```
Question: What is the current temperature in Tokyo, Japan?
Gold Answer: XX°C (real-time)

Without web: LLM can only give historical averages
With web: Fetches current weather API
```

**L4-Task-3: Recent News**
```
Question: What major tech company announced layoffs in the past 24 hours?
Gold Answer: [Current news]

Without web: LLM has no access to today's news
With web: Searches news, finds recent announcements
```

**L4-Task-4: Exchange Rates**
```
Question: What is the current USD to EUR exchange rate?
Gold Answer: 1 USD = X.XX EUR (real-time)

Without web: LLM knows outdated rates
With web: Fetches live forex data
```

**L4-Task-5: Website Status**
```
Question: Is github.com currently accessible? What is the HTTP status code?
Gold Answer: Yes, 200 OK (or current status)

Without web: LLM cannot check live status
With web: Makes HTTP request, reports status
```

---

## L5 Tasks: Multi-Agent Orchestration Required

### Why Single Agent Fails
Complex tasks require:
- Parallel information gathering
- Specialized sub-task delegation
- Coordination between experts
- Aggregation of multiple results

### Task Examples

**L5-Task-1: Research Synthesis**
```
Question: Compare the AI strategies of Google, Microsoft, and Meta
          based on their latest earnings calls.
          Summarize key differences in 200 words.
Gold Answer: [Synthesized comparison from 3 sources]

Without orchestration: Single agent struggles with breadth
With orchestration: 3 agents each analyze one company, coordinator synthesizes
```

**L5-Task-2: Multi-Domain Problem**
```
Question: A company wants to build a new data center.
          Consider: legal requirements, environmental impact,
          technical specifications, and cost analysis.
          Provide recommendations.
Gold Answer: [Multi-faceted analysis]

Without orchestration: Single agent misses domain expertise
With orchestration: Legal, Environmental, Technical, Financial agents each contribute
```

**L5-Task-3: Debate Simulation**
```
Question: Simulate a debate between a climate scientist and
          an economist about carbon tax policy.
          Present both sides fairly.
Gold Answer: [Balanced debate with domain-specific arguments]

Without orchestration: Single agent has bias
With orchestration: Two specialized agents present opposing views
```

**L5-Task-4: Code Review Pipeline**
```
Question: Review this 500-line codebase for: security vulnerabilities,
          performance issues, code style, and documentation gaps.
Gold Answer: [Multi-aspect review]

Without orchestration: Single agent misses some aspects
With orchestration: Security, Performance, Style, Docs agents each review
```

**L5-Task-5: Competitive Analysis**
```
Question: Analyze the competitive landscape for EV charging stations
          in California, including market share, pricing, and
          technology comparison for the top 5 providers.
Gold Answer: [Comprehensive competitive analysis]

Without orchestration: Too broad for single agent
With orchestration: 5 agents each research one provider, coordinator compares
```

---

## Implementation Architecture

### How Tasks Are Gated

```python
class TaskGate:
    """Ensures tasks require specific tool levels."""

    @staticmethod
    def verify_l1_required(task):
        """L1 task fails without code execution."""
        # Run with pure LLM
        llm_answer = pure_llm(task.question)
        llm_correct = evaluate(llm_answer, task.gold)

        # Run with code execution
        code_answer = llm_with_code(task.question)
        code_correct = evaluate(code_answer, task.gold)

        # Verify gap
        assert llm_correct < 0.3, "Task too easy for pure LLM"
        assert code_correct > 0.8, "Code doesn't help enough"
        return True
```

### Agent Tool Access by Level

```python
AGENT_CAPABILITIES = {
    'L0_agent': [],
    'L1_agent': ['python_executor'],
    'L2_agent': ['python_executor', 'vision_model'],
    'L3_agent': ['python_executor', 'vision_model', 'retriever'],
    'L4_agent': ['python_executor', 'vision_model', 'retriever', 'web_access'],
    'L5_agent': ['python_executor', 'vision_model', 'retriever', 'web_access', 'agent_coordinator'],
}
```

### Regime-Tool Mapping (Hidden from Agents)

```python
REGIME_OPTIMAL_TOOL = {
    'precise_math': 'L1',      # Needs code execution
    'visual_analysis': 'L2',   # Needs vision
    'document_qa': 'L3',       # Needs RAG
    'realtime_info': 'L4',     # Needs web
    'complex_synthesis': 'L5', # Needs orchestration
}
```

---

## Expected Results with Tool-Gated Tasks

| Regime | L0 Accuracy | Specialist Accuracy | Gap |
|--------|-------------|--------------------|----- |
| precise_math | 15% | 90% (L1) | 75% |
| visual_analysis | 0% | 85% (L2) | 85% |
| document_qa | 5% | 80% (L3) | 75% |
| realtime_info | 0% | 90% (L4) | 90% |
| complex_synthesis | 20% | 75% (L5) | 55% |

**Average Gap: 76%** - This is massive and easily detectable!

---

## Why This Design Works

1. **Tool creates hard barrier:** You literally CANNOT answer "current stock price" without web access
2. **Specialization = tool selection:** The specialist's value is knowing WHICH tool to use
3. **Generalist fails predictably:** L0 agent will fail on all tool-gated tasks
4. **Measurable advantage:** 75%+ gap is statistically significant with just 10 samples

---

## Implementation Priority

### Phase 1: L1 Tasks (Code Execution)
- Easiest to implement (just need Python sandbox)
- Large accuracy gap expected
- Good proof of concept

### Phase 2: L3 Tasks (RAG)
- Simulated with pre-loaded documents
- Clear specialist advantage
- Practical for enterprise use cases

### Phase 3: L4 Tasks (Web)
- Requires API integration
- Most impressive demos
- Real-time accuracy verification

### Phase 4: L2 and L5
- L2 requires vision model integration
- L5 requires multi-agent framework
- More complex, defer to v2.1

---

## Key Insight

> **The original tasks tested "reasoning difficulty." The new tasks test "capability access." You can't reason your way to a current stock price - you need the tool.**

---

## Success Criteria

After implementing tool-gated tasks:

1. **L0 baseline accuracy < 20%** on all regimes
2. **Specialist accuracy > 75%** on matching regimes
3. **Specialist advantage > 50%** (vs previous 0%)
4. **Clear tool-task correlation** (L1 agent wins math, L4 agent wins realtime)

---

## Next Steps

1. [ ] Implement L1 tasks with Python sandbox
2. [ ] Implement L3 tasks with mock document retrieval
3. [ ] Create task verification (ensure tool-gating)
4. [ ] Re-run Phase 1 tests
5. [ ] Document results
