# V3 Comprehensive Audit Report

**Date**: 2026-01-15  
**Auditor**: Automated Code Review  
**Scope**: All v3/ files for data simulation, improper methods, and cheating

---

## Executive Summary

**CRITICAL FINDINGS**: 2 major issues identified that invalidate current results.

| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 CRITICAL | 2 | Training bypasses real tools; Tasks too simple |
| 🟡 WARNING | 3 | Missing components for full functionality |
| 🟢 OK | 15+ | Properly implemented components |

---

## 🔴 CRITICAL ISSUES

### Issue 1: Training Script Bypasses Real Tools

**File**: `experiments/training/run_training_real.py`  
**Lines**: 55-64, 289-297

**Problem**: The `RealLLMClient` class only varies PROMPT TEXT, never calls actual tools.

```python
# CURRENT CODE (WRONG):
class RealLLMClient:
    async def evaluate_task(self, question, expected_answer, tool, ...):
        # Line 55-64: Tool selection only changes prompt text!
        if tool == "L0":
            prompt = f"Answer this question concisely:\n\n{question}"
        elif tool == "L1":
            prompt = f"Solve this problem. If it requires calculation..."
        elif tool == "L3":
            prompt = f"Use your knowledge to answer:\n\n{question}"
        # ...
        response = model.generate_content(prompt)  # Same LLM call for all tools!
```

**Impact**:
- RAGTool.execute() is NEVER called
- WebSearchTool.execute() is NEVER called  
- CodeExecutionTool.execute() is NEVER called
- All "tool specialization" results are INVALID

**Evidence**: The MCP infrastructure exists and works:
- `mcp/server.py` line 100-151: `call_tool()` properly routes to real tools
- `core/competition.py` line 137: `await agent.solve_with_tool(task, tool, mcp_tools)`
- BUT `run_training_real.py` doesn't use `MCPEnabledAgent` or pass `mcp_tools`!

**Fix Required**:
```python
# Use MCPEnabledAgent instead of raw agents
from mcp.client import wrap_population_with_mcp
mcp_population = wrap_population_with_mcp(population)

# OR pass mcp_tools to competition engine
mcp_server = get_mcp_server()
result = await engine.run_competition_round(task, mcp_tools={'L1': ..., 'L3': ...})
```

---

### Issue 2: Task Bank is Too Simple (Ceiling Effect)

**File**: `experiments/training/run_training_real.py`  
**Lines**: 232-263

**Problem**: Tasks can be solved by base LLM without any tools.

```python
TASK_BANK = {
    'code_math': [
        ('What is 17 * 23?', '391'),           # LLM can do this mentally
        ('What is the sum of first 10 natural numbers?', '55'),  # LLM knows this
    ],
    'rag': [
        ('Who wrote Romeo and Juliet?', 'Shakespeare'),  # LLM already knows
        ('What is the capital of Japan?', 'Tokyo'),      # LLM already knows
    ],
    'web': [
        ('What is the largest planet in our solar system?', 'Jupiter'),  # Static knowledge
    ],
}
```

**Impact**:
- No performance difference between tools (ceiling effect)
- Agents have no signal to learn correct tool preferences
- Test 2 showed only 43% tool selection accuracy because tools don't matter

**Fix Required**: Tool-gated tasks where ONLY the correct tool can succeed:

| Tool | Task Type | Example | Why Tool Required |
|------|-----------|---------|-------------------|
| L1 | Code execution | `sum([i**3 for i in range(1, 1001)])` | LLM can't compute 1000 cubes |
| L2 | Vision | Upload real image, ask about it | LLM can't see images |
| L3 | Private RAG | "Based on document X, what was Q3 revenue?" | LLM doesn't have doc X |
| L4 | Real-time | "What is the current BTC price?" | LLM has stale knowledge |

---

## 🟡 WARNING ISSUES

### Issue 3: Vision Tasks Have No Images

**File**: `experiments/training/run_training_real.py`

**Problem**: Vision tasks are text-only questions about hypothetical images.

```python
'vision': [
    ('What shape has 4 equal sides and 4 right angles?', 'square'),  # No image!
]
```

**Impact**: Vision tool is never actually used with images.

**Fix Required**: Include actual image files or URLs in vision tasks.

---

### Issue 4: RAG Has No Documents Indexed

**File**: `tools/rag.py`

**Problem**: RAGTool creates an empty ChromaDB collection. No documents are added.

```python
self._index = VectorStoreIndex(
    [],  # Empty document list!
    storage_context=storage_context
)
```

**Impact**: RAG queries return nothing because there's nothing to retrieve.

**Fix Required**: Index documents before training:
```python
rag_tool = RAGTool()
await rag_tool.add_documents([...knowledge base...])
```

---

### Issue 5: Competition Engine Has mcp_tools But Training Doesn't Pass It

**File**: `core/competition.py` vs `experiments/training/run_training_real.py`

**Problem**: CompetitionEngine supports `mcp_tools` parameter but training script doesn't use it.

```python
# competition.py line 137 (CORRECT infrastructure):
response = await agent.solve_with_tool(task, tool, mcp_tools)

# run_training_real.py (BYPASSES this):
response, correct, confidence = await llm.evaluate_task(...)  # No mcp_tools!
```

---

## 🟢 PROPERLY IMPLEMENTED (No Issues)

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Thompson Sampling | `core/thompson.py` | ✅ | Beta distributions, proper updates |
| Fitness Sharing | `core/fitness.py` | ✅ | 1/sqrt(n) penalty implemented |
| Beta Distribution | `core/thompson.py:14-40` | ✅ | sample(), update_success/failure() |
| Tool Selection | `core/thompson.py:70-99` | ✅ | Proper Thompson Sampling |
| Competition Engine | `core/competition.py` | ✅ | Subset K=3, epsilon exploration |
| Winner-Only Memory | `core/agent.py:142-167` | ✅ | Losers don't update memory |
| Episodic Memory | `memory/episodic.py` | ✅ | Sliding window, wins-only |
| MCP Server | `mcp/server.py` | ✅ | Proper tool routing |
| MCP Client | `mcp/client.py` | ✅ | MCPEnabledAgent wrapper |
| CodeExecutionTool | `tools/code.py` | ✅ | Gemini code execution API |
| VisionTool | `tools/vision.py` | ✅ | Gemini vision API |
| RAGTool | `tools/rag.py` | ✅ | LlamaIndex + ChromaDB |
| WebSearchTool | `tools/web.py` | ✅ | Tavily API with rate limiting |
| Regime Sampling | `core/regimes.py` | ✅ | Non-uniform frequencies |
| Agent State | `core/agent.py` | ✅ | Proper serialization |

---

## Root Cause Analysis

The infrastructure for real tool execution EXISTS and is properly implemented.
The bug is that `run_training_real.py` was written as a SHORTCUT that bypasses everything.

**Timeline**:
1. MCP tools implemented correctly (tools/*.py) ✓
2. MCP server implemented correctly (mcp/server.py) ✓
3. Competition engine supports mcp_tools (core/competition.py) ✓
4. BUT training script was written to "simulate" for quick testing
5. Simulation was never replaced with real tool calls

---

## Required Fixes (Priority Order)

### Priority 1: Integrate Real Tools in Training

```python
# In run_training_real.py, REPLACE:
results = []
for agent in competitors:
    tool = agent.beliefs.select(regime, available_tools)
    response, correct, confidence = await llm.evaluate_task(...)

# WITH:
from mcp.client import MCPEnabledAgent
from mcp.server import get_mcp_server

mcp_server = get_mcp_server()
for agent in competitors:
    mcp_agent = MCPEnabledAgent(agent, mcp_server)
    result = await mcp_agent.solve_with_tool(task, agent.select_tool(regime))
    correct = evaluate(result['answer'], task['answer'])
```

### Priority 2: Create Tool-Gated Tasks

```python
TOOL_GATED_TASKS = {
    'code_math': [
        {
            'question': 'Execute: sum([i**3 for i in range(1, 1001)])',
            'answer': '250500250000',  # Must execute code to get this
            'requires_tool': 'L1'
        },
    ],
    'web': [
        {
            'question': 'What is the current Bitcoin price in USD?',
            'answer': None,  # Dynamic - must use web search
            'requires_tool': 'L4'
        },
    ],
    'rag': [
        {
            'question': 'Based on the indexed documents, what was the Q3 2024 revenue?',
            'answer': None,  # Must retrieve from indexed docs
            'requires_tool': 'L3',
            'setup': 'index_financial_docs()'  # Pre-index documents
        },
    ],
}
```

### Priority 3: Index Documents for RAG

```python
# Before training:
rag_tool = RAGTool()
await rag_tool.add_documents([
    "Q3 2024 Financial Report: Revenue was $45.2 billion...",
    "Product Documentation: The API supports...",
    # ... more documents
])
```

### Priority 4: Add Real Images for Vision

```python
# Include actual image paths or URLs:
'vision': [
    {
        'question': 'How many people are in this image?',
        'image': 'data/test_images/crowd.jpg',  # Real image
        'answer': '12',
    },
]
```

---

## Verification Checklist

After fixes, verify:

- [ ] `CodeExecutionTool.execute()` called (check logs)
- [ ] `RAGTool.execute()` called (check logs)
- [ ] `WebSearchTool.execute()` called (check logs)
- [ ] Different tools produce different outputs for same question
- [ ] Tool-specific accuracy > base LLM by >30%
- [ ] Specialists prefer correct tools for their regimes

---

*Audit completed: 2026-01-15*
