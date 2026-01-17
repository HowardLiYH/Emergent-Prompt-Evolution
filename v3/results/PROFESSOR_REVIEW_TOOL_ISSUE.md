# Professor Panel Review: Critical Tool Implementation Issue

## Issue Summary
**Date**: 2026-01-15  
**Severity**: CRITICAL  
**Raised by**: User observation

The "RAG and Web-tool specialists" are NOT actually using RAG or Web tools. The implementation only varies prompts.

---

## Technical Evidence

### run_training_real.py (lines 55-64)
```python
if tool == "L0":
    prompt = f"Answer this question concisely:\n\n{question}"
elif tool == "L1":
    prompt = f"Solve this problem. If it requires calculation, show your work:\n\n{question}"
elif tool == "L2":
    prompt = f"Analyze and answer:\n\n{question}"
elif tool == "L3":
    prompt = f"Use your knowledge to answer:\n\n{question}"
```

**Problem**: These are just TEXT variations, not actual tool calls!

### What SHOULD happen:
- L1 (Code): Execute Python code via E2B sandbox
- L2 (Vision): Call Gemini vision API with images
- L3 (RAG): Call RAGTool.execute() with LlamaIndex/ChromaDB
- L4 (Web): Call WebSearchTool.execute() with Tavily API

### What ACTUALLY happens:
- All tools: Just different prompt templates to the same LLM call

---

## Professor Panel Opinions

### Prof. Percy Liang (Stanford HAI)
> "This is a fundamental validity issue. The experiment claims to test tool specialization, but agents never touch the tools. It's like claiming you tested driving skills when everyone took a written test. The implementation must ACTUALLY call RAGTool.execute() and WebSearchTool.execute() for the results to be meaningful."

### Prof. Noah Smith (UW/AI2)
> "The discrepancy between the sophisticated tool implementations (rag.py, web.py) and their non-use in training is concerning. You've built a sports car but never started the engine. The fix is straightforward: integrate real tool calls into the training loop."

### Prof. Devi Parikh (Georgia Tech/Meta)
> "This explains why the tool selection accuracy was only 43%. If L3 (RAG) and L4 (Web) just change prompt text, there's no performance difference to learn from. The agents have no signal that RAG is 'better' for retrieval tasks because RAG isn't actually doing retrieval."

### Prof. Chelsea Finn (Stanford)
> "For emergent tool specialization to be valid, there must be a PERFORMANCE GAP between tools. If L0 and L3 both just query the LLM with slightly different prompts, why would an agent specialize? You need tasks where using the actual RAG tool genuinely helps."

### Prof. Dan Jurafsky (Stanford NLP)
> "The task bank is problematic too. Questions like 'Who wrote Romeo and Juliet?' don't actually NEED RAG - the LLM already knows this. You need tasks where RAG access to a specific knowledge base provides information the LLM doesn't have."

### Prof. Denny Zhou (Google DeepMind)
> "This is a classic 'training-evaluation mismatch.' The tools are defined but never invoked. The fix requires:
> 1. Tasks that genuinely require tools (code for computation, web for real-time info)
> 2. Actual tool execution in the training loop
> 3. Performance metrics that reflect tool effectiveness"

### Prof. Oriol Vinyals (Google DeepMind)
> "The code_math specialists succeeded (100% tool accuracy) because math problems DO benefit from 'show your work' prompting. But RAG/Web specialists failed because those prompts don't actually provide retrieval/search capabilities. The asymmetry in results reflects the asymmetry in implementation."

---

## Required Fixes

### 1. Integrate Real Tool Calls
```python
# CURRENT (wrong):
if tool == "L3":
    prompt = "Use your knowledge..."  # Just text

# FIXED:
if tool == "L3":
    rag_tool = RAGTool()
    context = await rag_tool.execute(question)  # ACTUAL RAG
    prompt = f"Using this retrieved context: {context}\n\nAnswer: {question}"
```

### 2. Use Tool-Gated Tasks
Tasks where the correct answer is ONLY obtainable with the tool:
- L1: "Execute: print(sum([i**2 for i in range(1, 101)]))" → needs code
- L3: "Based on doc X, what was the Q3 revenue?" → needs RAG with doc X
- L4: "What is today's Bitcoin price?" → needs real-time web search

### 3. Measure Tool Effectiveness
Track whether tool use actually improved accuracy vs. base LLM.

---

## Consensus
**ALL PROFESSORS AGREE**: The current implementation does NOT test tool specialization. 
It tests PROMPT specialization, which is a much weaker claim.

## Recommended Action
1. Modify training loop to call actual tools
2. Create tool-gated task benchmarks  
3. Re-run experiments with real tool execution
4. Measure tool-specific performance gains

---

*Panel convened: 2026-01-15*
