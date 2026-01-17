# Professor Panel Review: Critical V3 Issues

**Date**: 2026-01-15  
**Context**: Comprehensive audit revealed 2 critical issues invalidating all results.

---

## Issues Under Review

### Issue 1: Training Bypasses Real Tools
- `run_training_real.py` only varies prompt text
- RAGTool, WebSearchTool, CodeExecutionTool NEVER called
- MCP infrastructure exists but is bypassed

### Issue 2: Tasks Too Simple (Ceiling Effect)  
- Tasks solvable by base LLM without tools
- No performance differentiation between L0-L5
- Agents have no signal to learn correct tools

---

## Professor Suggestions

### 1. Prof. Percy Liang (Stanford HAI)

> "This is a classic case of 'building the engine but never starting it.' The fix is straightforward:
> 
> **Suggestion**: Create a `run_training_mcp.py` that uses the existing `MCPEnabledAgent`:
> ```python
> from mcp.client import MCPEnabledAgent, wrap_population_with_mcp
> mcp_population = wrap_population_with_mcp(population)
> for agent in mcp_population:
>     result = await agent.solve_with_tool(task, selected_tool)
> ```
> The infrastructure is already there—just wire it up."

---

### 2. Prof. Christopher Manning (Stanford NLP)

> "The task design is fundamentally flawed. For meaningful tool specialization:
>
> **Suggestion**: Implement **capability-gated tasks**:
> - L1 (Code): Problems requiring >10 arithmetic operations (e.g., `sum([i**3 for i in range(1000)])`)
> - L3 (RAG): Questions about private documents the LLM has never seen
> - L4 (Web): Real-time queries (stock prices, weather, news from today)
> 
> If the LLM can answer without the tool, the task is invalid."

---

### 3. Prof. Chelsea Finn (Stanford)

> "The meta-learning perspective reveals why this failed:
>
> **Suggestion**: Implement a **tool effectiveness validator**:
> ```python
> def validate_task(task, regime):
>     l0_acc = evaluate_with_tool(task, 'L0')  # Base LLM
>     optimal_acc = evaluate_with_tool(task, optimal_tool[regime])
>     assert optimal_acc > l0_acc + 0.3, 'Task not tool-gated!'
> ```
> Run this on every task before including it in training."

---

### 4. Prof. Dan Jurafsky (Stanford NLP)

> "The linguistic design of tasks matters enormously:
>
> **Suggestion**: For each tool level, create tasks with **explicit tool requirements**:
> - L1: 'Execute this Python code: ...'
> - L2: 'Looking at the attached image, describe...'
> - L3: 'According to the document in context, what...'
> - L4: 'Search the web for the current...'
> 
> Make the tool requirement explicit in the task phrasing."

---

### 5. Prof. Devi Parikh (Georgia Tech/Meta)

> "For vision (L2), you need ACTUAL images:
>
> **Suggestion**: 
> 1. Download test images from MMMU or create synthetic charts
> 2. Store in `v3/data/images/`
> 3. Tasks should reference real files:
> ```python
> {'question': 'Count the people in this image', 
>  'image': 'data/images/crowd_001.jpg',
>  'answer': '12'}
> ```
> Without real images, vision specialists can never emerge."

---

### 6. Prof. Pieter Abbeel (UC Berkeley)

> "From an RL perspective, the reward signal is corrupted:
>
> **Suggestion**: Implement **tool-specific reward shaping**:
> ```python
> def compute_reward(correct, tool_used, optimal_tool):
>     base_reward = 1.0 if correct else 0.0
>     tool_bonus = 0.2 if tool_used == optimal_tool else 0.0
>     return base_reward + tool_bonus
> ```
> This gives agents explicit signal about tool choice quality."

---

### 7. Prof. Yann LeCun (Meta/NYU)

> "The architecture is sound; the training data is the problem:
>
> **Suggestion**: Create a **benchmark validation suite**:
> 1. For each regime, sample 10 tasks
> 2. Run with L0 (base) and optimal tool
> 3. Require >30% accuracy gap to include tasks
> 4. Log tool call traces to verify actual execution
> 
> This ensures data quality before training."

---

### 8. Prof. Sergey Levine (UC Berkeley)

> "The offline RL perspective suggests:
>
> **Suggestion**: Add **tool call logging and verification**:
> ```python
> class ToolCallLogger:
>     def log(self, tool, input, output, latency):
>         self.calls.append({...})
>     
>     def verify_real_execution(self):
>         for call in self.calls:
>             assert call['output'] != call['input']  # Not just echo
>             assert call['latency'] > 10  # Real API calls take time
> ```
> If you can't prove tools were called, you can't claim tool specialization."

---

### 9. Prof. Fei-Fei Li (Stanford)

> "For vision tasks specifically:
>
> **Suggestion**: Use existing vision benchmarks:
> 1. Download MMMU sample images
> 2. Use ChartQA for data visualization tasks
> 3. Use DocVQA for document understanding
> 
> These have ground truth that REQUIRES vision to solve."

---

### 10. Prof. Denny Zhou (Google DeepMind)

> "The fix should be systematic:
>
> **Suggestion**: Implement a **three-phase validation**:
> 1. **Tool Verification**: Run each tool standalone, verify it works
> 2. **Task Verification**: Confirm tasks are tool-gated (L0 fails, Lk succeeds)
> 3. **Training Verification**: Log every tool call, verify real execution
> 
> No training should proceed until all three pass."

---

### 11. Prof. Oriol Vinyals (Google DeepMind)

> "The reason code_math specialists succeeded is instructive:
>
> **Suggestion**: For L1 (Code), use tasks that REQUIRE execution:
> ```python
> # Good: Must execute
> 'sum([i**3 for i in range(1, 10001)])'  # Answer: 2500500025000000
> 
> # Bad: LLM can reason
> 'What is 17 * 23?'  # LLM knows: 391
> ```
> The task should exceed LLM's mental computation capacity."

---

### 12. Prof. Ilya Sutskever (OpenAI/SSI)

> "The fundamental issue is training-evaluation mismatch:
>
> **Suggestion**: Implement **identical evaluation and training paths**:
> ```python
> # WRONG: Different code paths
> training: llm.generate(prompt)
> evaluation: mcp_tool.execute(query)
> 
> # CORRECT: Same code path
> training: mcp_agent.solve_with_tool(task, tool)
> evaluation: mcp_agent.solve_with_tool(task, tool)
> ```
> Never have different code for training vs inference."

---

### 13. Prof. John Schulman (OpenAI/Anthropic)

> "From a policy gradient perspective:
>
> **Suggestion**: The gradient signal for tool selection requires:
> 1. **Counterfactual comparison**: What if different tool was used?
> 2. **Clear performance gap**: Tool choice must affect outcome
> 
> Implement:
> ```python
> # After each task, compute counterfactual
> actual_result = solve_with_tool(task, selected_tool)
> counterfactual = solve_with_tool(task, 'L0')
> tool_advantage = actual_result.score - counterfactual.score
> ```
> This gives agents clear signal about tool value."

---

### 14. Prof. Yoshua Bengio (Mila)

> "The causal structure is broken:
>
> **Suggestion**: Implement **tool intervention tests**:
> 1. Run task with correct tool → measure accuracy
> 2. Run same task with tool disabled → measure drop
> 3. The delta proves tool causation
> 
> If delta < 20%, the task doesn't require the tool."

---

### 15. Prof. Jan Leike (Anthropic)

> "From a safety perspective, this audit is valuable:
>
> **Suggestion**: Add **execution traces** for auditability:
> ```python
> @trace_execution
> async def solve_with_tool(task, tool):
>     # Traces: timestamp, tool_name, input, output, latency
>     ...
> ```
> This not only proves real execution but enables debugging and safety monitoring."

---

### 16. Prof. Dawn Song (UC Berkeley)

> "Security-conscious implementation:
>
> **Suggestion**: For web search (L4), implement **verifiable external calls**:
> ```python
> async def web_search(query):
>     result = await tavily.search(query)
>     # Log the actual HTTP request/response for verification
>     log_external_call(request=query, response=result, api='tavily')
>     return result
> ```
> This creates an audit trail proving real API calls."

---

### 17. Prof. Jason Wei (OpenAI)

> "Chain-of-thought reveals the issue:
>
> **Suggestion**: Require **tool-use reasoning** in responses:
> ```python
> prompt = '''
> Task: {question}
> 
> First, explain which tool you need and why.
> Then, show the tool invocation.
> Finally, give your answer based on the tool output.
> '''
> ```
> This makes tool use explicit and verifiable."

---

### 18. Prof. Samy Bengio (Apple)

> "The evaluation protocol needs fixing:
>
> **Suggestion**: Implement **tool-stratified evaluation**:
> ```python
> for tool in ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']:
>     acc = evaluate_population(tasks[tool], forced_tool=tool)
>     print(f'{tool}: {acc}')
> ```
> This reveals whether tools actually provide value."

---

### 19. Prof. Noam Brown (Meta FAIR)

> "From a game-theoretic view:
>
> **Suggestion**: The payoff matrix must reflect real tool value:
> 
> | Task Type | L0 | Correct Tool |
> |-----------|-----|--------------|
> | code_math | 20% | 95% |
> | rag | 30% | 85% |
> | web | 5% | 90% |
> 
> If L0 performs similarly to specialized tools, there's no game to play.
> **Create tasks where the payoff difference is >50%.**"

---

## Consensus Recommendations

All 19 professors agree on these critical fixes:

### Priority 1: Wire Up MCP Tools (1 day)
```python
# Replace RealLLMClient with MCPEnabledAgent
from mcp.client import wrap_population_with_mcp
mcp_population = wrap_population_with_mcp(population)
```

### Priority 2: Create Tool-Gated Tasks (2 days)
- L1: Computation beyond LLM capacity (1000+ operations)
- L2: Real images from MMMU/ChartQA
- L3: Private documents indexed in ChromaDB
- L4: Real-time queries (current prices, today's news)

### Priority 3: Add Verification Layer (1 day)
```python
class ExecutionVerifier:
    def verify_tool_called(self, traces):
        assert len(traces) > 0, "No tool calls recorded"
        for trace in traces:
            assert trace.latency_ms > 10, "Suspiciously fast"
            assert trace.output != trace.input, "Echo detected"
```

### Priority 4: Index RAG Documents (0.5 day)
```python
rag_tool = RAGTool()
await rag_tool.add_documents([...private knowledge...])
```

### Priority 5: Re-run All Experiments (3 days)
With verified real tool execution and tool-gated tasks.

---

## Estimated Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Wire MCP tools into training | 1 day |
| 2 | Create tool-gated task bank | 2 days |
| 3 | Add verification/logging | 1 day |
| 4 | Index RAG docs, add images | 0.5 day |
| 5 | Re-run experiments | 3 days |
| 6 | Validate results | 1 day |
| **Total** | | **~8.5 days** |

---

*Panel convened: 2026-01-15*
