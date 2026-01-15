# V3 Changelog

## v3.0.0 - Complete Reimplementation (2026-01-15)

### Overview
Complete reimplementation of the Competitive Specialist Ecosystem (CSE) with:
- Real MCP tools (Gemini Code, Vision, LlamaIndex RAG, Tavily Web)
- Thompson Sampling for tool selection
- Fitness sharing for diversity
- 4-layer memory system
- Constitutional safety constraints

### Core Algorithm Components

#### Thompson Sampling (`core/thompson.py`)
- Beta distribution beliefs for each (regime, tool) pair
- Posterior updates on wins/losses
- Exploration-exploitation balance

#### Fitness Sharing (`core/fitness.py`)
- 1/sqrt(n) penalty for niche crowding
- Theorem 4 equilibrium distribution
- Equilibrium error computation

#### Competition Loop (`core/competition.py`)
- Subset selection (K=3 competitors)
- Epsilon exploration (10% random selection)
- Winner-only updates (anti-leakage by design)

#### Agent (`core/agent.py`)
- ModernSpecialist with MCP integration
- OBSERVE-RETRIEVE-REASON-ACT-REFLECT loop
- Specialty tracking via win distribution

### MCP Tools (`mcp/`, `tools/`)

| Level | Tool | Implementation |
|-------|------|----------------|
| L0 | Base LLM | Direct Gemini call |
| L1 | Code | Gemini native code execution |
| L2 | Vision | Gemini 2.5 vision API |
| L3 | RAG | LlamaIndex + ChromaDB + BGE |
| L4 | Web | Tavily search API |
| L5 | Orchestrator | LangGraph state machine |

### 4-Layer Memory System (`memory/`)

1. **Working Memory** - In-context for current task
2. **Episodic Memory** - Win-only episodes indexed by regime
3. **Semantic Memory** - Compressed patterns from episodes
4. **Procedural Memory** - Tool preferences and locked strategies

### Safety & Monitoring (`safety/`)

- Constitutional constraints for harmful content
- Collusion detection (alternating, round-robin patterns)
- Confidence calibration checks
- Alignment tax measurement
- Emergence monitoring

### Deployment Layer (`deploy/`)

- Task router trained from competition outcomes
- Specialist profile extraction
- Specialist caching with O(1) lookup

### Training Results

**First Run (8 agents, 100 generations, seed=42):**
- SCI: 0.905 (strong specialization)
- Coverage: 60% (3/5 regimes)
- All agents verified identical at start
- Specialization emerged through competition

**Key Findings:**
- Thompson Sampling learns appropriate tool preferences
- Agents with optimal tools win more competitions
- Fitness sharing prevents complete crowding (most of the time)

### Known Issues

1. **Theorem 4 Equilibrium Error**: High (57-82%) due to:
   - Not all regimes covered (vision, pure_qa missing)
   - Possible need for stronger fitness penalty
   - May need more generations for convergence

2. **Router Training**: Currently relies on competition history population
   which needs to be connected to the engine

### Next Steps

1. [ ] Load real benchmarks (LiveCodeBench, MMMU, GPQA, etc.)
2. [ ] Run L0 baseline verification
3. [ ] Run 10-seed statistical validation
4. [ ] Tune fitness sharing parameter (gamma)
5. [ ] Run practical value tests
6. [ ] Generate publication figures
