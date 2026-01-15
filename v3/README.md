# Emergent Prompt Evolution V3

## Complete Reimplementation with Real Tools

This version implements the Competitive Specialist Ecosystem (CSE) with:
- **Real MCP Tools**: Gemini Code Execution, Vision, LlamaIndex RAG, Tavily Web Search
- **Thompson Sampling**: For tool selection with Beta distributions
- **Fitness Sharing**: 1/sqrt(n) penalty for niche crowding
- **4-Layer Memory**: Working → Episodic → Semantic → Procedural
- **Modern Benchmarks**: LiveCodeBench, MMMU, GPQA, RealTimeQA, GAIA (2024-2025)

## Directory Structure

```
v3/
├── mcp/                    # Model Context Protocol
│   ├── server.py          # MCPToolServer
│   ├── client.py          # MCPEnabledAgent
│   └── schemas.py         # Tool definitions
├── tools/                  # Real Tool Implementations
│   ├── code.py            # L1: Gemini Code Execution
│   ├── vision.py          # L2: Gemini Vision
│   ├── rag.py             # L3: LlamaIndex + ChromaDB
│   ├── web.py             # L4: Tavily Search
│   └── orchestrator.py    # L5: LangGraph
├── core/                   # CSE Algorithm
│   ├── thompson.py        # Thompson Sampling
│   ├── fitness.py         # Fitness Sharing
│   ├── competition.py     # Competition Loop
│   ├── regimes.py         # Non-uniform regime sampling
│   └── agent.py           # ModernSpecialist
├── memory/                 # 4-Layer Memory System
│   ├── working.py         # In-context memory
│   ├── episodic.py        # Episode storage (wins only)
│   ├── semantic.py        # Compressed patterns
│   ├── procedural.py      # Strategy levels / tools
│   ├── consolidator.py    # Episode → Semantic compression
│   └── changelog.py       # Behavioral changelog + compaction
├── safety/                 # Safety & Monitoring
│   ├── constitutional.py  # Competition constraints
│   ├── collusion.py       # Collusion detection
│   ├── calibration.py     # Confidence calibration
│   ├── alignment_tax.py   # Safety cost measurement
│   └── monitor.py         # Emergent behavior alerts
├── deploy/                 # Deployment Layer
│   ├── router.py          # Task routing classifier
│   ├── profiles.py        # Specialist profile extraction
│   └── cache.py           # Specialist caching
├── benchmarks/             # Modern Benchmarks (2024-2025)
│   ├── livecodebench.py   # L1 Code (updated weekly)
│   ├── mmmu.py            # L2 Vision
│   ├── gpqa.py            # L3 RAG
│   ├── realtimeqa.py      # L4 Web (weekly updated)
│   ├── gaia.py            # All levels
│   └── mcpmark.py         # MCP benchmark
├── experiments/
│   ├── training/          # Competition training
│   ├── evaluation/        # Benchmark evaluation
│   ├── practical_value/   # 19 practical tests
│   ├── scaling/           # N=4,8,16,32,64
│   └── ablations/         # Component ablations
└── results/               # Experimental results
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys in `.env`:
```
GEMINI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
E2B_API_KEY=your_key_here
```

3. Run training:
```bash
python -m experiments.training.run_training
```

## Key Algorithms

### Thompson Sampling for Tool Selection
Agents maintain Beta distribution beliefs over tool effectiveness per regime.
Selection samples from beliefs; updates are wins-only.

### Fitness Sharing
Penalty = 1/sqrt(n) where n = specialists in regime.
Prevents crowding, guarantees coverage.

### Competition Loop (Subset K=3)
- Sample regime according to non-uniform frequency
- Select K=3 competitors (with epsilon exploration)
- Each uses Thompson Sampling to select tool
- Winner updates beliefs and memory
- Losers update beliefs only (failure signal)

### Equilibrium Distribution (Theorem 4)
n_r ∝ (f_r × R_r × D_r)^(2/3)
where f=frequency, R=reward, D=difficulty

## Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Baseline matches published | ±5% |
| Agents start identical | 100% |
| Specialization emerges | SCI > 0.75 |
| Real tool advantage | >25% gap |
| Equilibrium error | <20% |
| Specialist > Generalist | >10% |
| Router accuracy | >80% |
| Sublinear scaling | b < 0.8 |
