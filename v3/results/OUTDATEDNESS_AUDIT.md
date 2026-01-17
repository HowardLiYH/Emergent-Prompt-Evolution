# Outdatedness Audit Report

**Date**: 2026-01-15
**Purpose**: Verify all components are up-to-date and not using deprecated methods

---

## Executive Summary

| Component | Current Version | Latest | Status | Priority |
|-----------|-----------------|--------|--------|----------|
| LlamaIndex | 0.14.12 | 0.14.x | ✅ Up-to-date | - |
| ChromaDB | 1.4.1 | 1.4.x | ✅ Up-to-date | - |
| RAGAS | 0.4.3 | 0.4.x | ✅ Up-to-date | - |
| sentence-transformers | 5.2.0 | 5.x | ✅ Up-to-date | - |
| LangGraph | 1.0.6 | 1.0.x | ✅ Up-to-date | - |
| Tavily | 0.7.18 | 0.7.x | ✅ Up-to-date | - |
| **google-genai** | 1.56.0 | 1.56.0 | ✅ Up-to-date (MIGRATED) | FIXED |
| **BGE Embeddings** | bge-small-en-v1.5 | Outdated | 🟡 **SUBOPTIMAL** | MEDIUM |
| **Hybrid Search** | Not implemented | Recommended | 🟡 **MISSING** | MEDIUM |
| **Reranker** | Not implemented | Recommended | 🟡 **MISSING** | LOW |

---

## ✅ FIXED: Migrated from google-generativeai to google.genai

### Issue (RESOLVED)
The `google-generativeai` package was **officially deprecated** and replaced with `google.genai`.

### Migration Completed

**Old (deprecated)**:
```python
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')
```

**New (current)**:
```python
from google import genai
from google.genai import types
client = genai.Client(api_key=GEMINI_API_KEY)
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Your prompt here'
)
```

### Files Updated
- ✅ `v3/tools/rigorous_rag.py` - Migrated
- ✅ `v3/experiments/training/run_training_v2.py` - Migrated

### Status
- ✅ Using `google-genai 1.56.0` (latest)
- ✅ All code migrated to new API

---

## 🟡 SUBOPTIMAL: BGE Embeddings May Be Outdated

### Current
```python
embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')
# 384 dimensions, released 2023
```

### Better Alternatives (2025-2026)

| Model | Dimensions | Context | MTEB Rank | Notes |
|-------|------------|---------|-----------|-------|
| **Qwen3-Embedding-8B** | Flexible | 32K | #1 | Open-weight, multilingual |
| **text-embedding-3-large** | 3072 (flexible) | 8K | Top-5 | OpenAI, costs $ |
| **llama-embed-nemotron-8B** | - | - | SOTA | Open-source, Nov 2025 |
| **GTE-Qwen2** | 1024 | 8K | Top-5 | Alibaba, open-source |
| bge-small-en-v1.5 | 384 | 512 | ~Top-30 | **Currently using** |

### Recommendation
For research purposes, `bge-small-en-v1.5` is acceptable because:
1. It's widely cited and reproducible
2. Lower compute requirements
3. Well-documented

For production/better performance, consider:
- `GTE-Qwen2-1.5B-instruct` (good balance)
- `Qwen3-Embedding-8B` (best performance)

---

## 🟡 MISSING: Hybrid Search (Vector + Keyword)

### Current
Only vector search (embeddings)

### Recommended
Combine vector search with BM25/keyword search for:
- Better handling of rare terms/names
- Improved precision on exact matches
- Higher recall overall

### Implementation Options

```python
# Option 1: ChromaDB with full-text search (built-in)
# ChromaDB 1.4+ supports hybrid search

# Option 2: LlamaIndex Hybrid Retriever
from llama_index.core.retrievers import QueryFusionRetriever
```

### Priority
MEDIUM - Current vector-only approach works for research, but hybrid is production standard.

---

## 🟡 MISSING: Reranker

### Current
No reranking step after retrieval

### Recommended
Add a cross-encoder reranker to improve precision:

```python
# Example with sentence-transformers
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Rerank retrieved documents
scores = reranker.predict([(query, doc) for doc in retrieved_docs])
```

### Better Options (2025)
- `Qwen3-Reranker` - Best open-source
- `bge-reranker-v2` - Good balance
- Cohere Rerank - Commercial, very good

### Priority
LOW - Reranking improves precision but adds latency. For research, current approach is acceptable.

---

## ✅ Up-to-Date Components

### LlamaIndex (0.14.12)
- Newer than 0.11 mentioned in docs
- Uses new Workflows API
- Property Graph Index supported
- Pydantic v2 compatible

### ChromaDB (1.4.1)
- Newer than 1.0.15
- Supports vector + metadata search
- Persistent storage working

### RAGAS (0.4.3)
- Latest version
- Supports all required metrics

### LangGraph (1.0.6)
- Latest stable version
- Proper agent orchestration

---

## Action Items

### ✅ HIGH Priority (FIXED)

1. **Migrate from google-generativeai to google.genai** ✅ DONE
   - All files updated to use `from google import genai`
   - Using `google.genai 1.56.0`

### MEDIUM Priority (Nice to Have for Production)

2. **Consider upgrading embedding model**
   - Current: `bge-small-en-v1.5` (384-dim)
   - Better: `GTE-Qwen2-1.5B-instruct` (1024-dim)
   - Note: For research, current is acceptable

3. **Add hybrid search**
   - Enable ChromaDB full-text search
   - Or use LlamaIndex QueryFusionRetriever
   - Note: Vector-only is sufficient for research

### LOW Priority (Optional)

4. **Add reranker for better precision**
   - `bge-reranker-v2` or `cross-encoder/ms-marco-MiniLM-L-6-v2`

5. **Consider long-context embeddings**
   - For documents > 512 tokens

---

## Conclusion

| Category | Status |
|----------|--------|
| Core packages | ✅ Up-to-date |
| LLM integration | ✅ Using google.genai 1.56.0 (FIXED) |
| Embedding model | 🟡 Functional (bge-small acceptable for research) |
| Search strategy | 🟡 Vector-only (acceptable for research) |
| Reranking | 🟡 Not implemented (optional) |

**Overall**: The system is **fully up-to-date** for research purposes. All critical issues fixed.

For production deployment, consider:
- Upgrading to `Qwen3-Embedding-8B` or `GTE-Qwen2`
- Adding hybrid search (vector + BM25)
- Adding reranker for precision

---

*Audit completed: 2026-01-15*
