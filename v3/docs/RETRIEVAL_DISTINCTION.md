# L3 (RAG) vs L4 (External) Retrieval Distinction

## Overview

This document explains the critical distinction between our two retrieval-based tool levels:

| Level | Name | Corpus Type | Content Type | Use Case |
|-------|------|-------------|--------------|----------|
| **L3** | RAG (Internal Retrieval) | Internal knowledge base | Policies, manuals, structured docs | Domain-specific Q&A |
| **L4** | External Retrieval | External news corpus | News articles, public facts | General knowledge lookup |

## Why Two Retrieval Levels?

While both L3 and L4 involve corpus retrieval, they test **different skills**:

### L3: RAG (Internal Knowledge Base)
- **Corpus**: Company policies, employee handbooks, technical specifications
- **Content**: Structured, domain-specific, authoritative
- **Skill tested**: Finding specific information in curated documentation
- **Example tasks**:
  - "What is the company vacation policy?"
  - "According to the handbook, what is the work-from-home procedure?"
  - "What are the security specifications for the new system?"

### L4: External Retrieval (News Corpus)
- **Corpus**: Synthetic news articles, public announcements, general facts
- **Content**: Unstructured, broad coverage, varied sources
- **Skill tested**: Synthesizing information from external sources
- **Example tasks**:
  - "What were Zephyrix Technologies Q4 2025 earnings?"
  - "Who won the 2025 Global Hockey League Championship?"
  - "How much did Nexus Robotics raise in their IPO?"

## Why Synthetic Corpus (Not Live Web)?

We use a **synthetic news corpus** instead of live web search for several critical reasons:

### 1. Reproducibility
Live web search results change constantly. Rankings shift, pages update, links break. A synthetic corpus ensures:
- Same results every run
- Comparable experiments across time
- No network variability confounds

### 2. Verifiable Ground Truth
With synthetic facts about fictional entities:
- We **know** the correct answer (we created it)
- LLM **cannot** hallucinate correct answers
- Verification is **exact match** (no ambiguity)

### 3. Zero Contamination
Fictional entities (Zephyrix Technologies, Novastrand, etc.) cannot exist in:
- LLM training data
- LLM's parametric memory
- Any prior fine-tuning

This guarantees the gap we measure is **real tool advantage**, not memorization.

### 4. Safety & Control
- No risk of accessing harmful content
- No API rate limits or costs
- Fully interpretable retrieval process

## Methodology: NeoQA-Style Synthetic Facts

Our approach follows the NeoQA methodology (Arxiv 2505.05949):

1. **Create fictional entities**: Companies, people, events that don't exist
2. **Write realistic news articles** about these entities
3. **Extract verifiable Q&A pairs** from the articles
4. **Index in searchable corpus** accessible only via L4 tool
5. **Verify gap**: L4 should achieve ~100%, L0 should achieve ~0%

## Empirical Validation

Our Phase 0 audit verifies:
- L4 (External Retrieval): 100% accuracy on synthetic tasks
- L0 (No tools): ~10% accuracy (random guessing on numbers)
- **Gap: 90%** - proves retrieval is REQUIRED

## Limitations & Transparency

This approach does **NOT** test:
- Real web crawling or ranking
- Robustness to noisy web pages
- Freshness of live data
- Multi-hop web navigation

We are transparent about this in our paper:
> "External retrieval tasks use a controlled synthetic corpus for reproducibility. This tests tool routing and retrieval mechanics, not live web search competence."

## Related Benchmarks

Our approach aligns with established methodologies:
- **NeoQA**: Synthetic news for contamination-free evaluation
- **WebArena**: Sandboxed web environment for reproducibility
- **KILT**: Versioned knowledge bases for controlled retrieval
- **FEVER**: Evidence-bound verification with fixed snapshots

## Conclusion

L3 (RAG) and L4 (External) are both retrieval-based but serve different purposes:
- L3 tests **internal knowledge base** retrieval (structured, domain-specific)
- L4 tests **external corpus** retrieval (unstructured, general knowledge)

Using synthetic facts for L4 ensures reproducibility, verifiability, and zero contamination while maintaining the essential distinction between internal and external information access.
