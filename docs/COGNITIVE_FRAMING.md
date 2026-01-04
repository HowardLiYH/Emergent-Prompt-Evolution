# Cognitive Science Framing Clarification

## Purpose

This document addresses concerns raised about the cognitive science framing in the paper. We acknowledge that applying human cognitive science findings to LLMs requires careful interpretation.

---

## The Issue

The original paper cited cognitive science studies (Treiman & Zukowski, 1991; Bradley & Bryant, 1983; etc.) as justification for the synthetic rule design. Reviewers correctly noted that:

1. **Category Error**: Human cognitive mechanisms don't directly apply to LLMs
2. **Mechanistic Claims**: We cannot claim LLMs process rules the same way humans do
3. **Interpretability vs Mechanism**: Using cognitive categories for interpretability ≠ claiming shared mechanism

---

## Our Position: Cognitive Inspiration, Not Mechanism

We adopt **Option 2: Honest Acknowledgment**

### What We Claim

> "We use cognitively-inspired rule categories for **interpretability and experimental design**, not claiming mechanistic similarity between LLM and human processing."

### What We Don't Claim

- LLMs have phonemic awareness like humans
- LLMs process animacy the same way as the human ventral stream
- The cognitive science citations prove anything about LLM internals

---

## Revised Framing for Paper

### Original (Problematic)

> "The VOWEL_START rule tests phonemic awareness (Treiman & Zukowski, 1991), a cognitive process that develops in early childhood..."

### Revised (Acceptable)

> "The VOWEL_START rule requires identifying words beginning with vowels. While this task relates to phonemic awareness in human cognition (Treiman & Zukowski, 1991), we use it here as a well-defined, unambiguous selection criterion. We make no claims about mechanistic similarity between LLM and human processing of this rule."

---

## Rule Categories: Revised Descriptions

### Category 1: Purely Arbitrary Rules

**Rules**: POSITION, PATTERN, MATH_MOD

**Description**:
> "These rules are deliberately arbitrary and cannot be solved using prior knowledge. They test rule-following ability independent of any domain knowledge."

**No cognitive science claims needed** - these are purely synthetic.

### Category 2: Linguistically-Defined Rules

**Rules**: RHYME, VOWEL_START, ALPHABET

**Original (Problematic)**:
> "These rules leverage phonological processing (Bradley & Bryant, 1983)..."

**Revised**:
> "These rules use well-defined linguistic properties (rhyming, initial phoneme, alphabetical order) that provide clear, unambiguous selection criteria. The choice of these properties was inspired by their use in cognitive science studies of reading development, though we make no claims about mechanistic similarity."

### Category 3: Semantic-Category Rules

**Rules**: ANIMATE, INVERSE

**Original (Problematic)**:
> "ANIMATE tests category-specific processing (Warrington & Shallice, 1984)..."

**Revised**:
> "ANIMATE uses the living/non-living distinction, a category boundary that is well-represented in LLM training data. We chose this category for its unambiguity, not to claim that LLMs process animacy through dedicated neural pathways as suggested in human neuropsychology studies."

---

## Why Cognitive Inspiration is Still Valuable

Even without mechanistic claims, cognitive science provides:

1. **Well-Studied Task Categories**: Decades of research on what makes tasks easy/hard
2. **Unambiguous Criteria**: Categories like "living vs non-living" are well-defined
3. **Interpretability**: Easier to explain rules to reviewers and readers
4. **Ecological Validity**: Rules that matter to humans may matter for useful LLM applications

---

## Evidence That LLMs ≠ Humans on These Tasks

We can actually **strengthen** our paper by showing LLMs differ from humans:

### Human vs LLM Error Patterns

| Rule | Human Error Type | LLM Error Type |
|------|-----------------|----------------|
| RHYME | Phonological confusion | Semantic confusion |
| ANIMATE | Borderline cases (virus?) | Training data frequency |
| POSITION | Primacy/recency effects | Random |

If LLMs made the same errors as humans, that would suggest shared mechanisms. Different error patterns support our claim that we're using cognitive categories for **task design**, not mechanistic modeling.

---

## Suggested Paper Text

### Section 2.2: Synthetic Rule Design

> **Cognitive-Inspired Design**: We designed 8 synthetic rule domains inspired by cognitive science literature, though we emphasize that our use of these categories is for **interpretability and experimental design** rather than claiming mechanistic similarity between LLM and human cognition.
>
> The rules fall into three categories based on their reliance on prior knowledge:
>
> 1. **Purely Arbitrary**: POSITION, PATTERN, MATH_MOD – synthetic rules with no prior knowledge advantage
> 2. **Linguistically-Defined**: RHYME, VOWEL_START, ALPHABET – use clear linguistic properties
> 3. **Semantic-Category**: ANIMATE, INVERSE – leverage well-defined category boundaries
>
> This categorization allows us to analyze whether the specialization mechanism operates differently across rule types (see Section 5.3).

### Limitations Section

> **Cognitive Science Framing**: Our rule categories are inspired by cognitive science literature but do not imply mechanistic similarity between LLM and human processing. The cognitive science citations provide context for why these task categories are well-studied and unambiguous, not evidence for shared computational mechanisms.

---

## Summary

| Aspect | Original | Revised |
|--------|----------|---------|
| Citation purpose | Mechanistic justification | Interpretability & design |
| Claims about LLMs | Implicit similarity | Explicit disclaimer |
| Category usage | Assumes shared mechanism | Tool for experimental design |
| Error analysis | Not discussed | Can show differences |

This revised framing is more intellectually honest and actually strengthens the paper by being precise about what we can and cannot claim.

---

## References

- Treiman, R., & Zukowski, A. (1991). Levels of phonological awareness. *Phonological processes in literacy*, 67-83.
- Bradley, L., & Bryant, P. E. (1983). Categorizing sounds and learning to read. *Nature*, 301(5899), 419-421.
- Warrington, E. K., & Shallice, T. (1984). Category specific semantic impairments. *Brain*, 107(3), 829-854.

*Note: These citations are retained for historical context and interpretability, not as evidence of shared mechanisms.*
