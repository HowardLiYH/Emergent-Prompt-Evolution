"""
Semantic Memory - Compressed patterns extracted from episodes.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class SemanticPattern:
    """A compressed pattern from multiple episodes."""
    pattern: str           # The pattern description
    regime: str            # Primary regime
    confidence: float      # Pattern confidence (0-1)
    support: int           # Number of episodes supporting this
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = ""
    use_count: int = 0


class SemanticMemory:
    """
    Layer 3: Semantic Memory

    Stores compressed patterns extracted from episodic memory.
    - Patterns are generalizations of multiple episodes
    - Generated via LLM summarization
    - Permanent with confidence decay

    This is the "wisdom" layer that captures what works across many experiences.
    """

    MAX_PATTERNS = 50
    DECAY_RATE = 0.95  # Confidence decay per consolidation cycle
    MIN_CONFIDENCE = 0.3  # Patterns below this are pruned

    def __init__(self):
        """Initialize semantic memory."""
        self.patterns: Dict[str, List[SemanticPattern]] = {}
        self.global_patterns: List[SemanticPattern] = []

    def add_pattern(
        self,
        pattern: str,
        regime: str,
        confidence: float = 0.8,
        support: int = 1
    ):
        """
        Add a new pattern.

        Args:
            pattern: Pattern description
            regime: Primary regime
            confidence: Initial confidence
            support: Number of episodes supporting this
        """
        sp = SemanticPattern(
            pattern=pattern,
            regime=regime,
            confidence=confidence,
            support=support
        )

        if regime not in self.patterns:
            self.patterns[regime] = []

        # Check for similar existing pattern
        for existing in self.patterns[regime]:
            if self._similar(pattern, existing.pattern):
                # Merge: increase confidence and support
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.support += support
                return

        self.patterns[regime].append(sp)

        # Prune if over limit
        if len(self.patterns[regime]) > self.MAX_PATTERNS:
            self._prune_regime(regime)

    def add_global_pattern(self, pattern: str, confidence: float = 0.8):
        """Add a regime-agnostic pattern."""
        sp = SemanticPattern(
            pattern=pattern,
            regime='global',
            confidence=confidence,
            support=1
        )
        self.global_patterns.append(sp)

    def retrieve(
        self,
        regime: str,
        top_k: int = 3,
        include_global: bool = True
    ) -> List[SemanticPattern]:
        """
        Retrieve relevant patterns for a regime.

        Args:
            regime: Regime to retrieve for
            top_k: Number of patterns to retrieve
            include_global: Whether to include global patterns

        Returns:
            List of relevant patterns
        """
        patterns = []

        # Get regime-specific patterns
        if regime in self.patterns:
            patterns.extend(self.patterns[regime])

        # Add global patterns
        if include_global:
            patterns.extend(self.global_patterns)

        # Sort by confidence × support
        patterns.sort(
            key=lambda p: p.confidence * (1 + 0.1 * p.support),
            reverse=True
        )

        # Mark as used
        for p in patterns[:top_k]:
            p.last_used = datetime.now().isoformat()
            p.use_count += 1

        return patterns[:top_k]

    def decay_confidence(self):
        """Apply confidence decay to all patterns."""
        for regime in self.patterns:
            for p in self.patterns[regime]:
                p.confidence *= self.DECAY_RATE

            # Prune low-confidence patterns
            self.patterns[regime] = [
                p for p in self.patterns[regime]
                if p.confidence >= self.MIN_CONFIDENCE
            ]

        # Decay global patterns
        self.global_patterns = [
            p for p in self.global_patterns
            if p.confidence * self.DECAY_RATE >= self.MIN_CONFIDENCE
        ]
        for p in self.global_patterns:
            p.confidence *= self.DECAY_RATE

    def _similar(self, p1: str, p2: str) -> bool:
        """Check if two patterns are similar."""
        # Simple word overlap check
        words1 = set(p1.lower().split())
        words2 = set(p2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        return overlap / union > 0.5

    def _prune_regime(self, regime: str):
        """Prune patterns for a regime."""
        # Keep patterns with highest confidence × support
        self.patterns[regime].sort(
            key=lambda p: p.confidence * (1 + 0.1 * p.support),
            reverse=True
        )
        self.patterns[regime] = self.patterns[regime][:self.MAX_PATTERNS]

    def get_stats(self) -> Dict:
        """Get semantic memory statistics."""
        return {
            'total_patterns': sum(len(ps) for ps in self.patterns.values()),
            'global_patterns': len(self.global_patterns),
            'regimes': list(self.patterns.keys()),
            'patterns_per_regime': {
                r: len(ps) for r, ps in self.patterns.items()
            }
        }

    def export(self) -> Dict:
        """Export to serializable dict."""
        return {
            'patterns': {
                regime: [
                    {
                        'pattern': p.pattern,
                        'confidence': p.confidence,
                        'support': p.support,
                        'created_at': p.created_at,
                        'use_count': p.use_count,
                    }
                    for p in patterns
                ]
                for regime, patterns in self.patterns.items()
            },
            'global': [
                {
                    'pattern': p.pattern,
                    'confidence': p.confidence,
                    'support': p.support,
                }
                for p in self.global_patterns
            ]
        }

    def import_from(self, data: Dict):
        """Import from dict."""
        for regime, patterns in data.get('patterns', {}).items():
            for p_data in patterns:
                self.add_pattern(
                    pattern=p_data['pattern'],
                    regime=regime,
                    confidence=p_data['confidence'],
                    support=p_data.get('support', 1)
                )

        for p_data in data.get('global', []):
            self.add_global_pattern(
                pattern=p_data['pattern'],
                confidence=p_data['confidence']
            )
