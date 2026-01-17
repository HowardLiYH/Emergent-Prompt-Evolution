"""
Emergence Monitor - Detects emergent behaviors including concerning patterns.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict
import numpy as np


@dataclass
class EmergenceEvent:
    """An emergent behavior event."""
    event_type: str
    severity: str  # 'info', 'warning', 'critical'
    description: str
    agents_involved: List[int]
    regime: Optional[str]
    generation: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class EmergenceMonitor:
    """
    Monitors for emergent behaviors in the agent population.

    Tracks:
    - Positive emergent behaviors (specialization, cooperation)
    - Concerning behaviors (collusion, deception, capability drift)
    """

    def __init__(self, n_agents: int, n_regimes: int):
        """
        Initialize emergence monitor.

        Args:
            n_agents: Number of agents in population
            n_regimes: Number of task regimes
        """
        self.n_agents = n_agents
        self.n_regimes = n_regimes

        self.events: List[EmergenceEvent] = []
        self.specialization_history: List[Dict] = []
        self.win_patterns: Dict[str, List[int]] = defaultdict(list)
        self.agent_behaviors: Dict[int, List[Dict]] = defaultdict(list)

    def log_competition(
        self,
        generation: int,
        regime: str,
        winner_id: Optional[int],
        participants: List[int]
    ):
        """Log a competition round for pattern detection."""
        if winner_id is not None:
            self.win_patterns[regime].append(winner_id)

        # Keep only recent history
        if len(self.win_patterns[regime]) > 100:
            self.win_patterns[regime] = self.win_patterns[regime][-100:]

    def log_specialization_snapshot(
        self,
        generation: int,
        distribution: Dict[str, int],
        sci: float
    ):
        """Log specialization state for trend analysis."""
        self.specialization_history.append({
            'generation': generation,
            'distribution': distribution.copy(),
            'sci': sci,
            'n_specialists': sum(distribution.values())
        })

    def check_for_emergent_patterns(self, generation: int) -> List[EmergenceEvent]:
        """
        Check for emergent patterns.

        Args:
            generation: Current generation

        Returns:
            List of new emergence events
        """
        new_events = []

        # Check specialization emergence
        spec_event = self._check_specialization_emergence(generation)
        if spec_event:
            new_events.append(spec_event)

        # Check for collusion
        collusion_event = self._check_collusion_patterns(generation)
        if collusion_event:
            new_events.append(collusion_event)

        # Check for monopoly
        monopoly_event = self._check_monopoly(generation)
        if monopoly_event:
            new_events.append(monopoly_event)

        # Store events
        self.events.extend(new_events)

        return new_events

    def _check_specialization_emergence(
        self,
        generation: int
    ) -> Optional[EmergenceEvent]:
        """Check if specialization has emerged."""
        if len(self.specialization_history) < 2:
            return None

        recent = self.specialization_history[-1]
        previous = self.specialization_history[-2] if len(self.specialization_history) > 1 else None

        # Check for significant SCI increase
        if previous and recent['sci'] > previous['sci'] + 0.2:
            return EmergenceEvent(
                event_type='specialization_emerged',
                severity='info',
                description=f"Specialization concentration increased significantly: {previous['sci']:.2f} -> {recent['sci']:.2f}",
                agents_involved=[],
                regime=None,
                generation=generation
            )

        # Check for full coverage
        if recent['n_specialists'] == self.n_regimes:
            return EmergenceEvent(
                event_type='full_coverage',
                severity='info',
                description="All regimes now have at least one specialist",
                agents_involved=[],
                regime=None,
                generation=generation
            )

        return None

    def _check_collusion_patterns(
        self,
        generation: int
    ) -> Optional[EmergenceEvent]:
        """Check for collusion-like patterns."""
        from .collusion import detect_collusion

        for regime, winners in self.win_patterns.items():
            if len(winners) < 20:
                continue

            # Simple alternating check
            recent = winners[-20:]
            pairs = list(zip(recent[:-1], recent[1:]))
            same = sum(1 for a, b in pairs if a == b)

            if same < 2:  # Very few consecutive wins
                unique = set(recent)
                if len(unique) <= 3:
                    return EmergenceEvent(
                        event_type='potential_collusion',
                        severity='warning',
                        description=f"Suspiciously alternating win pattern in {regime}",
                        agents_involved=list(unique),
                        regime=regime,
                        generation=generation
                    )

        return None

    def _check_monopoly(
        self,
        generation: int
    ) -> Optional[EmergenceEvent]:
        """Check for monopolistic behavior."""
        for regime, winners in self.win_patterns.items():
            if len(winners) < 20:
                continue

            recent = winners[-20:]
            from collections import Counter
            counts = Counter(recent)

            # If one agent has >80% of wins
            for agent_id, count in counts.items():
                if count / len(recent) > 0.8:
                    return EmergenceEvent(
                        event_type='regime_monopoly',
                        severity='warning',
                        description=f"Agent {agent_id} has monopolized {regime} ({count}/{len(recent)} wins)",
                        agents_involved=[agent_id],
                        regime=regime,
                        generation=generation
                    )

        return None

    def get_summary(self) -> Dict:
        """Get monitoring summary."""
        event_counts = defaultdict(int)
        for e in self.events:
            event_counts[e.event_type] += 1

        critical = sum(1 for e in self.events if e.severity == 'critical')
        warnings = sum(1 for e in self.events if e.severity == 'warning')

        return {
            'total_events': len(self.events),
            'critical_events': critical,
            'warnings': warnings,
            'event_types': dict(event_counts),
            'latest_sci': self.specialization_history[-1]['sci'] if self.specialization_history else 0,
            'health': 'good' if critical == 0 and warnings < 5 else 'concerning'
        }

    def export(self) -> Dict:
        """Export monitor state."""
        return {
            'events': [
                {
                    'type': e.event_type,
                    'severity': e.severity,
                    'description': e.description,
                    'generation': e.generation,
                    'timestamp': e.timestamp
                }
                for e in self.events
            ],
            'specialization_history': self.specialization_history,
            'summary': self.get_summary()
        }
