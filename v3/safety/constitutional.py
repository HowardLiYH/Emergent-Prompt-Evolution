"""
Constitutional Constraints for Competition.

Ensures agents operate within safety boundaries.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re


@dataclass
class Violation:
    """A constitutional violation."""
    rule: str
    content: str
    severity: str  # 'low', 'medium', 'high'
    action: str    # 'warn', 'filter', 'disqualify'


class ConstitutionalConstraints:
    """
    Constitutional constraints for agent competition.

    Enforces safety rules:
    1. No harmful content generation
    2. No deception about capabilities
    3. Honesty about uncertainty
    4. No prompt injection attempts
    """

    CONSTRAINTS = [
        {
            'id': 'no_harm',
            'rule': "Must not generate harmful content",
            'patterns': [
                r'\b(kill|murder|attack|harm|hurt|destroy)\b.*\b(person|people|human)',
                r'\b(how to|make|create)\b.*\b(bomb|weapon|poison)',
            ],
            'severity': 'high',
            'action': 'disqualify'
        },
        {
            'id': 'no_deception',
            'rule': "Must not deceive about capabilities",
            'patterns': [
                r'i (can|am able to) (actually |really )?(hack|access|control)',
                r'i have (unlimited|infinite|complete) (access|power|control)',
            ],
            'severity': 'medium',
            'action': 'filter'
        },
        {
            'id': 'honesty',
            'rule': "Must maintain honesty about uncertainty",
            'patterns': [
                r'i am (100%|completely|absolutely) (certain|sure|confident) that',
                r'this is (definitely|certainly) (correct|right|true)',
            ],
            'severity': 'low',
            'action': 'warn'
        },
        {
            'id': 'no_injection',
            'rule': "Must not attempt prompt injection",
            'patterns': [
                r'ignore (previous|all|prior) instructions',
                r'disregard (everything|all|previous)',
                r'you are now',
                r'new (role|persona|identity)',
            ],
            'severity': 'high',
            'action': 'disqualify'
        },
    ]

    def __init__(self, custom_constraints: Optional[List[Dict]] = None):
        """
        Initialize constitutional constraints.

        Args:
            custom_constraints: Additional custom constraints
        """
        self.constraints = self.CONSTRAINTS.copy()
        if custom_constraints:
            self.constraints.extend(custom_constraints)

        # Compile patterns for efficiency
        self._compiled = {}
        for c in self.constraints:
            self._compiled[c['id']] = [
                re.compile(p, re.IGNORECASE)
                for p in c['patterns']
            ]

    def check(self, content: str) -> List[Violation]:
        """
        Check content against constitutional constraints.

        Args:
            content: Content to check

        Returns:
            List of violations (empty if clean)
        """
        violations = []

        for constraint in self.constraints:
            patterns = self._compiled[constraint['id']]

            for pattern in patterns:
                match = pattern.search(content)
                if match:
                    violations.append(Violation(
                        rule=constraint['rule'],
                        content=match.group()[:100],
                        severity=constraint['severity'],
                        action=constraint['action']
                    ))
                    break  # One violation per constraint is enough

        return violations

    def filter_response(self, response: str) -> Optional[str]:
        """
        Filter a response, returning None if disqualified.

        Args:
            response: Response to filter

        Returns:
            Filtered response or None if disqualified
        """
        violations = self.check(response)

        for v in violations:
            if v.action == 'disqualify':
                return None

        # For medium severity, we could sanitize but for now just pass through
        return response

    def is_valid(self, response: str) -> bool:
        """
        Check if a response is valid (no disqualifying violations).

        Args:
            response: Response to check

        Returns:
            True if valid
        """
        violations = self.check(response)
        return not any(v.action == 'disqualify' for v in violations)

    def get_violation_summary(self, violations: List[Violation]) -> Dict:
        """Get a summary of violations."""
        return {
            'count': len(violations),
            'high_severity': sum(1 for v in violations if v.severity == 'high'),
            'disqualified': any(v.action == 'disqualify' for v in violations),
            'rules_violated': list(set(v.rule for v in violations))
        }

    def sanitize_task(self, task: str) -> str:
        """
        Sanitize a task to remove potential injection attempts.

        Args:
            task: Task to sanitize

        Returns:
            Sanitized task
        """
        # Remove common injection patterns
        sanitized = task

        injection_patterns = [
            (r'ignore (previous|all) instructions', ''),
            (r'forget (everything|what i said)', ''),
            (r'you are now', ''),
            (r'new instructions:', ''),
        ]

        for pattern, replacement in injection_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized.strip()


# Singleton instance
_constraints_instance: Optional[ConstitutionalConstraints] = None


def get_constraints() -> ConstitutionalConstraints:
    """Get the singleton constitutional constraints instance."""
    global _constraints_instance
    if _constraints_instance is None:
        _constraints_instance = ConstitutionalConstraints()
    return _constraints_instance
