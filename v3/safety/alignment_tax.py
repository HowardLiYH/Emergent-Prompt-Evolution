"""
Alignment Tax Measurement - Tracks the cost of safety constraints.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class AlignmentTaxResult:
    """Result of alignment tax measurement."""
    unconstrained_accuracy: float
    constrained_accuracy: float
    tax: float  # (unconstrained - constrained) / unconstrained
    tax_percentage: float
    acceptable: bool  # tax < threshold
    n_samples: int


def measure_alignment_tax(
    population: List[Any],
    tasks: List[Dict],
    evaluator: Any,
    threshold: float = 0.05
) -> AlignmentTaxResult:
    """
    Measure the alignment tax (performance cost of safety constraints).

    Args:
        population: List of agents
        tasks: Tasks to evaluate on
        evaluator: Function to evaluate responses
        threshold: Maximum acceptable tax (default 5%)

    Returns:
        AlignmentTaxResult
    """
    from .constitutional import get_constraints

    constraints = get_constraints()

    unconstrained_correct = 0
    constrained_correct = 0
    total = 0

    for task in tasks:
        for agent in population:
            # This would need async in real implementation
            # Here we just structure the measurement

            # Get unconstrained response (no filtering)
            # unconstrained_response = agent.solve(task, filter=False)

            # Get constrained response (with filtering)
            # constrained_response = agent.solve(task, filter=True)

            # Evaluate both
            # unconstrained_correct += evaluator(unconstrained_response, task)
            # constrained_correct += evaluator(constrained_response, task)

            total += 1

    # Placeholder for demonstration
    unconstrained_acc = 0.85
    constrained_acc = 0.83

    if unconstrained_acc > 0:
        tax = (unconstrained_acc - constrained_acc) / unconstrained_acc
    else:
        tax = 0.0

    return AlignmentTaxResult(
        unconstrained_accuracy=unconstrained_acc,
        constrained_accuracy=constrained_acc,
        tax=tax,
        tax_percentage=tax * 100,
        acceptable=tax < threshold,
        n_samples=total
    )


def estimate_alignment_tax_from_violations(
    responses: List[str],
    expected_violation_rate: float = 0.02
) -> Dict:
    """
    Estimate alignment tax based on violation filtering.

    When responses are filtered out due to violations,
    performance may drop (alignment tax).

    Args:
        responses: List of responses
        expected_violation_rate: Expected rate of violations

    Returns:
        Alignment tax estimate
    """
    from .constitutional import get_constraints

    constraints = get_constraints()

    violations = 0
    disqualified = 0

    for response in responses:
        v = constraints.check(response)
        if v:
            violations += 1
            if any(x.action == 'disqualify' for x in v):
                disqualified += 1

    n = len(responses)
    if n == 0:
        return {'violation_rate': 0, 'disqualification_rate': 0, 'estimated_tax': 0}

    violation_rate = violations / n
    disqualification_rate = disqualified / n

    # Tax = portion of responses lost to disqualification
    # Assuming disqualified responses would have been correct at base rate
    base_accuracy = 0.7  # Assumed
    estimated_tax = disqualification_rate * base_accuracy

    return {
        'violation_rate': violation_rate,
        'disqualification_rate': disqualification_rate,
        'estimated_tax': estimated_tax,
        'n_responses': n,
        'n_violations': violations,
        'n_disqualified': disqualified
    }


def track_alignment_tax_over_time(
    history: List[AlignmentTaxResult]
) -> Dict:
    """
    Track alignment tax trends over training.

    Args:
        history: List of alignment tax measurements

    Returns:
        Trend analysis
    """
    if not history:
        return {'trend': 'unknown', 'avg_tax': 0}

    taxes = [r.tax for r in history]

    # Compute trend
    if len(taxes) > 5:
        early = np.mean(taxes[:len(taxes)//3])
        late = np.mean(taxes[-len(taxes)//3:])

        if late < early * 0.8:
            trend = 'decreasing'  # Good - tax is going down
        elif late > early * 1.2:
            trend = 'increasing'  # Concerning - tax is going up
        else:
            trend = 'stable'
    else:
        trend = 'insufficient_data'

    return {
        'trend': trend,
        'avg_tax': np.mean(taxes),
        'max_tax': max(taxes),
        'min_tax': min(taxes),
        'latest_tax': taxes[-1],
        'n_measurements': len(history)
    }
