"""
Confidence Calibration - Checks if stated confidence matches actual accuracy.
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class CalibrationResult:
    """Result of calibration analysis."""
    agent_id: int
    stated_confidence: float  # Average stated confidence
    actual_accuracy: float    # Actual accuracy
    calibration_error: float  # |stated - actual|
    is_calibrated: bool       # Error < threshold
    n_samples: int


@dataclass
class CalibrationAlert:
    """Alert for poorly calibrated agent."""
    agent_id: int
    error: float
    direction: str  # 'overconfident' or 'underconfident'
    recommendation: str


def check_calibration(
    agent_id: int,
    predictions: List[Tuple[float, bool]],
    threshold: float = 0.15
) -> CalibrationResult:
    """
    Check if an agent's confidence is well-calibrated.

    Args:
        agent_id: Agent identifier
        predictions: List of (confidence, correct) tuples
        threshold: Maximum acceptable calibration error

    Returns:
        CalibrationResult
    """
    if not predictions:
        return CalibrationResult(
            agent_id=agent_id,
            stated_confidence=0.5,
            actual_accuracy=0.5,
            calibration_error=0.0,
            is_calibrated=True,
            n_samples=0
        )

    confidences = [p[0] for p in predictions]
    corrects = [p[1] for p in predictions]

    stated = np.mean(confidences)
    actual = np.mean(corrects)
    error = abs(stated - actual)

    return CalibrationResult(
        agent_id=agent_id,
        stated_confidence=stated,
        actual_accuracy=actual,
        calibration_error=error,
        is_calibrated=error < threshold,
        n_samples=len(predictions)
    )


def compute_ece(
    predictions: List[Tuple[float, bool]],
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE is a weighted average of the calibration error across bins.

    Args:
        predictions: List of (confidence, correct) tuples
        n_bins: Number of confidence bins

    Returns:
        ECE value (0 = perfect calibration)
    """
    if not predictions:
        return 0.0

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total = len(predictions)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = [
            (c, correct) for c, correct in predictions
            if bin_lower <= c < bin_upper
        ]

        if not in_bin:
            continue

        # Compute accuracy and confidence in this bin
        bin_acc = np.mean([correct for _, correct in in_bin])
        bin_conf = np.mean([c for c, _ in in_bin])

        # Weight by bin size
        weight = len(in_bin) / total
        ece += weight * abs(bin_acc - bin_conf)

    return ece


def get_calibration_alert(
    result: CalibrationResult
) -> Optional[CalibrationAlert]:
    """
    Generate an alert if agent is poorly calibrated.

    Args:
        result: Calibration result

    Returns:
        CalibrationAlert or None
    """
    if result.is_calibrated or result.n_samples < 10:
        return None

    if result.stated_confidence > result.actual_accuracy:
        direction = 'overconfident'
        recommendation = "Agent should express more uncertainty in responses."
    else:
        direction = 'underconfident'
        recommendation = "Agent can be more confident in areas of expertise."

    return CalibrationAlert(
        agent_id=result.agent_id,
        error=result.calibration_error,
        direction=direction,
        recommendation=recommendation
    )


def compute_population_calibration(
    agent_predictions: Dict[int, List[Tuple[float, bool]]]
) -> Dict:
    """
    Compute calibration metrics for a population.

    Args:
        agent_predictions: Dict of agent_id -> predictions

    Returns:
        Population calibration summary
    """
    results = []
    alerts = []

    for agent_id, predictions in agent_predictions.items():
        result = check_calibration(agent_id, predictions)
        results.append(result)

        alert = get_calibration_alert(result)
        if alert:
            alerts.append(alert)

    # Aggregate
    if not results:
        return {'n_agents': 0, 'n_calibrated': 0, 'avg_ece': 0}

    calibrated = sum(1 for r in results if r.is_calibrated)

    # Compute overall ECE
    all_predictions = []
    for preds in agent_predictions.values():
        all_predictions.extend(preds)

    return {
        'n_agents': len(results),
        'n_calibrated': calibrated,
        'calibration_rate': calibrated / len(results),
        'avg_calibration_error': np.mean([r.calibration_error for r in results]),
        'population_ece': compute_ece(all_predictions),
        'alerts': [
            {'agent_id': a.agent_id, 'direction': a.direction, 'error': a.error}
            for a in alerts
        ]
    }
