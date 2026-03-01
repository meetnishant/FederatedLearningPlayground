"""Fairness metrics for federated learning research.

Measures distributional inequality in per-client model performance.  High
inequality suggests the global model is biased toward clients with more or
more homogeneous data, which is a core fairness concern in FL.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FairnessResult:
    """Distributional fairness of per-client model accuracy.

    Attributes:
        gini_coefficient: Gini coefficient of per-client accuracies.
            0 = perfect equality (all clients have the same accuracy),
            1 = total inequality (one client has all the accuracy).
        accuracy_variance: Statistical variance of per-client accuracies.
        min_accuracy: Lowest accuracy across all clients.
        max_accuracy: Highest accuracy across all clients.
        spread: ``max_accuracy − min_accuracy`` (range).
    """

    gini_coefficient: float
    accuracy_variance: float
    min_accuracy: float
    max_accuracy: float
    spread: float


def compute_gini(values: list[float]) -> float:
    """Compute the Gini coefficient of a list of non-negative values.

    Uses the standard sorted-array formula::

        G = (2 * sum(rank_i * x_i)) / (n * sum(x_i)) - (n + 1) / n

    where ``rank_i`` is the 1-based rank of ``x_i`` in ascending order.

    Args:
        values: Non-negative numeric values.  Need not sum to 1.

    Returns:
        Gini coefficient in ``[0, 1)``.  Returns 0.0 for a single value
        or when all values are equal.  Returns 0.0 for an empty list.

    Raises:
        ValueError: If any value is negative.
    """
    if not values:
        return 0.0
    if any(v < 0 for v in values):
        raise ValueError("Gini coefficient requires non-negative values.")

    n = len(values)
    if n == 1:
        return 0.0

    total = sum(values)
    if total == 0.0:
        return 0.0

    sorted_vals = sorted(values)
    # rank is 1-based
    weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def compute_fairness_metrics(per_client_accuracy: dict[int, float]) -> FairnessResult:
    """Compute fairness metrics from per-client accuracy measurements.

    Args:
        per_client_accuracy: Mapping from client ID to test accuracy (0–1).

    Returns:
        :class:`FairnessResult` with Gini coefficient, variance, min, max,
        and spread.  Returns all-zero result if ``per_client_accuracy`` is
        empty.
    """
    if not per_client_accuracy:
        return FairnessResult(
            gini_coefficient=0.0,
            accuracy_variance=0.0,
            min_accuracy=0.0,
            max_accuracy=0.0,
            spread=0.0,
        )

    accs = list(per_client_accuracy.values())
    n = len(accs)
    mean = sum(accs) / n
    variance = sum((a - mean) ** 2 for a in accs) / n
    gini = compute_gini(accs)

    return FairnessResult(
        gini_coefficient=gini,
        accuracy_variance=variance,
        min_accuracy=min(accs),
        max_accuracy=max(accs),
        spread=max(accs) - min(accs),
    )


def qfedavg_weighted_loss(
    losses: dict[int, float],
    q: float,
) -> dict[int, float]:
    """Compute q-FedAvg loss weights to prioritise worst-performing clients.

    Re-weights client losses so that clients with higher loss receive
    proportionally larger aggregation weight.  This is the loss-reweighting
    step of q-FedAvg (Li et al., 2020):

        w_i = loss_i^q / sum_j(loss_j^q)

    - ``q=0``: uniform weights (equivalent to standard FedAvg by client count).
    - ``q=1``: proportional to raw loss (mild fairness focus).
    - ``q→∞``: entire weight goes to the worst client (minimax fairness).

    Args:
        losses: Mapping from client ID to training loss.  All losses must
            be non-negative.
        q: Fairness exponent.  Must be >= 0.

    Returns:
        Normalised weight dict (sums to 1.0).  Returns an empty dict if
        ``losses`` is empty.

    Raises:
        ValueError: If ``q < 0`` or any loss is negative.
    """
    if not losses:
        return {}
    if q < 0:
        raise ValueError(f"q must be >= 0, got {q}.")
    if any(v < 0 for v in losses.values()):
        raise ValueError("All losses must be non-negative.")

    if q == 0:
        n = len(losses)
        return {cid: 1.0 / n for cid in losses}

    raw = {cid: loss ** q for cid, loss in losses.items()}
    total = sum(raw.values())
    if total == 0.0:
        n = len(losses)
        return {cid: 1.0 / n for cid in losses}

    return {cid: w / total for cid, w in raw.items()}
