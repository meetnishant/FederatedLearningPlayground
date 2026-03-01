"""Research metrics for federated learning experimentation.

Public API::

    from flp.research.divergence import DivergenceResult, compute_weight_divergence, cosine_similarity_between_updates
    from flp.research.fairness import FairnessResult, compute_fairness_metrics, compute_gini, qfedavg_weighted_loss
"""

from flp.research.divergence import (
    DivergenceResult,
    compute_weight_divergence,
    cosine_similarity_between_updates,
)
from flp.research.fairness import (
    FairnessResult,
    compute_fairness_metrics,
    compute_gini,
    qfedavg_weighted_loss,
)

__all__ = [
    "DivergenceResult",
    "compute_weight_divergence",
    "cosine_similarity_between_updates",
    "FairnessResult",
    "compute_fairness_metrics",
    "compute_gini",
    "qfedavg_weighted_loss",
]
