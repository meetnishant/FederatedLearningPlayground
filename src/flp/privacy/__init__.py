"""Privacy layer: differential privacy noise injection and gradient clipping.

Public API::

    from flp.privacy.clipping import clip_model_update, clip_gradients, compute_update_norm, ClipResult
    from flp.privacy.dp import GaussianMechanism, DPAccountant, DPRoundRecord, compute_noise_multiplier
"""

from flp.privacy.clipping import ClipResult, clip_gradients, clip_model_update, compute_update_norm
from flp.privacy.dp import DPAccountant, DPRoundRecord, GaussianMechanism, compute_noise_multiplier

__all__ = [
    "clip_model_update",
    "clip_gradients",
    "compute_update_norm",
    "ClipResult",
    "GaussianMechanism",
    "DPAccountant",
    "DPRoundRecord",
    "compute_noise_multiplier",
]
