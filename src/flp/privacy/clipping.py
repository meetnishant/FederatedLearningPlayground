"""Per-update L2 norm clipping for differential privacy in federated learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ClipResult:
    """Result of clipping a single client model update.

    Attributes:
        state_dict: Clipped model state dict (float32 for float tensors;
            integer buffers are passed through from the updated state unchanged).
        original_norm: L2 norm of the unclipped update delta.
        clipped_norm: L2 norm of the delta after clipping (≤ max_norm).
        was_clipped: True if the original norm exceeded max_norm.
        scale: Scaling factor applied to the update delta (1.0 if not clipped).
    """

    state_dict: dict[str, torch.Tensor]
    original_norm: float
    clipped_norm: float
    was_clipped: bool
    scale: float


def compute_update_norm(
    original_state: dict[str, torch.Tensor],
    updated_state: dict[str, torch.Tensor],
) -> float:
    """Compute the L2 norm of the model update delta (updated − original).

    Only floating-point parameters are included; integer buffers such as
    BatchNorm ``num_batches_tracked`` are excluded.

    Args:
        original_state: Model state dict before local training (global weights).
        updated_state: Model state dict after local training.

    Returns:
        Scalar L2 norm of the concatenated flattened parameter delta.
        Returns 0.0 if the state dict contains no floating-point tensors.
    """
    deltas = [
        (updated_state[k].float() - original_state[k].float()).flatten()
        for k in original_state
        if original_state[k].is_floating_point()
    ]
    if not deltas:
        return 0.0
    return torch.cat(deltas).norm(p=2).item()


def clip_model_update(
    original_state: dict[str, torch.Tensor],
    updated_state: dict[str, torch.Tensor],
    max_norm: float,
) -> ClipResult:
    """Clip the L2 norm of the model update delta to at most ``max_norm``.

    Computes ``delta = updated - original`` for all floating-point tensors,
    measures the aggregate L2 norm, and rescales the delta if it exceeds
    ``max_norm``.  Integer buffers (e.g. ``num_batches_tracked``) are taken
    from ``updated_state`` unchanged.

    The clipped state dict contains:
    - Float keys: ``original + clip_scale * delta``
    - Integer keys: copied from ``updated_state``

    Args:
        original_state: Global model weights before local training.
        updated_state: Local model weights after local training.
        max_norm: Maximum allowed L2 norm for the update vector (sensitivity).

    Returns:
        :class:`ClipResult` with the clipped state dict and clipping statistics.

    Raises:
        ValueError: If ``max_norm`` is not positive.
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}.")

    float_keys = [k for k in original_state if original_state[k].is_floating_point()]

    if float_keys:
        deltas = {
            k: updated_state[k].float() - original_state[k].float()
            for k in float_keys
        }
        flat = torch.cat([d.flatten() for d in deltas.values()])
        norm = flat.norm(p=2).item()
    else:
        deltas = {}
        norm = 0.0

    scale = min(1.0, max_norm / (norm + 1e-8))
    was_clipped = norm > max_norm

    clipped: dict[str, torch.Tensor] = {}
    for k in original_state:
        if original_state[k].is_floating_point():
            clipped[k] = original_state[k].float() + deltas[k] * scale
        else:
            # Integer buffers (BatchNorm counters, etc.) pass through unchanged.
            clipped[k] = updated_state[k].clone()

    return ClipResult(
        state_dict=clipped,
        original_norm=norm,
        clipped_norm=norm * scale,
        was_clipped=was_clipped,
        scale=scale,
    )


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip all parameter gradients of ``model`` in-place to L2 norm ≤ ``max_norm``.

    This performs *global* gradient clipping across all parameter tensors
    simultaneously (equivalent to ``torch.nn.utils.clip_grad_norm_``).

    For DP-FedAvg, use :func:`clip_model_update` instead — it clips the
    *update* (weight delta) after local training rather than per-step gradients.

    Args:
        model: Model whose ``.grad`` attributes will be clipped in-place.
        max_norm: Maximum allowed total L2 gradient norm.

    Returns:
        Pre-clipping total gradient norm (useful for monitoring).
    """
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
