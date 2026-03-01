"""Gradient and update clipping for differential privacy."""

from __future__ import annotations

import torch
import torch.nn as nn


def clip_model_update(
    original_state: dict[str, torch.Tensor],
    updated_state: dict[str, torch.Tensor],
    max_norm: float,
) -> dict[str, torch.Tensor]:
    """Clip the L2 norm of the model update (delta weights).

    Computes the per-parameter delta between ``updated_state`` and
    ``original_state``, clips the aggregate L2 norm to ``max_norm``,
    and returns the clipped updated state dict.

    Args:
        original_state: Global model weights before local training.
        updated_state: Local model weights after local training.
        max_norm: Maximum L2 norm for the update vector.

    Returns:
        Clipped state dict (updated weights after bounded update is applied
        back to ``original_state``).
    """
    deltas = {
        k: updated_state[k].float() - original_state[k].float()
        for k in original_state
    }

    flat_delta = torch.cat([d.flatten() for d in deltas.values()])
    norm = flat_delta.norm(p=2).item()
    scale = min(1.0, max_norm / (norm + 1e-8))

    clipped: dict[str, torch.Tensor] = {}
    for k in original_state:
        clipped[k] = original_state[k].float() + deltas[k] * scale

    return clipped


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip gradients of ``model`` in-place to have L2 norm <= ``max_norm``.

    Args:
        model: Model whose gradients will be clipped.
        max_norm: Maximum allowed L2 norm.

    Returns:
        The pre-clipping gradient norm.
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(total_norm)
