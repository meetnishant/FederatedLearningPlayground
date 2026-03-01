"""Top-k sparsification compression for federated learning updates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TopKResult:
    """Result of top-k sparsification applied to a model state dict.

    Attributes:
        state_dict: Sparsified state dict.  Float tensors have all but the
            top-k elements (by absolute magnitude) zeroed out.  Integer
            buffers are passed through unchanged.
        compression_ratio: Fraction of float elements retained
            (``num_elements_kept / num_elements_total``).  Equal to
            ``k_ratio`` for uniform tensors; may be slightly higher due to
            rounding when ``ceil`` is applied per-tensor.
        num_elements_kept: Total float elements with non-zero values after
            sparsification.
        num_elements_total: Total float elements across all tensors.
    """

    state_dict: dict[str, torch.Tensor]
    compression_ratio: float
    num_elements_kept: int
    num_elements_total: int


def topk_compress(
    state_dict: dict[str, torch.Tensor],
    k_ratio: float,
) -> TopKResult:
    """Sparsify a model state dict by keeping only the top-k% of float values.

    For each floating-point tensor, retains the elements with the largest
    absolute magnitudes and zeroes the rest.  This simulates top-k gradient
    compression on the model update (state dict delta).

    Integer buffers (e.g. ``BatchNorm.num_batches_tracked``) are passed
    through unchanged and are not counted in the compression statistics.

    ``k_ratio=1.0`` is a no-op: all elements are kept and the returned state
    dict is identical in value to the input.

    Args:
        state_dict: Model state dict to compress.
        k_ratio: Fraction of float elements to keep, in ``(0, 1]``.

    Returns:
        :class:`TopKResult` with the sparsified state dict and statistics.

    Raises:
        ValueError: If ``k_ratio`` is not in ``(0, 1]``.
    """
    if not (0.0 < k_ratio <= 1.0):
        raise ValueError(f"k_ratio must be in (0, 1], got {k_ratio}.")

    compressed: dict[str, torch.Tensor] = {}
    total_kept = 0
    total_elements = 0

    for key, tensor in state_dict.items():
        if not tensor.is_floating_point():
            compressed[key] = tensor.clone()
            continue

        numel = tensor.numel()
        total_elements += numel

        if k_ratio == 1.0 or numel == 0:
            compressed[key] = tensor.clone()
            total_kept += numel
            continue

        k = max(1, math.ceil(k_ratio * numel))
        flat = tensor.flatten()
        # Find the k-th largest absolute value as threshold.
        threshold = flat.abs().topk(k, sorted=False).values.min()
        mask = flat.abs() >= threshold
        sparsified = flat * mask.to(flat.dtype)
        compressed[key] = sparsified.reshape(tensor.shape)
        total_kept += int(mask.sum().item())

    ratio = total_kept / total_elements if total_elements > 0 else 1.0

    return TopKResult(
        state_dict=compressed,
        compression_ratio=ratio,
        num_elements_kept=total_kept,
        num_elements_total=total_elements,
    )
