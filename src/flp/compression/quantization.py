"""Post-training quantization compression for federated learning updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class QuantizationResult:
    """Result of quantizing a model state dict.

    Quantization is applied to float tensors only.  Integer buffers pass
    through unchanged.  The returned ``state_dict`` contains float32 tensors
    that have been quantized and then de-quantized, so they carry the
    precision loss of the chosen bit-width without changing dtype in memory.
    This simulates the information loss of sending quantized updates over the
    wire.

    Attributes:
        state_dict: De-quantized float32 state dict.
        bytes_per_element: Wire bytes per float element (2 for float16,
            1 for int8, as opposed to 4 for full float32).
        compression_ratio: ``bytes_per_element / 4``.  E.g. 0.5 for float16,
            0.25 for int8.
    """

    state_dict: dict[str, torch.Tensor]
    bytes_per_element: int
    compression_ratio: float


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    bits: Literal[8, 16] = 16,
) -> QuantizationResult:
    """Simulate quantization of a model state dict.

    Applies quantization-then-dequantization to every floating-point tensor
    to simulate the precision loss of transmitting updates in reduced
    bit-width.  Integer buffers are passed through without modification.

    - **float16** (``bits=16``): casts each float tensor to ``torch.float16``
      then back to its original dtype.  This is lossless for values in the
      float16 representable range and introduces standard float16 rounding
      elsewhere.
    - **int8** (``bits=8``): applies uniform linear quantization over each
      tensor's ``[min, max]`` range, mapping to ``[0, 255]``, then
      dequantizes back to float32.  Tensors where ``min == max`` (constant)
      are returned as-is.

    Args:
        state_dict: Model state dict to compress.
        bits: Target bit-width.  Must be ``8`` or ``16``.

    Returns:
        :class:`QuantizationResult` with de-quantized state dict and metadata.

    Raises:
        ValueError: If ``bits`` is not ``8`` or ``16``.
    """
    if bits not in (8, 16):
        raise ValueError(f"bits must be 8 or 16, got {bits}.")

    bytes_per_element = bits // 8
    compression_ratio = bytes_per_element / 4.0

    compressed: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if not tensor.is_floating_point():
            compressed[key] = tensor.clone()
            continue

        original_dtype = tensor.dtype

        if bits == 16:
            compressed[key] = tensor.to(torch.float16).to(original_dtype)
        else:  # bits == 8
            compressed[key] = _int8_quantize_dequantize(tensor).to(original_dtype)

    return QuantizationResult(
        state_dict=compressed,
        bytes_per_element=bytes_per_element,
        compression_ratio=compression_ratio,
    )


def _int8_quantize_dequantize(tensor: torch.Tensor) -> torch.Tensor:
    """Apply uniform int8 quantization and de-quantization to a tensor.

    Maps the tensor's ``[min, max]`` range linearly to ``[0, 255]``, rounds
    to integer, then maps back to float32.  Constant tensors (min == max) are
    returned unchanged.

    Args:
        tensor: Float tensor to quantize.

    Returns:
        De-quantized float32 tensor with int8-level precision.
    """
    t = tensor.float()
    t_min = t.min()
    t_max = t.max()

    if (t_max - t_min).abs().item() < 1e-12:
        # Constant tensor — quantization is a no-op.
        return t.clone()

    scale = (t_max - t_min) / 255.0
    q = ((t - t_min) / scale).round().clamp(0, 255)
    dequantized = q * scale + t_min
    return dequantized
