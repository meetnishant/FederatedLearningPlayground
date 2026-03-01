"""Error-feedback buffer for top-k gradient compression.

Error feedback (also called error accumulation or memory correction) compensates
for information lost during sparsification by carrying residuals forward across
rounds.  The residual from round t is added back to the update at round t+1
before compression, so no gradient signal is permanently discarded.
"""

from __future__ import annotations

from typing import Callable

import torch


class ErrorFeedbackBuffer:
    """Per-client residual accumulation buffer for error-feedback compression.

    Maintains a dictionary of accumulated residual tensors, one entry per
    client.  On each call to :meth:`apply_and_compress`, the stored residual
    for that client is added to the incoming update before compression, and the
    new residual (corrected - compressed) is stored for the next round.

    This ensures that gradient information dropped in one round is recovered in
    subsequent rounds, improving convergence compared to vanilla top-k without
    error feedback.

    Args:
        client_ids: All client IDs that may submit updates.  Residuals are
            tracked separately per client.
    """

    def __init__(self, client_ids: list[int]) -> None:
        # Maps client_id → per-key residual tensors (float only).
        self._buffers: dict[int, dict[str, torch.Tensor]] = {
            cid: {} for cid in client_ids
        }

    def apply_and_compress(
        self,
        client_id: int,
        state_dict: dict[str, torch.Tensor],
        compress_fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Add accumulated error, compress, and update the residual buffer.

        The compression pipeline for client ``client_id`` at each round:

        1. ``corrected = state_dict + accumulated_error``
        2. ``compressed = compress_fn(corrected)``
        3. ``accumulated_error = corrected − compressed``  (new residual)
        4. Return ``compressed``

        Only floating-point tensors participate in error feedback.  Integer
        buffers are passed through ``compress_fn`` without modification and
        are not accumulated in the buffer.

        Args:
            client_id: Client whose residual buffer to use and update.
            state_dict: Raw model update from this client.
            compress_fn: Compression callable (e.g. a closure over
                :func:`~flp.compression.topk.topk_compress`) that takes a
                state dict and returns a compressed state dict.

        Returns:
            Compressed state dict (after correcting for accumulated error).

        Raises:
            KeyError: If ``client_id`` was not registered at construction time.
        """
        if client_id not in self._buffers:
            raise KeyError(
                f"client_id={client_id} was not registered in ErrorFeedbackBuffer. "
                f"Registered IDs: {sorted(self._buffers.keys())}"
            )

        accumulated = self._buffers[client_id]

        # Step 1: add accumulated residual to current update (float keys only).
        corrected: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            if tensor.is_floating_point() and key in accumulated:
                corrected[key] = tensor + accumulated[key].to(tensor.device)
            else:
                corrected[key] = tensor.clone()

        # Step 2: compress the corrected update.
        compressed = compress_fn(corrected)

        # Step 3: compute and store new residual.
        self._buffers[client_id] = {
            key: (corrected[key] - compressed[key]).detach()
            for key in corrected
            if corrected[key].is_floating_point()
        }

        return compressed

    def reset(self, client_id: int) -> None:
        """Clear the accumulated residual for a specific client.

        Call this when a client drops out or is re-initialised so that stale
        residuals do not corrupt future updates.

        Args:
            client_id: Client whose buffer to clear.

        Raises:
            KeyError: If ``client_id`` was not registered at construction time.
        """
        if client_id not in self._buffers:
            raise KeyError(
                f"client_id={client_id} was not registered in ErrorFeedbackBuffer."
            )
        self._buffers[client_id] = {}

    def has_residual(self, client_id: int) -> bool:
        """Return True if ``client_id`` has a non-empty accumulated residual."""
        return bool(self._buffers.get(client_id))
