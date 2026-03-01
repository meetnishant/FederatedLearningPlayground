"""Gradient compression for federated learning communication cost reduction.

Public API::

    from flp.compression import GradientCompressor
    from flp.compression.topk import TopKResult, topk_compress
    from flp.compression.quantization import QuantizationResult, quantize_state_dict
    from flp.compression.error_feedback import ErrorFeedbackBuffer
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from flp.compression.error_feedback import ErrorFeedbackBuffer
from flp.compression.quantization import QuantizationResult, quantize_state_dict
from flp.compression.topk import TopKResult, topk_compress

if TYPE_CHECKING:
    from flp.core.client import ClientUpdate
    from flp.experiments.config_loader import CompressionConfig

logger = logging.getLogger(__name__)


class GradientCompressor:
    """Unified compression facade used by :class:`~flp.core.server.FLServer`.

    Wraps top-k sparsification or quantization and optionally maintains
    an :class:`~flp.compression.error_feedback.ErrorFeedbackBuffer` for
    residual accumulation across rounds.

    This class is the single entry point for compression in the server's
    round loop.  Initialise it once per experiment and call :meth:`compress`
    for each client update in every round.

    Args:
        config: Compression configuration section from the experiment YAML.
        all_client_ids: All client IDs that may ever submit updates.  Required
            to pre-allocate the error feedback buffer.
    """

    def __init__(
        self,
        config: "CompressionConfig",
        all_client_ids: list[int],
    ) -> None:
        self._config = config
        self._error_buffer: ErrorFeedbackBuffer | None = (
            ErrorFeedbackBuffer(all_client_ids) if config.error_feedback else None
        )
        logger.info(
            "GradientCompressor: strategy=%s | error_feedback=%s%s",
            config.strategy,
            config.error_feedback,
            f" | topk_ratio={config.topk_ratio}" if config.strategy == "topk" else
            f" | bits={config.quantization_bits}",
        )

    def compress(self, update: "ClientUpdate") -> tuple["ClientUpdate", float]:
        """Compress one client update, returning the compressed update and ratio.

        Args:
            update: Client model update to compress.

        Returns:
            Tuple of ``(compressed_update, compression_ratio)`` where
            ``compression_ratio`` is the fraction of bytes relative to a full
            float32 transmission (e.g. 0.1 for top-10%, 0.5 for float16).
        """
        from flp.core.client import ClientUpdate  # local to avoid circular import

        cfg = self._config

        if cfg.strategy == "topk":
            if self._error_buffer is not None:
                def _compress_fn(sd: dict) -> dict:
                    return topk_compress(sd, cfg.topk_ratio).state_dict
                compressed_sd = self._error_buffer.apply_and_compress(
                    update.client_id, update.state_dict, _compress_fn
                )
                ratio = cfg.topk_ratio  # kept fraction ≈ k_ratio
            else:
                result = topk_compress(update.state_dict, cfg.topk_ratio)
                compressed_sd = result.state_dict
                ratio = result.compression_ratio

        else:  # quantization
            result_q = quantize_state_dict(update.state_dict, cfg.quantization_bits)
            compressed_sd = result_q.state_dict
            ratio = result_q.compression_ratio

        compressed_update = ClientUpdate(
            client_id=update.client_id,
            state_dict=compressed_sd,
            num_samples=update.num_samples,
            train_result=update.train_result,
        )
        return compressed_update, ratio


__all__ = [
    "GradientCompressor",
    "TopKResult",
    "topk_compress",
    "QuantizationResult",
    "quantize_state_dict",
    "ErrorFeedbackBuffer",
]
