"""Communication cost tracking for federated learning experiments."""

from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Total number of trainable scalar parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_bytes(model: nn.Module, dtype: torch.dtype = torch.float32) -> int:
    """Estimate model size in bytes assuming all parameters use ``dtype``.

    Args:
        model: PyTorch model.
        dtype: Data type used for parameter storage.

    Returns:
        Total size in bytes.
    """
    bytes_per_element = torch.finfo(dtype).bits // 8
    return count_parameters(model) * bytes_per_element


class CommunicationTracker:
    """Tracks cumulative communication cost across federated rounds.

    Measures upload (client -> server) and download (server -> client) costs
    in bytes assuming full-precision (float32) transmission of model weights.

    Args:
        model: The global model (used to determine parameter count).
    """

    def __init__(self, model: nn.Module) -> None:
        self._param_bytes = model_size_bytes(model)
        self._total_upload_bytes: int = 0
        self._total_download_bytes: int = 0
        self._rounds: list[dict[str, int]] = []

    def record_round(self, num_clients_upload: int, num_clients_download: int) -> None:
        """Record communication for one round.

        Args:
            num_clients_upload: Number of clients that uploaded updates.
            num_clients_download: Number of clients that received the global model.
        """
        upload = num_clients_upload * self._param_bytes
        download = num_clients_download * self._param_bytes
        self._total_upload_bytes += upload
        self._total_download_bytes += download
        self._rounds.append({"upload_bytes": upload, "download_bytes": download})

    @property
    def total_bytes(self) -> int:
        """Total bytes transferred (upload + download) across all rounds."""
        return self._total_upload_bytes + self._total_download_bytes

    @property
    def total_mb(self) -> float:
        """Total megabytes transferred."""
        return self.total_bytes / (1024 ** 2)

    def summary(self) -> dict[str, object]:
        """Return a communication cost summary.

        Returns:
            Dict with ``total_upload_mb``, ``total_download_mb``, ``total_mb``,
            and ``num_rounds``.
        """
        return {
            "total_upload_mb": self._total_upload_bytes / (1024 ** 2),
            "total_download_mb": self._total_download_bytes / (1024 ** 2),
            "total_mb": self.total_mb,
            "num_rounds": len(self._rounds),
        }
