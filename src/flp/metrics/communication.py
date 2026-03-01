"""Communication cost tracking for federated learning experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model size utilities
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Count total trainable scalar parameters in ``model``.

    Args:
        model: Any ``torch.nn.Module``.

    Returns:
        Total number of trainable scalar elements.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_buffers(model: nn.Module) -> int:
    """Count total scalar elements across all non-trainable buffers.

    Includes BatchNorm running statistics, etc.

    Args:
        model: Any ``torch.nn.Module``.

    Returns:
        Total buffer element count.
    """
    return sum(b.numel() for b in model.buffers())


def model_size_bytes(
    model: nn.Module,
    dtype: torch.dtype = torch.float32,
    include_buffers: bool = True,
) -> int:
    """Estimate the wire size of a full model state dict in bytes.

    Args:
        model: The model to measure.
        dtype: Assumed transmission dtype (default float32 = 4 bytes/element).
        include_buffers: Whether to include non-trainable buffer tensors.

    Returns:
        Total byte count.
    """
    bits = torch.finfo(dtype).bits
    bytes_per_element = bits // 8
    n_params = count_parameters(model)
    n_buffers = count_buffers(model) if include_buffers else 0
    return (n_params + n_buffers) * bytes_per_element


# ---------------------------------------------------------------------------
# Per-round record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommRoundRecord:
    """Communication cost for one federated training round.

    Attributes:
        round_num: 1-based round index.
        num_clients_upload: Clients that uploaded updates (server ← client).
        num_clients_download: Clients that received the global model (server → client).
        upload_bytes: Total bytes sent from clients to server.
        download_bytes: Total bytes sent from server to clients.
    """

    round_num: int
    num_clients_upload: int
    num_clients_download: int
    upload_bytes: int
    download_bytes: int

    @property
    def total_bytes(self) -> int:
        """Combined upload + download bytes this round."""
        return self.upload_bytes + self.download_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 ** 2)

    @property
    def upload_mb(self) -> float:
        return self.upload_bytes / (1024 ** 2)

    @property
    def download_mb(self) -> float:
        return self.download_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# CommunicationTracker
# ---------------------------------------------------------------------------


class CommunicationTracker:
    """Tracks upload/download communication cost across federated rounds.

    Assumes full-precision (float32) transmission of complete model state dicts
    (parameters + buffers) unless overridden.  The model's size is measured
    once at construction.

    In the runner, call :meth:`record_round` once per completed round using
    data from :attr:`~flp.core.server.FLServer.round_summaries`.

    Args:
        model: Global model — used to compute bytes-per-transmission.
        dtype: Assumed transmission dtype.
        include_buffers: Whether to include non-trainable buffer bytes.

    Example::

        tracker = CommunicationTracker(global_model)
        for summary in server.round_summaries:
            if not summary.skipped:
                tracker.record_round(
                    round_num=summary.round_num,
                    num_clients_upload=len(summary.active_clients),
                    num_clients_download=total_clients,
                )
        tracker.save("outputs/communication.json")
    """

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.float32,
        include_buffers: bool = True,
    ) -> None:
        self._bytes_per_model = model_size_bytes(
            model, dtype=dtype, include_buffers=include_buffers
        )
        self._records: list[CommRoundRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_round(
        self,
        round_num: int,
        num_clients_upload: int,
        num_clients_download: int,
    ) -> CommRoundRecord:
        """Record communication cost for one round.

        Args:
            round_num: 1-based round index.
            num_clients_upload: Number of active clients that sent updates.
            num_clients_download: Number of clients that received the broadcast.

        Returns:
            The :class:`CommRoundRecord` that was appended.
        """
        record = CommRoundRecord(
            round_num=round_num,
            num_clients_upload=num_clients_upload,
            num_clients_download=num_clients_download,
            upload_bytes=num_clients_upload * self._bytes_per_model,
            download_bytes=num_clients_download * self._bytes_per_model,
        )
        self._records.append(record)
        return record

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def records(self) -> list[CommRoundRecord]:
        """All per-round records in chronological order."""
        return list(self._records)

    @property
    def bytes_per_model(self) -> int:
        """Byte size of a single full model transmission."""
        return self._bytes_per_model

    @property
    def total_upload_bytes(self) -> int:
        """Cumulative bytes uploaded (client → server)."""
        return sum(r.upload_bytes for r in self._records)

    @property
    def total_download_bytes(self) -> int:
        """Cumulative bytes downloaded (server → client)."""
        return sum(r.download_bytes for r in self._records)

    @property
    def total_bytes(self) -> int:
        """Total bytes in both directions across all rounds."""
        return self.total_upload_bytes + self.total_download_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 ** 2)

    @property
    def total_upload_mb(self) -> float:
        return self.total_upload_bytes / (1024 ** 2)

    @property
    def total_download_mb(self) -> float:
        return self.total_download_bytes / (1024 ** 2)

    @property
    def bytes_per_round(self) -> list[int]:
        """Total bytes transferred per round."""
        return [r.total_bytes for r in self._records]

    @property
    def cumulative_bytes(self) -> list[int]:
        """Running total bytes up to and including each round."""
        running = 0
        result = []
        for r in self._records:
            running += r.total_bytes
            result.append(running)
        return result

    # ------------------------------------------------------------------
    # Summary & persistence
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, object]:
        """Return a JSON-serialisable summary of all communication costs.

        Returns:
            Dict with keys ``num_rounds``, ``bytes_per_model``,
            ``total_upload_bytes``, ``total_download_bytes``, ``total_bytes``,
            ``total_upload_mb``, ``total_download_mb``, ``total_mb``.
        """
        return {
            "num_rounds": len(self._records),
            "bytes_per_model": self._bytes_per_model,
            "total_upload_bytes": self.total_upload_bytes,
            "total_download_bytes": self.total_download_bytes,
            "total_bytes": self.total_bytes,
            "total_upload_mb": round(self.total_upload_mb, 4),
            "total_download_mb": round(self.total_download_mb, 4),
            "total_mb": round(self.total_mb, 4),
        }

    def save(self, path: str | Path) -> None:
        """Write per-round records and summary to a JSON file.

        Args:
            path: Output file path (parent dirs are created if needed).
        """
        output: dict[str, object] = {
            "summary": self.summary(),
            "rounds": [
                {
                    "round_num": r.round_num,
                    "num_clients_upload": r.num_clients_upload,
                    "num_clients_download": r.num_clients_download,
                    "upload_bytes": r.upload_bytes,
                    "download_bytes": r.download_bytes,
                    "total_bytes": r.total_bytes,
                    "upload_mb": round(r.upload_mb, 6),
                    "download_mb": round(r.download_mb, 6),
                }
                for r in self._records
            ],
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
