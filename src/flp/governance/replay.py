"""Deterministic replay manifest: all metadata needed to reproduce an experiment."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_SCHEMA_VERSION = "1.1"


# ---------------------------------------------------------------------------
# Per-round lineage record
# ---------------------------------------------------------------------------


@dataclass
class RoundLineageRecord:
    """Model lineage entry for one federated round.

    Together with the global ``seed`` and ``selection_seed``, an auditor can
    independently recompute which clients were selected and verify that the
    pre- and post-round model hashes match a replay run.

    Attributes:
        round_num: 1-based round index.
        selection_seed: Seed used for client selection this round.
            Derived as ``config.seed + round_num * 997``.
        selected_clients: Client IDs sampled this round (pre-dropout).
        active_clients: Client IDs that completed local training.
        pre_round_model_hash: SHA-256 of the global model *before* this round.
        post_round_model_hash: SHA-256 of the global model *after* aggregation.
            Equals ``pre_round_model_hash`` for skipped rounds.
        skipped: True if all selected clients dropped out (no model update).
    """

    round_num: int
    selection_seed: int
    selected_clients: list[int]
    active_clients: list[int]
    pre_round_model_hash: str
    post_round_model_hash: str
    skipped: bool


# ---------------------------------------------------------------------------
# Replay manifest
# ---------------------------------------------------------------------------


class ReplayManifest:
    """Captures all metadata required to deterministically replay an experiment.

    A replay manifest answers: *"Given only this file, can I reproduce every
    training decision made during this run?"*  The answer is yes when:

    - The same ``seed`` is used.
    - The config identified by ``config_hash`` is applied unchanged.
    - Environment library versions match (``torch``, Python).
    - The model identified by ``initial_model_hash`` is recreated
      (same architecture, same global RNG state at initialisation time).

    Use :meth:`verify_config` and :meth:`verify_initial_model` to
    programmatically validate a replay setup against a recorded manifest.

    Usage::

        manifest = ReplayManifest(config_snapshot, config_hash, name, seed)
        manifest.set_initial_model("cnn", num_params, initial_hash)
        manifest.set_data_info("mnist", 60000, 10000, "dirichlet", 0.5, 10, counts)
        for round_record in ...:
            manifest.add_round(round_record)
        manifest.save(output_dir)

    Args:
        config_snapshot: Plain-dict snapshot of the full experiment config.
        config_hash: SHA-256 of the serialised config snapshot.
        experiment_name: Human-readable experiment name.
        seed: Global random seed used for this run.
    """

    def __init__(
        self,
        config_snapshot: dict[str, Any],
        config_hash: str,
        experiment_name: str,
        seed: int,
    ) -> None:
        self.config_snapshot = config_snapshot
        self.config_hash = config_hash
        self.experiment_name = experiment_name
        self.seed = seed
        self.generated_at = datetime.now(timezone.utc).isoformat()

        self._model_info: dict[str, Any] = {}
        self._data_info: dict[str, Any] = {}
        self._round_lineage: list[RoundLineageRecord] = []
        self._features: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def set_initial_model(
        self,
        architecture: str,
        num_params: int,
        initial_model_hash: str,
    ) -> None:
        """Record the initial (pre-training) model's identity.

        Args:
            architecture: Architecture name (e.g. ``"cnn"``).
            num_params: Total trainable parameter count.
            initial_model_hash: SHA-256 of the initial state dict.
        """
        self._model_info = {
            "architecture": architecture,
            "num_trainable_params": num_params,
            "initial_model_hash": initial_model_hash,
        }

    def set_data_info(
        self,
        dataset: str,
        train_samples: int,
        test_samples: int,
        partitioning: str,
        alpha: float,
        num_clients: int,
        client_sample_counts: list[int],
    ) -> None:
        """Record dataset and partitioning metadata.

        Args:
            dataset: Dataset name (e.g. ``"mnist"``).
            train_samples: Total training set size.
            test_samples: Total test set size.
            partitioning: Partitioning strategy (``"iid"``, ``"dirichlet"``, ``"shard"``).
            alpha: Dirichlet concentration parameter.
            num_clients: Number of simulated clients.
            client_sample_counts: Per-client sample counts, indexed by client ID.
        """
        self._data_info = {
            "dataset": dataset,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "partitioning": partitioning,
            "alpha": alpha,
            "num_clients": num_clients,
            "client_sample_counts": client_sample_counts,
        }

    def add_round(self, record: RoundLineageRecord) -> None:
        """Append a per-round lineage record.

        Args:
            record: Lineage data for one completed or skipped round.
        """
        self._round_lineage.append(record)

    def set_feature_flags(self, config: Any) -> None:
        """Capture which Phase-2 features were active in this experiment.

        Extracts ``async_fl``, ``compression``, and ``privacy`` settings from
        the config and records them in a top-level ``features`` section of the
        manifest so reviewers can see at a glance what was enabled without
        digging through the full config blob.

        Args:
            config: Fully validated :class:`~flp.experiments.config_loader.ExperimentConfig`
                instance from the experiment runner.
        """
        self._features = {
            "async_fl": {
                "enabled": config.async_fl.enabled,
                "delay_min": config.async_fl.delay_min,
                "delay_max": config.async_fl.delay_max,
                "staleness_threshold": config.async_fl.staleness_threshold,
                "staleness_strategy": config.async_fl.staleness_strategy,
                "staleness_decay_factor": config.async_fl.staleness_decay_factor,
            },
            "compression": {
                "enabled": config.compression.enabled,
                "strategy": config.compression.strategy,
                "topk_ratio": config.compression.topk_ratio,
                "quantization_bits": config.compression.quantization_bits,
                "error_feedback": config.compression.error_feedback,
            },
            "differential_privacy": {
                "enabled": config.privacy.enabled,
                "epsilon": config.privacy.epsilon,
                "delta": config.privacy.delta,
                "max_grad_norm": config.privacy.max_grad_norm,
            },
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the manifest to a JSON-compatible plain dict."""
        d: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "experiment": {
                "name": self.experiment_name,
                "seed": self.seed,
                "config_hash": self.config_hash,
            },
            "environment": _capture_environment(),
            "model": self._model_info,
            "data": self._data_info,
            "config": self.config_snapshot,
            "round_lineage": [
                {
                    "round_num": r.round_num,
                    "selection_seed": r.selection_seed,
                    "selected_clients": r.selected_clients,
                    "active_clients": r.active_clients,
                    "pre_round_model_hash": r.pre_round_model_hash,
                    "post_round_model_hash": r.post_round_model_hash,
                    "skipped": r.skipped,
                }
                for r in self._round_lineage
            ],
        }
        if self._features:
            d["features"] = self._features
        return d

    def save(self, output_dir: Path) -> None:
        """Write the manifest to ``replay_manifest.json``.

        Args:
            output_dir: Target directory; created recursively if absent.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "replay_manifest.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def verify_config(self, config_hash: str) -> bool:
        """Return ``True`` if ``config_hash`` matches the recorded config hash.

        Args:
            config_hash: SHA-256 hash of the candidate config.
        """
        return config_hash == self.config_hash

    def verify_initial_model(self, model_hash: str) -> bool:
        """Return ``True`` if ``model_hash`` matches the recorded initial model hash.

        Args:
            model_hash: SHA-256 hash of the candidate model.
        """
        return model_hash == self._model_info.get("initial_model_hash", "")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _capture_environment() -> dict[str, str]:
    """Capture the runtime environment for reproducibility auditing."""
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "hostname": hostname,
        "git_commit_hash": _get_git_commit_hash(),
    }


def _get_git_commit_hash() -> str:
    """Return the current git HEAD commit hash, or ``"unknown"`` if unavailable.

    Uses ``git rev-parse HEAD`` in the working directory.  Returns
    ``"unknown"`` if git is not installed, the working directory is not a
    repository, or any other error occurs, so the manifest is always
    serialisable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"
