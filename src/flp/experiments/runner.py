"""Experiment runner: wires together all FLP components and executes a full run."""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from flp.core.client import FLClient
from flp.core.models import build_model
from flp.core.server import FLServer, RoundSummary
from flp.experiments.config_loader import ExperimentConfig
from flp.governance.audit import AuditLog
from flp.governance.hashing import hash_config, hash_state_dict
from flp.governance.replay import ReplayManifest, RoundLineageRecord
from flp.metrics.communication import CommunicationTracker
from flp.metrics.tracker import MetricsTracker
from flp.simulation.partitioning import DataPartitioner
from flp.visualization.plots import save_all_plots

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates a complete federated learning experiment.

    Loads the dataset, partitions data across simulated clients, runs all
    federated rounds via :class:`~flp.core.server.FLServer`, and persists
    metrics, communication cost, plots, a human-readable summary JSON, and
    (when ``config.governance.enabled``) a full governance package.

    Args:
        config: Fully validated experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._set_seeds()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config.output.dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        round_callback: Callable[[RoundSummary], None] | None = None,
    ) -> MetricsTracker:
        """Execute the full experiment end-to-end.

        Args:
            round_callback: Optional callable invoked after each completed
                (non-skipped) federated round. Receives the :class:`RoundSummary`
                for that round. Useful for live progress displays.

        Returns:
            Fully populated :class:`~flp.metrics.tracker.MetricsTracker`.
        """
        t_experiment_start = time.perf_counter()

        logger.info("=" * 60)
        logger.info("Experiment : %s", self.config.name)
        logger.info("Device     : %s | Seed: %d", self.device, self.config.seed)
        logger.info(
            "Training   : %d rounds | %d clients | fraction=%.2f",
            self.config.training.num_rounds,
            self.config.training.num_clients,
            self.config.training.client_fraction,
        )
        logger.info(
            "Partition  : %s | alpha=%.2f | dropout=%.2f",
            self.config.simulation.partitioning,
            self.config.simulation.alpha,
            self.config.simulation.dropout_rate,
        )
        if self.config.governance.enabled:
            logger.info("Governance : ENABLED (audit log + replay manifest)")
        logger.info("=" * 60)

        # ----- Data -----
        logger.info("Loading dataset '%s'...", self.config.dataset)
        train_dataset, test_loader = self._load_data()
        logger.info(
            "Dataset loaded: %d train samples, %d test batches.",
            len(train_dataset),  # type: ignore[arg-type]
            len(test_loader),
        )

        # ----- Partition -----
        logger.info(
            "Partitioning data across %d clients using '%s'...",
            self.config.training.num_clients,
            self.config.simulation.partitioning,
        )
        partitioner = DataPartitioner(
            dataset=train_dataset,
            num_clients=self.config.training.num_clients,
            strategy=self.config.simulation.partitioning,
            seed=self.config.seed,
            alpha=self.config.simulation.alpha,
            num_shards_per_client=self.config.simulation.num_shards_per_client,
        )
        client_indices = partitioner.partition()
        total_partitioned = sum(len(idx) for idx in client_indices)
        logger.info(
            "Partitioned %d samples across %d clients (min=%d, max=%d).",
            total_partitioned,
            self.config.training.num_clients,
            min(len(idx) for idx in client_indices),
            max(len(idx) for idx in client_indices),
        )

        # ----- Model -----
        global_model = build_model("cnn")
        n_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
        logger.info("Global model: %s | trainable params: %d", type(global_model).__name__, n_params)

        # ----- Governance setup (before any training) -----
        gov_enabled = self.config.governance.enabled
        audit_log: AuditLog | None = None
        replay_manifest: ReplayManifest | None = None
        config_snapshot = self._build_config_snapshot()

        if gov_enabled:
            initial_model_hash = hash_state_dict(global_model.state_dict())
            config_hash = hash_config(config_snapshot)
            audit_log = AuditLog()
            replay_manifest = ReplayManifest(
                config_snapshot=config_snapshot,
                config_hash=config_hash,
                experiment_name=self.config.name,
                seed=self.config.seed,
            )
            replay_manifest.set_initial_model(
                architecture="cnn",
                num_params=n_params,
                initial_model_hash=initial_model_hash,
            )
            replay_manifest.set_data_info(
                dataset=self.config.dataset,
                train_samples=len(train_dataset),  # type: ignore[arg-type]
                test_samples=len(test_loader.dataset),  # type: ignore[arg-type]
                partitioning=self.config.simulation.partitioning,
                alpha=self.config.simulation.alpha,
                num_clients=self.config.training.num_clients,
                client_sample_counts=[len(idx) for idx in client_indices],
            )
            logger.info(
                "Governance: initial model hash = %s", initial_model_hash[:20] + "..."
            )

        # ----- Clients -----
        clients = [
            FLClient(
                client_id=i,
                dataset=train_dataset,
                indices=client_indices[i],
                model=global_model,
                config=self.config.client,
                device=self.device,
                seed=self.config.seed,
            )
            for i in range(self.config.training.num_clients)
        ]
        logger.info("Initialised %d clients.", len(clients))

        # ----- Communication tracker -----
        comm_tracker = CommunicationTracker(global_model)
        logger.info(
            "Model size: %.3f MB per transmission.",
            comm_tracker.bytes_per_model / (1024 ** 2),
        )

        # ----- Server -----
        if self.config.async_fl.enabled:
            from flp.core.async_server import AsyncFLServer
            logger.info(
                "Async FL : ENABLED (delay=[%.1f, %.1f] rounds, staleness_threshold=%d)",
                self.config.async_fl.delay_min,
                self.config.async_fl.delay_max,
                self.config.async_fl.staleness_threshold,
            )
            server: FLServer = AsyncFLServer(
                model=global_model,
                clients=clients,
                config=self.config,
                test_loader=test_loader,
                device=self.device,
                round_callback=round_callback,
                audit_log=audit_log,
            )
        else:
            server = FLServer(
                model=global_model,
                clients=clients,
                config=self.config,
                test_loader=test_loader,
                device=self.device,
                round_callback=round_callback,
                audit_log=audit_log,
            )

        # ----- Training -----
        metrics = server.run()

        # ----- Comm cost from round_summaries -----
        for rs in server.round_summaries:
            comm_tracker.record_round(
                round_num=rs.round_num,
                num_clients_upload=len(rs.active_clients),
                num_clients_download=len(rs.selected_clients),
            )

        elapsed = time.perf_counter() - t_experiment_start

        # ----- Summaries -----
        summary = metrics.summary()
        comm_summary = comm_tracker.summary()
        dropout_summary = server.dropout_sim.metrics.summary()

        logger.info("=" * 60)
        logger.info("Experiment complete: %s  (%.1fs)", self.config.name, elapsed)
        logger.info(
            "Accuracy  — best: %.4f (round %s) | final: %.4f | delta: %+.4f",
            summary["best_accuracy"],
            summary["best_round"],
            summary["final_accuracy"],
            summary["accuracy_improvement"],
        )
        logger.info(
            "Comm      — %.2f MB total (↑ %.2f MB upload, ↓ %.2f MB download)",
            comm_summary["total_mb"],
            comm_summary["total_upload_mb"],
            comm_summary["total_download_mb"],
        )
        logger.info(
            "Dropout   — overall: %.1f%% | skipped rounds: %d",
            dropout_summary["overall_dropout_rate"] * 100,
            dropout_summary["total_skipped_rounds"],
        )
        if server.dp_accountant is not None:
            dp_summary = server.dp_accountant.summary()
            logger.info(
                "Privacy   — total ε=%.4f | total δ=%.2e | clipped updates: %d",
                dp_summary["total_epsilon"],
                dp_summary["total_delta"],
                dp_summary["total_clients_clipped"],
            )
        if gov_enabled and audit_log is not None:
            gov_summary = audit_log.summary()
            logger.info(
                "Governance — %d events | %d unique model hashes | %d rounds with dropout",
                gov_summary["num_rounds_recorded"],
                gov_summary["unique_model_hashes"],
                gov_summary["num_rounds_with_dropout"],
            )
        logger.info("=" * 60)

        # ----- Persist outputs -----
        if self.config.output.save_metrics:
            metrics.save(self.output_dir / "metrics.json")
            comm_tracker.save(self.output_dir / "communication.json")
            logger.info("Metrics saved to %s/", self.output_dir)

        # summary.json is always written — it's the top-level experiment report
        self._save_summary(
            metrics=metrics,
            comm_summary=comm_summary,
            dropout_summary=dropout_summary,
            server=server,
            elapsed_seconds=elapsed,
            config_snapshot=config_snapshot,
        )
        logger.info("Summary saved to %s/summary.json", self.output_dir)

        if self.config.output.save_plots:
            plots_dir = self.output_dir / "plots"
            save_all_plots(metrics, str(plots_dir), comm_tracker=comm_tracker)
            logger.info("Plots saved to %s/", plots_dir)

        if self.config.output.save_model:
            model_path = self.output_dir / "global_model.pt"
            torch.save(server.model.state_dict(), model_path)
            logger.info("Global model saved to %s", model_path)

        # ----- Governance outputs -----
        if gov_enabled and audit_log is not None and replay_manifest is not None:
            gov_dir = self.output_dir / "governance"

            # Populate replay manifest with per-round lineage from audit log
            for event in audit_log.events:
                replay_manifest.add_round(RoundLineageRecord(
                    round_num=event.round_num,
                    selection_seed=self.config.seed + event.round_num * 997,
                    selected_clients=event.selected_clients,
                    active_clients=event.active_clients,
                    pre_round_model_hash=event.pre_round_model_hash,
                    post_round_model_hash=event.post_round_model_hash,
                    skipped=event.skipped,
                ))

            if self.config.governance.save_audit_log:
                audit_log.save(gov_dir)
                logger.info("Audit log saved to %s/", gov_dir)

            if self.config.governance.save_replay_manifest:
                replay_manifest.save(gov_dir)
                logger.info("Replay manifest saved to %s/replay_manifest.json", gov_dir)

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _set_seeds(self) -> None:
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_data(
        self,
    ) -> tuple[torchvision.datasets.MNIST, DataLoader]:  # type: ignore[type-arg]
        """Download (if needed) and load the configured dataset.

        Returns:
            ``(train_dataset, test_loader)`` tuple.
        """
        data_dir = Path(self.config.data_dir).expanduser()
        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

        train_dataset = torchvision.datasets.MNIST(
            root=str(data_dir), train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=str(data_dir), train=False, download=True, transform=transform
        )
        test_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
            test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
        )
        return train_dataset, test_loader

    def _build_config_snapshot(self) -> dict[str, object]:
        """Build a JSON-serialisable snapshot of the full experiment config."""
        return {
            "name": self.config.name,
            "seed": self.config.seed,
            "dataset": self.config.dataset,
            "data_dir": self.config.data_dir,
            "training": {
                "num_rounds": self.config.training.num_rounds,
                "num_clients": self.config.training.num_clients,
                "client_fraction": self.config.training.client_fraction,
            },
            "client": {
                "batch_size": self.config.client.batch_size,
                "local_epochs": self.config.client.local_epochs,
                "lr": self.config.client.lr,
                "momentum": self.config.client.momentum,
                "weight_decay": self.config.client.weight_decay,
            },
            "simulation": {
                "partitioning": self.config.simulation.partitioning,
                "alpha": self.config.simulation.alpha,
                "num_shards_per_client": self.config.simulation.num_shards_per_client,
                "dropout_rate": self.config.simulation.dropout_rate,
            },
            "privacy": {
                "enabled": self.config.privacy.enabled,
                "epsilon": self.config.privacy.epsilon,
                "delta": self.config.privacy.delta,
                "max_grad_norm": self.config.privacy.max_grad_norm,
            },
            "governance": {
                "enabled": self.config.governance.enabled,
                "save_audit_log": self.config.governance.save_audit_log,
                "save_replay_manifest": self.config.governance.save_replay_manifest,
            },
            "output": {
                "dir": self.config.output.dir,
                "save_plots": self.config.output.save_plots,
                "save_metrics": self.config.output.save_metrics,
                "save_model": self.config.output.save_model,
            },
            "async_fl": {
                "enabled": self.config.async_fl.enabled,
                "delay_min": self.config.async_fl.delay_min,
                "delay_max": self.config.async_fl.delay_max,
                "staleness_threshold": self.config.async_fl.staleness_threshold,
            },
        }

    def _save_summary(
        self,
        metrics: MetricsTracker,
        comm_summary: dict[str, object],
        dropout_summary: dict[str, object],
        server: FLServer,
        elapsed_seconds: float,
        config_snapshot: dict[str, object] | None = None,
    ) -> None:
        """Write a comprehensive ``summary.json`` to the output directory."""
        training_summary = metrics.summary()

        timing_by_round = {rs.round_num: rs.elapsed_seconds for rs in server.round_summaries}
        skipped_rounds = {rs.round_num for rs in server.round_summaries if rs.skipped}

        round_history = []
        for r in metrics.rounds:
            round_history.append({
                "round": r.round_num,
                "global_accuracy": round(r.global_accuracy, 6),
                "global_loss": round(r.global_loss, 6),
                "active_clients": r.num_active_clients,
                "total_samples": r.total_samples,
                "avg_client_loss": round(r.avg_client_loss, 6),
                "min_client_accuracy": round(r.min_client_accuracy, 6),
                "max_client_accuracy": round(r.max_client_accuracy, 6),
                "elapsed_seconds": round(timing_by_round.get(r.round_num, 0.0), 3),
            })

        snapshot = config_snapshot if config_snapshot is not None else self._build_config_snapshot()

        output: dict[str, object] = {
            "experiment": {
                "name": self.config.name,
                "seed": self.config.seed,
                "dataset": self.config.dataset,
                "device": str(self.device),
                "output_dir": str(self.output_dir),
                "elapsed_seconds": round(elapsed_seconds, 2),
            },
            "config": snapshot,
            "results": {
                "num_rounds_completed": int(training_summary["num_rounds"]),  # type: ignore[arg-type]
                "num_rounds_skipped": len(skipped_rounds),
                "best_accuracy": training_summary["best_accuracy"],
                "best_round": training_summary["best_round"],
                "final_accuracy": training_summary["final_accuracy"],
                "final_loss": training_summary["final_loss"],
                "accuracy_improvement": training_summary["accuracy_improvement"],
                "avg_active_clients": training_summary["avg_active_clients"],
            },
            "communication": comm_summary,
            "dropout": dropout_summary,
            "round_history": round_history,
        }

        summary_path = self.output_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(output, f, indent=2)
