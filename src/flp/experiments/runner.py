"""Experiment runner: wires together all FLP components and executes a full run."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict
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
from flp.metrics.communication import CommunicationTracker
from flp.metrics.tracker import MetricsTracker
from flp.simulation.partitioning import DataPartitioner
from flp.visualization.plots import save_all_plots

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates a complete federated learning experiment.

    Loads the dataset, partitions data across simulated clients, runs all
    federated rounds via :class:`~flp.core.server.FLServer`, and persists
    metrics, communication cost, plots, and a human-readable summary JSON.

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
        # Default to CNN for MNIST; architecture could be made configurable later.
        global_model = build_model("cnn")
        n_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
        logger.info("Global model: %s | trainable params: %d", type(global_model).__name__, n_params)

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
        server = FLServer(
            model=global_model,
            clients=clients,
            config=self.config,
            test_loader=test_loader,
            device=self.device,
            round_callback=round_callback,
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

    def _save_summary(
        self,
        metrics: MetricsTracker,
        comm_summary: dict[str, object],
        dropout_summary: dict[str, object],
        server: FLServer,
        elapsed_seconds: float,
    ) -> None:
        """Write a comprehensive ``summary.json`` to the output directory.

        The summary contains everything needed to understand the experiment
        outcome without opening the verbose ``metrics.json``:

        - Experiment metadata (name, seed, device, config snapshot)
        - High-level training results (best/final accuracy, improvement)
        - Communication cost totals
        - Dropout statistics
        - Per-round history (round, accuracy, loss, active_clients, elapsed)

        Args:
            metrics: Populated metrics tracker from the completed run.
            comm_summary: Output of :meth:`~flp.metrics.communication.CommunicationTracker.summary`.
            dropout_summary: Output of :meth:`~flp.simulation.dropout.DropoutMetrics.summary`.
            server: Completed server (used for round_summaries timing data).
            elapsed_seconds: Total wall-clock time for the experiment.
        """
        training_summary = metrics.summary()

        # Build a per-round history table for quick inspection
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

        # Config snapshot — serialise as plain dicts
        config_snapshot = {
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
            "output": {
                "dir": self.config.output.dir,
                "save_plots": self.config.output.save_plots,
                "save_metrics": self.config.output.save_metrics,
                "save_model": self.config.output.save_model,
            },
        }

        output: dict[str, object] = {
            "experiment": {
                "name": self.config.name,
                "seed": self.config.seed,
                "dataset": self.config.dataset,
                "device": str(self.device),
                "output_dir": str(self.output_dir),
                "elapsed_seconds": round(elapsed_seconds, 2),
            },
            "config": config_snapshot,
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
