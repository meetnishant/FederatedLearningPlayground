"""Experiment runner: wires together all FLP components and executes a full run."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from flp.core.client import FLClient
from flp.core.models import build_model
from flp.core.server import FLServer
from flp.experiments.config_loader import ExperimentConfig
from flp.metrics.communication import CommunicationTracker
from flp.metrics.tracker import MetricsTracker
from flp.simulation.partitioning import DataPartitioner
from flp.visualization.plots import save_all_plots

logger = logging.getLogger(__name__)



class ExperimentRunner:
    """Orchestrates a complete federated learning experiment.

    Args:
        config: Fully validated experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._set_seeds()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config.output.dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _set_seeds(self) -> None:
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self) -> MetricsTracker:
        """Execute the full experiment.

        Returns:
            Populated :class:`~flp.metrics.tracker.MetricsTracker`.
        """
        logger.info("Starting experiment: %s", self.config.name)
        logger.info("Device: %s | Seed: %d", self.device, self.config.seed)

        # ----- Data -----
        train_dataset, test_loader = self._load_data()

        # ----- Partition -----
        partitioner = DataPartitioner(
            dataset=train_dataset,
            num_clients=self.config.training.num_clients,
            strategy=self.config.simulation.partitioning,
            seed=self.config.seed,
            alpha=self.config.simulation.alpha,
            num_shards_per_client=self.config.simulation.num_shards_per_client,
        )
        client_indices = partitioner.partition()
        logger.info(
            "Partitioned %d samples across %d clients using '%s'.",
            sum(len(idx) for idx in client_indices),
            self.config.training.num_clients,
            self.config.simulation.partitioning,
        )

        # ----- Model -----
        global_model = build_model("cnn")

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

        # ----- Communication tracking -----
        comm_tracker = CommunicationTracker(global_model)

        # ----- Server -----
        server = FLServer(
            model=global_model,
            clients=clients,
            config=self.config,
            test_loader=test_loader,
            device=self.device,
        )

        metrics = server.run()

        # Record comm cost from round_summaries so skipped rounds get 0-upload entries
        for rs in server.round_summaries:
            comm_tracker.record_round(
                round_num=rs.round_num,
                num_clients_upload=len(rs.active_clients),
                num_clients_download=len(rs.selected_clients),
            )

        # ----- Summary -----
        summary = metrics.summary()
        comm_summary = comm_tracker.summary()
        dropout_summary = server.dropout_sim.metrics.summary()

        logger.info("=" * 60)
        logger.info("Experiment complete: %s", self.config.name)
        logger.info(
            "Accuracy  — best: %.4f | final: %.4f | improvement: %+.4f",
            summary["best_accuracy"],
            summary["final_accuracy"],
            summary["accuracy_improvement"],
        )
        logger.info(
            "Communication — %.2f MB total (↑ %.2f MB upload, ↓ %.2f MB download)",
            comm_summary["total_mb"],
            comm_summary["total_upload_mb"],
            comm_summary["total_download_mb"],
        )
        logger.info(
            "Dropout   — overall rate: %.1f%% | skipped rounds: %d",
            dropout_summary["overall_dropout_rate"] * 100,
            dropout_summary["total_skipped_rounds"],
        )

        # ----- Save outputs -----
        if self.config.output.save_metrics:
            metrics_path = self.output_dir / "metrics.json"
            metrics.save(metrics_path)
            comm_tracker.save(self.output_dir / "communication.json")
            logger.info("Metrics saved to %s", self.output_dir)

        if self.config.output.save_plots:
            plots_dir = self.output_dir / "plots"
            save_all_plots(metrics, str(plots_dir))
            logger.info("Plots saved to %s", plots_dir)

        if self.config.output.save_model:
            model_path = self.output_dir / "global_model.pt"
            torch.save(server.model.state_dict(), model_path)
            logger.info("Global model saved to %s", model_path)

        return metrics

    def _load_data(
        self,
    ) -> tuple[torchvision.datasets.MNIST, DataLoader]:  # type: ignore[type-arg]
        data_dir = Path(self.config.data_dir).expanduser()
        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

        train_dataset = torchvision.datasets.MNIST(
            root=str(data_dir), train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=str(data_dir), train=False, download=True, transform=transform
        )
        test_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
            test_dataset, batch_size=256, shuffle=False, num_workers=0
        )
        return train_dataset, test_loader
