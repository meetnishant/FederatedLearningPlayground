"""Federated learning server that orchestrates training rounds."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flp.core.aggregator import FedAvgAggregator
from flp.core.client import FLClient
from flp.metrics.tracker import MetricsTracker
from flp.simulation.dropout import DropoutSimulator

if TYPE_CHECKING:
    from flp.experiments.config_loader import ExperimentConfig

logger = logging.getLogger(__name__)


class FLServer:
    """Central server coordinating federated training rounds.

    The server:
    1. Broadcasts the global model to selected clients each round.
    2. Collects and aggregates local updates.
    3. Evaluates the new global model on a held-out test set.
    4. Records all metrics via :class:`~flp.metrics.tracker.MetricsTracker`.

    Args:
        model: The global model (acts as a blueprint; weights are updated in-place).
        clients: All available federated clients.
        config: Full experiment configuration.
        test_loader: DataLoader for global evaluation.
        device: Torch device for server-side evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        clients: list[FLClient],
        config: ExperimentConfig,
        test_loader: DataLoader,  # type: ignore[type-arg]
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.clients = clients
        self.config = config
        self.test_loader = test_loader
        self.device = device
        self.aggregator = FedAvgAggregator()
        self.dropout_sim = DropoutSimulator(
            dropout_rate=config.simulation.dropout_rate,
            seed=config.seed,
        )
        self.metrics = MetricsTracker()

    def run(self) -> MetricsTracker:
        """Execute all federated training rounds.

        Returns:
            Populated :class:`~flp.metrics.tracker.MetricsTracker` instance.
        """
        num_rounds = self.config.training.num_rounds
        fraction = self.config.training.client_fraction
        num_select = max(1, int(len(self.clients) * fraction))

        for round_num in range(1, num_rounds + 1):
            logger.info("--- Round %d / %d ---", round_num, num_rounds)

            # Client selection
            selected = self._select_clients(num_select, round_num)
            active = self.dropout_sim.apply(selected, round_num)

            if not active:
                logger.warning("Round %d: all selected clients dropped out.", round_num)
                continue

            logger.info(
                "Round %d: %d/%d clients active after dropout.",
                round_num,
                len(active),
                len(selected),
            )

            # Broadcast global weights and collect updates
            global_state = copy.deepcopy(self.model.state_dict())
            updates: list[dict[str, object]] = []

            for client in active:
                client.set_global_weights(global_state)
                update = client.train()
                updates.append(update)
                logger.debug(
                    "  Client %d | samples=%d | loss=%.4f",
                    update["client_id"],
                    update["num_samples"],
                    update["loss"],
                )

            # Aggregate
            new_state = self.aggregator.aggregate(updates)
            self.model.load_state_dict(new_state)

            # Global evaluation
            global_eval = self._evaluate_global()
            logger.info(
                "Round %d | global_acc=%.4f | global_loss=%.4f",
                round_num,
                global_eval["accuracy"],
                global_eval["loss"],
            )

            # Per-client evaluation
            per_client_acc: dict[int, float] = {}
            for client in active:
                client.set_global_weights(new_state)
                result = client.evaluate()
                per_client_acc[client.client_id] = result["accuracy"]

            # Record metrics
            self.metrics.record_round(
                round_num=round_num,
                global_accuracy=global_eval["accuracy"],
                global_loss=global_eval["loss"],
                per_client_accuracy=per_client_acc,
                num_active_clients=len(active),
                client_updates=updates,
            )

        return self.metrics

    def _select_clients(self, k: int, round_num: int) -> list[FLClient]:
        """Randomly select ``k`` clients for this round."""
        rng = torch.Generator()
        rng.manual_seed(self.config.seed + round_num)
        indices = torch.randperm(len(self.clients), generator=rng)[:k].tolist()
        return [self.clients[i] for i in indices]

    def _evaluate_global(self) -> dict[str, float]:
        """Evaluate the global model on the held-out test set."""
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += inputs.size(0)

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "loss": total_loss / total if total > 0 else 0.0,
        }
