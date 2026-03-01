"""Federated learning server: round orchestration, aggregation, and evaluation."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flp.core.aggregator import AggregationResult, FedAvgAggregator
from flp.core.client import ClientUpdate, FLClient
from flp.metrics.tracker import MetricsTracker
from flp.simulation.dropout import DropoutSimulator

if TYPE_CHECKING:
    from flp.experiments.config_loader import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class RoundSummary:
    """Diagnostic snapshot of a single federated training round.

    Attributes:
        round_num: 1-based round index.
        selected_clients: IDs of clients sampled this round.
        active_clients: IDs of clients that completed training (post-dropout).
        dropped_clients: IDs of clients that dropped out.
        aggregation: Result returned by the aggregator (None if round skipped).
        global_accuracy: Test-set accuracy of the new global model.
        global_loss: Test-set cross-entropy loss of the new global model.
        elapsed_seconds: Wall-clock time for this round.
        skipped: True if all selected clients dropped out (no update produced).
    """

    round_num: int
    selected_clients: list[int]
    active_clients: list[int]
    dropped_clients: list[int]
    aggregation: AggregationResult | None
    global_accuracy: float
    global_loss: float
    elapsed_seconds: float
    skipped: bool = False


class FLServer:
    """Central server that orchestrates federated training rounds.

    The server implements the outer loop of FedAvg:

    1. **Select** a random subset of clients each round.
    2. **Broadcast** the current global model weights to each selected client.
    3. **Collect** local model updates (post local SGD).
    4. **Aggregate** updates via :class:`~flp.core.aggregator.FedAvgAggregator`.
    5. **Evaluate** the new global model on a held-out test set.
    6. **Record** all metrics in :class:`~flp.metrics.tracker.MetricsTracker`.

    Client selection uses a seeded :func:`torch.randperm` keyed on
    ``seed + round_num * 997``, making the selection order fully reproducible
    without affecting the global RNG state.

    Args:
        model: Global model.  Weights are updated in-place after each round.
        clients: All available federated clients.
        config: Full experiment configuration.
        test_loader: DataLoader for server-side global evaluation.
        device: Torch device used for the global model and evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        clients: list[FLClient],
        config: ExperimentConfig,
        test_loader: DataLoader,  # type: ignore[type-arg]
        device: torch.device,
        round_callback: Callable[[RoundSummary], None] | None = None,
    ) -> None:
        if len(clients) < 2:
            raise ValueError(
                f"FLServer requires at least 2 clients, got {len(clients)}."
            )

        self.model = model.to(device)
        self.clients = clients
        self.config = config
        self.test_loader = test_loader
        self.device = device
        self._round_callback = round_callback

        self.aggregator = FedAvgAggregator()
        self.dropout_sim = DropoutSimulator(
            dropout_rate=config.simulation.dropout_rate,
            seed=config.seed,
        )
        self.metrics = MetricsTracker()
        self._round_summaries: list[RoundSummary] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> MetricsTracker:
        """Execute all federated training rounds defined in the config.

        Returns:
            Fully populated :class:`~flp.metrics.tracker.MetricsTracker`.
        """
        num_rounds = self.config.training.num_rounds
        num_select = max(1, int(len(self.clients) * self.config.training.client_fraction))

        logger.info(
            "Starting federated training: %d rounds, %d/%d clients per round.",
            num_rounds,
            num_select,
            len(self.clients),
        )

        for round_num in range(1, num_rounds + 1):
            summary = self._run_round(round_num, num_select)
            self._round_summaries.append(summary)

            if summary.skipped:
                logger.warning(
                    "Round %d/%d SKIPPED — all %d selected clients dropped out.",
                    round_num,
                    num_rounds,
                    len(summary.selected_clients),
                )
                continue

            logger.info(
                "Round %d/%d | acc=%.4f | loss=%.4f | clients=%d/%d | %.1fs",
                round_num,
                num_rounds,
                summary.global_accuracy,
                summary.global_loss,
                len(summary.active_clients),
                len(summary.selected_clients),
                summary.elapsed_seconds,
            )

            if self._round_callback is not None:
                self._round_callback(summary)

        logger.info("Federated training complete.")
        return self.metrics

    @property
    def round_summaries(self) -> list[RoundSummary]:
        """Per-round diagnostic summaries (available after :meth:`run`)."""
        return list(self._round_summaries)

    # ------------------------------------------------------------------
    # Round implementation
    # ------------------------------------------------------------------

    def _run_round(self, round_num: int, num_select: int) -> RoundSummary:
        """Execute a single federated round and return its summary."""
        t_start = time.perf_counter()

        # ---- Client selection ----
        selected = self._select_clients(num_select, round_num)
        selected_ids = [c.client_id for c in selected]

        # ---- Dropout simulation ----
        dropout_result = self.dropout_sim.apply(selected, round_num)
        self.dropout_sim.record(dropout_result)

        active = dropout_result.active
        active_ids = dropout_result.active_ids
        dropped_ids = dropout_result.dropped_ids

        if dropped_ids:
            logger.debug(
                "Round %d: %d/%d clients dropped out %s.",
                round_num,
                len(dropped_ids),
                len(selected_ids),
                dropped_ids,
            )

        # ---- All dropped out — skip round ----
        if dropout_result.all_dropped:
            return RoundSummary(
                round_num=round_num,
                selected_clients=selected_ids,
                active_clients=[],
                dropped_clients=dropped_ids,
                aggregation=None,
                global_accuracy=0.0,
                global_loss=0.0,
                elapsed_seconds=time.perf_counter() - t_start,
                skipped=True,
            )

        # ---- Broadcast & collect updates ----
        global_state = copy.deepcopy(self.model.state_dict())
        updates: list[ClientUpdate] = []

        for client in active:
            client.set_global_weights(global_state)
            update = client.train()
            updates.append(update)
            logger.debug(
                "  Client %d | samples=%d | loss=%.4f",
                update.client_id,
                update.num_samples,
                update.train_result.loss,
            )

        # ---- Aggregate ----
        agg_result = self.aggregator.aggregate(updates)
        self.model.load_state_dict(agg_result.state_dict)

        # ---- Global evaluation ----
        global_eval = self._evaluate_global()

        # ---- Per-client evaluation (against refreshed global weights) ----
        per_client_acc: dict[int, float] = {}
        new_state = self.model.state_dict()
        for client in active:
            client.set_global_weights(new_state)
            result = client.evaluate()
            per_client_acc[client.client_id] = result.accuracy

        # ---- Record metrics ----
        self.metrics.record_round(
            round_num=round_num,
            global_accuracy=global_eval["accuracy"],
            global_loss=global_eval["loss"],
            per_client_accuracy=per_client_acc,
            num_active_clients=len(active),
            client_updates=[
                {
                    "client_id": u.client_id,
                    "num_samples": u.num_samples,
                    "loss": u.train_result.loss,
                }
                for u in updates
            ],
        )

        return RoundSummary(
            round_num=round_num,
            selected_clients=selected_ids,
            active_clients=active_ids,
            dropped_clients=dropped_ids,
            aggregation=agg_result,
            global_accuracy=global_eval["accuracy"],
            global_loss=global_eval["loss"],
            elapsed_seconds=time.perf_counter() - t_start,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_clients(self, k: int, round_num: int) -> list[FLClient]:
        """Randomly select ``k`` clients using a deterministic per-round seed.

        Uses an isolated :class:`torch.Generator` so selection does not
        consume entropy from the global RNG.

        Args:
            k: Number of clients to select.
            round_num: Current round number (mixed into the seed).

        Returns:
            List of ``k`` selected :class:`~flp.core.client.FLClient` instances.
        """
        rng = torch.Generator()
        # Multiply round_num by a prime to avoid seed collisions between rounds.
        rng.manual_seed(self.config.seed + round_num * 997)
        indices: list[int] = torch.randperm(len(self.clients), generator=rng)[:k].tolist()
        return [self.clients[i] for i in indices]

    def _evaluate_global(self) -> dict[str, float]:
        """Evaluate the current global model on the held-out test loader.

        Returns:
            Dict with keys ``"accuracy"`` (float in [0, 1]) and ``"loss"``.
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                logits = self.model(inputs)
                loss = criterion(logits, targets)
                total_loss += loss.item() * inputs.size(0)
                correct += (logits.argmax(dim=1) == targets).sum().item()
                total += inputs.size(0)

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "loss": total_loss / total if total > 0 else 0.0,
        }
