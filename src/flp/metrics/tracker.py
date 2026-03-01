"""Metrics tracking for federated learning experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RoundMetrics:
    """Metrics recorded for a single federated training round."""

    round_num: int
    global_accuracy: float
    global_loss: float
    per_client_accuracy: dict[int, float]
    num_active_clients: int
    avg_client_loss: float
    total_samples: int


class MetricsTracker:
    """Accumulates and summarises metrics across all federated training rounds.

    Usage::

        tracker = MetricsTracker()
        tracker.record_round(round_num=1, global_accuracy=0.85, ...)
        tracker.summary()
        tracker.save("outputs/metrics.json")
    """

    def __init__(self) -> None:
        self._rounds: list[RoundMetrics] = []

    def record_round(
        self,
        round_num: int,
        global_accuracy: float,
        global_loss: float,
        per_client_accuracy: dict[int, float],
        num_active_clients: int,
        client_updates: list[dict[str, object]],
    ) -> None:
        """Record metrics for one federated round.

        Args:
            round_num: The current round index (1-based).
            global_accuracy: Accuracy of the aggregated global model on the test set.
            global_loss: Cross-entropy loss of the global model on the test set.
            per_client_accuracy: Mapping of client_id -> local accuracy.
            num_active_clients: Number of clients that completed training.
            client_updates: Raw update dicts from each participating client.
        """
        losses = [float(u["loss"]) for u in client_updates if "loss" in u]  # type: ignore[arg-type]
        samples = [int(u["num_samples"]) for u in client_updates if "num_samples" in u]  # type: ignore[arg-type]
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        total_samples = sum(samples)

        self._rounds.append(
            RoundMetrics(
                round_num=round_num,
                global_accuracy=global_accuracy,
                global_loss=global_loss,
                per_client_accuracy=per_client_accuracy,
                num_active_clients=num_active_clients,
                avg_client_loss=avg_loss,
                total_samples=total_samples,
            )
        )

    @property
    def rounds(self) -> list[RoundMetrics]:
        """All recorded round metrics in chronological order."""
        return list(self._rounds)

    @property
    def global_accuracies(self) -> list[float]:
        """Global accuracy per round."""
        return [r.global_accuracy for r in self._rounds]

    @property
    def global_losses(self) -> list[float]:
        """Global loss per round."""
        return [r.global_loss for r in self._rounds]

    def best_accuracy(self) -> float:
        """Return the highest global accuracy achieved across all rounds."""
        return max(self.global_accuracies) if self._rounds else 0.0

    def summary(self) -> dict[str, object]:
        """Return a high-level summary dictionary.

        Returns:
            Dict with keys: ``num_rounds``, ``best_accuracy``, ``final_accuracy``,
            ``final_loss``.
        """
        if not self._rounds:
            return {"num_rounds": 0, "best_accuracy": 0.0, "final_accuracy": 0.0, "final_loss": 0.0}

        last = self._rounds[-1]
        return {
            "num_rounds": len(self._rounds),
            "best_accuracy": self.best_accuracy(),
            "final_accuracy": last.global_accuracy,
            "final_loss": last.global_loss,
        }

    def save(self, path: str | Path) -> None:
        """Serialise all metrics to a JSON file.

        Args:
            path: Output file path.
        """
        output = {
            "summary": self.summary(),
            "rounds": [
                {
                    "round_num": r.round_num,
                    "global_accuracy": r.global_accuracy,
                    "global_loss": r.global_loss,
                    "per_client_accuracy": {str(k): v for k, v in r.per_client_accuracy.items()},
                    "num_active_clients": r.num_active_clients,
                    "avg_client_loss": r.avg_client_loss,
                    "total_samples": r.total_samples,
                }
                for r in self._rounds
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
