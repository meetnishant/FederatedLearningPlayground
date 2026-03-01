"""Per-round and aggregate metrics tracking for federated learning experiments."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClientRoundMetrics:
    """Metrics for one client in one round.

    Attributes:
        client_id: Client identifier.
        accuracy: Local model accuracy on this client's data after the round.
        loss: Local training loss for this round.
        num_samples: Number of samples used for training.
    """

    client_id: int
    accuracy: float
    loss: float
    num_samples: int


@dataclass
class RoundMetrics:
    """All metrics recorded for a single federated training round.

    Attributes:
        round_num: 1-based round index.
        global_accuracy: Test-set accuracy of the aggregated global model.
        global_loss: Test-set cross-entropy loss of the global model.
        per_client_accuracy: Mapping of client_id → local accuracy.
        num_active_clients: Clients that completed training this round.
        avg_client_loss: Sample-weighted mean training loss across active clients.
        weighted_client_loss: Alias for ``avg_client_loss`` (sample-weighted).
        total_samples: Total training samples processed this round.
        min_client_accuracy: Lowest per-client accuracy (fairness indicator).
        max_client_accuracy: Highest per-client accuracy.
        std_client_accuracy: Std-dev of per-client accuracies (spread indicator).
        client_records: Full per-client breakdown for this round.
    """

    round_num: int
    global_accuracy: float
    global_loss: float
    per_client_accuracy: dict[int, float]
    num_active_clients: int
    avg_client_loss: float
    weighted_client_loss: float
    total_samples: int
    min_client_accuracy: float
    max_client_accuracy: float
    std_client_accuracy: float
    client_records: list[ClientRoundMetrics]


class MetricsTracker:
    """Accumulates and queries metrics across all federated training rounds.

    The tracker is the single source of truth for all training outcomes.
    It is populated by the server each round and consumed by the runner
    for logging, JSON export, and visualisation.

    Usage::

        tracker = MetricsTracker()
        tracker.record_round(round_num=1, global_accuracy=0.85, ...)
        print(tracker.summary())
        tracker.save("outputs/metrics.json")
    """

    def __init__(self) -> None:
        self._rounds: list[RoundMetrics] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_round(
        self,
        round_num: int,
        global_accuracy: float,
        global_loss: float,
        per_client_accuracy: dict[int, float],
        num_active_clients: int,
        client_updates: list[dict[str, object]],
    ) -> None:
        """Record all metrics for one federated round.

        Args:
            round_num: 1-based round index.
            global_accuracy: Test-set accuracy of the aggregated global model.
            global_loss: Test-set cross-entropy loss.
            per_client_accuracy: ``{client_id: accuracy}`` for active clients.
            num_active_clients: Count of clients that completed training.
            client_updates: List of dicts each containing at minimum
                ``"client_id"``, ``"num_samples"``, and ``"loss"``.
        """
        # Build per-client records
        client_records: list[ClientRoundMetrics] = []
        for u in client_updates:
            cid = int(u["client_id"])  # type: ignore[arg-type]
            client_records.append(
                ClientRoundMetrics(
                    client_id=cid,
                    accuracy=per_client_accuracy.get(cid, 0.0),
                    loss=float(u.get("loss", 0.0)),  # type: ignore[arg-type]
                    num_samples=int(u.get("num_samples", 0)),  # type: ignore[arg-type]
                )
            )

        # Weighted average training loss
        total_samples = sum(r.num_samples for r in client_records)
        if total_samples > 0:
            weighted_loss = sum(
                r.loss * r.num_samples for r in client_records
            ) / total_samples
        else:
            weighted_loss = (
                statistics.mean(r.loss for r in client_records)
                if client_records else 0.0
            )

        # Per-client accuracy statistics
        accs = list(per_client_accuracy.values())
        min_acc = min(accs) if accs else 0.0
        max_acc = max(accs) if accs else 0.0
        std_acc = statistics.stdev(accs) if len(accs) >= 2 else 0.0

        self._rounds.append(
            RoundMetrics(
                round_num=round_num,
                global_accuracy=global_accuracy,
                global_loss=global_loss,
                per_client_accuracy=per_client_accuracy,
                num_active_clients=num_active_clients,
                avg_client_loss=weighted_loss,
                weighted_client_loss=weighted_loss,
                total_samples=total_samples,
                min_client_accuracy=min_acc,
                max_client_accuracy=max_acc,
                std_client_accuracy=std_acc,
                client_records=client_records,
            )
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def rounds(self) -> list[RoundMetrics]:
        """All recorded round metrics in chronological order."""
        return list(self._rounds)

    @property
    def global_accuracies(self) -> list[float]:
        """Global accuracy for each recorded round."""
        return [r.global_accuracy for r in self._rounds]

    @property
    def global_losses(self) -> list[float]:
        """Global loss for each recorded round."""
        return [r.global_loss for r in self._rounds]

    @property
    def per_client_accuracies(self) -> dict[int, list[float]]:
        """Per-client accuracy time series: ``{client_id: [acc_r1, acc_r2, ...]}``.

        Only rounds in which a client was active contribute an entry; rounds
        where the client was dropped or not selected are omitted.
        """
        result: dict[int, list[float]] = {}
        for r in self._rounds:
            for cid, acc in r.per_client_accuracy.items():
                result.setdefault(cid, []).append(acc)
        return result

    @property
    def active_client_counts(self) -> list[int]:
        """Number of active clients per round."""
        return [r.num_active_clients for r in self._rounds]

    def best_accuracy(self) -> float:
        """Highest global accuracy achieved across all rounds."""
        return max(self.global_accuracies) if self._rounds else 0.0

    def best_round(self) -> int | None:
        """Round number (1-based) at which best accuracy was achieved."""
        if not self._rounds:
            return None
        return max(self._rounds, key=lambda r: r.global_accuracy).round_num

    def accuracy_improvement(self) -> float:
        """Total accuracy gain from round 1 to the final round."""
        if len(self._rounds) < 2:
            return 0.0
        return self._rounds[-1].global_accuracy - self._rounds[0].global_accuracy

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, object]:
        """Return a JSON-serialisable high-level summary.

        Returns:
            Dict with keys:
            ``num_rounds``, ``best_accuracy``, ``best_round``,
            ``final_accuracy``, ``final_loss``,
            ``accuracy_improvement``, ``avg_active_clients``.
        """
        if not self._rounds:
            return {
                "num_rounds": 0,
                "best_accuracy": 0.0,
                "best_round": None,
                "final_accuracy": 0.0,
                "final_loss": 0.0,
                "accuracy_improvement": 0.0,
                "avg_active_clients": 0.0,
            }

        last = self._rounds[-1]
        avg_active = statistics.mean(r.num_active_clients for r in self._rounds)

        return {
            "num_rounds": len(self._rounds),
            "best_accuracy": self.best_accuracy(),
            "best_round": self.best_round(),
            "final_accuracy": last.global_accuracy,
            "final_loss": last.global_loss,
            "accuracy_improvement": self.accuracy_improvement(),
            "avg_active_clients": avg_active,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise all metrics to a JSON file.

        The file contains two top-level keys:

        - ``"summary"``: high-level experiment summary.
        - ``"rounds"``: list of per-round detail records.

        Args:
            path: Output file path (parent directories are created if needed).
        """
        output: dict[str, object] = {
            "summary": self.summary(),
            "rounds": [
                {
                    "round_num": r.round_num,
                    "global_accuracy": r.global_accuracy,
                    "global_loss": r.global_loss,
                    "num_active_clients": r.num_active_clients,
                    "total_samples": r.total_samples,
                    "avg_client_loss": r.avg_client_loss,
                    "min_client_accuracy": r.min_client_accuracy,
                    "max_client_accuracy": r.max_client_accuracy,
                    "std_client_accuracy": r.std_client_accuracy,
                    "per_client_accuracy": {
                        str(k): v for k, v in r.per_client_accuracy.items()
                    },
                    "clients": [
                        {
                            "client_id": c.client_id,
                            "accuracy": c.accuracy,
                            "loss": c.loss,
                            "num_samples": c.num_samples,
                        }
                        for c in r.client_records
                    ],
                }
                for r in self._rounds
            ],
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> MetricsTracker:
        """Reconstruct a :class:`MetricsTracker` from a saved JSON file.

        Only the data needed to re-populate ``rounds`` is restored;
        per-client detail records are reconstructed from the ``"clients"`` list.

        Args:
            path: Path to a JSON file previously written by :meth:`save`.

        Returns:
            Populated :class:`MetricsTracker` instance.
        """
        with open(path) as f:
            data: dict[str, object] = json.load(f)

        tracker = cls()
        for r in data.get("rounds", []):  # type: ignore[union-attr]
            per_client_acc = {
                int(k): v for k, v in r["per_client_accuracy"].items()  # type: ignore[index]
            }
            client_updates = [
                {
                    "client_id": c["client_id"],
                    "loss": c["loss"],
                    "num_samples": c["num_samples"],
                }
                for c in r.get("clients", [])  # type: ignore[index]
            ]
            tracker.record_round(
                round_num=int(r["round_num"]),  # type: ignore[arg-type]
                global_accuracy=float(r["global_accuracy"]),  # type: ignore[arg-type]
                global_loss=float(r["global_loss"]),  # type: ignore[arg-type]
                per_client_accuracy=per_client_acc,
                num_active_clients=int(r["num_active_clients"]),  # type: ignore[arg-type]
                client_updates=client_updates,
            )
        return tracker
