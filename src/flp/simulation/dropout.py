"""Client dropout simulation for federated learning rounds."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flp.core.client import FLClient


@dataclass(frozen=True)
class DropoutResult:
    """Structured outcome of applying dropout to a set of selected clients.

    Attributes:
        active: Clients that survived dropout and will train this round.
        active_ids: Client IDs of surviving clients (convenience view).
        dropped_ids: Client IDs of clients that dropped out.
        num_selected: Total clients offered this round before dropout.
        actual_rate: Observed dropout fraction this round
            (``len(dropped_ids) / num_selected``).
        round_num: The federated round this result belongs to.
        all_dropped: True when every selected client dropped out (round must be skipped).
    """

    active: list[FLClient]
    active_ids: list[int]
    dropped_ids: list[int]
    num_selected: int
    actual_rate: float
    round_num: int
    all_dropped: bool


@dataclass
class DropoutRoundRecord:
    """Per-round dropout record stored by :class:`DropoutMetrics`.

    Attributes:
        round_num: 1-based round index.
        num_selected: Clients offered to the round.
        num_active: Clients that remained after dropout.
        num_dropped: Clients that dropped out.
        actual_rate: Observed dropout fraction.
        all_dropped: True if the entire round was skipped.
    """

    round_num: int
    num_selected: int
    num_active: int
    num_dropped: int
    actual_rate: float
    all_dropped: bool


@dataclass
class DropoutMetrics:
    """Cumulative dropout statistics collected across all training rounds.

    Populated in-place by :meth:`DropoutSimulator.record` after each round.

    Attributes:
        records: Ordered list of per-round records.
        total_selected: Total client-round slots offered.
        total_dropped: Total client-round dropouts.
        total_skipped_rounds: Rounds where all selected clients dropped out.
    """

    records: list[DropoutRoundRecord] = field(default_factory=list)
    total_selected: int = 0
    total_dropped: int = 0
    total_skipped_rounds: int = 0

    @property
    def overall_dropout_rate(self) -> float:
        """Fraction of client-round slots lost to dropout across all rounds."""
        return self.total_dropped / self.total_selected if self.total_selected > 0 else 0.0

    @property
    def dropout_rates_per_round(self) -> list[float]:
        """Observed dropout rate for each recorded round (in order)."""
        return [r.actual_rate for r in self.records]

    @property
    def active_counts_per_round(self) -> list[int]:
        """Number of active clients per round (in order)."""
        return [r.num_active for r in self.records]

    def summary(self) -> dict[str, object]:
        """Return a JSON-serialisable summary dictionary."""
        return {
            "total_rounds": len(self.records),
            "total_selected": self.total_selected,
            "total_dropped": self.total_dropped,
            "total_skipped_rounds": self.total_skipped_rounds,
            "overall_dropout_rate": self.overall_dropout_rate,
        }


class DropoutSimulator:
    """Simulates independent random client dropout each federated round.

    Each selected client independently drops out with probability
    ``dropout_rate``.  The RNG is seeded as ``seed + round_num * 31`` so that:

    - The same (seed, round_num) pair always produces the same dropout pattern.
    - Different rounds produce different patterns.
    - The global Python/NumPy/PyTorch RNG state is never mutated.

    Use :meth:`apply` to get a :class:`DropoutResult` for a round, then call
    :meth:`record` to persist that result in :attr:`metrics`.  The server calls
    both methods in sequence each round so metrics are always up-to-date.

    Args:
        dropout_rate: Per-client dropout probability in ``[0, 1)``.
            ``0.0`` disables dropout entirely.
        seed: Base random seed for reproducibility.
    """

    def __init__(self, dropout_rate: float = 0.0, seed: int = 42) -> None:
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}. "
                "Use 0.0 to disable dropout."
            )
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.metrics = DropoutMetrics()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def apply(self, clients: list[FLClient], round_num: int) -> DropoutResult:
        """Determine which clients survive dropout for this round.

        Each client independently survives with probability
        ``1 - dropout_rate``, decided by a seeded RNG that is isolated
        from the global random state.

        Args:
            clients: Clients selected to participate this round.
            round_num: Current 1-based round index (mixed into the seed).

        Returns:
            :class:`DropoutResult` describing survivors and dropouts.
        """
        num_selected = len(clients)

        if self.dropout_rate == 0.0 or num_selected == 0:
            return DropoutResult(
                active=list(clients),
                active_ids=[c.client_id for c in clients],
                dropped_ids=[],
                num_selected=num_selected,
                actual_rate=0.0,
                round_num=round_num,
                all_dropped=(num_selected == 0),
            )

        rng = random.Random(self.seed + round_num * 31)
        active: list[FLClient] = []
        dropped_ids: list[int] = []

        for client in clients:
            if rng.random() < self.dropout_rate:
                dropped_ids.append(client.client_id)
            else:
                active.append(client)

        actual_rate = len(dropped_ids) / num_selected if num_selected > 0 else 0.0

        return DropoutResult(
            active=active,
            active_ids=[c.client_id for c in active],
            dropped_ids=dropped_ids,
            num_selected=num_selected,
            actual_rate=actual_rate,
            round_num=round_num,
            all_dropped=(len(active) == 0),
        )

    def record(self, result: DropoutResult) -> None:
        """Persist a :class:`DropoutResult` in :attr:`metrics`.

        Call this once per round immediately after :meth:`apply`.

        Args:
            result: The dropout result to record.
        """
        rec = DropoutRoundRecord(
            round_num=result.round_num,
            num_selected=result.num_selected,
            num_active=len(result.active),
            num_dropped=len(result.dropped_ids),
            actual_rate=result.actual_rate,
            all_dropped=result.all_dropped,
        )
        self.metrics.records.append(rec)
        self.metrics.total_selected += result.num_selected
        self.metrics.total_dropped += len(result.dropped_ids)
        if result.all_dropped:
            self.metrics.total_skipped_rounds += 1

    def reset_metrics(self) -> None:
        """Clear all accumulated dropout metrics (useful between experiments)."""
        self.metrics = DropoutMetrics()
