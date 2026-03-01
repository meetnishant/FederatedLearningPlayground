"""Per-round audit log for federated learning governance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class AuditEvent:
    """Immutable audit record for a single federated round.

    Every field is populated regardless of whether DP is active.  When DP is
    disabled, ``dp_epsilon_spent`` and ``dp_delta_spent`` are ``0.0``.

    Attributes:
        round_num: 1-based round index.
        timestamp_utc: ISO 8601 UTC timestamp when the round completed.
        selected_clients: Client IDs sampled this round (pre-dropout).
        active_clients: Client IDs that completed local training.
        dropped_clients: Client IDs that dropped out before uploading.
        pre_round_model_hash: SHA-256 of the global model *before* this round.
        post_round_model_hash: SHA-256 of the global model *after* aggregation.
            Equals ``pre_round_model_hash`` for skipped rounds.
        global_accuracy: Test-set accuracy of the updated global model.
        global_loss: Test-set cross-entropy loss of the updated global model.
        num_clients_clipped: Updates whose L2 norm exceeded the DP clip bound.
            Always ``0`` when DP is disabled.
        dp_epsilon_spent: Privacy budget consumed this round (``0.0`` if DP off).
        dp_delta_spent: Failure probability consumed this round (``0.0`` if DP off).
        elapsed_seconds: Wall-clock time for the full round.
        skipped: True if all selected clients dropped out (no model update).
    """

    round_num: int
    timestamp_utc: str
    selected_clients: list[int]
    active_clients: list[int]
    dropped_clients: list[int]
    pre_round_model_hash: str
    post_round_model_hash: str
    global_accuracy: float
    global_loss: float
    num_clients_clipped: int
    dp_epsilon_spent: float
    dp_delta_spent: float
    elapsed_seconds: float
    skipped: bool


class AuditLog:
    """Append-only per-round audit log for federated learning experiments.

    Records are accumulated in insertion order.  After training completes,
    call :meth:`save` to persist to disk in two complementary formats:

    - ``audit_log.json``  — indented JSON array (human-readable, ``jq``-queryable)
    - ``audit_log.jsonl`` — one JSON object per line (stream-friendly, ``grep``-able)

    Example::

        log = AuditLog()
        log.record(AuditEvent(...))
        log.save(Path("outputs/my_exp/governance"))
    """

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[AuditEvent]:
        """All recorded events in round order (returns a defensive copy)."""
        return list(self._events)

    def record(self, event: AuditEvent) -> None:
        """Append a completed round's audit event.

        Args:
            event: Audit record for the completed round.
        """
        self._events.append(event)

    def to_records(self) -> list[dict[str, Any]]:
        """Return all events as a list of plain dicts (JSON-serialisable)."""
        return [asdict(e) for e in self._events]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return aggregate governance statistics over all recorded rounds.

        Returns:
            Dict with keys: ``num_rounds_recorded``, ``num_rounds_skipped``,
            ``num_rounds_with_dropout``, ``total_clients_clipped``,
            ``total_dp_epsilon``, ``total_dp_delta``, ``unique_model_hashes``.
        """
        hashes = {e.post_round_model_hash for e in self._events if not e.skipped}
        return {
            "num_rounds_recorded": len(self._events),
            "num_rounds_skipped": sum(1 for e in self._events if e.skipped),
            "num_rounds_with_dropout": sum(
                1 for e in self._events if e.dropped_clients
            ),
            "total_clients_clipped": sum(e.num_clients_clipped for e in self._events),
            "total_dp_epsilon": round(
                sum(e.dp_epsilon_spent for e in self._events), 8
            ),
            "total_dp_delta": sum(e.dp_delta_spent for e in self._events),
            "unique_model_hashes": len(hashes),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: Path) -> None:
        """Persist the audit log to disk.

        Writes two complementary files under ``output_dir``:

        - ``audit_log.json``  — full indented JSON array
        - ``audit_log.jsonl`` — one JSON object per line

        Args:
            output_dir: Target directory; created recursively if absent.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        records = self.to_records()

        with open(output_dir / "audit_log.json", "w") as f:
            json.dump(records, f, indent=2)

        with open(output_dir / "audit_log.jsonl", "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
