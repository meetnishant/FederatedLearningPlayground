"""Virtual-time event queue for asynchronous federated learning simulation."""

from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass(frozen=True)
class FLEvent:
    """A scheduled federated learning event.

    Represents a client update queued for delivery at a specific virtual round.
    The ``virtual_round`` field determines when the server can process this
    update; the ``model_version`` field records which server checkpoint the
    client trained on (used to compute staleness).

    Attributes:
        virtual_round: The server round at which this update becomes available.
        client_id: ID of the client that produced this update.
        model_version: Server model version the client trained on.
            Staleness = current_server_version − model_version.
        update: The :class:`~flp.core.client.ClientUpdate` payload.
            Typed as ``object`` to avoid a circular import at runtime;
            callers treat it as ``ClientUpdate``.
    """

    virtual_round: int
    client_id: int
    model_version: int
    update: object  # ClientUpdate at runtime


class FLEventLoop:
    """Priority queue of federated learning events ordered by virtual round.

    Events are stored in ascending ``virtual_round`` order so that
    :meth:`pop_ready` is O(k) where *k* is the number of ready events.
    Insertion is O(n) in the worst case via :func:`bisect.bisect_right`,
    which is acceptable for the small per-round client counts typical of
    FL simulations (10–1000 clients).

    Ties at the same ``virtual_round`` are broken by insertion order
    (stable), ensuring deterministic processing when the server iterates
    clients in a fixed sequence.

    Example::

        loop = FLEventLoop()
        loop.push(FLEvent(virtual_round=3, client_id=0, model_version=2, update=upd))
        ready = loop.pop_ready(current_round=3)   # returns the event above
    """

    def __init__(self) -> None:
        self._queue: list[FLEvent] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def push(self, event: FLEvent) -> None:
        """Insert an event maintaining ascending ``virtual_round`` order.

        Args:
            event: The event to schedule.
        """
        keys = [e.virtual_round for e in self._queue]
        idx = bisect.bisect_right(keys, event.virtual_round)
        self._queue.insert(idx, event)

    def pop_ready(self, current_round: int) -> list[FLEvent]:
        """Remove and return all events with ``virtual_round <= current_round``.

        Events are returned in ascending ``virtual_round`` order (i.e. the
        oldest-queued updates come first), which gives the aggregator a
        deterministic ordering regardless of push sequence.

        Args:
            current_round: The current server round number.

        Returns:
            List of ready events; empty if none are available.
        """
        cut = 0
        for event in self._queue:
            if event.virtual_round <= current_round:
                cut += 1
            else:
                break
        ready = self._queue[:cut]
        self._queue = self._queue[cut:]
        return ready

    def discard_stale(self, min_virtual_round: int) -> int:
        """Remove events whose ``virtual_round`` is strictly less than ``min_virtual_round``.

        Useful for enforcing a hard deadline: any event that was scheduled
        for an earlier round but was not yet collected can be purged.

        Args:
            min_virtual_round: Events with ``virtual_round < min_virtual_round``
                are discarded.

        Returns:
            Number of events removed.
        """
        before = len(self._queue)
        self._queue = [e for e in self._queue if e.virtual_round >= min_virtual_round]
        return before - len(self._queue)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def pending_count(self) -> int:
        """Number of events currently waiting in the queue."""
        return len(self._queue)
