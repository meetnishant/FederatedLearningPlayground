"""Communication delay simulation for federated learning clients."""

from __future__ import annotations

import random


class DelaySimulator:
    """Simulates heterogeneous communication delays across clients.

    Delay values are sampled from a uniform distribution
    [``min_delay``, ``max_delay``].  Clients whose delay exceeds
    ``deadline`` are treated as stragglers and excluded from the round.

    Args:
        min_delay: Minimum delay in simulated time units.
        max_delay: Maximum delay in simulated time units.
        deadline: Maximum tolerated delay; clients exceeding this are dropped.
        seed: Base random seed.
    """

    def __init__(
        self,
        min_delay: float = 0.0,
        max_delay: float = 1.0,
        deadline: float | None = None,
        seed: int = 42,
    ) -> None:
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.deadline = deadline
        self.seed = seed

    def sample_delays(self, num_clients: int, round_num: int) -> list[float]:
        """Sample a delay value for each client.

        Args:
            num_clients: Number of clients to assign delays to.
            round_num: Current round (used to vary the RNG seed).

        Returns:
            List of delay values, one per client.
        """
        rng = random.Random(self.seed + round_num * 1000)
        return [rng.uniform(self.min_delay, self.max_delay) for _ in range(num_clients)]

    def filter_stragglers(
        self,
        client_ids: list[int],
        delays: list[float],
    ) -> tuple[list[int], list[int]]:
        """Separate on-time clients from stragglers.

        Args:
            client_ids: Client IDs in the same order as ``delays``.
            delays: Delay values per client.

        Returns:
            Tuple of (on_time_ids, straggler_ids).
        """
        if self.deadline is None:
            return client_ids, []

        on_time = [cid for cid, d in zip(client_ids, delays) if d <= self.deadline]
        stragglers = [cid for cid, d in zip(client_ids, delays) if d > self.deadline]
        return on_time, stragglers
