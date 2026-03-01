"""Client dropout simulation for federated learning rounds."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flp.core.client import FLClient


class DropoutSimulator:
    """Simulates random client dropout per training round.

    Each selected client independently drops out with probability
    ``dropout_rate``.  The seed is offset by round number so that dropout
    patterns differ between rounds while remaining reproducible.

    Args:
        dropout_rate: Probability [0, 1) that an individual client drops out.
        seed: Base random seed.
    """

    def __init__(self, dropout_rate: float = 0.0, seed: int = 42) -> None:
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}.")
        self.dropout_rate = dropout_rate
        self.seed = seed

    def apply(self, clients: list[FLClient], round_num: int) -> list[FLClient]:
        """Return the subset of clients that remain active this round.

        Args:
            clients: Clients selected to participate this round.
            round_num: Current round number (used to vary the RNG seed).

        Returns:
            Clients that did not drop out.
        """
        if self.dropout_rate == 0.0:
            return clients

        rng = random.Random(self.seed + round_num)
        return [c for c in clients if rng.random() >= self.dropout_rate]
