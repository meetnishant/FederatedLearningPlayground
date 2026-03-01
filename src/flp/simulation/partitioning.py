"""Data partitioning strategies for simulating non-IID federated datasets."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class DataPartitioner:
    """Partitions a dataset across N clients using configurable strategies.

    Supported strategies:

    - ``iid``: Uniform random shuffling and equal-sized partitions.
    - ``dirichlet``: Non-IID partitioning via Dirichlet distribution over class
      labels, controlled by concentration parameter ``alpha``.  Smaller alpha
      means more heterogeneous data distribution.
    - ``shard``: Each client receives exactly ``num_shards_per_client`` sorted
      class shards, producing a deterministic non-IID split.

    Args:
        dataset: The full training dataset with a ``targets`` attribute.
        num_clients: Number of clients to partition data across.
        strategy: One of ``"iid"``, ``"dirichlet"``, or ``"shard"``.
        seed: Random seed for reproducibility.
        alpha: Dirichlet concentration parameter (used when ``strategy="dirichlet"``).
        num_shards_per_client: Number of class shards per client
            (used when ``strategy="shard"``).
    """

    def __init__(
        self,
        dataset: Dataset,  # type: ignore[type-arg]
        num_clients: int,
        strategy: str = "dirichlet",
        seed: int = 42,
        alpha: float = 0.5,
        num_shards_per_client: int = 2,
    ) -> None:
        self.dataset = dataset
        self.num_clients = num_clients
        self.strategy = strategy
        self.seed = seed
        self.alpha = alpha
        self.num_shards_per_client = num_shards_per_client

        targets = getattr(dataset, "targets", None)
        if targets is None:
            raise AttributeError("Dataset must expose a 'targets' attribute.")
        if isinstance(targets, torch.Tensor):
            self._targets = targets.numpy()
        else:
            self._targets = np.array(targets)

    def partition(self) -> list[list[int]]:
        """Compute per-client index lists.

        Returns:
            A list of length ``num_clients`` where each element is a list of
            dataset indices belonging to that client.
        """
        if self.strategy == "iid":
            return self._iid_partition()
        elif self.strategy == "dirichlet":
            return self._dirichlet_partition()
        elif self.strategy == "shard":
            return self._shard_partition()
        else:
            raise ValueError(
                f"Unknown partitioning strategy '{self.strategy}'. "
                "Choose from: 'iid', 'dirichlet', 'shard'."
            )

    def _iid_partition(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(self.dataset)).tolist()
        chunk_size = len(indices) // self.num_clients
        return [
            indices[i * chunk_size : (i + 1) * chunk_size]
            for i in range(self.num_clients)
        ]

    def _dirichlet_partition(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed)
        classes = np.unique(self._targets)
        client_indices: list[list[int]] = [[] for _ in range(self.num_clients)]

        for cls in classes:
            cls_indices = np.where(self._targets == cls)[0]
            rng.shuffle(cls_indices)
            proportions = rng.dirichlet(alpha=[self.alpha] * self.num_clients)
            # Convert proportions to split points
            splits = (np.cumsum(proportions) * len(cls_indices)).astype(int)
            splits = np.clip(splits, 0, len(cls_indices))
            partitioned = np.split(cls_indices, splits[:-1])
            for client_id, chunk in enumerate(partitioned):
                client_indices[client_id].extend(chunk.tolist())

        # Shuffle each client's indices
        for indices in client_indices:
            rng.shuffle(indices)

        return client_indices

    def _shard_partition(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed)
        sorted_indices = np.argsort(self._targets)
        num_shards = self.num_clients * self.num_shards_per_client
        shard_size = len(sorted_indices) // num_shards

        shards = [
            sorted_indices[i * shard_size : (i + 1) * shard_size].tolist()
            for i in range(num_shards)
        ]
        shard_order = rng.permutation(num_shards).tolist()
        client_indices: list[list[int]] = []

        for i in range(self.num_clients):
            assigned: list[int] = []
            for j in range(self.num_shards_per_client):
                assigned.extend(shards[shard_order[i * self.num_shards_per_client + j]])
            client_indices.append(assigned)

        return client_indices
