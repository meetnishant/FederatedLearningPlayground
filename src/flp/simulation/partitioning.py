"""Data partitioning strategies for simulating IID and non-IID federated datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

Strategy = Literal["iid", "dirichlet", "shard", "label_skew"]


@dataclass(frozen=True)
class PartitionStats:
    """Descriptive statistics for a completed partitioning.

    Attributes:
        num_clients: Number of clients produced.
        total_samples: Total number of assigned samples (may be < dataset size
            if the dataset is not evenly divisible).
        samples_per_client: List of sample counts, one per client.
        classes_per_client: List of unique class counts, one per client.
        min_samples: Smallest partition size.
        max_samples: Largest partition size.
        heterogeneity: Coefficient of variation of per-client class distributions
            (higher = more heterogeneous).  Defined as std(class_fracs) / mean(class_fracs)
            averaged across all clients and classes.
    """

    num_clients: int
    total_samples: int
    samples_per_client: list[int]
    classes_per_client: list[int]
    min_samples: int
    max_samples: int
    heterogeneity: float


class DataPartitioner:
    """Partition a labelled dataset across N federated clients.

    Supported strategies
    --------------------
    ``"iid"``
        Uniform random shuffling; every client gets an equal-sized slice of the
        full dataset.  Class distributions are approximately identical across
        clients (homogeneous baseline).

    ``"dirichlet"``
        Non-IID via Dirichlet distribution.  For each class, samples are split
        across clients according to proportions drawn from
        ``Dir(alpha, ..., alpha)``.  Smaller *alpha* → more skewed distributions
        (extreme non-IID at alpha ≈ 0.05; near-IID at alpha ≈ 10).

    ``"shard"``
        Deterministic label-sorted shard split.  The dataset is sorted by label,
        sliced into ``num_clients * num_shards_per_client`` equal shards, and
        each client is randomly assigned ``num_shards_per_client`` shards.
        Guarantees every client sees at most ``num_shards_per_client`` classes.

    ``"label_skew"``
        Explicit label skew.  Each client is assigned a *primary* class that
        makes up ``primary_class_ratio`` of its data; the remainder is filled
        from other classes uniformly.  Unlike Dirichlet, the skew magnitude is
        directly controlled and reproducible.

    Args:
        dataset: A PyTorch ``Dataset`` that exposes a ``targets`` attribute
            (``torch.Tensor`` or list of integer labels).
        num_clients: Number of clients to partition data across (>= 2).
        strategy: Partitioning strategy name.
        seed: Base random seed for full reproducibility.
        alpha: Dirichlet concentration parameter (``"dirichlet"`` only).
        num_shards_per_client: Shards per client (``"shard"`` only).
        primary_class_ratio: Fraction of each client's data drawn from its
            primary class (``"label_skew"`` only).  Must be in (0, 1).
        min_samples_per_client: If > 0, clients with fewer samples than this
            threshold after partitioning will be topped up by borrowing samples
            from the largest partition.  Applied to all strategies.
    """

    def __init__(
        self,
        dataset: Dataset,  # type: ignore[type-arg]
        num_clients: int,
        strategy: Strategy = "dirichlet",
        seed: int = 42,
        alpha: float = 0.5,
        num_shards_per_client: int = 2,
        primary_class_ratio: float = 0.8,
        min_samples_per_client: int = 0,
    ) -> None:
        if num_clients < 2:
            raise ValueError(f"num_clients must be >= 2, got {num_clients}.")

        self.dataset = dataset
        self.num_clients = num_clients
        self.strategy = strategy
        self.seed = seed
        self.alpha = alpha
        self.num_shards_per_client = num_shards_per_client
        self.primary_class_ratio = primary_class_ratio
        self.min_samples_per_client = min_samples_per_client

        targets = getattr(dataset, "targets", None)
        if targets is None:
            raise AttributeError(
                "Dataset must expose a 'targets' attribute (torch.Tensor or list of ints). "
                "MNIST, CIFAR-10, and most torchvision datasets satisfy this."
            )
        self._targets: np.ndarray = (
            targets.numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
        )
        self._classes: np.ndarray = np.unique(self._targets)
        self._num_classes: int = len(self._classes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def partition(self) -> list[list[int]]:
        """Compute per-client index lists using the configured strategy.

        Returns:
            A list of length ``num_clients``.  Each element is a list of integer
            dataset indices assigned to that client.  Indices within each list
            are shuffled.

        Raises:
            ValueError: If the strategy name is not recognised.
        """
        dispatch: dict[str, object] = {
            "iid": self._iid_partition,
            "dirichlet": self._dirichlet_partition,
            "shard": self._shard_partition,
            "label_skew": self._label_skew_partition,
        }
        if self.strategy not in dispatch:
            raise ValueError(
                f"Unknown partitioning strategy '{self.strategy}'. "
                f"Valid options: {sorted(dispatch)}."
            )
        client_indices: list[list[int]] = dispatch[self.strategy]()  # type: ignore[operator]

        if self.min_samples_per_client > 0:
            client_indices = self._enforce_min_samples(client_indices)

        return client_indices

    def compute_stats(self, client_indices: list[list[int]]) -> PartitionStats:
        """Compute descriptive statistics for a completed partition.

        Args:
            client_indices: Output of :meth:`partition`.

        Returns:
            :class:`PartitionStats` instance.
        """
        samples_per_client = [len(p) for p in client_indices]
        classes_per_client: list[int] = []
        all_class_fracs: list[np.ndarray] = []

        for indices in client_indices:
            if not indices:
                classes_per_client.append(0)
                all_class_fracs.append(np.zeros(self._num_classes))
                continue
            labels = self._targets[indices]
            unique_cls = np.unique(labels)
            classes_per_client.append(len(unique_cls))
            fracs = np.array(
                [np.sum(labels == c) / len(labels) for c in self._classes]
            )
            all_class_fracs.append(fracs)

        # Heterogeneity: mean coefficient of variation across class dimensions
        frac_matrix = np.stack(all_class_fracs, axis=0)  # shape (num_clients, num_classes)
        col_means = frac_matrix.mean(axis=0)
        col_stds = frac_matrix.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = np.where(col_means > 0, col_stds / col_means, 0.0)
        heterogeneity = float(cv.mean())

        return PartitionStats(
            num_clients=self.num_clients,
            total_samples=sum(samples_per_client),
            samples_per_client=samples_per_client,
            classes_per_client=classes_per_client,
            min_samples=min(samples_per_client) if samples_per_client else 0,
            max_samples=max(samples_per_client) if samples_per_client else 0,
            heterogeneity=heterogeneity,
        )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _iid_partition(self) -> list[list[int]]:
        """Uniform random partition with equal-sized splits."""
        rng = np.random.default_rng(self.seed)
        all_indices: list[int] = rng.permutation(len(self.dataset)).tolist()
        chunk = len(all_indices) // self.num_clients
        partitions: list[list[int]] = []
        for i in range(self.num_clients):
            partitions.append(all_indices[i * chunk : (i + 1) * chunk])
        return partitions

    def _dirichlet_partition(self) -> list[list[int]]:
        """Non-IID partition via per-class Dirichlet proportions.

        For each class c, samples are split across clients according to
        Dir(alpha) proportions, where smaller alpha produces more extreme
        class imbalance.
        """
        rng = np.random.default_rng(self.seed)
        client_indices: list[list[int]] = [[] for _ in range(self.num_clients)]

        for cls in self._classes:
            cls_idx = np.where(self._targets == cls)[0]
            rng.shuffle(cls_idx)

            # Draw proportions from Dirichlet; retry if degenerate
            for _ in range(100):
                proportions = rng.dirichlet(np.full(self.num_clients, self.alpha))
                if proportions.max() < 1.0:
                    break

            # Convert proportions to cumulative split points
            cum = (np.cumsum(proportions) * len(cls_idx)).astype(int)
            cum = np.clip(cum, 0, len(cls_idx))
            splits = np.split(cls_idx, cum[:-1])
            for cid, chunk in enumerate(splits):
                client_indices[cid].extend(chunk.tolist())

        # Shuffle each client's full index list
        for indices in client_indices:
            rng.shuffle(indices)

        return client_indices

    def _shard_partition(self) -> list[list[int]]:
        """Deterministic label-sorted shard split.

        Sorts all dataset indices by label, slices into equal shards, then
        randomly assigns ``num_shards_per_client`` shards to each client.
        Each client therefore sees at most ``num_shards_per_client`` distinct
        classes, producing a strong non-IID effect.
        """
        rng = np.random.default_rng(self.seed)
        sorted_idx = np.argsort(self._targets, kind="stable")

        num_shards = self.num_clients * self.num_shards_per_client
        shard_size = len(sorted_idx) // num_shards
        if shard_size == 0:
            raise ValueError(
                f"shard_size=0 with num_clients={self.num_clients}, "
                f"num_shards_per_client={self.num_shards_per_client}, "
                f"dataset_size={len(sorted_idx)}. "
                "Reduce num_shards_per_client or use a larger dataset."
            )

        shards: list[list[int]] = [
            sorted_idx[i * shard_size : (i + 1) * shard_size].tolist()
            for i in range(num_shards)
        ]
        shard_order: list[int] = rng.permutation(num_shards).tolist()
        client_indices: list[list[int]] = []

        for cid in range(self.num_clients):
            assigned: list[int] = []
            for j in range(self.num_shards_per_client):
                shard_idx = shard_order[cid * self.num_shards_per_client + j]
                assigned.extend(shards[shard_idx])
            client_indices.append(assigned)

        return client_indices

    def _label_skew_partition(self) -> list[list[int]]:
        """Explicit label-skew partition.

        Each client is assigned a *primary* class that contributes
        ``primary_class_ratio`` of its data.  The remaining fraction is filled
        uniformly from all other classes.  Clients are assigned primary classes
        in round-robin order, so with 10 clients and 10 classes each class is
        one client's primary class.

        This gives direct control over skew magnitude, unlike Dirichlet where
        the exact per-client distribution is stochastic.
        """
        if not 0.0 < self.primary_class_ratio < 1.0:
            raise ValueError(
                f"primary_class_ratio must be in (0, 1), got {self.primary_class_ratio}."
            )

        rng = np.random.default_rng(self.seed)

        # Build per-class index pools (shuffled)
        class_pools: dict[int, list[int]] = {}
        for cls in self._classes:
            idx = np.where(self._targets == cls)[0]
            rng.shuffle(idx)
            class_pools[int(cls)] = idx.tolist()

        # Target samples per client: evenly divide dataset
        total = len(self.dataset)
        target_per_client = total // self.num_clients

        # Assign a primary class to each client (round-robin over sorted classes)
        primary_classes = [
            int(self._classes[cid % self._num_classes]) for cid in range(self.num_clients)
        ]

        client_indices: list[list[int]] = []
        class_cursors: dict[int, int] = {int(c): 0 for c in self._classes}

        for cid in range(self.num_clients):
            primary = primary_classes[cid]
            n_primary = int(target_per_client * self.primary_class_ratio)
            n_other = target_per_client - n_primary

            # Draw primary samples
            pool = class_pools[primary]
            cur = class_cursors[primary]
            primary_samples = pool[cur : cur + n_primary]
            class_cursors[primary] = cur + len(primary_samples)

            # Draw other-class samples uniformly
            other_classes = [int(c) for c in self._classes if int(c) != primary]
            other_samples: list[int] = []
            if other_classes:
                per_class = max(1, n_other // len(other_classes))
                for oc in other_classes:
                    oc_pool = class_pools[oc]
                    oc_cur = class_cursors[oc]
                    chunk = oc_pool[oc_cur : oc_cur + per_class]
                    other_samples.extend(chunk)
                    class_cursors[oc] = oc_cur + len(chunk)
                    if len(other_samples) >= n_other:
                        break
                other_samples = other_samples[:n_other]

            combined: list[int] = primary_samples + other_samples
            rng.shuffle(combined)
            client_indices.append(combined)

        return client_indices

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _enforce_min_samples(
        self, client_indices: list[list[int]]
    ) -> list[list[int]]:
        """Top up clients below ``min_samples_per_client`` from the largest partition.

        Samples are moved (not duplicated) from the largest partition to
        under-provisioned clients.  If the largest partition cannot cover all
        deficits the threshold is silently reduced to what is available.

        Args:
            client_indices: Partition to adjust (modified in-place).

        Returns:
            Adjusted ``client_indices``.
        """
        threshold = self.min_samples_per_client
        for cid, indices in enumerate(client_indices):
            deficit = threshold - len(indices)
            if deficit <= 0:
                continue
            # Find the client with the most samples
            donor_id = max(range(self.num_clients), key=lambda i: len(client_indices[i]))
            donor = client_indices[donor_id]
            transfer = min(deficit, len(donor) - threshold)
            if transfer <= 0:
                continue
            client_indices[cid].extend(donor[-transfer:])
            client_indices[donor_id] = donor[:-transfer]

        return client_indices
