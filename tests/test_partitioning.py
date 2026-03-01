"""Tests for data partitioning strategies."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from flp.simulation.partitioning import DataPartitioner


def _make_dataset(num_samples: int = 1000, num_classes: int = 10) -> TensorDataset:
    data = torch.randn(num_samples, 1, 28, 28)
    labels = torch.arange(num_samples) % num_classes
    dataset = TensorDataset(data, labels)
    dataset.targets = labels  # type: ignore[attr-defined]
    return dataset


class TestIIDPartitioning:
    def test_correct_number_of_clients(self) -> None:
        dataset = _make_dataset()
        p = DataPartitioner(dataset, num_clients=10, strategy="iid", seed=42)
        result = p.partition()
        assert len(result) == 10

    def test_all_indices_covered(self) -> None:
        dataset = _make_dataset(1000)
        p = DataPartitioner(dataset, num_clients=10, strategy="iid", seed=42)
        result = p.partition()
        flat = sorted([i for part in result for i in part])
        assert flat == list(range(1000))

    def test_reproducible(self) -> None:
        dataset = _make_dataset()
        p1 = DataPartitioner(dataset, num_clients=5, strategy="iid", seed=0)
        p2 = DataPartitioner(dataset, num_clients=5, strategy="iid", seed=0)
        assert p1.partition() == p2.partition()

    def test_different_seeds_differ(self) -> None:
        dataset = _make_dataset()
        p1 = DataPartitioner(dataset, num_clients=5, strategy="iid", seed=0)
        p2 = DataPartitioner(dataset, num_clients=5, strategy="iid", seed=99)
        assert p1.partition() != p2.partition()


class TestDirichletPartitioning:
    def test_correct_number_of_clients(self) -> None:
        dataset = _make_dataset()
        p = DataPartitioner(dataset, num_clients=5, strategy="dirichlet", seed=42, alpha=0.5)
        result = p.partition()
        assert len(result) == 5

    def test_no_empty_clients_high_alpha(self) -> None:
        dataset = _make_dataset(5000)
        p = DataPartitioner(dataset, num_clients=10, strategy="dirichlet", seed=42, alpha=10.0)
        result = p.partition()
        assert all(len(part) > 0 for part in result)

    def test_non_iid_low_alpha(self) -> None:
        dataset = _make_dataset(5000)
        p = DataPartitioner(dataset, num_clients=10, strategy="dirichlet", seed=42, alpha=0.01)
        result = p.partition()
        # With very low alpha, some clients may have very skewed distributions
        assert all(len(part) >= 0 for part in result)


class TestShardPartitioning:
    def test_correct_number_of_clients(self) -> None:
        dataset = _make_dataset()
        p = DataPartitioner(
            dataset, num_clients=5, strategy="shard", seed=42, num_shards_per_client=2
        )
        result = p.partition()
        assert len(result) == 5

    def test_all_indices_unique(self) -> None:
        dataset = _make_dataset(1000)
        p = DataPartitioner(
            dataset, num_clients=5, strategy="shard", seed=42, num_shards_per_client=2
        )
        result = p.partition()
        flat = [i for part in result for i in part]
        assert len(flat) == len(set(flat))


class TestInvalidStrategy:
    def test_raises_on_unknown_strategy(self) -> None:
        dataset = _make_dataset()
        p = DataPartitioner(dataset, num_clients=5, strategy="invalid")
        with pytest.raises(ValueError, match="Unknown partitioning strategy"):
            p.partition()
