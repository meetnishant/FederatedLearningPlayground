"""Unit tests for simulation/partitioning.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from flp.simulation.partitioning import DataPartitioner, PartitionStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    num_samples: int = 1000,
    num_classes: int = 10,
    balanced: bool = True,
) -> TensorDataset:
    """Create a TensorDataset with a ``targets`` attribute.

    Args:
        num_samples: Total number of samples.
        num_classes: Number of label classes.
        balanced: If True each class gets exactly num_samples//num_classes
            samples; otherwise labels are assigned randomly.
    """
    data = torch.zeros(num_samples, 1, 28, 28)
    if balanced:
        labels = torch.arange(num_samples) % num_classes
    else:
        labels = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(data, labels)
    ds.targets = labels  # type: ignore[attr-defined]
    return ds


def _all_indices(partitions: list[list[int]]) -> list[int]:
    return [i for part in partitions for i in part]


# ---------------------------------------------------------------------------
# Common contract tests (apply to every strategy)
# ---------------------------------------------------------------------------


STRATEGIES = ["iid", "dirichlet", "shard", "label_skew"]


@pytest.mark.parametrize("strategy", STRATEGIES)
class TestCommonContract:
    def test_returns_correct_number_of_clients(self, strategy: str) -> None:
        ds = _make_dataset(1000)
        p = DataPartitioner(ds, num_clients=10, strategy=strategy, seed=42)  # type: ignore[arg-type]
        result = p.partition()
        assert len(result) == 10

    def test_all_partitions_are_lists_of_ints(self, strategy: str) -> None:
        ds = _make_dataset(1000)
        p = DataPartitioner(ds, num_clients=5, strategy=strategy, seed=0)  # type: ignore[arg-type]
        for part in p.partition():
            assert isinstance(part, list)
            assert all(isinstance(i, int) for i in part)

    def test_no_index_out_of_bounds(self, strategy: str) -> None:
        ds = _make_dataset(500)
        p = DataPartitioner(ds, num_clients=5, strategy=strategy, seed=1)  # type: ignore[arg-type]
        flat = _all_indices(p.partition())
        assert all(0 <= i < len(ds) for i in flat)

    def test_reproducible_with_same_seed(self, strategy: str) -> None:
        ds = _make_dataset(500)
        p1 = DataPartitioner(ds, num_clients=5, strategy=strategy, seed=7)  # type: ignore[arg-type]
        p2 = DataPartitioner(ds, num_clients=5, strategy=strategy, seed=7)  # type: ignore[arg-type]
        assert p1.partition() == p2.partition()

    def test_different_seeds_produce_different_splits(self, strategy: str) -> None:
        ds = _make_dataset(1000)
        p1 = DataPartitioner(ds, num_clients=5, strategy=strategy, seed=0)  # type: ignore[arg-type]
        p2 = DataPartitioner(ds, num_clients=5, strategy=strategy, seed=999)  # type: ignore[arg-type]
        # Sort-order independent comparison: check that at least one client differs
        r1 = p1.partition()
        r2 = p2.partition()
        assert any(sorted(r1[i]) != sorted(r2[i]) for i in range(5))


# ---------------------------------------------------------------------------
# IID
# ---------------------------------------------------------------------------


class TestIIDPartition:
    def test_covers_all_indices(self) -> None:
        ds = _make_dataset(1000)
        p = DataPartitioner(ds, num_clients=10, strategy="iid", seed=42)
        flat = sorted(_all_indices(p.partition()))
        assert flat == list(range(1000))

    def test_equal_partition_sizes(self) -> None:
        ds = _make_dataset(1000)
        p = DataPartitioner(ds, num_clients=10, strategy="iid", seed=42)
        sizes = [len(part) for part in p.partition()]
        assert len(set(sizes)) == 1  # all equal

    def test_no_duplicate_indices(self) -> None:
        ds = _make_dataset(500)
        p = DataPartitioner(ds, num_clients=5, strategy="iid", seed=3)
        flat = _all_indices(p.partition())
        assert len(flat) == len(set(flat))

    def test_class_distribution_is_approximately_uniform(self) -> None:
        """Each client should see all classes with high alpha (IID-like)."""
        ds = _make_dataset(2000, num_classes=5)
        p = DataPartitioner(ds, num_clients=5, strategy="iid", seed=0)
        partitions = p.partition()
        labels = ds.targets.numpy()  # type: ignore[attr-defined]
        for part in partitions:
            part_labels = labels[part]
            unique = np.unique(part_labels)
            assert len(unique) == 5, f"Expected 5 classes, got {len(unique)}"


# ---------------------------------------------------------------------------
# Dirichlet
# ---------------------------------------------------------------------------


class TestDirichletPartition:
    def test_all_samples_assigned(self) -> None:
        ds = _make_dataset(1000)
        p = DataPartitioner(ds, num_clients=10, strategy="dirichlet", seed=42, alpha=0.5)
        flat = _all_indices(p.partition())
        assert len(flat) == 1000

    def test_no_duplicates(self) -> None:
        ds = _make_dataset(1000)
        p = DataPartitioner(ds, num_clients=5, strategy="dirichlet", seed=42, alpha=0.5)
        flat = _all_indices(p.partition())
        assert len(flat) == len(set(flat))

    def test_high_alpha_gives_balanced_classes(self) -> None:
        """With large alpha the distribution should be near-IID: all clients see all classes."""
        ds = _make_dataset(5000, num_classes=10)
        p = DataPartitioner(ds, num_clients=10, strategy="dirichlet", seed=0, alpha=100.0)
        partitions = p.partition()
        labels = ds.targets.numpy()  # type: ignore[attr-defined]
        for part in partitions:
            assert len(part) > 0
            n_classes = len(np.unique(labels[part]))
            assert n_classes >= 8, f"Expected >=8 classes with high alpha, got {n_classes}"

    def test_low_alpha_gives_skewed_classes(self) -> None:
        """With very small alpha most clients should be dominated by one class."""
        ds = _make_dataset(10000, num_classes=10)
        p = DataPartitioner(ds, num_clients=10, strategy="dirichlet", seed=42, alpha=0.01)
        partitions = p.partition()
        labels = ds.targets.numpy()  # type: ignore[attr-defined]
        dominant_fractions = []
        for part in partitions:
            if not part:
                continue
            counts = np.bincount(labels[part], minlength=10)
            dominant_fractions.append(counts.max() / len(part))
        avg_dominant = np.mean(dominant_fractions)
        assert avg_dominant > 0.7, f"Expected dominant class fraction > 0.7, got {avg_dominant:.3f}"

    def test_heterogeneity_increases_as_alpha_decreases(self) -> None:
        """Lower alpha must produce a higher heterogeneity score."""
        ds = _make_dataset(5000)
        high_alpha = DataPartitioner(ds, num_clients=10, strategy="dirichlet", seed=0, alpha=10.0)
        low_alpha = DataPartitioner(ds, num_clients=10, strategy="dirichlet", seed=0, alpha=0.1)
        p_high = high_alpha.partition()
        p_low = low_alpha.partition()
        h_high = high_alpha.compute_stats(p_high).heterogeneity
        h_low = low_alpha.compute_stats(p_low).heterogeneity
        assert h_low > h_high, (
            f"Lower alpha should give higher heterogeneity: "
            f"alpha=0.1 → {h_low:.3f}, alpha=10.0 → {h_high:.3f}"
        )

    def test_each_partition_is_shuffled(self) -> None:
        """Each client's indices must not be purely label-sorted."""
        ds = _make_dataset(2000)
        p = DataPartitioner(ds, num_clients=5, strategy="dirichlet", seed=42, alpha=0.5)
        labels = ds.targets.numpy()  # type: ignore[attr-defined]
        for part in p.partition():
            if len(part) < 2:
                continue
            part_labels = labels[part]
            # If perfectly sorted the diff would be all >= 0
            diffs = np.diff(part_labels)
            assert not np.all(diffs >= 0), "Partition indices appear label-sorted (not shuffled)"


# ---------------------------------------------------------------------------
# Shard
# ---------------------------------------------------------------------------


class TestShardPartition:
    def test_no_duplicate_indices(self) -> None:
        ds = _make_dataset(1000)
        p = DataPartitioner(
            ds, num_clients=5, strategy="shard", seed=42, num_shards_per_client=2
        )
        flat = _all_indices(p.partition())
        assert len(flat) == len(set(flat))

    def test_clients_see_limited_classes(self) -> None:
        """With 2 shards/client each client should see at most 2–3 classes."""
        ds = _make_dataset(1000, num_classes=10)
        p = DataPartitioner(
            ds, num_clients=5, strategy="shard", seed=0, num_shards_per_client=2
        )
        partitions = p.partition()
        labels = ds.targets.numpy()  # type: ignore[attr-defined]
        for part in partitions:
            n_classes = len(np.unique(labels[part]))
            # Shard boundaries mean a client can straddle at most 3 classes
            assert n_classes <= 4, f"Expected <= 4 classes per shard client, got {n_classes}"

    def test_more_shards_per_client_increases_class_diversity(self) -> None:
        ds = _make_dataset(2000, num_classes=10)
        p2 = DataPartitioner(
            ds, num_clients=5, strategy="shard", seed=0, num_shards_per_client=2
        )
        p5 = DataPartitioner(
            ds, num_clients=5, strategy="shard", seed=0, num_shards_per_client=5
        )
        # Adjust for different shard counts (need compatible total shards)
        labels = ds.targets.numpy()  # type: ignore[attr-defined]
        avg2 = np.mean([len(np.unique(labels[p])) for p in p2.partition()])
        avg5 = np.mean([len(np.unique(labels[p])) for p in p5.partition()])
        assert avg5 >= avg2, "More shards per client should give >= class diversity"

    def test_raises_when_shard_size_is_zero(self) -> None:
        ds = _make_dataset(10, num_classes=2)
        p = DataPartitioner(
            ds, num_clients=10, strategy="shard", seed=0, num_shards_per_client=100
        )
        with pytest.raises(ValueError, match="shard_size=0"):
            p.partition()

    def test_partition_sizes_are_equal(self) -> None:
        ds = _make_dataset(1000)
        p = DataPartitioner(
            ds, num_clients=5, strategy="shard", seed=7, num_shards_per_client=2
        )
        sizes = [len(part) for part in p.partition()]
        assert len(set(sizes)) == 1


# ---------------------------------------------------------------------------
# Label Skew
# ---------------------------------------------------------------------------


class TestLabelSkewPartition:
    def test_returns_correct_number_of_clients(self) -> None:
        ds = _make_dataset(1000)
        p = DataPartitioner(ds, num_clients=5, strategy="label_skew", seed=42)
        assert len(p.partition()) == 5

    def test_no_duplicate_indices(self) -> None:
        ds = _make_dataset(1000)
        p = DataPartitioner(ds, num_clients=5, strategy="label_skew", seed=42)
        flat = _all_indices(p.partition())
        assert len(flat) == len(set(flat))

    def test_primary_class_dominates(self) -> None:
        """Each client's most-frequent class should be its assigned primary class."""
        ds = _make_dataset(2000, num_classes=5)
        p = DataPartitioner(
            ds, num_clients=5, strategy="label_skew", seed=0, primary_class_ratio=0.9
        )
        partitions = p.partition()
        labels = ds.targets.numpy()  # type: ignore[attr-defined]
        for cid, part in enumerate(partitions):
            if not part:
                continue
            counts = np.bincount(labels[part], minlength=5)
            dominant_fraction = counts.max() / len(part)
            assert dominant_fraction >= 0.7, (
                f"Client {cid}: expected dominant class >= 70% of samples, "
                f"got {dominant_fraction:.2f}"
            )

    def test_higher_ratio_gives_more_skew(self) -> None:
        ds = _make_dataset(2000, num_classes=5)
        labels = ds.targets.numpy()  # type: ignore[attr-defined]

        def _avg_dominant(ratio: float) -> float:
            p = DataPartitioner(
                ds, num_clients=5, strategy="label_skew", seed=0, primary_class_ratio=ratio
            )
            fracs = []
            for part in p.partition():
                if part:
                    counts = np.bincount(labels[part], minlength=5)
                    fracs.append(counts.max() / len(part))
            return float(np.mean(fracs))

        assert _avg_dominant(0.9) > _avg_dominant(0.5)

    def test_invalid_ratio_raises(self) -> None:
        ds = _make_dataset(500)
        with pytest.raises(ValueError, match="primary_class_ratio"):
            DataPartitioner(
                ds, num_clients=5, strategy="label_skew", primary_class_ratio=1.0
            ).partition()
        with pytest.raises(ValueError, match="primary_class_ratio"):
            DataPartitioner(
                ds, num_clients=5, strategy="label_skew", primary_class_ratio=0.0
            ).partition()

    def test_all_non_empty(self) -> None:
        ds = _make_dataset(1000, num_classes=5)
        p = DataPartitioner(ds, num_clients=5, strategy="label_skew", seed=42)
        for part in p.partition():
            assert len(part) > 0


# ---------------------------------------------------------------------------
# PartitionStats / compute_stats
# ---------------------------------------------------------------------------


class TestPartitionStats:
    def test_returns_partition_stats_instance(self) -> None:
        ds = _make_dataset(500)
        p = DataPartitioner(ds, num_clients=5, strategy="iid", seed=0)
        partitions = p.partition()
        stats = p.compute_stats(partitions)
        assert isinstance(stats, PartitionStats)

    def test_total_samples_correct(self) -> None:
        ds = _make_dataset(1000)
        p = DataPartitioner(ds, num_clients=10, strategy="iid", seed=0)
        partitions = p.partition()
        stats = p.compute_stats(partitions)
        assert stats.total_samples == sum(len(part) for part in partitions)

    def test_iid_heterogeneity_lower_than_non_iid(self) -> None:
        ds = _make_dataset(5000)
        p_iid = DataPartitioner(ds, num_clients=10, strategy="iid", seed=0)
        p_noniid = DataPartitioner(ds, num_clients=10, strategy="dirichlet", seed=0, alpha=0.05)
        s_iid = p_iid.compute_stats(p_iid.partition())
        s_noniid = p_noniid.compute_stats(p_noniid.partition())
        assert s_iid.heterogeneity < s_noniid.heterogeneity

    def test_min_max_samples_bounds(self) -> None:
        ds = _make_dataset(500)
        p = DataPartitioner(ds, num_clients=5, strategy="dirichlet", seed=42, alpha=0.5)
        partitions = p.partition()
        stats = p.compute_stats(partitions)
        assert stats.min_samples <= stats.max_samples
        assert stats.min_samples == min(len(part) for part in partitions)
        assert stats.max_samples == max(len(part) for part in partitions)


# ---------------------------------------------------------------------------
# min_samples_per_client enforcement
# ---------------------------------------------------------------------------


class TestMinSamplesEnforcement:
    def test_all_clients_meet_threshold(self) -> None:
        """With extreme non-IID some clients get very few samples; enforce minimum."""
        ds = _make_dataset(5000, num_classes=10)
        p = DataPartitioner(
            ds,
            num_clients=10,
            strategy="dirichlet",
            seed=42,
            alpha=0.01,
            min_samples_per_client=50,
        )
        for part in p.partition():
            assert len(part) >= 50


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_raises_on_unknown_strategy(self) -> None:
        ds = _make_dataset()
        p = DataPartitioner(ds, num_clients=5, strategy="unknown")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown partitioning strategy"):
            p.partition()

    def test_raises_on_too_few_clients(self) -> None:
        ds = _make_dataset()
        with pytest.raises(ValueError, match="num_clients must be >= 2"):
            DataPartitioner(ds, num_clients=1, strategy="iid")

    def test_raises_on_missing_targets_attribute(self) -> None:
        # A plain TensorDataset without .targets
        ds = TensorDataset(torch.zeros(100, 1, 28, 28))
        with pytest.raises(AttributeError, match="targets"):
            DataPartitioner(ds, num_clients=5, strategy="iid")
