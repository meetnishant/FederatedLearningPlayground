"""Tests for staleness-aware aggregation weighting (M2)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from flp.core.aggregator import AggregationResult, FedAvgAggregator
from flp.core.client import ClientUpdate
from flp.core.staleness import StalenessWeighter
from flp.experiments.config_loader import AsyncConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_update(client_id: int, num_samples: int, loss: float = 0.5) -> ClientUpdate:
    """Factory for a minimal ClientUpdate with a 1-element weight tensor."""
    train_result = MagicMock()
    train_result.loss = loss
    return ClientUpdate(
        client_id=client_id,
        state_dict={"w": torch.tensor([1.0 + client_id * 0.1])},
        num_samples=num_samples,
        train_result=train_result,
    )


# ===========================================================================
# StalenessWeighter — construction
# ===========================================================================


class TestStalenessWeighterInit:
    def test_default_strategy_is_uniform(self) -> None:
        sw = StalenessWeighter()
        assert sw.strategy == "uniform"

    def test_default_decay_factor(self) -> None:
        sw = StalenessWeighter()
        assert sw.decay_factor == pytest.approx(0.9)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            StalenessWeighter(strategy="bad_strategy")  # type: ignore[arg-type]

    def test_zero_decay_factor_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            StalenessWeighter(strategy="exponential_decay", decay_factor=0.0)

    def test_negative_decay_factor_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            StalenessWeighter(strategy="exponential_decay", decay_factor=-0.5)

    def test_decay_factor_one_valid(self) -> None:
        sw = StalenessWeighter(strategy="exponential_decay", decay_factor=1.0)
        assert sw.decay_factor == pytest.approx(1.0)


# ===========================================================================
# StalenessWeighter — compute_weights input validation
# ===========================================================================


class TestStalenessWeighterValidation:
    def test_empty_inputs_return_empty(self) -> None:
        sw = StalenessWeighter()
        assert sw.compute_weights([], []) == []

    def test_mismatched_lengths_raise(self) -> None:
        sw = StalenessWeighter()
        with pytest.raises(ValueError, match="same length"):
            sw.compute_weights([0, 1], [100])

    def test_negative_num_samples_raises(self) -> None:
        sw = StalenessWeighter()
        with pytest.raises(ValueError):
            sw.compute_weights([0], [-1])

    def test_negative_staleness_raises(self) -> None:
        sw = StalenessWeighter()
        with pytest.raises(ValueError):
            sw.compute_weights([-1], [100])

    def test_all_zero_num_samples_raises(self) -> None:
        sw = StalenessWeighter()
        with pytest.raises(ValueError, match="zero"):
            sw.compute_weights([0, 0], [0, 0])


# ===========================================================================
# StalenessWeighter — uniform strategy
# ===========================================================================


class TestStalenessWeighterUniform:
    def test_weights_sum_to_one(self) -> None:
        sw = StalenessWeighter(strategy="uniform")
        w = sw.compute_weights([0, 1, 2], [100, 200, 300])
        assert sum(w) == pytest.approx(1.0)

    def test_weights_proportional_to_num_samples(self) -> None:
        sw = StalenessWeighter(strategy="uniform")
        # Equal num_samples → equal weights
        w = sw.compute_weights([0, 3, 9], [100, 100, 100])
        assert w[0] == pytest.approx(w[1])
        assert w[1] == pytest.approx(w[2])

    def test_staleness_does_not_affect_weights(self) -> None:
        sw = StalenessWeighter(strategy="uniform")
        w_fresh = sw.compute_weights([0, 0], [100, 200])
        w_stale = sw.compute_weights([5, 5], [100, 200])
        assert w_fresh == pytest.approx(w_stale)

    def test_single_update_weight_is_one(self) -> None:
        sw = StalenessWeighter(strategy="uniform")
        w = sw.compute_weights([3], [500])
        assert w[0] == pytest.approx(1.0)

    def test_proportional_values(self) -> None:
        sw = StalenessWeighter(strategy="uniform")
        w = sw.compute_weights([0, 0], [100, 300])
        assert w[0] == pytest.approx(0.25)
        assert w[1] == pytest.approx(0.75)


# ===========================================================================
# StalenessWeighter — inverse_staleness strategy
# ===========================================================================


class TestStalenessWeighterInverseStaleness:
    def test_weights_sum_to_one(self) -> None:
        sw = StalenessWeighter(strategy="inverse_staleness")
        w = sw.compute_weights([0, 1, 2], [100, 100, 100])
        assert sum(w) == pytest.approx(1.0)

    def test_fresher_update_higher_weight_equal_samples(self) -> None:
        sw = StalenessWeighter(strategy="inverse_staleness")
        w = sw.compute_weights([0, 3], [100, 100])
        assert w[0] > w[1]

    def test_zero_staleness_full_sample_weight(self) -> None:
        sw = StalenessWeighter(strategy="inverse_staleness")
        # Only one update with zero staleness → weight = 1.0
        w = sw.compute_weights([0], [100])
        assert w[0] == pytest.approx(1.0)

    def test_single_stale_update_weight_is_one(self) -> None:
        sw = StalenessWeighter(strategy="inverse_staleness")
        w = sw.compute_weights([5], [200])
        assert w[0] == pytest.approx(1.0)

    def test_penalty_formula(self) -> None:
        sw = StalenessWeighter(strategy="inverse_staleness")
        # staleness=[0,1], num_samples=[100,100]
        # raw = [100/1, 100/2] = [100, 50] → normalised = [2/3, 1/3]
        w = sw.compute_weights([0, 1], [100, 100])
        assert w[0] == pytest.approx(2 / 3)
        assert w[1] == pytest.approx(1 / 3)

    def test_larger_samples_can_outweigh_freshness(self) -> None:
        sw = StalenessWeighter(strategy="inverse_staleness")
        # fresh update: 10 samples at staleness=0 → raw=10
        # stale update: 1000 samples at staleness=1 → raw=500
        w = sw.compute_weights([0, 1], [10, 1000])
        assert w[1] > w[0]


# ===========================================================================
# StalenessWeighter — exponential_decay strategy
# ===========================================================================


class TestStalenessWeighterExponentialDecay:
    def test_weights_sum_to_one(self) -> None:
        sw = StalenessWeighter(strategy="exponential_decay", decay_factor=0.9)
        w = sw.compute_weights([0, 1, 2], [100, 100, 100])
        assert sum(w) == pytest.approx(1.0)

    def test_fresher_update_higher_weight_equal_samples(self) -> None:
        sw = StalenessWeighter(strategy="exponential_decay", decay_factor=0.8)
        w = sw.compute_weights([0, 2], [100, 100])
        assert w[0] > w[1]

    def test_decay_factor_one_matches_uniform(self) -> None:
        # decay_factor=1.0 → 1^staleness = 1 for any staleness → identical to uniform
        sw_exp = StalenessWeighter(strategy="exponential_decay", decay_factor=1.0)
        sw_uni = StalenessWeighter(strategy="uniform")
        staleness = [0, 1, 3]
        samples = [100, 200, 150]
        assert sw_exp.compute_weights(staleness, samples) == pytest.approx(
            sw_uni.compute_weights(staleness, samples)
        )

    def test_decay_formula(self) -> None:
        sw = StalenessWeighter(strategy="exponential_decay", decay_factor=0.5)
        # staleness=[0,1], num_samples=[100,100]
        # raw = [100*0.5^0, 100*0.5^1] = [100, 50] → [2/3, 1/3]
        w = sw.compute_weights([0, 1], [100, 100])
        assert w[0] == pytest.approx(2 / 3)
        assert w[1] == pytest.approx(1 / 3)

    def test_single_update_weight_is_one(self) -> None:
        sw = StalenessWeighter(strategy="exponential_decay", decay_factor=0.7)
        w = sw.compute_weights([4], [300])
        assert w[0] == pytest.approx(1.0)


# ===========================================================================
# FedAvgAggregator — explicit weights parameter
# ===========================================================================


class TestFedAvgAggregatorExplicitWeights:
    def _aggregator(self) -> FedAvgAggregator:
        return FedAvgAggregator()

    def test_none_weights_uses_num_samples(self) -> None:
        agg = self._aggregator()
        u1 = _make_update(0, num_samples=100)
        u2 = _make_update(1, num_samples=100)
        result = agg.aggregate([u1, u2], weights=None)
        assert isinstance(result, AggregationResult)
        assert result.num_clients == 2

    def test_explicit_equal_weights_same_as_default_equal_samples(self) -> None:
        agg = self._aggregator()
        u1 = _make_update(0, num_samples=100)
        u2 = _make_update(1, num_samples=100)
        result_default = agg.aggregate([u1, u2])
        result_explicit = agg.aggregate([u1, u2], weights=[0.5, 0.5])
        for key in result_default.state_dict:
            assert torch.allclose(
                result_default.state_dict[key],
                result_explicit.state_dict[key],
            )

    def test_explicit_weights_affect_aggregated_values(self) -> None:
        agg = self._aggregator()
        u1 = _make_update(0, num_samples=100)
        u2 = _make_update(1, num_samples=100)
        # Weight entirely on client 0
        result = agg.aggregate([u1, u2], weights=[1.0, 0.0])
        assert torch.allclose(result.state_dict["w"], u1.state_dict["w"])

    def test_explicit_weights_length_mismatch_raises(self) -> None:
        agg = self._aggregator()
        u1 = _make_update(0, num_samples=100)
        u2 = _make_update(1, num_samples=100)
        with pytest.raises(ValueError, match="match"):
            agg.aggregate([u1, u2], weights=[0.5, 0.3, 0.2])

    def test_weighted_loss_uses_explicit_weights(self) -> None:
        agg = self._aggregator()
        u1 = _make_update(0, num_samples=100, loss=1.0)
        u2 = _make_update(1, num_samples=100, loss=0.0)
        # All weight on client 0 (loss=1.0)
        result = agg.aggregate([u1, u2], weights=[1.0, 0.0])
        assert result.weighted_loss == pytest.approx(1.0)

    def test_staleness_weighter_integrates_with_aggregator(self) -> None:
        """End-to-end: StalenessWeighter weights flow into FedAvgAggregator."""
        sw = StalenessWeighter(strategy="inverse_staleness")
        agg = self._aggregator()
        u1 = _make_update(0, num_samples=100)
        u2 = _make_update(1, num_samples=100)
        weights = sw.compute_weights(staleness_values=[0, 2], num_samples=[100, 100])
        result = agg.aggregate([u1, u2], weights=weights)
        # Fresh update (client 0) contributes more → result closer to u1's weight (1.0)
        expected = u1.state_dict["w"].item() * weights[0] + u2.state_dict["w"].item() * weights[1]
        assert result.state_dict["w"].item() == pytest.approx(expected, abs=1e-5)


# ===========================================================================
# AsyncConfig — new fields
# ===========================================================================


class TestAsyncConfigNewFields:
    def test_default_staleness_strategy_is_uniform(self) -> None:
        cfg = AsyncConfig()
        assert cfg.staleness_strategy == "uniform"

    def test_default_decay_factor(self) -> None:
        cfg = AsyncConfig()
        assert cfg.staleness_decay_factor == pytest.approx(0.9)

    def test_inverse_staleness_strategy_valid(self) -> None:
        cfg = AsyncConfig(staleness_strategy="inverse_staleness")
        assert cfg.staleness_strategy == "inverse_staleness"

    def test_exponential_decay_strategy_valid(self) -> None:
        cfg = AsyncConfig(staleness_strategy="exponential_decay", staleness_decay_factor=0.8)
        assert cfg.staleness_strategy == "exponential_decay"
        assert cfg.staleness_decay_factor == pytest.approx(0.8)

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(Exception):
            AsyncConfig(staleness_strategy="bad")  # type: ignore[arg-type]

    def test_decay_factor_zero_raises(self) -> None:
        with pytest.raises(Exception):
            AsyncConfig(staleness_decay_factor=0.0)

    def test_decay_factor_above_one_raises(self) -> None:
        with pytest.raises(Exception):
            AsyncConfig(staleness_decay_factor=1.1)

    def test_decay_factor_one_valid(self) -> None:
        cfg = AsyncConfig(staleness_decay_factor=1.0)
        assert cfg.staleness_decay_factor == pytest.approx(1.0)

    def test_existing_fields_unchanged(self) -> None:
        cfg = AsyncConfig(
            enabled=True,
            delay_min=0.0,
            delay_max=3.0,
            staleness_threshold=3,
            staleness_strategy="exponential_decay",
            staleness_decay_factor=0.85,
        )
        assert cfg.enabled is True
        assert cfg.staleness_threshold == 3
