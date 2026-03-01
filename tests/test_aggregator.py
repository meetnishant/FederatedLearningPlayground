"""Tests for the FedAvg aggregator."""

import pytest
import torch

from flp.core.aggregator import FedAvgAggregator
from flp.core.client import ClientUpdate
from flp.core.trainer import TrainResult


def _make_update(value: float, num_samples: int, client_id: int = 0) -> ClientUpdate:
    state_dict = {
        "weight": torch.tensor([[value, value], [value, value]]),
        "bias": torch.tensor([value, value]),
    }
    return ClientUpdate(
        client_id=client_id,
        state_dict=state_dict,
        num_samples=num_samples,
        train_result=TrainResult(loss=0.5, total_samples=num_samples, epochs=1),
    )


def _make_update_with_bn(value: float, num_samples: int, client_id: int = 0) -> ClientUpdate:
    """Update whose state dict includes a BatchNorm num_batches_tracked (Long)."""
    state_dict = {
        "weight": torch.tensor([value, value]),
        "num_batches_tracked": torch.tensor(10, dtype=torch.long),
    }
    return ClientUpdate(
        client_id=client_id,
        state_dict=state_dict,
        num_samples=num_samples,
        train_result=TrainResult(loss=0.4, total_samples=num_samples, epochs=1),
    )


class TestFedAvgAggregator:
    def setup_method(self) -> None:
        self.agg = FedAvgAggregator()

    def test_raises_on_empty_updates(self) -> None:
        with pytest.raises(ValueError, match="empty update list"):
            self.agg.aggregate([])

    def test_raises_on_zero_total_samples(self) -> None:
        update = _make_update(1.0, num_samples=0)
        with pytest.raises(ValueError, match="num_samples=0"):
            self.agg.aggregate([update])

    def test_single_client_returns_same_weights(self) -> None:
        update = _make_update(3.0, num_samples=100)
        result = self.agg.aggregate([update])
        assert torch.allclose(result.state_dict["weight"], update.state_dict["weight"].float())

    def test_equal_weight_average(self) -> None:
        updates = [
            _make_update(2.0, num_samples=50, client_id=0),
            _make_update(4.0, num_samples=50, client_id=1),
        ]
        result = self.agg.aggregate(updates)
        expected = torch.full((2, 2), 3.0)
        assert torch.allclose(result.state_dict["weight"], expected)

    def test_weighted_average(self) -> None:
        updates = [
            _make_update(0.0, num_samples=75, client_id=0),
            _make_update(4.0, num_samples=25, client_id=1),
        ]
        result = self.agg.aggregate(updates)
        # 0 * 0.75 + 4 * 0.25 = 1.0
        expected = torch.ones(2, 2)
        assert torch.allclose(result.state_dict["weight"], expected)

    def test_preserves_all_float_keys(self) -> None:
        update = _make_update(1.0, num_samples=10)
        result = self.agg.aggregate([update])
        assert set(result.state_dict.keys()) == {"weight", "bias"}

    def test_integer_buffers_not_averaged(self) -> None:
        updates = [
            _make_update_with_bn(1.0, num_samples=100, client_id=0),
            _make_update_with_bn(2.0, num_samples=50, client_id=1),
        ]
        result = self.agg.aggregate(updates)
        # num_batches_tracked is Long and must not be converted to float
        assert result.state_dict["num_batches_tracked"].dtype == torch.long

    def test_total_samples_reported_correctly(self) -> None:
        updates = [
            _make_update(1.0, num_samples=60, client_id=0),
            _make_update(2.0, num_samples=40, client_id=1),
        ]
        result = self.agg.aggregate(updates)
        assert result.total_samples == 100
        assert result.num_clients == 2

    def test_weighted_loss_computed(self) -> None:
        updates = [
            _make_update(1.0, num_samples=100, client_id=0),
            _make_update(2.0, num_samples=100, client_id=1),
        ]
        # Both have loss=0.5 → weighted_loss should be 0.5
        result = self.agg.aggregate(updates)
        assert result.weighted_loss == pytest.approx(0.5)

    def test_output_dtype_matches_input(self) -> None:
        update = _make_update(1.0, num_samples=10)
        result = self.agg.aggregate([update])
        for key in result.state_dict:
            assert result.state_dict[key].dtype == update.state_dict[key].dtype
