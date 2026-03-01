"""Tests for the FedAvg aggregator."""

import pytest
import torch

from flp.core.aggregator import FedAvgAggregator


def _make_state(value: float) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor([[value, value], [value, value]]),
        "bias": torch.tensor([value, value]),
    }


class TestFedAvgAggregator:
    def setup_method(self) -> None:
        self.aggregator = FedAvgAggregator()

    def test_raises_on_empty_updates(self) -> None:
        with pytest.raises(ValueError, match="empty update list"):
            self.aggregator.aggregate([])

    def test_raises_on_zero_samples(self) -> None:
        updates = [{"state_dict": _make_state(1.0), "num_samples": 0}]
        with pytest.raises(ValueError, match="zero"):
            self.aggregator.aggregate(updates)

    def test_single_client_returns_same_weights(self) -> None:
        state = _make_state(3.0)
        updates = [{"state_dict": state, "num_samples": 100}]
        result = self.aggregator.aggregate(updates)
        assert torch.allclose(result["weight"], state["weight"].float())

    def test_equal_weight_average(self) -> None:
        updates = [
            {"state_dict": _make_state(2.0), "num_samples": 50},
            {"state_dict": _make_state(4.0), "num_samples": 50},
        ]
        result = self.aggregator.aggregate(updates)
        expected = torch.full((2, 2), 3.0)
        assert torch.allclose(result["weight"], expected)

    def test_weighted_average(self) -> None:
        updates = [
            {"state_dict": _make_state(0.0), "num_samples": 75},
            {"state_dict": _make_state(4.0), "num_samples": 25},
        ]
        result = self.aggregator.aggregate(updates)
        # 0 * 0.75 + 4 * 0.25 = 1.0
        expected = torch.ones(2, 2)
        assert torch.allclose(result["weight"], expected)

    def test_preserves_all_keys(self) -> None:
        updates = [{"state_dict": _make_state(1.0), "num_samples": 10}]
        result = self.aggregator.aggregate(updates)
        assert set(result.keys()) == {"weight", "bias"}
