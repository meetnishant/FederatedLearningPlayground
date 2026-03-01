"""Tests for research metrics: divergence and fairness (M4)."""

from __future__ import annotations

import math

import pytest
import torch

from flp.research.divergence import (
    DivergenceResult,
    compute_weight_divergence,
    cosine_similarity_between_updates,
)
from flp.research.fairness import (
    FairnessResult,
    compute_fairness_metrics,
    compute_gini,
    qfedavg_weighted_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sd(w: list[float], bias: list[float] | None = None) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {"weight": torch.tensor(w, dtype=torch.float32)}
    if bias is not None:
        sd["bias"] = torch.tensor(bias, dtype=torch.float32)
    return sd


def _sd_with_int(w: list[float]) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor(w, dtype=torch.float32),
        "num_batches_tracked": torch.tensor([5], dtype=torch.int64),
    }


# ===========================================================================
# compute_weight_divergence
# ===========================================================================


class TestComputeWeightDivergence:
    def test_returns_divergence_result(self) -> None:
        global_sd = _sd([1.0, 2.0, 3.0])
        client_states = {0: _sd([1.5, 2.5, 3.5])}
        result = compute_weight_divergence(global_sd, client_states)
        assert isinstance(result, DivergenceResult)

    def test_identical_states_zero_divergence(self) -> None:
        sd = _sd([1.0, 2.0, 3.0])
        result = compute_weight_divergence(sd, {0: _sd([1.0, 2.0, 3.0])})
        assert result.per_client_l2[0] == pytest.approx(0.0, abs=1e-6)
        assert result.mean_divergence == pytest.approx(0.0, abs=1e-6)
        assert result.max_divergence == pytest.approx(0.0, abs=1e-6)

    def test_known_l2_norm(self) -> None:
        # delta = [1, 0, 0] → L2 = 1.0
        global_sd = _sd([0.0, 0.0, 0.0])
        client_states = {0: _sd([1.0, 0.0, 0.0])}
        result = compute_weight_divergence(global_sd, client_states)
        assert result.per_client_l2[0] == pytest.approx(1.0)

    def test_known_l2_norm_pythagorean(self) -> None:
        # delta = [3, 4] → L2 = 5
        global_sd = _sd([0.0, 0.0])
        client_states = {0: _sd([3.0, 4.0])}
        result = compute_weight_divergence(global_sd, client_states)
        assert result.per_client_l2[0] == pytest.approx(5.0)

    def test_multiple_clients_mean_and_max(self) -> None:
        global_sd = _sd([0.0])
        client_states = {
            0: _sd([1.0]),   # L2 = 1.0
            1: _sd([3.0]),   # L2 = 3.0
            2: _sd([2.0]),   # L2 = 2.0
        }
        result = compute_weight_divergence(global_sd, client_states)
        assert result.mean_divergence == pytest.approx(2.0)
        assert result.max_divergence == pytest.approx(3.0)

    def test_empty_client_states_returns_zeros(self) -> None:
        result = compute_weight_divergence(_sd([1.0, 2.0]), {})
        assert result.per_client_l2 == {}
        assert result.mean_divergence == pytest.approx(0.0)
        assert result.max_divergence == pytest.approx(0.0)

    def test_integer_buffers_ignored(self) -> None:
        global_sd = _sd_with_int([0.0, 0.0])
        client_sd = _sd_with_int([0.0, 0.0])
        # Make the integer buffer differ — should not affect divergence.
        client_sd["num_batches_tracked"] = torch.tensor([99], dtype=torch.int64)
        result = compute_weight_divergence(global_sd, {0: client_sd})
        assert result.per_client_l2[0] == pytest.approx(0.0)

    def test_monotone_with_distance(self) -> None:
        global_sd = _sd([0.0, 0.0, 0.0])
        small = compute_weight_divergence(global_sd, {0: _sd([0.1, 0.0, 0.0])})
        large = compute_weight_divergence(global_sd, {0: _sd([10.0, 0.0, 0.0])})
        assert large.per_client_l2[0] > small.per_client_l2[0]

    def test_missing_key_raises(self) -> None:
        global_sd = {"w": torch.tensor([1.0]), "b": torch.tensor([0.0])}
        client_sd = {"w": torch.tensor([2.0])}  # missing "b"
        with pytest.raises(ValueError, match="missing keys"):
            compute_weight_divergence(global_sd, {0: client_sd})

    def test_multiple_float_tensors(self) -> None:
        global_sd = _sd([0.0, 0.0], bias=[0.0])
        client_sd = _sd([3.0, 4.0], bias=[0.0])  # weight delta = 5
        result = compute_weight_divergence(global_sd, {0: client_sd})
        assert result.per_client_l2[0] == pytest.approx(5.0)


# ===========================================================================
# cosine_similarity_between_updates
# ===========================================================================


class TestCosineSimilarityBetweenUpdates:
    def test_identical_updates_similarity_one(self) -> None:
        u = _sd([1.0, 2.0, 3.0])
        assert cosine_similarity_between_updates(u, u) == pytest.approx(1.0)

    def test_opposite_updates_similarity_minus_one(self) -> None:
        a = _sd([1.0, 0.0])
        b = _sd([-1.0, 0.0])
        assert cosine_similarity_between_updates(a, b) == pytest.approx(-1.0)

    def test_orthogonal_updates_similarity_zero(self) -> None:
        a = _sd([1.0, 0.0])
        b = _sd([0.0, 1.0])
        assert cosine_similarity_between_updates(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_returns_zero(self) -> None:
        zero = _sd([0.0, 0.0, 0.0])
        other = _sd([1.0, 2.0, 3.0])
        assert cosine_similarity_between_updates(zero, other) == pytest.approx(0.0)

    def test_result_in_range(self) -> None:
        import random
        random.seed(42)
        a = _sd([random.gauss(0, 1) for _ in range(20)])
        b = _sd([random.gauss(0, 1) for _ in range(20)])
        sim = cosine_similarity_between_updates(a, b)
        assert -1.0 <= sim <= 1.0

    def test_scale_invariant(self) -> None:
        a = _sd([1.0, 2.0, 3.0])
        b = _sd([2.0, 4.0, 6.0])  # 2x scale of a
        assert cosine_similarity_between_updates(a, b) == pytest.approx(1.0)

    def test_integer_buffers_ignored(self) -> None:
        a = _sd_with_int([1.0, 0.0])
        b = _sd_with_int([0.0, 1.0])
        # Should be orthogonal regardless of integer buffer values.
        assert cosine_similarity_between_updates(a, b) == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# compute_gini
# ===========================================================================


class TestComputeGini:
    def test_empty_returns_zero(self) -> None:
        assert compute_gini([]) == pytest.approx(0.0)

    def test_single_value_returns_zero(self) -> None:
        assert compute_gini([5.0]) == pytest.approx(0.0)

    def test_equal_values_returns_zero(self) -> None:
        assert compute_gini([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-6)

    def test_all_zeros_returns_zero(self) -> None:
        assert compute_gini([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_one_has_all_value(self) -> None:
        # One client has all the "accuracy"; others have 0 → near 1.
        n = 5
        vals = [0.0] * (n - 1) + [1.0]
        g = compute_gini(vals)
        # Exact value: G = (2*n - n - 1) / (n*(n-1)/(n-1)) = (n-1)/n for one-hot
        expected = (n - 1) / n
        assert g == pytest.approx(expected, abs=1e-6)

    def test_negative_value_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_gini([1.0, -0.1, 0.5])

    def test_gini_increases_with_inequality(self) -> None:
        low_ineq = compute_gini([0.7, 0.8, 0.9])
        high_ineq = compute_gini([0.1, 0.1, 1.5])
        assert high_ineq > low_ineq

    def test_result_non_negative(self) -> None:
        vals = [0.5, 0.6, 0.7, 0.8, 0.9]
        assert compute_gini(vals) >= 0.0

    def test_known_value_two_elements(self) -> None:
        # [0, 1] → sorted: [0, 1]; weighted_sum = 1*0 + 2*1 = 2; total = 1; n=2
        # G = (2*2)/(2*1) - 3/2 = 2 - 1.5 = 0.5
        assert compute_gini([0.0, 1.0]) == pytest.approx(0.5, abs=1e-6)


# ===========================================================================
# compute_fairness_metrics
# ===========================================================================


class TestComputeFairnessMetrics:
    def test_returns_fairness_result(self) -> None:
        result = compute_fairness_metrics({0: 0.9, 1: 0.8, 2: 0.7})
        assert isinstance(result, FairnessResult)

    def test_empty_returns_zeros(self) -> None:
        result = compute_fairness_metrics({})
        assert result.gini_coefficient == pytest.approx(0.0)
        assert result.accuracy_variance == pytest.approx(0.0)
        assert result.min_accuracy == pytest.approx(0.0)
        assert result.max_accuracy == pytest.approx(0.0)
        assert result.spread == pytest.approx(0.0)

    def test_single_client(self) -> None:
        result = compute_fairness_metrics({0: 0.85})
        assert result.gini_coefficient == pytest.approx(0.0)
        assert result.min_accuracy == pytest.approx(0.85)
        assert result.max_accuracy == pytest.approx(0.85)
        assert result.spread == pytest.approx(0.0)

    def test_equal_accuracies_zero_gini(self) -> None:
        result = compute_fairness_metrics({0: 0.8, 1: 0.8, 2: 0.8})
        assert result.gini_coefficient == pytest.approx(0.0, abs=1e-6)
        assert result.accuracy_variance == pytest.approx(0.0, abs=1e-6)

    def test_spread_is_max_minus_min(self) -> None:
        result = compute_fairness_metrics({0: 0.5, 1: 0.9, 2: 0.7})
        assert result.spread == pytest.approx(0.9 - 0.5)

    def test_variance_known(self) -> None:
        # Values [0, 1] → mean=0.5, variance=0.25
        result = compute_fairness_metrics({0: 0.0, 1: 1.0})
        assert result.accuracy_variance == pytest.approx(0.25)

    def test_gini_matches_standalone_function(self) -> None:
        accs = {0: 0.5, 1: 0.7, 2: 0.9}
        result = compute_fairness_metrics(accs)
        expected_gini = compute_gini(list(accs.values()))
        assert result.gini_coefficient == pytest.approx(expected_gini)


# ===========================================================================
# qfedavg_weighted_loss
# ===========================================================================


class TestQFedAvgWeightedLoss:
    def test_q_zero_uniform_weights(self) -> None:
        losses = {0: 0.1, 1: 0.5, 2: 1.0}
        weights = qfedavg_weighted_loss(losses, q=0)
        for w in weights.values():
            assert w == pytest.approx(1 / 3)

    def test_weights_sum_to_one(self) -> None:
        losses = {0: 0.2, 1: 0.5, 2: 0.8}
        for q in [0.0, 1.0, 2.0, 5.0]:
            weights = qfedavg_weighted_loss(losses, q=q)
            assert sum(weights.values()) == pytest.approx(1.0)

    def test_q_one_proportional_to_loss(self) -> None:
        losses = {0: 1.0, 1: 3.0}
        weights = qfedavg_weighted_loss(losses, q=1)
        assert weights[0] == pytest.approx(0.25)
        assert weights[1] == pytest.approx(0.75)

    def test_higher_q_focuses_on_worst_client(self) -> None:
        losses = {0: 0.1, 1: 1.0}
        w_low = qfedavg_weighted_loss(losses, q=1)
        w_high = qfedavg_weighted_loss(losses, q=5)
        # With higher q, the worst client (1) gets even more weight.
        assert w_high[1] > w_low[1]

    def test_empty_returns_empty(self) -> None:
        assert qfedavg_weighted_loss({}, q=1.0) == {}

    def test_negative_q_raises(self) -> None:
        with pytest.raises(ValueError):
            qfedavg_weighted_loss({0: 0.5}, q=-1.0)

    def test_negative_loss_raises(self) -> None:
        with pytest.raises(ValueError):
            qfedavg_weighted_loss({0: -0.1, 1: 0.5}, q=1.0)

    def test_all_zero_losses_uniform(self) -> None:
        # All zeros → total is 0 → fall back to uniform.
        losses = {0: 0.0, 1: 0.0}
        weights = qfedavg_weighted_loss(losses, q=1.0)
        assert weights[0] == pytest.approx(0.5)
        assert weights[1] == pytest.approx(0.5)

    def test_single_client_weight_one(self) -> None:
        weights = qfedavg_weighted_loss({0: 0.5}, q=2.0)
        assert weights[0] == pytest.approx(1.0)

    def test_returned_keys_match_input(self) -> None:
        losses = {3: 0.1, 7: 0.9, 12: 0.5}
        weights = qfedavg_weighted_loss(losses, q=1.0)
        assert set(weights.keys()) == {3, 7, 12}
