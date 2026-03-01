"""Unit tests for simulation/dropout.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from flp.simulation.dropout import (
    DropoutMetrics,
    DropoutResult,
    DropoutRoundRecord,
    DropoutSimulator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clients(n: int) -> list[MagicMock]:
    clients = []
    for i in range(n):
        c = MagicMock()
        c.client_id = i
        clients.append(c)
    return clients


def _apply_and_record(
    sim: DropoutSimulator, clients: list[MagicMock], round_num: int
) -> DropoutResult:
    result = sim.apply(clients, round_num)
    sim.record(result)
    return result


# ---------------------------------------------------------------------------
# DropoutResult structure
# ---------------------------------------------------------------------------


class TestDropoutResult:
    def test_active_ids_match_active_clients(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.5, seed=0)
        clients = _make_clients(20)
        result = sim.apply(clients, round_num=1)
        assert result.active_ids == [c.client_id for c in result.active]

    def test_dropped_plus_active_equals_selected(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.4, seed=42)
        clients = _make_clients(15)
        result = sim.apply(clients, round_num=5)
        assert len(result.active) + len(result.dropped_ids) == result.num_selected

    def test_no_overlap_between_active_and_dropped(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.5, seed=1)
        clients = _make_clients(30)
        result = sim.apply(clients, round_num=2)
        assert set(result.active_ids).isdisjoint(set(result.dropped_ids))

    def test_actual_rate_bounds(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.3, seed=7)
        clients = _make_clients(100)
        result = sim.apply(clients, round_num=1)
        assert 0.0 <= result.actual_rate <= 1.0

    def test_all_dropped_flag_when_empty_active(self) -> None:
        # Force all-dropout by using rate=0.99 and a specific seed/round
        # that produces all drops.  Use a large client pool to make it unlikely
        # to fail, then check the flag is consistent with the active list.
        sim = DropoutSimulator(dropout_rate=0.99, seed=0)
        clients = _make_clients(5)
        result = sim.apply(clients, round_num=1)
        assert result.all_dropped == (len(result.active) == 0)

    def test_all_dropped_false_when_some_survive(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.0, seed=0)
        clients = _make_clients(10)
        result = sim.apply(clients, round_num=1)
        assert result.all_dropped is False

    def test_round_num_stored_correctly(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.2, seed=3)
        clients = _make_clients(10)
        for rn in [1, 5, 10]:
            assert sim.apply(clients, round_num=rn).round_num == rn


# ---------------------------------------------------------------------------
# Zero dropout (disabled)
# ---------------------------------------------------------------------------


class TestZeroDropout:
    def test_all_clients_survive(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.0, seed=42)
        clients = _make_clients(20)
        result = sim.apply(clients, round_num=1)
        assert len(result.active) == 20

    def test_no_dropped_ids(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.0, seed=42)
        result = sim.apply(_make_clients(10), round_num=3)
        assert result.dropped_ids == []

    def test_actual_rate_is_zero(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.0, seed=0)
        result = sim.apply(_make_clients(10), round_num=1)
        assert result.actual_rate == 0.0

    def test_empty_client_list(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.0, seed=0)
        result = sim.apply([], round_num=1)
        assert result.active == []
        assert result.all_dropped is True


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        clients = _make_clients(20)
        s1 = DropoutSimulator(dropout_rate=0.4, seed=7)
        s2 = DropoutSimulator(dropout_rate=0.4, seed=7)
        r1 = s1.apply(clients, round_num=3)
        r2 = s2.apply(clients, round_num=3)
        assert r1.active_ids == r2.active_ids
        assert r1.dropped_ids == r2.dropped_ids

    def test_different_seeds_differ(self) -> None:
        clients = _make_clients(50)
        s1 = DropoutSimulator(dropout_rate=0.5, seed=0)
        s2 = DropoutSimulator(dropout_rate=0.5, seed=999)
        r1 = s1.apply(clients, round_num=1)
        r2 = s2.apply(clients, round_num=1)
        assert r1.active_ids != r2.active_ids

    def test_different_rounds_differ(self) -> None:
        clients = _make_clients(30)
        sim = DropoutSimulator(dropout_rate=0.4, seed=42)
        r1 = sim.apply(clients, round_num=1)
        r2 = sim.apply(clients, round_num=2)
        # Statistically near-certain to differ with 30 clients and 0.4 rate
        assert r1.active_ids != r2.active_ids

    def test_apply_is_stateless(self) -> None:
        """Calling apply twice for the same round must return identical results."""
        clients = _make_clients(20)
        sim = DropoutSimulator(dropout_rate=0.3, seed=5)
        r1 = sim.apply(clients, round_num=4)
        r2 = sim.apply(clients, round_num=4)
        assert r1.active_ids == r2.active_ids


# ---------------------------------------------------------------------------
# Statistical behaviour
# ---------------------------------------------------------------------------


class TestStatisticalBehaviour:
    def test_high_dropout_reduces_clients(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.9, seed=42)
        clients = _make_clients(100)
        result = sim.apply(clients, round_num=1)
        assert len(result.active) < 50  # expect ~10 survivors from 100

    def test_dropout_rate_approximately_correct(self) -> None:
        """Over many rounds, mean actual_rate should be close to configured rate."""
        rate = 0.3
        sim = DropoutSimulator(dropout_rate=rate, seed=0)
        clients = _make_clients(200)
        actual_rates = [sim.apply(clients, round_num=r).actual_rate for r in range(1, 101)]
        mean_rate = sum(actual_rates) / len(actual_rates)
        assert abs(mean_rate - rate) < 0.05, (
            f"Mean dropout rate {mean_rate:.3f} too far from configured {rate}"
        )

    def test_no_client_appears_twice_in_active(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.3, seed=0)
        clients = _make_clients(20)
        result = sim.apply(clients, round_num=1)
        assert len(result.active_ids) == len(set(result.active_ids))

    def test_order_of_clients_preserved_in_active(self) -> None:
        """Active clients must preserve the original selection order."""
        sim = DropoutSimulator(dropout_rate=0.5, seed=99)
        clients = _make_clients(20)
        result = sim.apply(clients, round_num=1)
        original_order = [c.client_id for c in clients]
        active_in_original = [cid for cid in original_order if cid in set(result.active_ids)]
        assert result.active_ids == active_in_original


# ---------------------------------------------------------------------------
# DropoutMetrics tracking
# ---------------------------------------------------------------------------


class TestDropoutMetrics:
    def test_empty_metrics_defaults(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.2, seed=0)
        assert sim.metrics.total_selected == 0
        assert sim.metrics.total_dropped == 0
        assert sim.metrics.total_skipped_rounds == 0
        assert sim.metrics.overall_dropout_rate == 0.0

    def test_record_accumulates_totals(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.3, seed=42)
        clients = _make_clients(10)
        for rn in range(1, 6):
            result = sim.apply(clients, round_num=rn)
            sim.record(result)
        assert sim.metrics.total_selected == 50  # 10 clients × 5 rounds
        assert sim.metrics.total_dropped + sum(sim.metrics.active_counts_per_round) == 50

    def test_records_length_matches_rounds(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.2, seed=0)
        clients = _make_clients(10)
        for rn in range(1, 8):
            _apply_and_record(sim, clients, rn)
        assert len(sim.metrics.records) == 7

    def test_record_tracks_skipped_rounds(self) -> None:
        """A round where all clients drop out must increment total_skipped_rounds."""
        sim = DropoutSimulator(dropout_rate=0.99, seed=0)
        clients = _make_clients(2)
        # Run many rounds until at least one all-dropout occurs
        skipped = 0
        for rn in range(1, 50):
            result = sim.apply(clients, round_num=rn)
            sim.record(result)
            if result.all_dropped:
                skipped += 1
        assert sim.metrics.total_skipped_rounds == skipped

    def test_dropout_rates_per_round_length(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.3, seed=1)
        clients = _make_clients(10)
        for rn in range(1, 6):
            _apply_and_record(sim, clients, rn)
        rates = sim.metrics.dropout_rates_per_round
        assert len(rates) == 5
        assert all(0.0 <= r <= 1.0 for r in rates)

    def test_overall_dropout_rate_within_bounds(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.4, seed=7)
        clients = _make_clients(20)
        for rn in range(1, 21):
            _apply_and_record(sim, clients, rn)
        assert 0.0 <= sim.metrics.overall_dropout_rate <= 1.0

    def test_summary_keys(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.2, seed=0)
        summary = sim.metrics.summary()
        for key in [
            "total_rounds",
            "total_selected",
            "total_dropped",
            "total_skipped_rounds",
            "overall_dropout_rate",
        ]:
            assert key in summary, f"Missing key '{key}' in summary"

    def test_reset_metrics_clears_state(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.3, seed=0)
        clients = _make_clients(10)
        for rn in range(1, 6):
            _apply_and_record(sim, clients, rn)
        assert sim.metrics.total_selected > 0
        sim.reset_metrics()
        assert sim.metrics.total_selected == 0
        assert sim.metrics.records == []

    def test_apply_does_not_mutate_metrics(self) -> None:
        """apply() alone must not change metrics; only record() does."""
        sim = DropoutSimulator(dropout_rate=0.3, seed=0)
        clients = _make_clients(10)
        sim.apply(clients, round_num=1)
        sim.apply(clients, round_num=2)
        assert sim.metrics.total_selected == 0

    def test_per_round_records_store_correct_fields(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.5, seed=42)
        clients = _make_clients(10)
        result = _apply_and_record(sim, clients, round_num=3)
        rec: DropoutRoundRecord = sim.metrics.records[0]
        assert rec.round_num == 3
        assert rec.num_selected == 10
        assert rec.num_active == len(result.active)
        assert rec.num_dropped == len(result.dropped_ids)
        assert rec.num_active + rec.num_dropped == rec.num_selected


# ---------------------------------------------------------------------------
# Config-driven behaviour
# ---------------------------------------------------------------------------


class TestConfigDrivenBehaviour:
    def test_dropout_rate_attribute_stored(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.25, seed=0)
        assert sim.dropout_rate == 0.25

    def test_seed_attribute_stored(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.1, seed=99)
        assert sim.seed == 99

    def test_rate_zero_is_accepted(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.0)
        assert sim.dropout_rate == 0.0

    def test_rate_near_one_is_accepted(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.999)
        assert sim.dropout_rate == pytest.approx(0.999)


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rate_of_one_raises(self) -> None:
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            DropoutSimulator(dropout_rate=1.0)

    def test_rate_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            DropoutSimulator(dropout_rate=1.5)

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            DropoutSimulator(dropout_rate=-0.1)

    def test_empty_client_list_does_not_raise(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.5, seed=0)
        result = sim.apply([], round_num=1)
        assert result.active == []
        assert result.num_selected == 0
