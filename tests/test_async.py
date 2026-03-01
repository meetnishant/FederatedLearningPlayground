"""Tests for the async FL event loop and async server components."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from flp.core.event_loop import FLEvent, FLEventLoop
from flp.core.async_server import AsyncRoundSummary
from flp.core.server import RoundSummary
from flp.experiments.config_loader import AsyncConfig, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    virtual_round: int,
    client_id: int = 0,
    model_version: int = 0,
) -> FLEvent:
    """Factory for a minimal FLEvent with a mock update payload."""
    return FLEvent(
        virtual_round=virtual_round,
        client_id=client_id,
        model_version=model_version,
        update=MagicMock(),
    )


# ===========================================================================
# FLEvent
# ===========================================================================


class TestFLEvent:
    def test_fields_accessible(self) -> None:
        ev = _make_event(virtual_round=5, client_id=3, model_version=2)
        assert ev.virtual_round == 5
        assert ev.client_id == 3
        assert ev.model_version == 2

    def test_is_frozen(self) -> None:
        ev = _make_event(1)
        with pytest.raises((AttributeError, TypeError)):
            ev.virtual_round = 99  # type: ignore[misc]

    def test_equality_on_same_values(self) -> None:
        mock = MagicMock()
        e1 = FLEvent(virtual_round=1, client_id=0, model_version=0, update=mock)
        e2 = FLEvent(virtual_round=1, client_id=0, model_version=0, update=mock)
        assert e1 == e2

    def test_update_field_accepts_any_object(self) -> None:
        # Ensure no runtime type error for arbitrary payload
        ev = FLEvent(virtual_round=1, client_id=0, model_version=0, update={"key": "val"})
        assert ev.update == {"key": "val"}


# ===========================================================================
# FLEventLoop — initialisation
# ===========================================================================


class TestFLEventLoopInit:
    def test_empty_on_init(self) -> None:
        loop = FLEventLoop()
        assert loop.pending_count == 0

    def test_pop_ready_on_empty_returns_empty(self) -> None:
        loop = FLEventLoop()
        assert loop.pop_ready(current_round=5) == []

    def test_discard_stale_on_empty_returns_zero(self) -> None:
        loop = FLEventLoop()
        assert loop.discard_stale(min_virtual_round=1) == 0


# ===========================================================================
# FLEventLoop — push & pending_count
# ===========================================================================


class TestFLEventLoopPush:
    def test_pending_count_increments(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(1))
        loop.push(_make_event(2))
        assert loop.pending_count == 2

    def test_push_maintains_sorted_order(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(5))
        loop.push(_make_event(1))
        loop.push(_make_event(3))
        # Internal queue should be sorted ascending
        rounds = [e.virtual_round for e in loop._queue]
        assert rounds == sorted(rounds)

    def test_push_stable_for_same_virtual_round(self) -> None:
        loop = FLEventLoop()
        e1 = _make_event(virtual_round=2, client_id=0)
        e2 = _make_event(virtual_round=2, client_id=1)
        loop.push(e1)
        loop.push(e2)
        # Both present; insertion order preserved within same round
        assert loop.pending_count == 2
        ids = [e.client_id for e in loop._queue]
        assert ids == [0, 1]


# ===========================================================================
# FLEventLoop — pop_ready
# ===========================================================================


class TestFLEventLoopPopReady:
    def test_pop_ready_returns_events_at_exact_round(self) -> None:
        loop = FLEventLoop()
        ev = _make_event(virtual_round=3)
        loop.push(ev)
        ready = loop.pop_ready(current_round=3)
        assert len(ready) == 1
        assert ready[0] is ev

    def test_pop_ready_returns_events_before_current_round(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(virtual_round=1))
        loop.push(_make_event(virtual_round=2))
        ready = loop.pop_ready(current_round=5)
        assert len(ready) == 2

    def test_pop_ready_leaves_future_events(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(virtual_round=10))
        ready = loop.pop_ready(current_round=3)
        assert ready == []
        assert loop.pending_count == 1

    def test_pop_ready_removes_from_queue(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(virtual_round=2))
        loop.pop_ready(current_round=2)
        assert loop.pending_count == 0

    def test_pop_ready_partial_drain(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(virtual_round=1))
        loop.push(_make_event(virtual_round=3))
        loop.push(_make_event(virtual_round=5))
        ready = loop.pop_ready(current_round=3)
        assert len(ready) == 2
        assert loop.pending_count == 1

    def test_pop_ready_returns_events_in_ascending_order(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(virtual_round=3))
        loop.push(_make_event(virtual_round=1))
        loop.push(_make_event(virtual_round=2))
        ready = loop.pop_ready(current_round=5)
        rounds = [e.virtual_round for e in ready]
        assert rounds == [1, 2, 3]

    def test_pop_ready_boundary_at_zero(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(virtual_round=0))
        ready = loop.pop_ready(current_round=0)
        assert len(ready) == 1

    def test_multiple_pops_are_idempotent_on_empty(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(virtual_round=1))
        loop.pop_ready(1)
        assert loop.pop_ready(1) == []
        assert loop.pending_count == 0

    def test_events_at_same_round_all_returned(self) -> None:
        loop = FLEventLoop()
        for cid in range(5):
            loop.push(_make_event(virtual_round=2, client_id=cid))
        ready = loop.pop_ready(current_round=2)
        assert len(ready) == 5

    def test_interleaved_push_and_pop(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(1))
        loop.push(_make_event(4))
        loop.pop_ready(2)         # drains round-1 event
        loop.push(_make_event(3))
        ready = loop.pop_ready(3)  # should get rounds 3 (round-4 stays)
        assert len(ready) == 1
        assert ready[0].virtual_round == 3
        assert loop.pending_count == 1


# ===========================================================================
# FLEventLoop — discard_stale
# ===========================================================================


class TestFLEventLoopDiscardStale:
    def test_discard_removes_old_events(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(virtual_round=1))
        loop.push(_make_event(virtual_round=2))
        loop.push(_make_event(virtual_round=5))
        removed = loop.discard_stale(min_virtual_round=3)
        assert removed == 2
        assert loop.pending_count == 1

    def test_discard_keeps_events_at_min_round(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(virtual_round=3))
        removed = loop.discard_stale(min_virtual_round=3)
        assert removed == 0
        assert loop.pending_count == 1

    def test_discard_all(self) -> None:
        loop = FLEventLoop()
        for r in range(5):
            loop.push(_make_event(virtual_round=r))
        removed = loop.discard_stale(min_virtual_round=10)
        assert removed == 5
        assert loop.pending_count == 0

    def test_discard_none_when_all_fresh(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(virtual_round=5))
        loop.push(_make_event(virtual_round=6))
        removed = loop.discard_stale(min_virtual_round=3)
        assert removed == 0
        assert loop.pending_count == 2

    def test_discard_returns_correct_count(self) -> None:
        loop = FLEventLoop()
        for r in [1, 1, 2, 3, 5]:
            loop.push(_make_event(virtual_round=r))
        removed = loop.discard_stale(min_virtual_round=3)
        assert removed == 3   # rounds 1, 1, 2 are all < 3 → 3 events discarded


# ===========================================================================
# AsyncRoundSummary
# ===========================================================================


class TestAsyncRoundSummary:
    def _make(self, **kwargs) -> AsyncRoundSummary:  # type: ignore[return]
        defaults = dict(
            round_num=1,
            selected_clients=[0, 1],
            active_clients=[0],
            dropped_clients=[1],
            aggregation=None,
            global_accuracy=0.85,
            global_loss=0.3,
            elapsed_seconds=1.2,
            skipped=False,
        )
        defaults.update(kwargs)
        return AsyncRoundSummary(**defaults)

    def test_inherits_round_summary_fields(self) -> None:
        s = self._make()
        assert s.round_num == 1
        assert s.global_accuracy == pytest.approx(0.85)
        assert s.skipped is False

    def test_async_fields_default_zero(self) -> None:
        s = self._make()
        assert s.stale_updates_used == 0
        assert s.stale_updates_discarded == 0
        assert s.pending_updates == 0

    def test_async_fields_settable(self) -> None:
        s = self._make(stale_updates_used=2, stale_updates_discarded=1, pending_updates=4)
        assert s.stale_updates_used == 2
        assert s.stale_updates_discarded == 1
        assert s.pending_updates == 4

    def test_is_subclass_of_round_summary(self) -> None:
        s = self._make()
        assert isinstance(s, RoundSummary)

    def test_skipped_summary_has_async_fields(self) -> None:
        s = self._make(skipped=True, stale_updates_discarded=3, pending_updates=5)
        assert s.skipped is True
        assert s.stale_updates_discarded == 3
        assert s.pending_updates == 5


# ===========================================================================
# AsyncConfig
# ===========================================================================


class TestAsyncConfig:
    def test_defaults(self) -> None:
        cfg = AsyncConfig()
        assert cfg.enabled is False
        assert cfg.delay_min == 0.0
        assert cfg.delay_max == pytest.approx(3.0)
        assert cfg.staleness_threshold == 3

    def test_enabled_true(self) -> None:
        cfg = AsyncConfig(enabled=True)
        assert cfg.enabled is True

    def test_invalid_delay_range_raises(self) -> None:
        with pytest.raises(Exception):
            AsyncConfig(enabled=True, delay_min=5.0, delay_max=1.0)

    def test_equal_delay_min_max_valid(self) -> None:
        # delay_min == delay_max is fine (all clients get the same delay)
        cfg = AsyncConfig(enabled=True, delay_min=2.0, delay_max=2.0)
        assert cfg.delay_min == cfg.delay_max

    def test_staleness_threshold_zero_valid(self) -> None:
        cfg = AsyncConfig(staleness_threshold=0)
        assert cfg.staleness_threshold == 0

    def test_negative_staleness_threshold_raises(self) -> None:
        with pytest.raises(Exception):
            AsyncConfig(staleness_threshold=-1)

    def test_delay_range_check_only_when_enabled(self) -> None:
        # When disabled, invalid range should not raise
        cfg = AsyncConfig(enabled=False, delay_min=10.0, delay_max=1.0)
        assert cfg.enabled is False


# ===========================================================================
# FLEventLoop — determinism
# ===========================================================================


class TestFLEventLoopDeterminism:
    def test_same_push_order_same_pop_order(self) -> None:
        def build_and_pop() -> list[int]:
            loop = FLEventLoop()
            for r in [3, 1, 4, 1, 5, 2]:
                loop.push(_make_event(virtual_round=r))
            return [e.virtual_round for e in loop.pop_ready(10)]

        assert build_and_pop() == build_and_pop()

    def test_pop_then_push_then_pop(self) -> None:
        loop = FLEventLoop()
        loop.push(_make_event(1))
        loop.push(_make_event(2))
        first = loop.pop_ready(1)
        assert len(first) == 1
        loop.push(_make_event(2))
        second = loop.pop_ready(2)
        # Should contain the original round-2 event AND the newly pushed one
        assert len(second) == 2
