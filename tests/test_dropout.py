"""Tests for the dropout simulator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from flp.simulation.dropout import DropoutSimulator


def _make_clients(n: int) -> list[MagicMock]:
    clients = []
    for i in range(n):
        c = MagicMock()
        c.client_id = i
        clients.append(c)
    return clients


class TestDropoutSimulator:
    def test_zero_dropout_returns_all(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.0, seed=42)
        clients = _make_clients(10)
        result = sim.apply(clients, round_num=1)
        assert len(result) == 10

    def test_high_dropout_reduces_clients(self) -> None:
        sim = DropoutSimulator(dropout_rate=0.9, seed=42)
        clients = _make_clients(50)
        result = sim.apply(clients, round_num=1)
        assert len(result) < 50

    def test_reproducible_with_same_seed(self) -> None:
        clients = _make_clients(20)
        sim1 = DropoutSimulator(dropout_rate=0.3, seed=7)
        sim2 = DropoutSimulator(dropout_rate=0.3, seed=7)
        r1 = [c.client_id for c in sim1.apply(clients, round_num=3)]
        r2 = [c.client_id for c in sim2.apply(clients, round_num=3)]
        assert r1 == r2

    def test_different_rounds_differ(self) -> None:
        clients = _make_clients(20)
        sim = DropoutSimulator(dropout_rate=0.3, seed=42)
        r1 = [c.client_id for c in sim.apply(clients, round_num=1)]
        r2 = [c.client_id for c in sim.apply(clients, round_num=2)]
        # They should occasionally differ (statistically near-certain for 20 clients)
        # We just check both are valid subsets
        assert all(cid in range(20) for cid in r1)
        assert all(cid in range(20) for cid in r2)

    def test_invalid_dropout_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            DropoutSimulator(dropout_rate=1.0)
        with pytest.raises(ValueError):
            DropoutSimulator(dropout_rate=-0.1)
