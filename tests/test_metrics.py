"""Tests for the MetricsTracker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flp.metrics.tracker import MetricsTracker


def _sample_updates(n: int = 3) -> list[dict[str, object]]:
    return [{"client_id": i, "loss": 0.5, "num_samples": 100} for i in range(n)]


class TestMetricsTracker:
    def setup_method(self) -> None:
        self.tracker = MetricsTracker()

    def test_empty_summary(self) -> None:
        s = self.tracker.summary()
        assert s["num_rounds"] == 0
        assert s["best_accuracy"] == 0.0

    def test_record_and_retrieve(self) -> None:
        self.tracker.record_round(
            round_num=1,
            global_accuracy=0.80,
            global_loss=0.5,
            per_client_accuracy={0: 0.78, 1: 0.82},
            num_active_clients=2,
            client_updates=_sample_updates(2),
        )
        assert len(self.tracker.rounds) == 1
        assert self.tracker.rounds[0].global_accuracy == pytest.approx(0.80)

    def test_best_accuracy(self) -> None:
        for acc in [0.5, 0.9, 0.7]:
            self.tracker.record_round(
                round_num=1,
                global_accuracy=acc,
                global_loss=0.3,
                per_client_accuracy={},
                num_active_clients=5,
                client_updates=_sample_updates(),
            )
        assert self.tracker.best_accuracy() == pytest.approx(0.9)

    def test_save_creates_json(self, tmp_path: Path) -> None:
        self.tracker.record_round(
            round_num=1,
            global_accuracy=0.75,
            global_loss=0.4,
            per_client_accuracy={0: 0.75},
            num_active_clients=1,
            client_updates=_sample_updates(1),
        )
        out = tmp_path / "metrics.json"
        self.tracker.save(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "summary" in data
        assert "rounds" in data
        assert data["summary"]["num_rounds"] == 1

    def test_global_accuracies_list(self) -> None:
        for i, acc in enumerate([0.6, 0.7, 0.8]):
            self.tracker.record_round(
                round_num=i + 1,
                global_accuracy=acc,
                global_loss=0.3,
                per_client_accuracy={},
                num_active_clients=3,
                client_updates=_sample_updates(),
            )
        assert self.tracker.global_accuracies == pytest.approx([0.6, 0.7, 0.8])
