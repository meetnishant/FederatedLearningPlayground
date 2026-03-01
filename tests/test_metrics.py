"""Unit tests for metrics/tracker.py and metrics/communication.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from flp.metrics.communication import (
    CommRoundRecord,
    CommunicationTracker,
    count_buffers,
    count_parameters,
    model_size_bytes,
)
from flp.metrics.tracker import ClientRoundMetrics, MetricsTracker, RoundMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_linear() -> nn.Module:
    """Single linear layer: 10 params (2×5) + 2 bias = 12 trainable scalars, no buffers."""
    return nn.Linear(5, 2)


def _tiny_bn() -> nn.Module:
    """Linear + BatchNorm1d so we get both params and buffers."""
    return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))


def _make_updates(n: int, loss: float = 0.5, samples: int = 100) -> list[dict[str, object]]:
    return [{"client_id": i, "loss": loss, "num_samples": samples} for i in range(n)]


def _record_round(
    tracker: MetricsTracker,
    round_num: int = 1,
    global_accuracy: float = 0.8,
    global_loss: float = 0.4,
    per_client_accuracy: dict[int, float] | None = None,
    num_active_clients: int = 3,
    client_updates: list[dict[str, object]] | None = None,
) -> None:
    if per_client_accuracy is None:
        per_client_accuracy = {0: 0.75, 1: 0.80, 2: 0.85}
    if client_updates is None:
        client_updates = _make_updates(num_active_clients)
    tracker.record_round(
        round_num=round_num,
        global_accuracy=global_accuracy,
        global_loss=global_loss,
        per_client_accuracy=per_client_accuracy,
        num_active_clients=num_active_clients,
        client_updates=client_updates,
    )


# ---------------------------------------------------------------------------
# MetricsTracker — empty state
# ---------------------------------------------------------------------------


class TestMetricsTrackerEmpty:
    def test_rounds_empty(self) -> None:
        t = MetricsTracker()
        assert t.rounds == []

    def test_global_accuracies_empty(self) -> None:
        assert MetricsTracker().global_accuracies == []

    def test_global_losses_empty(self) -> None:
        assert MetricsTracker().global_losses == []

    def test_active_client_counts_empty(self) -> None:
        assert MetricsTracker().active_client_counts == []

    def test_per_client_accuracies_empty(self) -> None:
        assert MetricsTracker().per_client_accuracies == {}

    def test_best_accuracy_zero(self) -> None:
        assert MetricsTracker().best_accuracy() == 0.0

    def test_best_round_none(self) -> None:
        assert MetricsTracker().best_round() is None

    def test_accuracy_improvement_zero(self) -> None:
        assert MetricsTracker().accuracy_improvement() == 0.0

    def test_summary_defaults(self) -> None:
        s = MetricsTracker().summary()
        assert s["num_rounds"] == 0
        assert s["best_accuracy"] == 0.0
        assert s["best_round"] is None
        assert s["final_accuracy"] == 0.0
        assert s["final_loss"] == 0.0
        assert s["accuracy_improvement"] == 0.0
        assert s["avg_active_clients"] == 0.0


# ---------------------------------------------------------------------------
# MetricsTracker — recording
# ---------------------------------------------------------------------------


class TestMetricsTrackerRecording:
    def test_record_adds_one_round(self) -> None:
        t = MetricsTracker()
        _record_round(t, round_num=1)
        assert len(t.rounds) == 1

    def test_round_num_stored(self) -> None:
        t = MetricsTracker()
        _record_round(t, round_num=7)
        assert t.rounds[0].round_num == 7

    def test_global_accuracy_stored(self) -> None:
        t = MetricsTracker()
        _record_round(t, global_accuracy=0.92)
        assert t.rounds[0].global_accuracy == pytest.approx(0.92)

    def test_global_loss_stored(self) -> None:
        t = MetricsTracker()
        _record_round(t, global_loss=0.33)
        assert t.rounds[0].global_loss == pytest.approx(0.33)

    def test_num_active_clients_stored(self) -> None:
        t = MetricsTracker()
        _record_round(t, num_active_clients=5, per_client_accuracy={i: 0.8 for i in range(5)},
                      client_updates=_make_updates(5))
        assert t.rounds[0].num_active_clients == 5

    def test_multiple_rounds_preserved_in_order(self) -> None:
        t = MetricsTracker()
        accs = [0.5, 0.65, 0.78]
        for i, acc in enumerate(accs):
            _record_round(t, round_num=i + 1, global_accuracy=acc)
        assert [r.global_accuracy for r in t.rounds] == pytest.approx(accs)

    def test_per_client_accuracy_stored(self) -> None:
        t = MetricsTracker()
        pca = {0: 0.70, 1: 0.85}
        _record_round(t, per_client_accuracy=pca, num_active_clients=2,
                      client_updates=_make_updates(2))
        assert t.rounds[0].per_client_accuracy == pca

    def test_client_records_built(self) -> None:
        t = MetricsTracker()
        updates = [{"client_id": 0, "loss": 0.4, "num_samples": 120}]
        _record_round(t, per_client_accuracy={0: 0.77}, num_active_clients=1,
                      client_updates=updates)
        assert len(t.rounds[0].client_records) == 1
        rec: ClientRoundMetrics = t.rounds[0].client_records[0]
        assert rec.client_id == 0
        assert rec.num_samples == 120
        assert rec.loss == pytest.approx(0.4)

    def test_weighted_loss_single_client(self) -> None:
        t = MetricsTracker()
        updates = [{"client_id": 0, "loss": 0.6, "num_samples": 200}]
        _record_round(t, per_client_accuracy={0: 0.8}, num_active_clients=1,
                      client_updates=updates)
        assert t.rounds[0].avg_client_loss == pytest.approx(0.6)

    def test_weighted_loss_two_clients(self) -> None:
        t = MetricsTracker()
        updates = [
            {"client_id": 0, "loss": 0.4, "num_samples": 100},
            {"client_id": 1, "loss": 0.8, "num_samples": 100},
        ]
        _record_round(t, per_client_accuracy={0: 0.8, 1: 0.7}, num_active_clients=2,
                      client_updates=updates)
        assert t.rounds[0].avg_client_loss == pytest.approx(0.6)

    def test_weighted_loss_unequal_samples(self) -> None:
        t = MetricsTracker()
        updates = [
            {"client_id": 0, "loss": 1.0, "num_samples": 100},
            {"client_id": 1, "loss": 0.0, "num_samples": 300},
        ]
        _record_round(t, per_client_accuracy={0: 0.5, 1: 0.9}, num_active_clients=2,
                      client_updates=updates)
        # (1.0*100 + 0.0*300) / 400 = 0.25
        assert t.rounds[0].avg_client_loss == pytest.approx(0.25)

    def test_min_max_std_accuracy(self) -> None:
        t = MetricsTracker()
        pca = {0: 0.6, 1: 0.8, 2: 0.9}
        _record_round(t, per_client_accuracy=pca, num_active_clients=3,
                      client_updates=_make_updates(3))
        r = t.rounds[0]
        assert r.min_client_accuracy == pytest.approx(0.6)
        assert r.max_client_accuracy == pytest.approx(0.9)
        assert r.std_client_accuracy > 0.0

    def test_empty_per_client_accuracy(self) -> None:
        t = MetricsTracker()
        _record_round(t, per_client_accuracy={}, num_active_clients=0, client_updates=[])
        r = t.rounds[0]
        assert r.min_client_accuracy == 0.0
        assert r.max_client_accuracy == 0.0
        assert r.std_client_accuracy == 0.0

    def test_total_samples_summed(self) -> None:
        t = MetricsTracker()
        updates = [
            {"client_id": 0, "loss": 0.5, "num_samples": 80},
            {"client_id": 1, "loss": 0.5, "num_samples": 120},
        ]
        _record_round(t, per_client_accuracy={0: 0.8, 1: 0.7}, num_active_clients=2,
                      client_updates=updates)
        assert t.rounds[0].total_samples == 200

    def test_weighted_client_loss_alias(self) -> None:
        t = MetricsTracker()
        _record_round(t)
        r = t.rounds[0]
        assert r.avg_client_loss == r.weighted_client_loss


# ---------------------------------------------------------------------------
# MetricsTracker — accessors
# ---------------------------------------------------------------------------


class TestMetricsTrackerAccessors:
    def setup_method(self) -> None:
        self.t = MetricsTracker()
        for i, acc in enumerate([0.5, 0.7, 0.9]):
            _record_round(self.t, round_num=i + 1, global_accuracy=acc,
                          global_loss=0.5 - i * 0.1)

    def test_global_accuracies(self) -> None:
        assert self.t.global_accuracies == pytest.approx([0.5, 0.7, 0.9])

    def test_global_losses(self) -> None:
        assert self.t.global_losses == pytest.approx([0.5, 0.4, 0.3])

    def test_active_client_counts(self) -> None:
        assert self.t.active_client_counts == [3, 3, 3]

    def test_per_client_accuracies_keys(self) -> None:
        pca = self.t.per_client_accuracies
        assert set(pca.keys()) == {0, 1, 2}

    def test_per_client_accuracies_length(self) -> None:
        pca = self.t.per_client_accuracies
        # Each client recorded in all 3 rounds
        for vals in pca.values():
            assert len(vals) == 3

    def test_rounds_returns_copy(self) -> None:
        rounds = self.t.rounds
        rounds.clear()
        assert len(self.t.rounds) == 3

    def test_best_accuracy(self) -> None:
        assert self.t.best_accuracy() == pytest.approx(0.9)

    def test_best_round(self) -> None:
        assert self.t.best_round() == 3

    def test_accuracy_improvement(self) -> None:
        assert self.t.accuracy_improvement() == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# MetricsTracker — summary
# ---------------------------------------------------------------------------


class TestMetricsTrackerSummary:
    def test_summary_keys_present(self) -> None:
        t = MetricsTracker()
        _record_round(t)
        s = t.summary()
        for key in [
            "num_rounds", "best_accuracy", "best_round",
            "final_accuracy", "final_loss", "accuracy_improvement",
            "avg_active_clients",
        ]:
            assert key in s, f"Missing key: {key}"

    def test_num_rounds_correct(self) -> None:
        t = MetricsTracker()
        for i in range(5):
            _record_round(t, round_num=i + 1)
        assert t.summary()["num_rounds"] == 5

    def test_final_accuracy(self) -> None:
        t = MetricsTracker()
        for i, acc in enumerate([0.5, 0.6, 0.88]):
            _record_round(t, round_num=i + 1, global_accuracy=acc)
        assert t.summary()["final_accuracy"] == pytest.approx(0.88)

    def test_final_loss(self) -> None:
        t = MetricsTracker()
        _record_round(t, global_loss=0.123)
        assert t.summary()["final_loss"] == pytest.approx(0.123)

    def test_avg_active_clients(self) -> None:
        t = MetricsTracker()
        for n in [2, 4, 6]:
            _record_round(t, num_active_clients=n,
                          per_client_accuracy={i: 0.8 for i in range(n)},
                          client_updates=_make_updates(n))
        assert t.summary()["avg_active_clients"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# MetricsTracker — persistence
# ---------------------------------------------------------------------------


class TestMetricsTrackerPersistence:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        t = MetricsTracker()
        _record_round(t)
        p = tmp_path / "m.json"
        t.save(p)
        assert p.exists()

    def test_save_valid_json(self, tmp_path: Path) -> None:
        t = MetricsTracker()
        _record_round(t)
        p = tmp_path / "m.json"
        t.save(p)
        data = json.loads(p.read_text())
        assert "summary" in data
        assert "rounds" in data

    def test_save_round_count(self, tmp_path: Path) -> None:
        t = MetricsTracker()
        for i in range(3):
            _record_round(t, round_num=i + 1)
        p = tmp_path / "m.json"
        t.save(p)
        data = json.loads(p.read_text())
        assert len(data["rounds"]) == 3

    def test_save_includes_clients(self, tmp_path: Path) -> None:
        t = MetricsTracker()
        _record_round(t, num_active_clients=2,
                      per_client_accuracy={0: 0.7, 1: 0.8},
                      client_updates=_make_updates(2))
        p = tmp_path / "m.json"
        t.save(p)
        data = json.loads(p.read_text())
        assert "clients" in data["rounds"][0]
        assert len(data["rounds"][0]["clients"]) == 2

    def test_load_roundtrip(self, tmp_path: Path) -> None:
        t = MetricsTracker()
        for i in range(4):
            _record_round(t, round_num=i + 1, global_accuracy=0.5 + i * 0.1)
        p = tmp_path / "m.json"
        t.save(p)
        t2 = MetricsTracker.load(p)
        assert len(t2.rounds) == 4
        assert t2.global_accuracies == pytest.approx(t.global_accuracies)

    def test_load_restores_global_loss(self, tmp_path: Path) -> None:
        t = MetricsTracker()
        _record_round(t, global_loss=0.312)
        p = tmp_path / "m.json"
        t.save(p)
        t2 = MetricsTracker.load(p)
        assert t2.global_losses[0] == pytest.approx(0.312)

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        t = MetricsTracker()
        _record_round(t)
        p = tmp_path / "nested" / "deep" / "m.json"
        t.save(p)
        assert p.exists()


# ---------------------------------------------------------------------------
# Model size utilities
# ---------------------------------------------------------------------------


class TestModelSizeUtils:
    def test_count_parameters_linear(self) -> None:
        model = nn.Linear(5, 2)
        # weight: 5*2=10, bias: 2 → total 12
        assert count_parameters(model) == 12

    def test_count_parameters_frozen_excluded(self) -> None:
        model = nn.Linear(4, 4)
        for p in model.parameters():
            p.requires_grad = False
        assert count_parameters(model) == 0

    def test_count_buffers_no_buffers(self) -> None:
        model = _tiny_linear()
        assert count_buffers(model) == 0

    def test_count_buffers_batchnorm(self) -> None:
        bn = nn.BatchNorm1d(8)
        # running_mean(8) + running_var(8) + num_batches_tracked(1) = 17
        assert count_buffers(bn) == 17

    def test_model_size_bytes_float32(self) -> None:
        model = nn.Linear(5, 2, bias=False)  # 10 params, no buffers
        size = model_size_bytes(model, dtype=torch.float32, include_buffers=False)
        assert size == 10 * 4

    def test_model_size_bytes_float16(self) -> None:
        model = nn.Linear(5, 2, bias=False)
        size = model_size_bytes(model, dtype=torch.float16, include_buffers=False)
        assert size == 10 * 2

    def test_model_size_bytes_with_buffers(self) -> None:
        bn = nn.BatchNorm1d(4)  # 8 params (weight+bias), 9 buffers
        size_with = model_size_bytes(bn, dtype=torch.float32, include_buffers=True)
        size_without = model_size_bytes(bn, dtype=torch.float32, include_buffers=False)
        assert size_with > size_without

    def test_model_size_positive(self) -> None:
        model = _tiny_linear()
        assert model_size_bytes(model) > 0


# ---------------------------------------------------------------------------
# CommRoundRecord
# ---------------------------------------------------------------------------


class TestCommRoundRecord:
    def _make_record(
        self,
        upload: int = 1000,
        download: int = 2000,
        round_num: int = 1,
        up_clients: int = 5,
        down_clients: int = 10,
    ) -> CommRoundRecord:
        return CommRoundRecord(
            round_num=round_num,
            num_clients_upload=up_clients,
            num_clients_download=down_clients,
            upload_bytes=upload,
            download_bytes=download,
        )

    def test_total_bytes(self) -> None:
        r = self._make_record(upload=1000, download=2000)
        assert r.total_bytes == 3000

    def test_upload_mb(self) -> None:
        r = self._make_record(upload=1024 ** 2)
        assert r.upload_mb == pytest.approx(1.0)

    def test_download_mb(self) -> None:
        r = self._make_record(download=2 * 1024 ** 2)
        assert r.download_mb == pytest.approx(2.0)

    def test_total_mb(self) -> None:
        r = self._make_record(upload=1024 ** 2, download=1024 ** 2)
        assert r.total_mb == pytest.approx(2.0)

    def test_frozen(self) -> None:
        r = self._make_record()
        with pytest.raises((AttributeError, TypeError)):
            r.upload_bytes = 0  # type: ignore[misc]

    def test_round_num_stored(self) -> None:
        r = self._make_record(round_num=5)
        assert r.round_num == 5

    def test_client_counts_stored(self) -> None:
        r = self._make_record(up_clients=3, down_clients=7)
        assert r.num_clients_upload == 3
        assert r.num_clients_download == 7


# ---------------------------------------------------------------------------
# CommunicationTracker — initialisation
# ---------------------------------------------------------------------------


class TestCommunicationTrackerInit:
    def test_bytes_per_model_positive(self) -> None:
        tracker = CommunicationTracker(_tiny_linear())
        assert tracker.bytes_per_model > 0

    def test_records_empty_on_init(self) -> None:
        tracker = CommunicationTracker(_tiny_linear())
        assert tracker.records == []

    def test_totals_zero_on_init(self) -> None:
        tracker = CommunicationTracker(_tiny_linear())
        assert tracker.total_bytes == 0
        assert tracker.total_upload_bytes == 0
        assert tracker.total_download_bytes == 0

    def test_bytes_per_model_float16_smaller(self) -> None:
        model = _tiny_linear()
        t32 = CommunicationTracker(model, dtype=torch.float32)
        t16 = CommunicationTracker(model, dtype=torch.float16)
        assert t16.bytes_per_model < t32.bytes_per_model

    def test_bytes_per_model_matches_utility(self) -> None:
        model = _tiny_linear()
        tracker = CommunicationTracker(model)
        assert tracker.bytes_per_model == model_size_bytes(model)


# ---------------------------------------------------------------------------
# CommunicationTracker — recording
# ---------------------------------------------------------------------------


class TestCommunicationTrackerRecording:
    def setup_method(self) -> None:
        self.model = _tiny_linear()
        self.tracker = CommunicationTracker(self.model)
        self.bpm = self.tracker.bytes_per_model

    def test_record_returns_comm_round_record(self) -> None:
        r = self.tracker.record_round(round_num=1, num_clients_upload=2, num_clients_download=5)
        assert isinstance(r, CommRoundRecord)

    def test_record_adds_to_records(self) -> None:
        self.tracker.record_round(round_num=1, num_clients_upload=2, num_clients_download=5)
        assert len(self.tracker.records) == 1

    def test_upload_bytes_correct(self) -> None:
        r = self.tracker.record_round(round_num=1, num_clients_upload=3, num_clients_download=5)
        assert r.upload_bytes == 3 * self.bpm

    def test_download_bytes_correct(self) -> None:
        r = self.tracker.record_round(round_num=1, num_clients_upload=3, num_clients_download=10)
        assert r.download_bytes == 10 * self.bpm

    def test_zero_upload_clients(self) -> None:
        r = self.tracker.record_round(round_num=1, num_clients_upload=0, num_clients_download=5)
        assert r.upload_bytes == 0
        assert r.download_bytes == 5 * self.bpm

    def test_total_upload_bytes_accumulates(self) -> None:
        self.tracker.record_round(round_num=1, num_clients_upload=2, num_clients_download=5)
        self.tracker.record_round(round_num=2, num_clients_upload=3, num_clients_download=5)
        assert self.tracker.total_upload_bytes == 5 * self.bpm

    def test_total_download_bytes_accumulates(self) -> None:
        self.tracker.record_round(round_num=1, num_clients_upload=2, num_clients_download=5)
        self.tracker.record_round(round_num=2, num_clients_upload=2, num_clients_download=3)
        assert self.tracker.total_download_bytes == 8 * self.bpm

    def test_total_bytes_is_sum(self) -> None:
        self.tracker.record_round(round_num=1, num_clients_upload=2, num_clients_download=4)
        up = self.tracker.total_upload_bytes
        down = self.tracker.total_download_bytes
        assert self.tracker.total_bytes == up + down

    def test_bytes_per_round(self) -> None:
        self.tracker.record_round(round_num=1, num_clients_upload=1, num_clients_download=1)
        self.tracker.record_round(round_num=2, num_clients_upload=2, num_clients_download=2)
        bpr = self.tracker.bytes_per_round
        assert len(bpr) == 2
        assert bpr[1] == 2 * bpr[0]

    def test_cumulative_bytes_monotone(self) -> None:
        for rn in range(1, 6):
            self.tracker.record_round(round_num=rn, num_clients_upload=1, num_clients_download=1)
        cb = self.tracker.cumulative_bytes
        assert all(cb[i] <= cb[i + 1] for i in range(len(cb) - 1))

    def test_cumulative_bytes_last_equals_total(self) -> None:
        for rn in range(1, 4):
            self.tracker.record_round(round_num=rn, num_clients_upload=2, num_clients_download=3)
        assert self.tracker.cumulative_bytes[-1] == self.tracker.total_bytes

    def test_records_returns_copy(self) -> None:
        self.tracker.record_round(round_num=1, num_clients_upload=1, num_clients_download=1)
        records = self.tracker.records
        records.clear()
        assert len(self.tracker.records) == 1


# ---------------------------------------------------------------------------
# CommunicationTracker — summary
# ---------------------------------------------------------------------------


class TestCommunicationTrackerSummary:
    def setup_method(self) -> None:
        self.tracker = CommunicationTracker(_tiny_linear())
        self.tracker.record_round(round_num=1, num_clients_upload=3, num_clients_download=5)

    def test_summary_keys(self) -> None:
        s = self.tracker.summary()
        for key in [
            "num_rounds", "bytes_per_model",
            "total_upload_bytes", "total_download_bytes", "total_bytes",
            "total_upload_mb", "total_download_mb", "total_mb",
        ]:
            assert key in s, f"Missing key: {key}"

    def test_summary_num_rounds(self) -> None:
        assert self.tracker.summary()["num_rounds"] == 1

    def test_summary_bytes_per_model(self) -> None:
        assert self.tracker.summary()["bytes_per_model"] == self.tracker.bytes_per_model

    def test_summary_total_mb_is_float(self) -> None:
        assert isinstance(self.tracker.summary()["total_mb"], float)

    def test_empty_tracker_summary(self) -> None:
        t = CommunicationTracker(_tiny_linear())
        s = t.summary()
        assert s["num_rounds"] == 0
        assert s["total_bytes"] == 0


# ---------------------------------------------------------------------------
# CommunicationTracker — persistence
# ---------------------------------------------------------------------------


class TestCommunicationTrackerPersistence:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        tracker = CommunicationTracker(_tiny_linear())
        tracker.record_round(round_num=1, num_clients_upload=2, num_clients_download=4)
        p = tmp_path / "comm.json"
        tracker.save(p)
        assert p.exists()

    def test_save_valid_json(self, tmp_path: Path) -> None:
        tracker = CommunicationTracker(_tiny_linear())
        tracker.record_round(round_num=1, num_clients_upload=2, num_clients_download=4)
        p = tmp_path / "comm.json"
        tracker.save(p)
        data = json.loads(p.read_text())
        assert "summary" in data
        assert "rounds" in data

    def test_save_round_count(self, tmp_path: Path) -> None:
        tracker = CommunicationTracker(_tiny_linear())
        for rn in range(1, 4):
            tracker.record_round(round_num=rn, num_clients_upload=1, num_clients_download=3)
        p = tmp_path / "comm.json"
        tracker.save(p)
        data = json.loads(p.read_text())
        assert len(data["rounds"]) == 3

    def test_save_round_fields(self, tmp_path: Path) -> None:
        tracker = CommunicationTracker(_tiny_linear())
        tracker.record_round(round_num=1, num_clients_upload=2, num_clients_download=5)
        p = tmp_path / "comm.json"
        tracker.save(p)
        data = json.loads(p.read_text())
        r = data["rounds"][0]
        for field in ["round_num", "num_clients_upload", "num_clients_download",
                      "upload_bytes", "download_bytes", "total_bytes",
                      "upload_mb", "download_mb"]:
            assert field in r, f"Missing field: {field}"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        tracker = CommunicationTracker(_tiny_linear())
        tracker.record_round(round_num=1, num_clients_upload=1, num_clients_download=1)
        p = tmp_path / "deep" / "nested" / "comm.json"
        tracker.save(p)
        assert p.exists()
