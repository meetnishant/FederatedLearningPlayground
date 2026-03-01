"""Tests for the governance module: hashing, audit log, and replay manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from flp.governance.audit import AuditEvent, AuditLog
from flp.governance.hashing import hash_config, hash_state_dict
from flp.governance.replay import ReplayManifest, RoundLineageRecord, _capture_environment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _simple_state_dict() -> dict[str, torch.Tensor]:
    """Small state dict with float and integer tensors."""
    return {
        "weight": torch.tensor([1.0, 2.0, 3.0]),
        "bias": torch.tensor([0.5]),
        "num_batches_tracked": torch.tensor(5, dtype=torch.long),
    }


def _make_model_state(value: float = 1.0) -> dict[str, torch.Tensor]:
    model = nn.Linear(4, 2)
    nn.init.constant_(model.weight, value)
    nn.init.constant_(model.bias, value)
    return model.state_dict()


def _make_audit_event(
    round_num: int = 1,
    skipped: bool = False,
) -> AuditEvent:
    return AuditEvent(
        round_num=round_num,
        timestamp_utc="2026-01-01T00:00:00+00:00",
        selected_clients=[0, 1, 2],
        active_clients=[] if skipped else [0, 1],
        dropped_clients=[2] if skipped else [],
        pre_round_model_hash="sha256:aaa",
        post_round_model_hash="sha256:aaa" if skipped else "sha256:bbb",
        global_accuracy=0.0 if skipped else 0.85,
        global_loss=0.0 if skipped else 0.42,
        num_clients_clipped=0,
        dp_epsilon_spent=0.0,
        dp_delta_spent=0.0,
        elapsed_seconds=1.5,
        skipped=skipped,
    )


# ===========================================================================
# hashing.py
# ===========================================================================


class TestHashStateDict:
    def test_returns_sha256_prefix(self) -> None:
        sd = _simple_state_dict()
        h = hash_state_dict(sd)
        assert h.startswith("sha256:")

    def test_deterministic_same_dict(self) -> None:
        sd = _simple_state_dict()
        assert hash_state_dict(sd) == hash_state_dict(sd)

    def test_deterministic_across_calls(self) -> None:
        sd1 = _simple_state_dict()
        sd2 = _simple_state_dict()
        assert hash_state_dict(sd1) == hash_state_dict(sd2)

    def test_different_values_different_hash(self) -> None:
        sd1 = {"w": torch.tensor([1.0])}
        sd2 = {"w": torch.tensor([2.0])}
        assert hash_state_dict(sd1) != hash_state_dict(sd2)

    def test_key_order_independent(self) -> None:
        sd1 = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        sd2 = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
        assert hash_state_dict(sd1) == hash_state_dict(sd2)

    def test_integer_tensor_included(self) -> None:
        sd1 = {"n": torch.tensor(1, dtype=torch.long)}
        sd2 = {"n": torch.tensor(2, dtype=torch.long)}
        assert hash_state_dict(sd1) != hash_state_dict(sd2)

    def test_hex_digest_length(self) -> None:
        h = hash_state_dict(_simple_state_dict())
        # "sha256:" (7 chars) + 64 hex chars
        assert len(h) == 71

    def test_actual_model_state(self) -> None:
        sd1 = _make_model_state(1.0)
        sd2 = _make_model_state(2.0)
        assert hash_state_dict(sd1) != hash_state_dict(sd2)
        assert hash_state_dict(sd1) == hash_state_dict(sd1)


class TestHashConfig:
    def test_returns_sha256_prefix(self) -> None:
        h = hash_config({"key": "value"})
        assert h.startswith("sha256:")

    def test_deterministic(self) -> None:
        cfg = {"seed": 42, "rounds": 10}
        assert hash_config(cfg) == hash_config(cfg)

    def test_key_order_independent(self) -> None:
        cfg1 = {"a": 1, "b": 2}
        cfg2 = {"b": 2, "a": 1}
        assert hash_config(cfg1) == hash_config(cfg2)

    def test_different_configs_different_hash(self) -> None:
        assert hash_config({"seed": 42}) != hash_config({"seed": 99})

    def test_nested_dict(self) -> None:
        cfg = {"training": {"num_rounds": 10, "num_clients": 5}}
        h = hash_config(cfg)
        assert h.startswith("sha256:")
        assert len(h) == 71

    def test_empty_dict(self) -> None:
        h = hash_config({})
        assert h.startswith("sha256:")

    def test_value_change_changes_hash(self) -> None:
        cfg1 = {"lr": 0.01}
        cfg2 = {"lr": 0.001}
        assert hash_config(cfg1) != hash_config(cfg2)


# ===========================================================================
# audit.py — AuditEvent
# ===========================================================================


class TestAuditEvent:
    def test_fields_accessible(self) -> None:
        event = _make_audit_event(round_num=3)
        assert event.round_num == 3
        assert event.global_accuracy == 0.85
        assert event.skipped is False

    def test_skipped_event(self) -> None:
        event = _make_audit_event(skipped=True)
        assert event.skipped is True
        assert event.active_clients == []
        assert event.pre_round_model_hash == event.post_round_model_hash

    def test_is_dataclass(self) -> None:
        from dataclasses import fields
        event = _make_audit_event()
        field_names = {f.name for f in fields(event)}
        assert "round_num" in field_names
        assert "pre_round_model_hash" in field_names
        assert "post_round_model_hash" in field_names


# ===========================================================================
# audit.py — AuditLog
# ===========================================================================


class TestAuditLog:
    def test_empty_on_init(self) -> None:
        log = AuditLog()
        assert log.events == []

    def test_record_appends(self) -> None:
        log = AuditLog()
        log.record(_make_audit_event(1))
        log.record(_make_audit_event(2))
        assert len(log.events) == 2

    def test_events_returns_copy(self) -> None:
        log = AuditLog()
        log.record(_make_audit_event(1))
        events = log.events
        events.clear()
        assert len(log.events) == 1  # original unaffected

    def test_events_in_insertion_order(self) -> None:
        log = AuditLog()
        for i in range(1, 6):
            log.record(_make_audit_event(i))
        rounds = [e.round_num for e in log.events]
        assert rounds == [1, 2, 3, 4, 5]

    def test_to_records_json_serialisable(self) -> None:
        log = AuditLog()
        log.record(_make_audit_event(1))
        records = log.to_records()
        assert isinstance(records, list)
        # Must not raise
        json.dumps(records)

    def test_to_records_contains_all_fields(self) -> None:
        log = AuditLog()
        log.record(_make_audit_event(1))
        record = log.to_records()[0]
        assert "round_num" in record
        assert "pre_round_model_hash" in record
        assert "post_round_model_hash" in record
        assert "timestamp_utc" in record
        assert "skipped" in record

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def test_summary_empty_log(self) -> None:
        log = AuditLog()
        s = log.summary()
        assert s["num_rounds_recorded"] == 0
        assert s["num_rounds_skipped"] == 0
        assert s["unique_model_hashes"] == 0

    def test_summary_counts_skipped(self) -> None:
        log = AuditLog()
        log.record(_make_audit_event(1, skipped=False))
        log.record(_make_audit_event(2, skipped=True))
        log.record(_make_audit_event(3, skipped=False))
        s = log.summary()
        assert s["num_rounds_recorded"] == 3
        assert s["num_rounds_skipped"] == 1

    def test_summary_unique_model_hashes(self) -> None:
        log = AuditLog()
        # All non-skipped events get distinct post_round hashes except one repeated
        e1 = _make_audit_event(1)
        e2 = _make_audit_event(2)
        # Patch post hashes
        from dataclasses import replace
        e1 = replace(e1, post_round_model_hash="sha256:hash_a")
        e2 = replace(e2, post_round_model_hash="sha256:hash_b")
        log.record(e1)
        log.record(e2)
        assert log.summary()["unique_model_hashes"] == 2

    def test_summary_skipped_not_counted_in_unique_hashes(self) -> None:
        log = AuditLog()
        skipped = _make_audit_event(1, skipped=True)
        log.record(skipped)
        assert log.summary()["unique_model_hashes"] == 0

    def test_summary_dropout_count(self) -> None:
        log = AuditLog()
        log.record(_make_audit_event(1, skipped=False))  # no dropout (dropped_clients=[])
        log.record(_make_audit_event(2, skipped=True))   # has dropout (dropped_clients=[2])
        s = log.summary()
        assert s["num_rounds_with_dropout"] == 1

    def test_summary_dp_epsilon_accumulates(self) -> None:
        from dataclasses import replace
        log = AuditLog()
        e1 = replace(_make_audit_event(1), dp_epsilon_spent=1.0)
        e2 = replace(_make_audit_event(2), dp_epsilon_spent=1.0)
        log.record(e1)
        log.record(e2)
        assert log.summary()["total_dp_epsilon"] == pytest.approx(2.0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        log = AuditLog()
        log.record(_make_audit_event(1))
        log.save(tmp_path)
        assert (tmp_path / "audit_log.json").exists()

    def test_save_creates_jsonl_file(self, tmp_path: Path) -> None:
        log = AuditLog()
        log.record(_make_audit_event(1))
        log.save(tmp_path)
        assert (tmp_path / "audit_log.jsonl").exists()

    def test_save_json_is_valid_array(self, tmp_path: Path) -> None:
        log = AuditLog()
        log.record(_make_audit_event(1))
        log.record(_make_audit_event(2))
        log.save(tmp_path)
        data = json.loads((tmp_path / "audit_log.json").read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_save_jsonl_line_count_matches(self, tmp_path: Path) -> None:
        log = AuditLog()
        for i in range(1, 4):
            log.record(_make_audit_event(i))
        log.save(tmp_path)
        lines = (tmp_path / "audit_log.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3

    def test_save_jsonl_each_line_valid_json(self, tmp_path: Path) -> None:
        log = AuditLog()
        log.record(_make_audit_event(1))
        log.save(tmp_path)
        for line in (tmp_path / "audit_log.jsonl").read_text().strip().splitlines():
            obj = json.loads(line)
            assert "round_num" in obj

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        log = AuditLog()
        log.record(_make_audit_event(1))
        nested = tmp_path / "deep" / "nested" / "dir"
        log.save(nested)
        assert (nested / "audit_log.json").exists()

    def test_roundtrip_data_integrity(self, tmp_path: Path) -> None:
        log = AuditLog()
        event = _make_audit_event(round_num=7)
        log.record(event)
        log.save(tmp_path)
        data = json.loads((tmp_path / "audit_log.json").read_text())
        assert data[0]["round_num"] == 7
        assert data[0]["global_accuracy"] == pytest.approx(0.85)


# ===========================================================================
# replay.py — RoundLineageRecord
# ===========================================================================


class TestRoundLineageRecord:
    def test_fields(self) -> None:
        r = RoundLineageRecord(
            round_num=1,
            selection_seed=1039,
            selected_clients=[0, 2],
            active_clients=[0, 2],
            pre_round_model_hash="sha256:aaa",
            post_round_model_hash="sha256:bbb",
            skipped=False,
        )
        assert r.round_num == 1
        assert r.selection_seed == 1039
        assert r.skipped is False

    def test_skipped_record(self) -> None:
        r = RoundLineageRecord(
            round_num=2,
            selection_seed=2036,
            selected_clients=[1],
            active_clients=[],
            pre_round_model_hash="sha256:ccc",
            post_round_model_hash="sha256:ccc",
            skipped=True,
        )
        assert r.skipped is True
        assert r.pre_round_model_hash == r.post_round_model_hash


# ===========================================================================
# replay.py — ReplayManifest
# ===========================================================================


def _make_manifest() -> ReplayManifest:
    config_snapshot: dict = {"name": "test_exp", "seed": 42}
    config_hash = hash_config(config_snapshot)
    m = ReplayManifest(
        config_snapshot=config_snapshot,
        config_hash=config_hash,
        experiment_name="test_exp",
        seed=42,
    )
    m.set_initial_model("cnn", 863146, "sha256:init_hash")
    m.set_data_info("mnist", 60000, 10000, "dirichlet", 0.5, 10, [6000] * 10)
    return m


class TestReplayManifest:
    def test_to_dict_schema_version(self) -> None:
        d = _make_manifest().to_dict()
        assert d["schema_version"] == "1.0"

    def test_to_dict_experiment_fields(self) -> None:
        d = _make_manifest().to_dict()
        exp = d["experiment"]
        assert exp["name"] == "test_exp"
        assert exp["seed"] == 42
        assert "config_hash" in exp

    def test_to_dict_model_info(self) -> None:
        d = _make_manifest().to_dict()
        assert d["model"]["architecture"] == "cnn"
        assert d["model"]["num_trainable_params"] == 863146
        assert d["model"]["initial_model_hash"] == "sha256:init_hash"

    def test_to_dict_data_info(self) -> None:
        d = _make_manifest().to_dict()
        data = d["data"]
        assert data["dataset"] == "mnist"
        assert data["train_samples"] == 60000
        assert data["partitioning"] == "dirichlet"
        assert len(data["client_sample_counts"]) == 10

    def test_to_dict_environment_keys(self) -> None:
        d = _make_manifest().to_dict()
        env = d["environment"]
        assert "python_version" in env
        assert "torch_version" in env
        assert "platform" in env
        assert "hostname" in env

    def test_to_dict_config_snapshot_included(self) -> None:
        m = _make_manifest()
        d = m.to_dict()
        assert "config" in d
        assert d["config"]["seed"] == 42

    def test_to_dict_generated_at_is_iso(self) -> None:
        from datetime import datetime, timezone
        d = _make_manifest().to_dict()
        # Must parse without error
        ts = d["generated_at"]
        assert "T" in ts  # ISO 8601 format

    def test_add_round_populates_lineage(self) -> None:
        m = _make_manifest()
        m.add_round(RoundLineageRecord(
            round_num=1,
            selection_seed=1039,
            selected_clients=[0, 1],
            active_clients=[0, 1],
            pre_round_model_hash="sha256:aaa",
            post_round_model_hash="sha256:bbb",
            skipped=False,
        ))
        d = m.to_dict()
        assert len(d["round_lineage"]) == 1
        assert d["round_lineage"][0]["round_num"] == 1

    def test_add_multiple_rounds(self) -> None:
        m = _make_manifest()
        for i in range(1, 6):
            m.add_round(RoundLineageRecord(
                round_num=i,
                selection_seed=42 + i * 997,
                selected_clients=[0],
                active_clients=[0],
                pre_round_model_hash=f"sha256:pre{i}",
                post_round_model_hash=f"sha256:post{i}",
                skipped=False,
            ))
        d = m.to_dict()
        assert len(d["round_lineage"]) == 5

    def test_round_lineage_selection_seed(self) -> None:
        m = _make_manifest()
        seed = 42
        round_num = 3
        expected_seed = seed + round_num * 997
        m.add_round(RoundLineageRecord(
            round_num=round_num,
            selection_seed=expected_seed,
            selected_clients=[0],
            active_clients=[0],
            pre_round_model_hash="sha256:x",
            post_round_model_hash="sha256:y",
            skipped=False,
        ))
        assert m.to_dict()["round_lineage"][0]["selection_seed"] == expected_seed

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def test_verify_config_correct(self) -> None:
        config_snapshot: dict = {"name": "test_exp", "seed": 42}
        config_hash = hash_config(config_snapshot)
        m = ReplayManifest(config_snapshot, config_hash, "test_exp", 42)
        assert m.verify_config(config_hash) is True

    def test_verify_config_wrong_hash(self) -> None:
        config_snapshot: dict = {"name": "test_exp", "seed": 42}
        config_hash = hash_config(config_snapshot)
        m = ReplayManifest(config_snapshot, config_hash, "test_exp", 42)
        assert m.verify_config("sha256:wrong") is False

    def test_verify_initial_model_correct(self) -> None:
        m = _make_manifest()
        assert m.verify_initial_model("sha256:init_hash") is True

    def test_verify_initial_model_wrong(self) -> None:
        m = _make_manifest()
        assert m.verify_initial_model("sha256:wrong") is False

    def test_verify_initial_model_before_set(self) -> None:
        config_snapshot: dict = {}
        m = ReplayManifest(config_snapshot, "sha256:x", "test", 0)
        assert m.verify_initial_model("sha256:anything") is False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def test_save_creates_file(self, tmp_path: Path) -> None:
        m = _make_manifest()
        m.save(tmp_path)
        assert (tmp_path / "replay_manifest.json").exists()

    def test_save_valid_json(self, tmp_path: Path) -> None:
        m = _make_manifest()
        m.save(tmp_path)
        data = json.loads((tmp_path / "replay_manifest.json").read_text())
        assert isinstance(data, dict)

    def test_save_roundtrip_schema_version(self, tmp_path: Path) -> None:
        m = _make_manifest()
        m.save(tmp_path)
        data = json.loads((tmp_path / "replay_manifest.json").read_text())
        assert data["schema_version"] == "1.0"

    def test_save_roundtrip_round_lineage(self, tmp_path: Path) -> None:
        m = _make_manifest()
        m.add_round(RoundLineageRecord(
            round_num=1,
            selection_seed=1039,
            selected_clients=[0, 1],
            active_clients=[0, 1],
            pre_round_model_hash="sha256:pre",
            post_round_model_hash="sha256:post",
            skipped=False,
        ))
        m.save(tmp_path)
        data = json.loads((tmp_path / "replay_manifest.json").read_text())
        assert len(data["round_lineage"]) == 1
        assert data["round_lineage"][0]["pre_round_model_hash"] == "sha256:pre"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        m = _make_manifest()
        nested = tmp_path / "a" / "b" / "c"
        m.save(nested)
        assert (nested / "replay_manifest.json").exists()


# ===========================================================================
# capture_environment
# ===========================================================================


class TestCaptureEnvironment:
    def test_returns_expected_keys(self) -> None:
        env = _capture_environment()
        assert "python_version" in env
        assert "torch_version" in env
        assert "platform" in env
        assert "hostname" in env

    def test_torch_version_matches(self) -> None:
        env = _capture_environment()
        assert env["torch_version"] == torch.__version__


# ===========================================================================
# Integration: hash_state_dict round-trips through governance pipeline
# ===========================================================================


class TestGovernanceIntegration:
    def test_model_hash_changes_after_update(self) -> None:
        model = nn.Linear(4, 2)
        before = hash_state_dict(model.state_dict())
        # Mutate a parameter
        with torch.no_grad():
            model.weight.add_(0.1)
        after = hash_state_dict(model.state_dict())
        assert before != after

    def test_model_hash_stable_without_update(self) -> None:
        model = nn.Linear(4, 2)
        h1 = hash_state_dict(model.state_dict())
        h2 = hash_state_dict(model.state_dict())
        assert h1 == h2

    def test_audit_log_and_manifest_round_counts_match(self, tmp_path: Path) -> None:
        log = AuditLog()
        for i in range(1, 4):
            log.record(_make_audit_event(i))

        config_snapshot: dict = {"name": "x", "seed": 1}
        m = ReplayManifest(config_snapshot, hash_config(config_snapshot), "x", 1)
        m.set_initial_model("cnn", 100, "sha256:init")
        m.set_data_info("mnist", 60000, 10000, "iid", 0.5, 2, [30000, 30000])

        for event in log.events:
            m.add_round(RoundLineageRecord(
                round_num=event.round_num,
                selection_seed=1 + event.round_num * 997,
                selected_clients=event.selected_clients,
                active_clients=event.active_clients,
                pre_round_model_hash=event.pre_round_model_hash,
                post_round_model_hash=event.post_round_model_hash,
                skipped=event.skipped,
            ))

        log.save(tmp_path)
        m.save(tmp_path)

        audit_data = json.loads((tmp_path / "audit_log.json").read_text())
        manifest_data = json.loads((tmp_path / "replay_manifest.json").read_text())

        assert len(audit_data) == len(manifest_data["round_lineage"]) == 3

    def test_config_hash_in_manifest_verifies_correctly(self) -> None:
        cfg: dict = {"seed": 7, "name": "test"}
        h = hash_config(cfg)
        m = ReplayManifest(cfg, h, "test", 7)
        assert m.verify_config(h)
        assert not m.verify_config(hash_config({"seed": 8, "name": "test"}))
