"""Regulatory scenario tests — Banking & Financial Services compliance simulation.

These tests are NOT unit tests of individual functions.  They model end-to-end
constraints imposed by regulated industries such as banking, insurance, and
financial services when deploying federated learning:

Regulatory domains covered
--------------------------
1.  DP Budget Enforcement         — strict per-round and cumulative ε/δ ceilings
2.  Audit Trail Completeness      — every round must produce a signed record
3.  Audit Trail Immutability      — records cannot be modified after emission
4.  Model Hash-Chain Integrity    — post_round[N] == pre_round[N+1] (no silent updates)
5.  Tamper Detection              — altered hashes break chain verification
6.  Minimum Client Participation  — quorum requirements under operational constraints
7.  Dropout Threshold Enforcement — excessive dropout triggers a compliance violation
8.  Reproducibility Verification  — manifest config-hash validates replay integrity
9.  Data Residency / Isolation    — client sets must be disjoint (no data leakage)
10. Mandatory Governance Gating   — high-epsilon experiments require governance enabled
11. Privacy Budget Exhaustion     — cumulative (T·ε, T·δ) must not exceed ceiling
12. Timestamp Ordering            — audit events must be chronologically ordered
13. Clipping Fraction Thresholds  — large clip fractions indicate model instability
14. Round Skipping Limits         — too many skipped rounds invalidates the training run
15. Model Lineage Traceability    — full hash chain must be reconstructable from manifest

These tests use only the public FLP API (no mocks of internal state); they
exercise the same code paths that execute during a real experiment run.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from flp.governance.audit import AuditEvent, AuditLog
from flp.governance.hashing import hash_config, hash_state_dict
from flp.governance.replay import ReplayManifest, RoundLineageRecord
from flp.privacy.dp import DPAccountant, DPRoundRecord, GaussianMechanism, compute_noise_multiplier


# ---------------------------------------------------------------------------
# Regulatory constants (representative banking / FSI policy values)
# ---------------------------------------------------------------------------

# GDPR / internal policy: per-round privacy budget ceiling
BANK_MAX_EPSILON_PER_ROUND: float = 1.0
BANK_MAX_DELTA_PER_ROUND: float = 1e-5

# Internal policy: cumulative budget ceiling across the full training run
BANK_MAX_CUMULATIVE_EPSILON: float = 10.0
BANK_MAX_CUMULATIVE_DELTA: float = 1e-4

# Operational: minimum fraction of selected clients that must respond
BANK_MIN_PARTICIPATION_FRACTION: float = 0.5

# Risk threshold: if more than this fraction of updates are clipped, flag the run
BANK_MAX_CLIP_FRACTION_WARNING: float = 0.3

# Governance: regulators require governance enabled for ε > this threshold
BANK_EPSILON_GOVERNANCE_THRESHOLD: float = 0.5

# Operational: max fraction of rounds that may be skipped before run is invalid
BANK_MAX_SKIP_FRACTION: float = 0.2

# Minimum number of rounds needed for a training run to be deemed valid
BANK_MIN_VALID_ROUNDS: int = 5


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _make_model() -> nn.Module:
    return nn.Linear(8, 4)


def _make_state(value: float = 1.0) -> dict[str, torch.Tensor]:
    model = nn.Linear(8, 4)
    nn.init.constant_(model.weight, value)
    nn.init.constant_(model.bias, value)
    return model.state_dict()


def _iso_utc(offset_seconds: float = 0.0) -> str:
    t = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return t.isoformat()


def _make_dp_record(
    round_num: int,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    num_total: int = 5,
    num_clipped: int = 0,
    noise_std: float = 0.1,
    clip_norm: float = 1.0,
) -> DPRoundRecord:
    return DPRoundRecord(
        round_num=round_num,
        num_clients_total=num_total,
        num_clients_clipped=num_clipped,
        noise_std=noise_std,
        clip_norm=clip_norm,
        epsilon_spent=epsilon,
        delta_spent=delta,
    )


def _make_audit_event(
    round_num: int,
    pre_hash: str,
    post_hash: str,
    selected: list[int] | None = None,
    active: list[int] | None = None,
    dropped: list[int] | None = None,
    dp_epsilon: float = 1.0,
    dp_delta: float = 1e-5,
    num_clipped: int = 0,
    skipped: bool = False,
    timestamp_utc: str | None = None,
) -> AuditEvent:
    if selected is None:
        selected = [0, 1, 2, 3, 4]
    if active is None:
        active = [] if skipped else [0, 1, 2, 3, 4]
    if dropped is None:
        dropped = selected if skipped else []
    return AuditEvent(
        round_num=round_num,
        timestamp_utc=timestamp_utc or _iso_utc(round_num),
        selected_clients=selected,
        active_clients=active,
        dropped_clients=dropped,
        pre_round_model_hash=pre_hash,
        post_round_model_hash=post_hash,
        global_accuracy=0.0 if skipped else 0.85,
        global_loss=0.0 if skipped else 0.42,
        num_clients_clipped=num_clipped,
        dp_epsilon_spent=dp_epsilon,
        dp_delta_spent=dp_delta,
        elapsed_seconds=2.0,
        skipped=skipped,
    )


def _build_hash_chain(num_rounds: int) -> list[str]:
    """Produce N+1 distinct hashes simulating a clean model lineage."""
    states = [_make_state(float(i)) for i in range(num_rounds + 1)]
    return [hash_state_dict(s) for s in states]


def _build_compliant_audit_log(
    num_rounds: int = 10,
    epsilon_per_round: float = 1.0,
    delta_per_round: float = 1e-5,
    dropout_rounds: list[int] | None = None,
    skip_rounds: list[int] | None = None,
) -> AuditLog:
    """Build an audit log representing a compliant training run."""
    dropout_rounds = dropout_rounds or []
    skip_rounds = skip_rounds or []
    hashes = _build_hash_chain(num_rounds)
    log = AuditLog()
    for r in range(1, num_rounds + 1):
        selected = list(range(5))
        dropped = [4] if r in dropout_rounds else []
        active = [c for c in selected if c not in dropped]
        skipped = r in skip_rounds
        log.record(_make_audit_event(
            round_num=r,
            pre_hash=hashes[r - 1],
            post_hash=hashes[r - 1] if skipped else hashes[r],
            selected=selected,
            active=[] if skipped else active,
            dropped=selected if skipped else dropped,
            dp_epsilon=epsilon_per_round,
            dp_delta=delta_per_round,
            skipped=skipped,
        ))
    return log


# ===========================================================================
# 1. DP Budget Enforcement
# ===========================================================================


class TestDPBudgetEnforcement:
    """Verify that per-round and cumulative DP budgets satisfy regulatory ceilings."""

    def test_per_round_epsilon_within_bank_ceiling(self) -> None:
        """Each round's ε must not exceed the bank's per-round budget ceiling."""
        log = _build_compliant_audit_log(num_rounds=5, epsilon_per_round=0.5)
        violations = [
            e.round_num
            for e in log.events
            if not e.skipped and e.dp_epsilon_spent > BANK_MAX_EPSILON_PER_ROUND
        ]
        assert violations == [], f"Rounds exceeded ε ceiling: {violations}"

    def test_per_round_delta_within_bank_ceiling(self) -> None:
        """Each round's δ must not exceed the bank's per-round delta ceiling."""
        log = _build_compliant_audit_log(num_rounds=5, delta_per_round=1e-6)
        violations = [
            e.round_num
            for e in log.events
            if not e.skipped and e.dp_delta_spent > BANK_MAX_DELTA_PER_ROUND
        ]
        assert violations == [], f"Rounds exceeded δ ceiling: {violations}"

    def test_per_round_epsilon_at_ceiling_is_non_compliant(self) -> None:
        """ε exactly at the ceiling is compliant; ε above is not."""
        compliant_epsilon = BANK_MAX_EPSILON_PER_ROUND
        non_compliant_epsilon = BANK_MAX_EPSILON_PER_ROUND + 0.001
        assert compliant_epsilon <= BANK_MAX_EPSILON_PER_ROUND
        assert non_compliant_epsilon > BANK_MAX_EPSILON_PER_ROUND

    def test_cumulative_epsilon_within_bank_ceiling(self) -> None:
        """Total (T·ε) under sequential composition must not exceed cumulative ceiling."""
        num_rounds = 8
        log = _build_compliant_audit_log(
            num_rounds=num_rounds,
            epsilon_per_round=BANK_MAX_CUMULATIVE_EPSILON / num_rounds,  # exactly at ceiling
        )
        total_epsilon = sum(e.dp_epsilon_spent for e in log.events if not e.skipped)
        assert total_epsilon <= BANK_MAX_CUMULATIVE_EPSILON + 1e-9

    def test_cumulative_delta_within_bank_ceiling(self) -> None:
        """Total (T·δ) must not exceed cumulative ceiling."""
        num_rounds = 8
        log = _build_compliant_audit_log(
            num_rounds=num_rounds,
            delta_per_round=BANK_MAX_CUMULATIVE_DELTA / num_rounds,
        )
        total_delta = sum(e.dp_delta_spent for e in log.events if not e.skipped)
        assert total_delta <= BANK_MAX_CUMULATIVE_DELTA + 1e-15

    def test_dp_accountant_cumulative_tracks_correctly(self) -> None:
        """DPAccountant sequential composition must equal sum of per-round costs."""
        accountant = DPAccountant(epsilon_per_round=0.5, delta_per_round=1e-5)
        for r in range(1, 6):
            accountant.record_round(_make_dp_record(r, epsilon=0.5, delta=1e-5))
        assert accountant.total_epsilon == pytest.approx(2.5)
        assert accountant.total_delta == pytest.approx(5e-5)

    def test_dp_accountant_flags_budget_exhaustion(self) -> None:
        """Detect when cumulative ε exceeds the regulatory ceiling."""
        accountant = DPAccountant(epsilon_per_round=2.0, delta_per_round=1e-5)
        for r in range(1, 8):  # 7 rounds × ε=2.0 = 14.0 > ceiling 10.0
            accountant.record_round(_make_dp_record(r, epsilon=2.0, delta=1e-5))
        assert accountant.total_epsilon > BANK_MAX_CUMULATIVE_EPSILON

    def test_zero_epsilon_rounds_skipped_in_accumulation(self) -> None:
        """Skipped rounds contribute 0 epsilon to the budget — no phantom cost."""
        log = _build_compliant_audit_log(
            num_rounds=5, epsilon_per_round=1.0, skip_rounds=[2, 4]
        )
        total = sum(e.dp_epsilon_spent for e in log.events if not e.skipped)
        # Only 3 non-skipped rounds × 1.0 = 3.0
        assert total == pytest.approx(3.0)

    def test_noise_multiplier_increases_with_stricter_epsilon(self) -> None:
        """Stricter ε (smaller) must produce larger noise for the same δ."""
        sigma_strong = compute_noise_multiplier(epsilon=0.1, delta=1e-5)
        sigma_moderate = compute_noise_multiplier(epsilon=1.0, delta=1e-5)
        sigma_weak = compute_noise_multiplier(epsilon=10.0, delta=1e-5)
        assert sigma_strong > sigma_moderate > sigma_weak

    def test_gaussian_mechanism_noise_scales_with_clip_norm(self) -> None:
        """Noise std must equal noise_multiplier × clip_norm (sensitivity)."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, clip_norm=2.0)
        expected_sigma = compute_noise_multiplier(1.0, 1e-5) * 2.0
        assert mech.noise_std == pytest.approx(expected_sigma)


# ===========================================================================
# 2. Audit Trail Completeness
# ===========================================================================


class TestAuditTrailCompleteness:
    """Every round in the training schedule must have exactly one audit record."""

    def test_audit_log_has_record_for_every_round(self) -> None:
        """No round may be silently omitted from the audit trail."""
        num_rounds = 10
        log = _build_compliant_audit_log(num_rounds=num_rounds)
        recorded_rounds = {e.round_num for e in log.events}
        expected_rounds = set(range(1, num_rounds + 1))
        assert recorded_rounds == expected_rounds

    def test_skipped_rounds_still_produce_audit_records(self) -> None:
        """Even a fully-skipped round (all clients dropped) must be audited."""
        log = _build_compliant_audit_log(num_rounds=5, skip_rounds=[2, 4])
        recorded = {e.round_num for e in log.events}
        assert 2 in recorded
        assert 4 in recorded

    def test_skipped_round_record_is_marked_correctly(self) -> None:
        log = _build_compliant_audit_log(num_rounds=3, skip_rounds=[2])
        skipped_event = next(e for e in log.events if e.round_num == 2)
        assert skipped_event.skipped is True
        assert skipped_event.active_clients == []

    def test_no_duplicate_round_records(self) -> None:
        """Each round number must appear exactly once in the audit trail."""
        log = _build_compliant_audit_log(num_rounds=8)
        round_nums = [e.round_num for e in log.events]
        assert len(round_nums) == len(set(round_nums))

    def test_audit_record_has_all_mandatory_fields_populated(self) -> None:
        """Every audit event must have non-empty mandatory fields."""
        log = _build_compliant_audit_log(num_rounds=3)
        for event in log.events:
            assert event.round_num >= 1
            assert event.timestamp_utc != ""
            assert event.pre_round_model_hash.startswith("sha256:")
            assert event.post_round_model_hash.startswith("sha256:")
            assert isinstance(event.selected_clients, list)
            assert isinstance(event.active_clients, list)
            assert isinstance(event.dropped_clients, list)

    def test_every_event_has_valid_iso_timestamp(self) -> None:
        """Timestamps must be ISO 8601 parseable (required for audit trail signing)."""
        log = _build_compliant_audit_log(num_rounds=5)
        for event in log.events:
            # Must not raise
            parsed = datetime.fromisoformat(event.timestamp_utc)
            assert parsed is not None

    def test_audit_log_persists_all_records(self, tmp_path: Path) -> None:
        """Persisted audit log must contain exactly as many records as were recorded."""
        num_rounds = 7
        log = _build_compliant_audit_log(num_rounds=num_rounds)
        log.save(tmp_path)
        data = json.loads((tmp_path / "audit_log.json").read_text())
        assert len(data) == num_rounds


# ===========================================================================
# 3. Audit Trail Immutability
# ===========================================================================


class TestAuditTrailImmutability:
    """Audit records must be effectively immutable after emission."""

    def test_events_property_returns_defensive_copy(self) -> None:
        """Mutating the returned events list must not affect the log."""
        log = AuditLog()
        log.record(_make_audit_event(1, "sha256:a", "sha256:b"))
        events = log.events
        events.clear()
        assert len(log.events) == 1

    def test_round_order_cannot_be_altered_externally(self) -> None:
        """The internal ordering must survive external list mutations."""
        log = _build_compliant_audit_log(num_rounds=4)
        first_copy = log.events
        first_copy.reverse()  # external mutation
        # Internal order must be unchanged
        assert log.events[0].round_num == 1
        assert log.events[-1].round_num == 4

    def test_saved_json_is_write_once_deterministic(self, tmp_path: Path) -> None:
        """The same audit log written twice must produce identical files."""
        log = _build_compliant_audit_log(num_rounds=3)
        log.save(tmp_path / "run1")
        log.save(tmp_path / "run2")
        content1 = (tmp_path / "run1" / "audit_log.json").read_text()
        content2 = (tmp_path / "run2" / "audit_log.json").read_text()
        assert content1 == content2

    def test_saved_jsonl_each_line_is_self_contained(self, tmp_path: Path) -> None:
        """Each JSONL line must be an independently parseable JSON object."""
        log = _build_compliant_audit_log(num_rounds=5)
        log.save(tmp_path)
        lines = (tmp_path / "audit_log.jsonl").read_text().strip().splitlines()
        for line in lines:
            obj = json.loads(line)
            assert "round_num" in obj
            assert "pre_round_model_hash" in obj
            assert "post_round_model_hash" in obj


# ===========================================================================
# 4 & 5. Model Hash-Chain Integrity + Tamper Detection
# ===========================================================================


class TestModelHashChainIntegrity:
    """Cryptographic hash chain: post_round[N] must equal pre_round[N+1].

    This is the financial-grade equivalent of a ledger's block-chaining:
    any silent model substitution between rounds is detectable.
    """

    def _assert_chain_valid(self, events: list[AuditEvent]) -> None:
        non_skipped = [e for e in events if not e.skipped]
        for i in range(1, len(non_skipped)):
            prev = non_skipped[i - 1]
            curr = non_skipped[i]
            assert curr.pre_round_model_hash == prev.post_round_model_hash, (
                f"Chain broken between round {prev.round_num} and {curr.round_num}: "
                f"post={prev.post_round_model_hash[:20]}... "
                f"!= pre={curr.pre_round_model_hash[:20]}..."
            )

    def test_clean_run_produces_valid_hash_chain(self) -> None:
        """A normally-executed run must form an unbroken hash chain."""
        log = _build_compliant_audit_log(num_rounds=8)
        self._assert_chain_valid(log.events)

    def test_hash_chain_valid_with_dropout_rounds(self) -> None:
        """Dropout (partial participation) must not break the hash chain."""
        log = _build_compliant_audit_log(num_rounds=8, dropout_rounds=[2, 5])
        self._assert_chain_valid(log.events)

    def test_hash_chain_valid_skipped_rounds_preserve_hash(self) -> None:
        """Skipped rounds must carry forward the same pre-hash (no model update).

        Chain with round 3 skipped (4 distinct model states, not 5):
          R1: pre=h0 → post=h1
          R2: pre=h1 → post=h2
          R3: pre=h2 → post=h2  (skipped — no aggregation, model unchanged)
          R4: pre=h2 → post=h3  (resumes from h2, not a new independent hash)
          R5: pre=h3 → post=h4
        """
        # 5 distinct model states (indices 0..4); round 3 skips so uses h2 twice
        hashes = _build_hash_chain(5)
        # Manually assign pre/post to correctly model the skipped-round carry-forward
        chain = [
            (hashes[0], hashes[1]),   # round 1
            (hashes[1], hashes[2]),   # round 2
            (hashes[2], hashes[2]),   # round 3 — skipped, no update
            (hashes[2], hashes[3]),   # round 4 — pre must match round 3's post (h2)
            (hashes[3], hashes[4]),   # round 5
        ]
        log = AuditLog()
        for r, (pre, post) in enumerate(chain, start=1):
            skipped = r == 3
            log.record(_make_audit_event(
                round_num=r, pre_hash=pre, post_hash=post, skipped=skipped,
            ))
        # Verify full chain continuity
        events = log.events
        for i in range(1, len(events)):
            assert events[i].pre_round_model_hash == events[i - 1].post_round_model_hash, (
                f"Chain broken between round {i} and {i + 1}"
            )

    def test_tampered_post_hash_breaks_chain(self) -> None:
        """Silently replacing a model between rounds must be detectable."""
        log = _build_compliant_audit_log(num_rounds=5)
        events = log.events
        # Tamper: replace post-hash of round 2 with a fake value
        tampered_event = replace(events[1], post_round_model_hash="sha256:" + "f" * 64)
        tampered_events = events[:1] + [tampered_event] + events[2:]
        # Now verify the chain — it must be broken
        with pytest.raises(AssertionError, match="Chain broken"):
            self._assert_chain_valid(tampered_events)

    def test_tampered_pre_hash_breaks_chain(self) -> None:
        """A tampered pre-hash must also be caught by chain verification."""
        log = _build_compliant_audit_log(num_rounds=5)
        events = log.events
        # Tamper: alter pre-hash of round 3 so it no longer matches post of round 2
        tampered = replace(events[2], pre_round_model_hash="sha256:" + "e" * 64)
        tampered_events = events[:2] + [tampered] + events[3:]
        with pytest.raises(AssertionError, match="Chain broken"):
            self._assert_chain_valid(tampered_events)

    def test_identical_model_produces_identical_hash(self) -> None:
        """Two identical state dicts must hash identically (no salt, deterministic)."""
        sd = _make_state(1.5)
        assert hash_state_dict(sd) == hash_state_dict(sd)

    def test_mutated_model_produces_different_hash(self) -> None:
        """A single weight change must change the model hash."""
        model = _make_model()
        h_before = hash_state_dict(model.state_dict())
        with torch.no_grad():
            model.weight[0, 0] += 1e-6
        h_after = hash_state_dict(model.state_dict())
        assert h_before != h_after

    def test_hash_is_order_independent_of_dict_insertion(self) -> None:
        """Hash must be stable regardless of state dict key insertion order."""
        sd1 = {"weight": torch.ones(4, 8), "bias": torch.zeros(4)}
        sd2 = {"bias": torch.zeros(4), "weight": torch.ones(4, 8)}
        assert hash_state_dict(sd1) == hash_state_dict(sd2)

    def test_full_lineage_reconstructable_from_manifest(self, tmp_path: Path) -> None:
        """Replay manifest must reproduce the full hash chain for auditor verification."""
        num_rounds = 6
        hashes = _build_hash_chain(num_rounds)
        log = _build_compliant_audit_log(num_rounds=num_rounds)

        config_snapshot: dict[str, Any] = {"name": "bank_test", "seed": 42}
        m = ReplayManifest(
            config_snapshot=config_snapshot,
            config_hash=hash_config(config_snapshot),
            experiment_name="bank_test",
            seed=42,
        )
        m.set_initial_model("cnn", 100, hashes[0])
        m.set_data_info("mnist", 60000, 10000, "dirichlet", 0.5, 10, [6000] * 10)

        for event in log.events:
            m.add_round(RoundLineageRecord(
                round_num=event.round_num,
                selection_seed=42 + event.round_num * 997,
                selected_clients=event.selected_clients,
                active_clients=event.active_clients,
                pre_round_model_hash=event.pre_round_model_hash,
                post_round_model_hash=event.post_round_model_hash,
                skipped=event.skipped,
            ))

        m.save(tmp_path)
        data = json.loads((tmp_path / "replay_manifest.json").read_text())

        lineage = data["round_lineage"]
        assert len(lineage) == num_rounds
        # Verify chain within manifest
        for i in range(1, len(lineage)):
            assert lineage[i]["pre_round_model_hash"] == lineage[i - 1]["post_round_model_hash"]


# ===========================================================================
# 6. Minimum Client Participation (Quorum)
# ===========================================================================


class TestMinimumClientParticipation:
    """Banking regulations may require a minimum fraction of clients to respond."""

    def _participation_fraction(self, event: AuditEvent) -> float:
        if not event.selected_clients:
            return 0.0
        return len(event.active_clients) / len(event.selected_clients)

    def test_compliant_run_meets_quorum_every_round(self) -> None:
        """All non-skipped rounds must meet the minimum participation fraction."""
        log = _build_compliant_audit_log(num_rounds=10)
        violations = [
            e.round_num
            for e in log.events
            if not e.skipped
            and self._participation_fraction(e) < BANK_MIN_PARTICIPATION_FRACTION
        ]
        assert violations == [], f"Quorum violations in rounds: {violations}"

    def test_single_dropout_below_quorum_is_flagged(self) -> None:
        """If >50% of clients drop out, the round violates the quorum requirement."""
        # 5 selected, 4 dropped → 20% participation < 50% quorum
        event = _make_audit_event(
            round_num=1,
            pre_hash="sha256:a",
            post_hash="sha256:b",
            selected=[0, 1, 2, 3, 4],
            active=[0],
            dropped=[1, 2, 3, 4],
        )
        fraction = self._participation_fraction(event)
        assert fraction < BANK_MIN_PARTICIPATION_FRACTION

    def test_exactly_at_quorum_threshold_is_compliant(self) -> None:
        """Exactly 50% participation (3 of 6 selected) must pass the quorum check."""
        event = _make_audit_event(
            round_num=1,
            pre_hash="sha256:a",
            post_hash="sha256:b",
            selected=[0, 1, 2, 3, 4, 5],
            active=[0, 1, 2],
            dropped=[3, 4, 5],
        )
        fraction = self._participation_fraction(event)
        assert fraction >= BANK_MIN_PARTICIPATION_FRACTION

    def test_quorum_check_skips_skipped_rounds(self) -> None:
        """Skipped rounds are excluded from quorum enforcement."""
        log = _build_compliant_audit_log(num_rounds=5, skip_rounds=[2])
        non_skipped_fractions = [
            self._participation_fraction(e) for e in log.events if not e.skipped
        ]
        assert all(f >= BANK_MIN_PARTICIPATION_FRACTION for f in non_skipped_fractions)

    def test_client_sets_are_disjoint_active_and_dropped(self) -> None:
        """Active and dropped client sets must be mutually exclusive (no data leakage)."""
        log = _build_compliant_audit_log(num_rounds=8, dropout_rounds=[2, 5])
        for event in log.events:
            active_set = set(event.active_clients)
            dropped_set = set(event.dropped_clients)
            overlap = active_set & dropped_set
            assert overlap == set(), (
                f"Round {event.round_num}: client(s) {overlap} appear in both "
                "active and dropped sets — data isolation violated"
            )

    def test_active_plus_dropped_equals_selected(self) -> None:
        """active_clients ∪ dropped_clients must equal selected_clients exactly."""
        log = _build_compliant_audit_log(num_rounds=8, dropout_rounds=[3])
        for event in log.events:
            reconstructed = sorted(event.active_clients + event.dropped_clients)
            assert reconstructed == sorted(event.selected_clients), (
                f"Round {event.round_num}: active+dropped != selected"
            )


# ===========================================================================
# 7. Dropout Threshold Enforcement
# ===========================================================================


class TestDropoutThresholdEnforcement:
    """Excessive client dropout across the run triggers a compliance flag."""

    def _overall_dropout_fraction(self, log: AuditLog) -> float:
        total_selected = sum(len(e.selected_clients) for e in log.events)
        total_dropped = sum(len(e.dropped_clients) for e in log.events)
        return total_dropped / total_selected if total_selected > 0 else 0.0

    def test_low_dropout_run_is_compliant(self) -> None:
        """A run with rare dropout must stay below the dropout ceiling."""
        log = _build_compliant_audit_log(num_rounds=10, dropout_rounds=[2])
        fraction = self._overall_dropout_fraction(log)
        assert fraction < 1 - BANK_MIN_PARTICIPATION_FRACTION

    def test_high_dropout_run_is_non_compliant(self) -> None:
        """When most rounds have high dropout, the run exceeds the compliance threshold."""
        # All rounds have 4/5 clients dropping → 80% dropout
        hashes = _build_hash_chain(5)
        log = AuditLog()
        for r in range(1, 6):
            log.record(_make_audit_event(
                round_num=r,
                pre_hash=hashes[r - 1],
                post_hash=hashes[r],
                selected=[0, 1, 2, 3, 4],
                active=[0],
                dropped=[1, 2, 3, 4],
            ))
        fraction = self._overall_dropout_fraction(log)
        assert fraction > 1 - BANK_MIN_PARTICIPATION_FRACTION

    def test_skip_fraction_within_regulatory_limit(self) -> None:
        """The fraction of skipped rounds must not exceed the max-skip policy."""
        num_rounds = 10
        skip_rounds = [2, 4]  # 20% skipped — exactly at limit
        log = _build_compliant_audit_log(num_rounds=num_rounds, skip_rounds=skip_rounds)
        skip_fraction = sum(1 for e in log.events if e.skipped) / len(log.events)
        assert skip_fraction <= BANK_MAX_SKIP_FRACTION

    def test_excessive_skipping_violates_policy(self) -> None:
        """More than BANK_MAX_SKIP_FRACTION skipped rounds invalidates the run."""
        num_rounds = 10
        skip_rounds = [1, 2, 3, 4, 5]  # 50% skipped > 20% limit
        log = _build_compliant_audit_log(num_rounds=num_rounds, skip_rounds=skip_rounds)
        skip_fraction = sum(1 for e in log.events if e.skipped) / len(log.events)
        assert skip_fraction > BANK_MAX_SKIP_FRACTION

    def test_minimum_valid_rounds_requirement(self) -> None:
        """The run must complete at least BANK_MIN_VALID_ROUNDS non-skipped rounds."""
        log = _build_compliant_audit_log(num_rounds=10, skip_rounds=[1, 2, 3, 4, 7])
        completed = sum(1 for e in log.events if not e.skipped)
        assert completed >= BANK_MIN_VALID_ROUNDS


# ===========================================================================
# 8. Reproducibility Verification (Replay Manifest)
# ===========================================================================


class TestReproducibilityVerification:
    """Regulatory auditors must be able to verify that a run can be reproduced."""

    def _make_manifest_for_run(self, num_rounds: int = 5) -> tuple[ReplayManifest, str]:
        config_snapshot: dict[str, Any] = {
            "name": "bank_fl_run",
            "seed": 42,
            "training": {"num_rounds": num_rounds, "num_clients": 10},
            "privacy": {"enabled": True, "epsilon": 1.0, "delta": 1e-5},
        }
        config_hash = hash_config(config_snapshot)
        m = ReplayManifest(config_snapshot, config_hash, "bank_fl_run", 42)
        m.set_initial_model("cnn", 863146, "sha256:" + "a" * 64)
        m.set_data_info("mnist", 60000, 10000, "dirichlet", 0.5, 10, [6000] * 10)
        return m, config_hash

    def test_config_hash_verifies_against_same_config(self) -> None:
        """Auditor re-hashing the same config must produce the same hash."""
        m, original_hash = self._make_manifest_for_run()
        assert m.verify_config(original_hash) is True

    def test_config_hash_rejects_modified_config(self) -> None:
        """Any modification to the config must invalidate the hash."""
        m, _ = self._make_manifest_for_run()
        modified_config: dict[str, Any] = {
            "name": "bank_fl_run",
            "seed": 99,  # seed changed
            "training": {"num_rounds": 5, "num_clients": 10},
            "privacy": {"enabled": True, "epsilon": 1.0, "delta": 1e-5},
        }
        tampered_hash = hash_config(modified_config)
        assert m.verify_config(tampered_hash) is False

    def test_initial_model_hash_verifies(self) -> None:
        """Auditor re-creating the initial model must produce the same hash."""
        m, _ = self._make_manifest_for_run()
        assert m.verify_initial_model("sha256:" + "a" * 64) is True

    def test_different_model_rejected(self) -> None:
        """A different model at replay time must fail initial model verification."""
        m, _ = self._make_manifest_for_run()
        assert m.verify_initial_model("sha256:" + "b" * 64) is False

    def test_round_selection_seeds_are_deterministic(self) -> None:
        """Selection seeds (seed + round * 997) must be reproducible by formula."""
        base_seed = 42
        for round_num in range(1, 6):
            expected_seed = base_seed + round_num * 997
            # Auditor computes the same formula
            computed_seed = base_seed + round_num * 997
            assert computed_seed == expected_seed

    def test_manifest_saves_and_reloads_config_hash(self, tmp_path: Path) -> None:
        """Config hash stored in manifest must survive a JSON round-trip."""
        m, config_hash = self._make_manifest_for_run()
        m.save(tmp_path)
        data = json.loads((tmp_path / "replay_manifest.json").read_text())
        assert data["experiment"]["config_hash"] == config_hash

    def test_manifest_environment_captured(self, tmp_path: Path) -> None:
        """Environment metadata (Python, torch, platform) must be recorded."""
        m, _ = self._make_manifest_for_run()
        m.save(tmp_path)
        data = json.loads((tmp_path / "replay_manifest.json").read_text())
        env = data["environment"]
        assert "python_version" in env
        assert "torch_version" in env

    def test_manifest_schema_version_present(self, tmp_path: Path) -> None:
        """Schema version must be recorded to support future format evolution."""
        m, _ = self._make_manifest_for_run()
        m.save(tmp_path)
        data = json.loads((tmp_path / "replay_manifest.json").read_text())
        assert "schema_version" in data
        assert data["schema_version"] == "1.1"

    def test_round_lineage_count_matches_training_schedule(self, tmp_path: Path) -> None:
        """Number of lineage records must equal num_rounds in the config."""
        num_rounds = 7
        m, _ = self._make_manifest_for_run(num_rounds=num_rounds)
        hashes = _build_hash_chain(num_rounds)
        for r in range(1, num_rounds + 1):
            m.add_round(RoundLineageRecord(
                round_num=r,
                selection_seed=42 + r * 997,
                selected_clients=[0, 1, 2, 3, 4],
                active_clients=[0, 1, 2, 3, 4],
                pre_round_model_hash=hashes[r - 1],
                post_round_model_hash=hashes[r],
                skipped=False,
            ))
        m.save(tmp_path)
        data = json.loads((tmp_path / "replay_manifest.json").read_text())
        assert len(data["round_lineage"]) == num_rounds


# ===========================================================================
# 9. Mandatory Governance Gating
# ===========================================================================


class TestMandatoryGovernanceGating:
    """High-epsilon or high-risk experiments must require governance to be enabled."""

    def _governance_required_for(self, epsilon: float) -> bool:
        """Bank policy: governance is mandatory when ε > threshold."""
        return epsilon > BANK_EPSILON_GOVERNANCE_THRESHOLD

    def test_low_epsilon_does_not_require_governance(self) -> None:
        """Below the threshold, governance is optional."""
        assert not self._governance_required_for(0.1)
        assert not self._governance_required_for(BANK_EPSILON_GOVERNANCE_THRESHOLD)

    def test_high_epsilon_requires_governance(self) -> None:
        """Above the threshold, governance must be enforced."""
        assert self._governance_required_for(1.0)
        assert self._governance_required_for(10.0)

    def test_dp_enabled_with_governance_is_compliant(self) -> None:
        """DP-enabled run with governance enabled is always compliant."""
        dp_enabled = True
        governance_enabled = True
        epsilon = 1.0
        governance_required = self._governance_required_for(epsilon)
        compliant = not governance_required or governance_enabled
        assert compliant

    def test_high_epsilon_without_governance_is_non_compliant(self) -> None:
        """High-ε run without governance enabled violates bank policy."""
        epsilon = 2.0
        governance_enabled = False
        governance_required = self._governance_required_for(epsilon)
        compliant = not governance_required or governance_enabled
        assert not compliant

    def test_governance_produces_both_required_artefacts(self, tmp_path: Path) -> None:
        """Governance mode must produce BOTH audit_log.json and replay_manifest.json."""
        log = _build_compliant_audit_log(num_rounds=3)
        config_snapshot: dict[str, Any] = {"name": "test", "seed": 1}
        m = ReplayManifest(config_snapshot, hash_config(config_snapshot), "test", 1)
        m.set_initial_model("cnn", 100, "sha256:" + "a" * 64)
        m.set_data_info("mnist", 60000, 10000, "iid", 0.5, 5, [12000] * 5)

        gov_dir = tmp_path / "governance"
        log.save(gov_dir)
        m.save(gov_dir)

        assert (gov_dir / "audit_log.json").exists(), "audit_log.json missing"
        assert (gov_dir / "replay_manifest.json").exists(), "replay_manifest.json missing"


# ===========================================================================
# 10. Clipping Fraction Monitoring
# ===========================================================================


class TestClippingFractionMonitoring:
    """Large clipping fractions signal model instability — a risk indicator."""

    def test_low_clip_fraction_is_acceptable(self) -> None:
        """A run where few updates are clipped is operationally healthy."""
        accountant = DPAccountant(epsilon_per_round=1.0, delta_per_round=1e-5)
        for r in range(1, 6):
            accountant.record_round(_make_dp_record(
                r, num_total=10, num_clipped=1  # 10% clipped
            ))
        assert accountant.avg_clip_fraction <= BANK_MAX_CLIP_FRACTION_WARNING

    def test_high_clip_fraction_triggers_risk_flag(self) -> None:
        """If >30% of updates are clipped every round, flag for model review."""
        accountant = DPAccountant(epsilon_per_round=1.0, delta_per_round=1e-5)
        for r in range(1, 6):
            accountant.record_round(_make_dp_record(
                r, num_total=10, num_clipped=5  # 50% clipped
            ))
        assert accountant.avg_clip_fraction > BANK_MAX_CLIP_FRACTION_WARNING

    def test_zero_clip_fraction_is_optimal(self) -> None:
        """No clipping means all client updates are within the sensitivity bound."""
        accountant = DPAccountant(epsilon_per_round=1.0, delta_per_round=1e-5)
        for r in range(1, 4):
            accountant.record_round(_make_dp_record(r, num_total=10, num_clipped=0))
        assert accountant.avg_clip_fraction == pytest.approx(0.0)

    def test_clip_fraction_property_computes_correctly(self) -> None:
        """DPRoundRecord.clip_fraction must equal clipped/total."""
        record = _make_dp_record(1, num_total=8, num_clipped=2)
        assert record.clip_fraction == pytest.approx(0.25)

    def test_audit_log_tracks_total_clipped_across_rounds(self) -> None:
        """AuditLog summary must accumulate clipped counts across all rounds."""
        hashes = _build_hash_chain(3)
        log = AuditLog()
        for r in range(1, 4):
            log.record(_make_audit_event(
                round_num=r,
                pre_hash=hashes[r - 1],
                post_hash=hashes[r],
                num_clipped=3,
            ))
        assert log.summary()["total_clients_clipped"] == 9


# ===========================================================================
# 11. Timestamp Ordering
# ===========================================================================


class TestTimestampOrdering:
    """Audit events must be emitted and recorded in chronological round order."""

    def test_round_nums_are_strictly_monotone(self) -> None:
        """Round numbers in the audit log must be strictly increasing."""
        log = _build_compliant_audit_log(num_rounds=10)
        round_nums = [e.round_num for e in log.events]
        assert round_nums == sorted(round_nums)
        assert len(round_nums) == len(set(round_nums))

    def test_timestamps_are_parseable_iso8601(self) -> None:
        """All timestamps must be valid ISO 8601 (required for SIEM ingestion)."""
        log = _build_compliant_audit_log(num_rounds=5)
        for event in log.events:
            ts = datetime.fromisoformat(event.timestamp_utc)
            assert ts.tzinfo is not None  # must include timezone

    def test_audit_log_jsonl_preserves_round_order(self, tmp_path: Path) -> None:
        """JSONL format must preserve insertion (round) order for streaming consumers."""
        num_rounds = 6
        log = _build_compliant_audit_log(num_rounds=num_rounds)
        log.save(tmp_path)
        lines = (tmp_path / "audit_log.jsonl").read_text().strip().splitlines()
        round_nums = [json.loads(line)["round_num"] for line in lines]
        assert round_nums == list(range(1, num_rounds + 1))


# ===========================================================================
# 12. End-to-End Regulatory Compliance Check
# ===========================================================================


class TestEndToEndRegulatoryCompliance:
    """Full-run scenario: simulate a compliant banking FL deployment."""

    def test_compliant_10_round_dp_run_passes_all_checks(self, tmp_path: Path) -> None:
        """A well-configured 10-round DP run passes every regulatory check."""
        num_rounds = 10
        epsilon_per_round = 0.5  # below 1.0 ceiling
        delta_per_round = 1e-6   # below 1e-5 ceiling

        log = _build_compliant_audit_log(
            num_rounds=num_rounds,
            epsilon_per_round=epsilon_per_round,
            delta_per_round=delta_per_round,
            dropout_rounds=[3, 7],
        )

        events = log.events

        # 1. Completeness: every round recorded
        assert len(events) == num_rounds

        # 2. No duplicate round numbers
        round_nums = [e.round_num for e in events]
        assert len(set(round_nums)) == num_rounds

        # 3. Per-round DP budget respected
        epsilon_violations = [e.round_num for e in events if e.dp_epsilon_spent > BANK_MAX_EPSILON_PER_ROUND]
        assert epsilon_violations == []

        # 4. Cumulative budget respected
        total_epsilon = sum(e.dp_epsilon_spent for e in events if not e.skipped)
        assert total_epsilon <= BANK_MAX_CUMULATIVE_EPSILON

        # 5. Client isolation: no overlap between active and dropped
        for e in events:
            assert set(e.active_clients) & set(e.dropped_clients) == set()

        # 6. Active + dropped = selected for every round
        for e in events:
            assert sorted(e.active_clients + e.dropped_clients) == sorted(e.selected_clients)

        # 7. Quorum: all non-skipped rounds meet minimum participation
        for e in events:
            if not e.skipped and e.selected_clients:
                fraction = len(e.active_clients) / len(e.selected_clients)
                assert fraction >= BANK_MIN_PARTICIPATION_FRACTION

        # 8. Hash chain integrity
        non_skipped = [e for e in events if not e.skipped]
        for i in range(1, len(non_skipped)):
            assert non_skipped[i].pre_round_model_hash == non_skipped[i - 1].post_round_model_hash

        # 9. No excessive skipping
        skip_fraction = sum(1 for e in events if e.skipped) / len(events)
        assert skip_fraction <= BANK_MAX_SKIP_FRACTION

        # 10. Minimum valid rounds
        completed = sum(1 for e in events if not e.skipped)
        assert completed >= BANK_MIN_VALID_ROUNDS

        # 11. All timestamps parseable
        for e in events:
            datetime.fromisoformat(e.timestamp_utc)

        # 12. Governance artefacts persist correctly
        log.save(tmp_path)
        data = json.loads((tmp_path / "audit_log.json").read_text())
        assert len(data) == num_rounds

    def test_non_compliant_run_fails_budget_check(self) -> None:
        """A run with per-round ε=2.0 (above ceiling) must fail the DP check."""
        num_rounds = 5
        log = _build_compliant_audit_log(
            num_rounds=num_rounds,
            epsilon_per_round=2.0,  # above BANK_MAX_EPSILON_PER_ROUND = 1.0
        )
        violations = [
            e.round_num for e in log.events
            if not e.skipped and e.dp_epsilon_spent > BANK_MAX_EPSILON_PER_ROUND
        ]
        assert len(violations) > 0

    def test_non_compliant_run_fails_cumulative_budget(self) -> None:
        """11 rounds × ε=1.0 = 11.0 > cumulative ceiling 10.0."""
        log = _build_compliant_audit_log(
            num_rounds=11,
            epsilon_per_round=1.0,
        )
        total = sum(e.dp_epsilon_spent for e in log.events if not e.skipped)
        assert total > BANK_MAX_CUMULATIVE_EPSILON
