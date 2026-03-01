"""Asynchronous federated learning server with virtual-time event scheduling."""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

import torch.nn as nn
from torch.utils.data import DataLoader

from flp.core.aggregator import AggregationResult
from flp.core.client import ClientUpdate, FLClient
from flp.core.event_loop import FLEvent, FLEventLoop
from flp.core.server import FLServer, RoundSummary
from flp.core.staleness import StalenessWeighter
from flp.governance.audit import AuditEvent, AuditLog
from flp.governance.hashing import hash_state_dict
from flp.privacy.clipping import clip_model_update
from flp.simulation.delay import DelaySimulator

if TYPE_CHECKING:
    from flp.experiments.config_loader import ExperimentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AsyncRoundSummary
# ---------------------------------------------------------------------------


@dataclass
class AsyncRoundSummary(RoundSummary):
    """Round summary extended with async-specific diagnostics.

    Inherits all fields from :class:`~flp.core.server.RoundSummary` and adds:

    Attributes:
        stale_updates_used: Updates from previous rounds that were still within
            the staleness threshold and contributed to this round's aggregation.
        stale_updates_discarded: Updates that exceeded the staleness threshold
            and were dropped before aggregation.
        pending_updates: Events still queued in the event loop at the end of
            this round, awaiting future rounds.
    """

    stale_updates_used: int = 0
    stale_updates_discarded: int = 0
    pending_updates: int = 0


# ---------------------------------------------------------------------------
# AsyncFLServer
# ---------------------------------------------------------------------------


class AsyncFLServer(FLServer):
    """Asynchronous federated learning server with virtual-time scheduling.

    Extends :class:`~flp.core.server.FLServer` with an event-driven model:

    1. Each round, selected active clients train on the **current** global model
       and their updates are placed into an :class:`~flp.core.event_loop.FLEventLoop`
       with a per-client delivery delay sampled from a uniform distribution.
    2. The server collects all updates whose virtual delivery round is ≤ the
       current round (including delayed updates from previous rounds).
    3. Updates older than ``staleness_threshold`` server versions are discarded.
    4. If no updates are available (all queued, all stale, or all dropped),
       the round is skipped — the model is not updated.

    The ``server_version`` counter increments on every successful aggregation,
    providing a monotone clock for staleness measurement independent of skipped
    rounds:

        staleness = server_version_at_aggregation − model_version_used_by_client

    All randomness (delay sampling, client selection, dropout) uses isolated
    seeded generators so results are bit-exact across runs.

    Args:
        model: Global model. Weights are updated in-place after each round.
        clients: All available federated clients.
        config: Full experiment configuration. Must have ``config.async_fl``
            populated (set ``enabled: true`` in YAML).
        test_loader: DataLoader for server-side global evaluation.
        device: Torch device for the global model and evaluation.
        round_callback: Optional callable invoked after each non-skipped round.
        audit_log: Optional :class:`~flp.governance.audit.AuditLog`.

    Note:
        When DP is enabled alongside async FL, clipping uses the **current**
        server model as the reference (not the model version the client trained
        on). This is a known simplification in async DP-FL.
    """

    def __init__(
        self,
        model: nn.Module,
        clients: list[FLClient],
        config: ExperimentConfig,
        test_loader: DataLoader,  # type: ignore[type-arg]
        device: "torch.device",  # type: ignore[name-defined]
        round_callback: Callable[[RoundSummary], None] | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        super().__init__(
            model=model,
            clients=clients,
            config=config,
            test_loader=test_loader,
            device=device,
            round_callback=round_callback,
            audit_log=audit_log,
        )

        async_cfg = config.async_fl
        self._event_loop = FLEventLoop()
        self._delay_sim = DelaySimulator(
            min_delay=async_cfg.delay_min,
            max_delay=async_cfg.delay_max,
            seed=config.seed,
        )
        self._staleness_threshold: int = async_cfg.staleness_threshold
        self._staleness_weighter = StalenessWeighter(
            strategy=async_cfg.staleness_strategy,
            decay_factor=async_cfg.staleness_decay_factor,
        )
        self._server_version: int = 0   # increments on each successful aggregation
        self._client_lookup: dict[int, FLClient] = {c.client_id: c for c in clients}

        logger.info(
            "AsyncFLServer: delay=[%.1f, %.1f] rounds | staleness_threshold=%d | strategy=%s",
            async_cfg.delay_min,
            async_cfg.delay_max,
            async_cfg.staleness_threshold,
            async_cfg.staleness_strategy,
        )

    # ------------------------------------------------------------------
    # Round implementation (overrides FLServer._run_round)
    # ------------------------------------------------------------------

    def _run_round(self, round_num: int, num_select: int) -> AsyncRoundSummary:  # type: ignore[override]
        """Execute a single async federated round.

        Returns:
            :class:`AsyncRoundSummary` with both standard and async diagnostics.
        """
        t_start = time.perf_counter()

        # ---- Client selection + dropout ----
        selected = self._select_clients(num_select, round_num)
        selected_ids = [c.client_id for c in selected]

        dropout_result = self.dropout_sim.apply(selected, round_num)
        self.dropout_sim.record(dropout_result)

        active = dropout_result.active
        dropped_ids = dropout_result.dropped_ids

        # ---- Snapshot current model before queuing ----
        # Used as reference for DP clipping and broadcast to active clients.
        current_global_state = copy.deepcopy(self.model.state_dict())

        # ---- Governance: hash pre-round model ----
        pre_round_hash: str = ""
        if self._audit_log is not None:
            pre_round_hash = hash_state_dict(self.model.state_dict())

        # ---- Queue updates from active clients ----
        if active:
            delays = self._delay_sim.sample_delays(len(active), round_num)
            for client, delay in zip(active, delays):
                client.set_global_weights(current_global_state)
                update = client.train()
                delivery_round = round_num + math.ceil(delay)
                self._event_loop.push(FLEvent(
                    virtual_round=delivery_round,
                    client_id=client.client_id,
                    model_version=self._server_version,
                    update=update,
                ))
            logger.debug(
                "Round %d: queued %d updates; event loop now has %d pending.",
                round_num, len(active), self._event_loop.pending_count,
            )

        # ---- Collect ready events ----
        ready_events = self._event_loop.pop_ready(round_num)

        # ---- Filter stale updates ----
        # staleness = current server_version - model_version the client used
        live_events: list[FLEvent] = []
        stale_discarded = 0
        for event in ready_events:
            staleness = self._server_version - event.model_version
            if staleness <= self._staleness_threshold:
                live_events.append(event)
            else:
                stale_discarded += 1
                logger.debug(
                    "  Discarded stale update from client %d (staleness=%d > threshold=%d).",
                    event.client_id, staleness, self._staleness_threshold,
                )

        stale_used = sum(
            1 for e in live_events if e.model_version < self._server_version
        )

        # ---- Skip round if no usable updates ----
        if not live_events:
            elapsed = time.perf_counter() - t_start
            if self._audit_log is not None:
                self._audit_log.record(AuditEvent(
                    round_num=round_num,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    selected_clients=selected_ids,
                    active_clients=[],
                    dropped_clients=dropped_ids,
                    pre_round_model_hash=pre_round_hash,
                    post_round_model_hash=pre_round_hash,
                    global_accuracy=0.0,
                    global_loss=0.0,
                    num_clients_clipped=0,
                    dp_epsilon_spent=0.0,
                    dp_delta_spent=0.0,
                    elapsed_seconds=round(elapsed, 4),
                    skipped=True,
                ))
            return AsyncRoundSummary(
                round_num=round_num,
                selected_clients=selected_ids,
                active_clients=[],
                dropped_clients=dropped_ids,
                aggregation=None,
                global_accuracy=0.0,
                global_loss=0.0,
                elapsed_seconds=elapsed,
                skipped=True,
                stale_updates_used=stale_used,
                stale_updates_discarded=stale_discarded,
                pending_updates=self._event_loop.pending_count,
            )

        active_client_ids = [e.client_id for e in live_events]
        updates: list[ClientUpdate] = [e.update for e in live_events]  # type: ignore[misc]

        # ---- Staleness-aware aggregation weights ----
        staleness_values = [self._server_version - e.model_version for e in live_events]
        agg_weights = self._staleness_weighter.compute_weights(
            staleness_values=staleness_values,
            num_samples=[u.num_samples for u in updates],
        )

        # ---- DP clipping ----
        num_clipped = 0
        _dp_epsilon = 0.0
        _dp_delta = 0.0
        if self._dp_mech is not None:
            clipped_updates: list[ClientUpdate] = []
            for u in updates:
                # Reference is the current server model (async-DP simplification).
                clip_result = clip_model_update(
                    current_global_state, u.state_dict, self._dp_mech.clip_norm
                )
                if clip_result.was_clipped:
                    num_clipped += 1
                clipped_updates.append(ClientUpdate(
                    client_id=u.client_id,
                    state_dict=clip_result.state_dict,
                    num_samples=u.num_samples,
                    train_result=u.train_result,
                ))
            updates = clipped_updates

        # ---- Compression (optional) ----
        if self._compressor is not None:
            compressed_list: list[ClientUpdate] = []
            c_ratios: list[float] = []
            for u in updates:
                c_update, ratio = self._compressor.compress(u)
                compressed_list.append(c_update)
                c_ratios.append(ratio)
            updates = compressed_list
            logger.debug(
                "Round %d: compression ratio=%.3f (strategy=%s)",
                round_num,
                sum(c_ratios) / len(c_ratios),
                self.config.compression.strategy,
            )

        # ---- Aggregate ----
        agg_result: AggregationResult = self.aggregator.aggregate(updates, weights=agg_weights)
        aggregated_state = agg_result.state_dict

        # ---- DP noise ----
        if self._dp_mech is not None:
            aggregated_state = self._dp_mech.add_noise(
                aggregated_state, num_clients=len(updates)
            )
            dp_record = self._dp_mech.make_round_record(
                round_num=round_num,
                num_clients_total=len(updates),
                num_clients_clipped=num_clipped,
            )
            self.dp_accountant.record_round(dp_record)  # type: ignore[union-attr]
            _dp_epsilon = dp_record.epsilon_spent
            _dp_delta = dp_record.delta_spent

        self.model.load_state_dict(aggregated_state)
        self._server_version += 1

        # ---- Governance: hash post-round model ----
        post_round_hash: str = ""
        if self._audit_log is not None:
            post_round_hash = hash_state_dict(self.model.state_dict())

        # ---- Global evaluation ----
        global_eval = self._evaluate_global()

        # ---- Per-client evaluation ----
        per_client_acc: dict[int, float] = {}
        new_state = self.model.state_dict()
        for event in live_events:
            client = self._client_lookup.get(event.client_id)
            if client is not None:
                client.set_global_weights(new_state)
                result = client.evaluate()
                per_client_acc[event.client_id] = result.accuracy

        # ---- Record metrics ----
        self.metrics.record_round(
            round_num=round_num,
            global_accuracy=global_eval["accuracy"],
            global_loss=global_eval["loss"],
            per_client_accuracy=per_client_acc,
            num_active_clients=len(live_events),
            client_updates=[
                {
                    "client_id": u.client_id,
                    "num_samples": u.num_samples,
                    "loss": u.train_result.loss,
                }
                for u in [e.update for e in live_events]  # type: ignore[misc]
            ],
        )

        elapsed = time.perf_counter() - t_start

        # ---- Governance audit event ----
        if self._audit_log is not None:
            self._audit_log.record(AuditEvent(
                round_num=round_num,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                selected_clients=selected_ids,
                active_clients=active_client_ids,
                dropped_clients=dropped_ids,
                pre_round_model_hash=pre_round_hash,
                post_round_model_hash=post_round_hash,
                global_accuracy=round(global_eval["accuracy"], 6),
                global_loss=round(global_eval["loss"], 6),
                num_clients_clipped=num_clipped,
                dp_epsilon_spent=_dp_epsilon,
                dp_delta_spent=_dp_delta,
                elapsed_seconds=round(elapsed, 4),
                skipped=False,
            ))

        logger.info(
            "Round %d/%s | acc=%.4f | loss=%.4f | updates=%d (stale_used=%d, discarded=%d) | pending=%d | v%d | %.1fs",
            round_num,
            self.config.training.num_rounds,
            global_eval["accuracy"],
            global_eval["loss"],
            len(live_events),
            stale_used,
            stale_discarded,
            self._event_loop.pending_count,
            self._server_version,
            elapsed,
        )

        return AsyncRoundSummary(
            round_num=round_num,
            selected_clients=selected_ids,
            active_clients=active_client_ids,
            dropped_clients=dropped_ids,
            aggregation=agg_result,
            global_accuracy=global_eval["accuracy"],
            global_loss=global_eval["loss"],
            elapsed_seconds=elapsed,
            stale_updates_used=stale_used,
            stale_updates_discarded=stale_discarded,
            pending_updates=self._event_loop.pending_count,
        )
