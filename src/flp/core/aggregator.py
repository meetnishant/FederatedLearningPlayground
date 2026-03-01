"""FedAvg aggregation: weighted averaging of client model updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flp.core.client import ClientUpdate


@dataclass(frozen=True)
class AggregationResult:
    """Output of one FedAvg aggregation step.

    Attributes:
        state_dict: New global model state dict (float32 parameters; integer
            buffers such as BatchNorm ``num_batches_tracked`` are taken from
            the client with the most training samples).
        total_samples: Sum of all participating clients' sample counts.
        weighted_loss: Sample-weighted mean training loss across clients.
        num_clients: Number of clients included in this aggregation.
    """

    state_dict: dict[str, torch.Tensor]
    total_samples: int
    weighted_loss: float
    num_clients: int


class FedAvgAggregator:
    """Federated Averaging (FedAvg) aggregation strategy.

    Computes a weighted average of client model updates where each client's
    contribution is proportional to its number of training samples.

    This is the canonical aggregation rule from:

        McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data", AISTATS 2017.

    Design notes:

    - **Floating-point tensors** (weights, biases, BN running stats):
      aggregated via weighted mean.
    - **Integer tensors** (e.g. ``BatchNorm.num_batches_tracked``):
      taken from the client with the largest sample count to avoid
      dtype-conversion errors and preserve correct BN statistics.
    - All arithmetic is performed in float32 and cast back to the original
      dtype before returning, so the state dict is always compatible with
      ``model.load_state_dict()``.
    """

    def aggregate(
        self,
        updates: list[ClientUpdate],
        weights: list[float] | None = None,
    ) -> AggregationResult:
        """Aggregate client updates into a new global model state dict.

        Args:
            updates: Non-empty list of :class:`~flp.core.client.ClientUpdate`
                objects, one per participating client.
            weights: Optional explicit per-update aggregation weights that must
                sum to 1.0, one per entry in ``updates``.  When provided these
                replace the default sample-count-proportional weights, allowing
                staleness-aware or other custom aggregation schemes.  When
                ``None`` (default) the standard FedAvg rule is used:
                ``weight_i = num_samples_i / total_samples``.

        Returns:
            :class:`AggregationResult` with the aggregated state dict and
            summary statistics.

        Raises:
            ValueError: If ``updates`` is empty, all sample counts are zero,
                or ``weights`` has a different length than ``updates``.
        """
        if not updates:
            raise ValueError(
                "FedAvgAggregator.aggregate() received an empty update list. "
                "At least one client must participate in each round."
            )

        total_samples: int = sum(u.num_samples for u in updates)
        if total_samples == 0:
            raise ValueError(
                "All client updates have num_samples=0. "
                "Check that clients have non-empty data partitions."
            )

        # Resolve per-update weights.
        if weights is not None:
            if len(weights) != len(updates):
                raise ValueError(
                    f"weights has {len(weights)} entries but updates has "
                    f"{len(updates)}. They must match."
                )
            effective_weights = weights
        else:
            effective_weights = [u.num_samples / total_samples for u in updates]

        # Client whose state dict is used as the reference for non-float tensors.
        # Always based on num_samples regardless of custom weights.
        reference_update = max(updates, key=lambda u: u.num_samples)
        ref_state = reference_update.state_dict

        # Initialise accumulation buffers.
        # Float tensors → zeroed; integer tensors → cloned from reference.
        aggregated: dict[str, torch.Tensor] = {}
        for key, param in ref_state.items():
            if param.is_floating_point():
                aggregated[key] = torch.zeros_like(param, dtype=torch.float64)
            else:
                # Integer buffers (e.g. num_batches_tracked): use reference.
                aggregated[key] = param.clone()

        # Weighted accumulation over float tensors only.
        for update, w in zip(updates, effective_weights):
            for key, param in update.state_dict.items():
                if param.is_floating_point():
                    aggregated[key] += param.double() * w

        # Cast float64 accumulators back to the original dtype of each tensor.
        final_state: dict[str, torch.Tensor] = {}
        for key, param in aggregated.items():
            if param.is_floating_point():
                original_dtype = ref_state[key].dtype
                final_state[key] = param.to(dtype=original_dtype)
            else:
                final_state[key] = param

        # Weighted training loss across all clients.
        weighted_loss = sum(
            u.train_result.loss * w
            for u, w in zip(updates, effective_weights)
        )

        return AggregationResult(
            state_dict=final_state,
            total_samples=total_samples,
            weighted_loss=weighted_loss,
            num_clients=len(updates),
        )
