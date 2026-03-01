"""Aggregation strategies for combining client model updates."""

from __future__ import annotations

import torch


class FedAvgAggregator:
    """Federated Averaging (FedAvg) aggregation.

    Computes a weighted average of client model state dicts, where weights
    are proportional to each client's number of training samples.

    Reference:
        McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data", AISTATS 2017.
    """

    def aggregate(
        self,
        updates: list[dict[str, object]],
    ) -> dict[str, torch.Tensor]:
        """Aggregate model updates using weighted averaging.

        Args:
            updates: List of client update dicts, each containing:
                - ``state_dict``: client model weights (``dict[str, torch.Tensor]``)
                - ``num_samples``: number of training samples used

        Returns:
            Aggregated global model state dict.

        Raises:
            ValueError: If ``updates`` is empty.
        """
        if not updates:
            raise ValueError("Cannot aggregate: received empty update list.")

        total_samples: int = sum(int(u["num_samples"]) for u in updates)  # type: ignore[arg-type]
        if total_samples == 0:
            raise ValueError("Cannot aggregate: total sample count is zero.")

        # Initialise aggregated state with zeros mirroring the first client's shapes
        first_state: dict[str, torch.Tensor] = updates[0]["state_dict"]  # type: ignore[assignment]
        aggregated: dict[str, torch.Tensor] = {
            key: torch.zeros_like(param, dtype=torch.float32)
            for key, param in first_state.items()
        }

        for update in updates:
            state_dict: dict[str, torch.Tensor] = update["state_dict"]  # type: ignore[assignment]
            weight = int(update["num_samples"]) / total_samples  # type: ignore[arg-type]
            for key, param in state_dict.items():
                aggregated[key] += param.float() * weight

        return aggregated
