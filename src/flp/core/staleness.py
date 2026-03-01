"""Staleness-aware weight computation for async federated aggregation."""

from __future__ import annotations

from typing import Literal

StalenessStrategy = Literal["uniform", "inverse_staleness", "exponential_decay"]


class StalenessWeighter:
    """Computes per-update aggregation weights that account for update staleness.

    Combines data-quantity weights (``num_samples``) with a staleness penalty
    so that fresher updates have greater influence on the aggregated model.

    Strategies:

    - **uniform**: Ignores staleness; weights are proportional to
      ``num_samples`` only (identical to standard FedAvg).
    - **inverse_staleness**: Staleness penalty = ``1 / (1 + staleness)``.
      An update that is 0 rounds stale receives full weight; one that is
      3 rounds stale receives weight ``1/4`` relative to a fresh update
      of equal data size.
    - **exponential_decay**: Staleness penalty = ``decay_factor ** staleness``.
      With ``decay_factor=0.9``, each additional stale round reduces an
      update's effective influence by 10%.

    In all strategies the returned weights are normalised to sum to 1.0 so
    they can be used as direct aggregation coefficients.

    Args:
        strategy: Weighting strategy.  One of ``"uniform"``,
            ``"inverse_staleness"``, ``"exponential_decay"``.
        decay_factor: Base for exponential decay.  Only used when
            ``strategy="exponential_decay"``.  Must be in ``(0, 1]``.
            ``decay_factor=1.0`` is equivalent to ``"uniform"``.
    """

    def __init__(
        self,
        strategy: StalenessStrategy = "uniform",
        decay_factor: float = 0.9,
    ) -> None:
        if strategy not in ("uniform", "inverse_staleness", "exponential_decay"):
            raise ValueError(
                f"Unknown staleness strategy {strategy!r}. "
                "Choose from 'uniform', 'inverse_staleness', 'exponential_decay'."
            )
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError(
                f"decay_factor must be in (0, 1], got {decay_factor}."
            )
        self.strategy: StalenessStrategy = strategy
        self.decay_factor: float = decay_factor

    def compute_weights(
        self,
        staleness_values: list[int],
        num_samples: list[int],
    ) -> list[float]:
        """Compute normalised per-update aggregation weights.

        Args:
            staleness_values: Staleness of each update, defined as
                ``current_server_version − model_version_used_by_client``.
                Must be ``>= 0``.
            num_samples: Number of training samples for each update, in the
                same order as ``staleness_values``.

        Returns:
            List of floats that sum to 1.0, one weight per update.  Returns
            an empty list when both inputs are empty.

        Raises:
            ValueError: If the lists have different lengths, any
                ``num_samples`` entry is negative, any staleness value is
                negative, or all combined weights are zero (e.g. all
                ``num_samples`` are 0).
        """
        if len(staleness_values) != len(num_samples):
            raise ValueError(
                f"staleness_values and num_samples must have the same length; "
                f"got {len(staleness_values)} and {len(num_samples)}."
            )
        if not staleness_values:
            return []
        if any(ns < 0 for ns in num_samples):
            raise ValueError("All num_samples values must be >= 0.")
        if any(s < 0 for s in staleness_values):
            raise ValueError("All staleness values must be >= 0.")

        # Compute raw (unnormalised) combined weights.
        raw: list[float]
        if self.strategy == "uniform":
            raw = [float(ns) for ns in num_samples]
        elif self.strategy == "inverse_staleness":
            raw = [ns / (1.0 + s) for ns, s in zip(num_samples, staleness_values)]
        else:  # exponential_decay
            raw = [ns * (self.decay_factor ** s) for ns, s in zip(num_samples, staleness_values)]

        total = sum(raw)
        if total == 0.0:
            raise ValueError(
                "All combined weights are zero — cannot normalise. "
                "Check that at least one update has num_samples > 0."
            )
        return [w / total for w in raw]
