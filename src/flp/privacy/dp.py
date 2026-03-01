"""Gaussian mechanism and privacy accountant for DP-FedAvg."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Calibration utilities
# ---------------------------------------------------------------------------


def compute_noise_multiplier(epsilon: float, delta: float) -> float:
    """Compute the Gaussian noise multiplier for (epsilon, delta)-DP.

    Uses the analytic calibration formula for the Gaussian mechanism:

        sigma = sqrt(2 * ln(1.25 / delta)) / epsilon

    The actual per-element noise standard deviation applied to an update with
    sensitivity C is then ``sigma * C``.

    Args:
        epsilon: Privacy budget (must be > 0).
        delta: Failure probability (must be in (0, 1)).

    Returns:
        Noise multiplier ``sigma`` (dimensionless).

    Raises:
        ValueError: If constraints on epsilon or delta are violated.

    References:
        Dwork & Roth, "The Algorithmic Foundations of Differential Privacy",
        Foundations and Trends in TCS, 2014.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    return math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


# ---------------------------------------------------------------------------
# Per-round record
# ---------------------------------------------------------------------------


@dataclass
class DPRoundRecord:
    """Privacy cost and clipping statistics for one federated round.

    Attributes:
        round_num: 1-based round index.
        num_clients_total: Active clients that contributed updates.
        num_clients_clipped: Clients whose update norm exceeded the clip bound.
        noise_std: Absolute noise standard deviation added to the aggregated
            update (sigma * clip_norm / num_clients for averaged aggregates).
        clip_norm: L2 clipping norm (sensitivity) used this round.
        epsilon_spent: Privacy budget consumed this round.
        delta_spent: Delta consumed this round.
    """

    round_num: int
    num_clients_total: int
    num_clients_clipped: int
    noise_std: float
    clip_norm: float
    epsilon_spent: float
    delta_spent: float

    @property
    def clip_fraction(self) -> float:
        """Fraction of clients whose updates were clipped."""
        return (
            self.num_clients_clipped / self.num_clients_total
            if self.num_clients_total > 0
            else 0.0
        )


# ---------------------------------------------------------------------------
# Privacy accountant
# ---------------------------------------------------------------------------


class DPAccountant:
    """Accumulates privacy cost across federated rounds (basic composition).

    Under sequential composition, after T rounds the total privacy loss is
    at most (T*ε, T*δ).  This is a conservative upper bound; a moments
    accountant (Rényi DP) would give tighter bounds at the cost of complexity.

    Args:
        epsilon_per_round: Per-round privacy budget.
        delta_per_round: Per-round failure probability.
    """

    def __init__(self, epsilon_per_round: float, delta_per_round: float) -> None:
        self.epsilon_per_round = epsilon_per_round
        self.delta_per_round = delta_per_round
        self._records: list[DPRoundRecord] = []

    def record_round(self, record: DPRoundRecord) -> None:
        """Log one completed DP round.

        Args:
            record: Statistics from :class:`GaussianMechanism` application.
        """
        self._records.append(record)

    @property
    def records(self) -> list[DPRoundRecord]:
        """All per-round DP records in order."""
        return list(self._records)

    @property
    def total_epsilon(self) -> float:
        """Cumulative epsilon under basic sequential composition."""
        return sum(r.epsilon_spent for r in self._records)

    @property
    def total_delta(self) -> float:
        """Cumulative delta under basic sequential composition."""
        return sum(r.delta_spent for r in self._records)

    @property
    def total_clipped(self) -> int:
        """Total number of client updates clipped across all rounds."""
        return sum(r.num_clients_clipped for r in self._records)

    @property
    def avg_clip_fraction(self) -> float:
        """Mean fraction of clients clipped per round."""
        if not self._records:
            return 0.0
        return sum(r.clip_fraction for r in self._records) / len(self._records)

    def summary(self) -> dict[str, object]:
        """Return a JSON-serialisable privacy cost summary.

        Returns:
            Dict with keys: ``num_rounds``, ``total_epsilon``, ``total_delta``,
            ``epsilon_per_round``, ``delta_per_round``, ``total_clients_clipped``,
            ``avg_clip_fraction``.
        """
        return {
            "num_rounds": len(self._records),
            "total_epsilon": round(self.total_epsilon, 8),
            "total_delta": self.total_delta,
            "epsilon_per_round": self.epsilon_per_round,
            "delta_per_round": self.delta_per_round,
            "total_clients_clipped": self.total_clipped,
            "avg_clip_fraction": round(self.avg_clip_fraction, 4),
        }


# ---------------------------------------------------------------------------
# Gaussian mechanism
# ---------------------------------------------------------------------------


class GaussianMechanism:
    """Gaussian mechanism for (ε, δ)-differential privacy in DP-FedAvg.

    Implements the standard DP-FedAvg approach (Geyer et al., 2017):

    1. Each client's model update is **clipped** to L2 norm ≤ ``clip_norm``
       (this is handled in the server via :func:`~flp.privacy.clipping.clip_model_update`).
    2. The **aggregated** update has calibrated Gaussian noise added:
       ``N(0, σ²I)`` where ``σ = noise_multiplier × clip_norm``.

    Noise is applied to the *averaged* aggregate (as FedAvg outputs), so the
    effective noise per element is scaled by ``1 / num_clients`` to preserve
    the correct (ε, δ) guarantee.

    The noise multiplier is auto-computed as:

        sigma = sqrt(2 * ln(1.25 / delta)) / epsilon

    or can be overridden directly via ``noise_multiplier``.

    Args:
        epsilon: Per-round privacy budget (must be > 0).
        delta: Per-round failure probability (must be in (0, 1)).
        clip_norm: L2 sensitivity / clipping norm (must be > 0).
        noise_multiplier: Override the auto-computed noise multiplier.
            When provided, ``sigma = noise_multiplier * clip_norm``.
        seed: RNG seed for reproducible noise generation.

    Attributes:
        sigma: Absolute noise std = noise_multiplier * clip_norm.
        noise_std: Alias for ``sigma``.
        noise_multiplier: The dimensionless noise scale.

    References:
        - Dwork & Roth (2014), "Algorithmic Foundations of Differential Privacy"
        - Geyer et al. (2017), "Differentially Private Federated Learning:
          A Client Level Perspective"
        - Abadi et al. (2016), "Deep Learning with Differential Privacy"
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        clip_norm: float,
        noise_multiplier: float | None = None,
        seed: int = 42,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}.")
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}.")
        if clip_norm <= 0:
            raise ValueError(f"clip_norm must be > 0, got {clip_norm}.")

        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.seed = seed

        self._noise_multiplier: float = (
            noise_multiplier
            if noise_multiplier is not None
            else compute_noise_multiplier(epsilon, delta)
        )
        self.sigma: float = self._noise_multiplier * clip_norm

        # Isolated RNG — does not consume global torch random state.
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def noise_multiplier(self) -> float:
        """Dimensionless noise scale: sigma / clip_norm."""
        return self._noise_multiplier

    @property
    def noise_std(self) -> float:
        """Absolute noise standard deviation added per model element."""
        return self.sigma

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    def add_noise(
        self,
        state_dict: dict[str, torch.Tensor],
        num_clients: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Add calibrated Gaussian noise to an aggregated model state dict.

        Noise is added only to floating-point tensors; integer buffers such
        as BatchNorm ``num_batches_tracked`` are left unchanged.

        Because FedAvg produces a *weighted average* (not a sum), the noise
        std is divided by ``num_clients`` so the final (ε, δ) guarantee is
        maintained for the averaged update:

            effective_std = sigma / num_clients

        Args:
            state_dict: Aggregated model state dict (CPU float tensors).
            num_clients: Number of clients that contributed updates.
                Used to scale noise for averaged aggregates.

        Returns:
            New state dict with Gaussian noise added to floating-point values.
        """
        effective_sigma = self.sigma / max(num_clients, 1)
        noised: dict[str, torch.Tensor] = {}
        for key, param in state_dict.items():
            if param.is_floating_point():
                noise = torch.zeros_like(param, dtype=torch.float32)
                noise.normal_(mean=0.0, std=effective_sigma, generator=self._rng)
                noised[key] = param.float() + noise
            else:
                noised[key] = param.clone()
        return noised

    def privatize(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Add noise with ``num_clients=1`` (legacy single-model interface).

        Args:
            state_dict: Model state dict to privatize.

        Returns:
            Noised copy of the state dict.
        """
        return self.add_noise(state_dict, num_clients=1)

    def make_round_record(
        self,
        round_num: int,
        num_clients_total: int,
        num_clients_clipped: int,
    ) -> DPRoundRecord:
        """Construct a :class:`DPRoundRecord` for the current round.

        Args:
            round_num: 1-based round index.
            num_clients_total: Total active clients this round.
            num_clients_clipped: Clients whose updates were clipped.

        Returns:
            Populated :class:`DPRoundRecord`.
        """
        effective_sigma = self.sigma / max(num_clients_total, 1)
        return DPRoundRecord(
            round_num=round_num,
            num_clients_total=num_clients_total,
            num_clients_clipped=num_clients_clipped,
            noise_std=effective_sigma,
            clip_norm=self.clip_norm,
            epsilon_spent=self.epsilon,
            delta_spent=self.delta,
        )
