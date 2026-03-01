"""Differential privacy mechanisms for federated learning."""

from __future__ import annotations

import torch


class GaussianMechanism:
    """Add calibrated Gaussian noise to model updates for (epsilon, delta)-DP.

    The noise standard deviation is computed as:

        sigma = (sensitivity * sqrt(2 * ln(1.25 / delta))) / epsilon

    where ``sensitivity`` corresponds to the L2 clipping norm of the update.

    Args:
        epsilon: Privacy budget (smaller = stronger privacy).
        delta: Probability of privacy failure.
        sensitivity: L2 sensitivity of the update (typically equals clip norm).
        seed: Random seed for reproducibility.

    References:
        Dwork & Roth, "The Algorithmic Foundations of Differential Privacy", 2014.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float,
        seed: int = 42,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}.")
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.seed = seed

        import math

        self.sigma = (sensitivity * math.sqrt(2.0 * math.log(1.25 / delta))) / epsilon
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def privatize(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Add Gaussian noise to every parameter tensor in ``state_dict``.

        Args:
            state_dict: Model state dict to privatize.

        Returns:
            Noised copy of the state dict.
        """
        noised: dict[str, torch.Tensor] = {}
        for key, param in state_dict.items():
            noise = torch.normal(
                mean=0.0,
                std=self.sigma,
                size=param.shape,
                generator=self._rng,
                dtype=torch.float32,
            )
            noised[key] = param.float() + noise
        return noised

    @property
    def noise_std(self) -> float:
        """Computed Gaussian noise standard deviation."""
        return self.sigma
