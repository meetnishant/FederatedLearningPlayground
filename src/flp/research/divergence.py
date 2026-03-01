"""Weight divergence metrics for federated learning research.

Measures how far client model weights deviate from the global model after
local training.  High divergence indicates strong data heterogeneity or
aggressive local optimisation — a key signal for understanding non-IID
effects and client drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DivergenceResult:
    """Weight divergence of client models from the global model.

    Attributes:
        per_client_l2: L2 norm of ``(client_weights − global_weights)`` for
            each participating client.  Keys are client IDs.
        mean_divergence: Mean of all per-client L2 norms.
        max_divergence: Maximum per-client L2 norm (worst-case client drift).
    """

    per_client_l2: dict[int, float]
    mean_divergence: float
    max_divergence: float


def compute_weight_divergence(
    global_state: dict[str, torch.Tensor],
    client_states: dict[int, dict[str, torch.Tensor]],
) -> DivergenceResult:
    """Measure how far each client's model deviates from the global model.

    Computes the L2 norm of the update delta (client − global) for each
    client, considering only floating-point parameters.  Integer buffers such
    as BatchNorm ``num_batches_tracked`` are excluded.

    This is equivalent to calling :func:`~flp.privacy.clipping.compute_update_norm`
    for each client — but returns a structured result with aggregate statistics.

    Args:
        global_state: Global model state dict (the broadcast checkpoint that
            clients started from).
        client_states: Mapping from client ID to that client's post-training
            state dict.

    Returns:
        :class:`DivergenceResult` with per-client L2 norms and aggregates.
        Returns zero for all fields if ``client_states`` is empty.

    Raises:
        ValueError: If a client's state dict has keys that do not match
            ``global_state``.
    """
    if not client_states:
        return DivergenceResult(per_client_l2={}, mean_divergence=0.0, max_divergence=0.0)

    per_client_l2: dict[int, float] = {}

    for client_id, client_state in client_states.items():
        missing = set(global_state.keys()) - set(client_state.keys())
        if missing:
            raise ValueError(
                f"Client {client_id} state dict is missing keys: {missing}"
            )
        norm = _l2_norm_delta(global_state, client_state)
        per_client_l2[client_id] = norm

    norms = list(per_client_l2.values())
    mean_div = sum(norms) / len(norms)
    max_div = max(norms)

    return DivergenceResult(
        per_client_l2=per_client_l2,
        mean_divergence=mean_div,
        max_divergence=max_div,
    )


def cosine_similarity_between_updates(
    update_a: dict[str, torch.Tensor],
    update_b: dict[str, torch.Tensor],
) -> float:
    """Compute cosine similarity between two model update vectors.

    Flattens all floating-point tensors from both updates into a single
    vector each, then returns their cosine similarity.  Values close to 1
    indicate updates pointing in the same direction (clients agree); values
    near 0 indicate orthogonal updates; values near -1 indicate conflicting
    gradients.

    Args:
        update_a: First model state dict (e.g. a client update delta).
        update_b: Second model state dict.  Must share the same float keys
            as ``update_a``.

    Returns:
        Cosine similarity in ``[−1, 1]``.  Returns 0.0 if either vector has
        zero norm.
    """
    flat_a = _flatten_floats(update_a)
    flat_b = _flatten_floats(update_b)

    norm_a = flat_a.norm(p=2).item()
    norm_b = flat_b.norm(p=2).item()

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return (flat_a @ flat_b / (norm_a * norm_b)).item()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2_norm_delta(
    original: dict[str, torch.Tensor],
    updated: dict[str, torch.Tensor],
) -> float:
    """L2 norm of the update delta (float tensors only)."""
    deltas = [
        (updated[k].float() - original[k].float()).flatten()
        for k in original
        if original[k].is_floating_point() and k in updated
    ]
    if not deltas:
        return 0.0
    return torch.cat(deltas).norm(p=2).item()


def _flatten_floats(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate all floating-point tensors into a single 1-D vector."""
    parts = [
        t.float().flatten()
        for t in state_dict.values()
        if t.is_floating_point()
    ]
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)
