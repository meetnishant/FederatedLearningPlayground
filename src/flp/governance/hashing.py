"""Deterministic SHA-256 hashing of model state dicts and experiment configs."""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any

import torch


def hash_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """Compute a deterministic SHA-256 hash of a model state dict.

    Tensors are serialised in **sorted key order** via :func:`torch.save` into
    a bytes buffer, then SHA-256 hashed.  The resulting digest uniquely
    identifies the exact floating-point parameter values of a model.

    Sorting by key guarantees that two state dicts built from the same
    architecture but with differently ordered insertion histories produce the
    same hash.

    Args:
        state_dict: Model state dict (key → tensor).

    Returns:
        Hex digest string, prefixed with ``"sha256:"``.
    """
    hasher = hashlib.sha256()
    buf = io.BytesIO()
    ordered = {k: state_dict[k] for k in sorted(state_dict)}
    torch.save(ordered, buf)
    hasher.update(buf.getvalue())
    return f"sha256:{hasher.hexdigest()}"


def hash_config(config_dict: dict[str, Any]) -> str:
    """Compute a SHA-256 hash of a JSON-serialisable config snapshot.

    The dict is serialised with sorted keys so that insertion-order differences
    between two equivalent configs do not affect the digest.

    Args:
        config_dict: Config as a plain Python dict (must be JSON-serialisable).

    Returns:
        Hex digest string, prefixed with ``"sha256:"``.
    """
    canonical = json.dumps(config_dict, sort_keys=True, ensure_ascii=True)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"
