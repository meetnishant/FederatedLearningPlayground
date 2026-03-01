"""YAML-driven experiment configuration with Pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ClientConfig(BaseModel):
    """Per-client training hyperparameters."""

    batch_size: int = Field(32, ge=1, description="Local mini-batch size.")
    local_epochs: int = Field(1, ge=1, description="Local training epochs per round.")
    lr: float = Field(0.01, gt=0, description="SGD learning rate.")
    momentum: float = Field(0.9, ge=0, le=1, description="SGD momentum.")
    weight_decay: float = Field(1e-4, ge=0, description="L2 regularisation.")


class TrainingConfig(BaseModel):
    """Global federated training parameters."""

    num_rounds: int = Field(10, ge=1, description="Total federated rounds.")
    num_clients: int = Field(10, ge=2, description="Number of simulated clients.")
    client_fraction: float = Field(
        0.5, gt=0, le=1, description="Fraction of clients selected per round."
    )


class SimulationConfig(BaseModel):
    """Simulation layer configuration."""

    partitioning: Literal["iid", "dirichlet", "shard"] = "dirichlet"
    alpha: float = Field(0.5, gt=0, description="Dirichlet concentration parameter.")
    num_shards_per_client: int = Field(2, ge=1)
    dropout_rate: float = Field(0.0, ge=0, lt=1, description="Per-client dropout probability.")


class PrivacyConfig(BaseModel):
    """Differential privacy configuration."""

    enabled: bool = False
    epsilon: float = Field(1.0, gt=0)
    delta: float = Field(1e-5, gt=0, lt=1)
    max_grad_norm: float = Field(1.0, gt=0, description="L2 clipping norm.")


class OutputConfig(BaseModel):
    """Output paths for experiment artefacts."""

    dir: str = "outputs"
    save_plots: bool = True
    save_metrics: bool = True
    save_model: bool = False


class ExperimentConfig(BaseModel):
    """Root configuration for an FLP experiment."""

    name: str = "unnamed_experiment"
    seed: int = 42
    dataset: Literal["mnist"] = "mnist"
    data_dir: str = "~/.flp/data"

    training: TrainingConfig = Field(default_factory=TrainingConfig)
    client: ClientConfig = Field(default_factory=ClientConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("seed")
    @classmethod
    def seed_must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("seed must be >= 0.")
        return v


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated :class:`ExperimentConfig` instance.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the config fails Pydantic validation.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig.model_validate(raw or {})
