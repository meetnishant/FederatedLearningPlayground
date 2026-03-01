"""YAML-driven experiment configuration with Pydantic v2 validation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import AfterValidator


# ---------------------------------------------------------------------------
# Custom annotated types with embedded error messages
# ---------------------------------------------------------------------------

def _positive_float(v: float) -> float:
    if v <= 0:
        raise ValueError(f"must be > 0, got {v}")
    return v


def _unit_fraction(v: float) -> float:
    if not 0.0 < v <= 1.0:
        raise ValueError(f"must be in (0, 1], got {v}")
    return v


PositiveFloat = Annotated[float, AfterValidator(_positive_float)]
UnitFraction = Annotated[float, AfterValidator(_unit_fraction)]


# ---------------------------------------------------------------------------
# ClientConfig
# ---------------------------------------------------------------------------


class ClientConfig(BaseModel):
    """Per-client local training hyperparameters.

    All fields are optional; defaults produce a sensible MNIST baseline.
    """

    batch_size: int = Field(
        32,
        ge=1,
        description="Local mini-batch size. Must be >= 1.",
    )
    local_epochs: int = Field(
        1,
        ge=1,
        description="Number of full passes over local data per round. Must be >= 1.",
    )
    lr: PositiveFloat = Field(
        0.01,
        description="SGD learning rate. Must be > 0.",
    )
    momentum: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="SGD momentum coefficient. Must be in [0, 1].",
    )
    weight_decay: float = Field(
        1e-4,
        ge=0.0,
        description="L2 regularisation coefficient. Must be >= 0.",
    )

    @field_validator("batch_size")
    @classmethod
    def batch_size_is_power_of_two_hint(cls, v: int) -> int:
        if v & (v - 1) != 0:
            import warnings
            warnings.warn(
                f"client.batch_size={v} is not a power of two. "
                "Power-of-two sizes are typically more efficient on GPU.",
                stacklevel=2,
            )
        return v


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TrainingConfig(BaseModel):
    """Global federated training schedule."""

    num_rounds: int = Field(
        10,
        ge=1,
        description="Total number of federated communication rounds. Must be >= 1.",
    )
    num_clients: int = Field(
        10,
        ge=2,
        description="Total number of simulated clients. Must be >= 2.",
    )
    client_fraction: UnitFraction = Field(
        0.5,
        description=(
            "Fraction of clients randomly selected each round. "
            "Must be in (0, 1]. At least one client is always selected."
        ),
    )

    @model_validator(mode="after")
    def at_least_one_client_selected(self) -> "TrainingConfig":
        selected = max(1, int(self.num_clients * self.client_fraction))
        if selected < 1:
            raise ValueError(
                f"training.client_fraction={self.client_fraction} with "
                f"training.num_clients={self.num_clients} results in 0 clients "
                "selected per round. Increase client_fraction or num_clients."
            )
        return self


# ---------------------------------------------------------------------------
# SimulationConfig
# ---------------------------------------------------------------------------

PartitioningStrategy = Literal["iid", "dirichlet", "shard"]


class SimulationConfig(BaseModel):
    """Data heterogeneity and fault simulation settings."""

    partitioning: PartitioningStrategy = Field(
        "dirichlet",
        description=(
            "Data partitioning strategy across clients. "
            "Options: 'iid' (uniform random), 'dirichlet' (non-IID via Dirichlet "
            "distribution), 'shard' (deterministic class-shard split)."
        ),
    )
    alpha: PositiveFloat = Field(
        0.5,
        description=(
            "Dirichlet concentration parameter. Only used when partitioning='dirichlet'. "
            "Smaller values (e.g. 0.1) produce extreme non-IID distributions; "
            "larger values (e.g. 10.0) approach IID. Must be > 0."
        ),
    )
    num_shards_per_client: int = Field(
        2,
        ge=1,
        description=(
            "Number of class shards assigned to each client. "
            "Only used when partitioning='shard'. Must be >= 1."
        ),
    )
    dropout_rate: float = Field(
        0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "Probability that each selected client drops out before uploading "
            "its update. Must be in [0, 1). Set to 0 to disable dropout."
        ),
    )

    @model_validator(mode="after")
    def warn_alpha_ignored_for_non_dirichlet(self) -> "SimulationConfig":
        if self.partitioning != "dirichlet" and self.alpha != 0.5:
            import warnings
            warnings.warn(
                f"simulation.alpha={self.alpha} is set but partitioning="
                f"'{self.partitioning}'. alpha is only used with 'dirichlet' "
                "and will be ignored.",
                stacklevel=2,
            )
        return self


# ---------------------------------------------------------------------------
# PrivacyConfig
# ---------------------------------------------------------------------------


class PrivacyConfig(BaseModel):
    """(ε, δ)-differential privacy via the Gaussian mechanism."""

    enabled: bool = Field(
        False,
        description="Enable DP-FedAvg. When False all other privacy fields are ignored.",
    )
    epsilon: PositiveFloat = Field(
        1.0,
        description=(
            "Privacy budget ε. Smaller values provide stronger privacy guarantees "
            "at the cost of accuracy. Must be > 0."
        ),
    )
    delta: float = Field(
        1e-5,
        gt=0.0,
        lt=1.0,
        description=(
            "Probability of privacy failure δ. Should be smaller than 1/n where "
            "n is the training set size. Must be in (0, 1)."
        ),
    )
    max_grad_norm: PositiveFloat = Field(
        1.0,
        description=(
            "Maximum L2 norm for per-client update clipping. "
            "Also used as the sensitivity for noise calibration. Must be > 0."
        ),
    )

    @model_validator(mode="after")
    def epsilon_delta_sanity(self) -> "PrivacyConfig":
        if not self.enabled:
            return self
        if self.epsilon > 10.0:
            import warnings
            warnings.warn(
                f"privacy.epsilon={self.epsilon} is very large (>10). "
                "This provides only weak privacy guarantees. "
                "Consider using epsilon <= 10 for meaningful DP.",
                stacklevel=2,
            )
        if self.delta > 1e-3:
            import warnings
            warnings.warn(
                f"privacy.delta={self.delta} is large (>1e-3). "
                "Typical values are 1e-5 or smaller.",
                stacklevel=2,
            )
        return self


# ---------------------------------------------------------------------------
# OutputConfig
# ---------------------------------------------------------------------------


class OutputConfig(BaseModel):
    """Paths and flags controlling experiment output artefacts."""

    dir: str = Field(
        "outputs",
        description=(
            "Root directory for all experiment outputs. A sub-directory named "
            "after the experiment will be created inside this path."
        ),
    )
    save_plots: bool = Field(True, description="Save matplotlib PNG plots after training.")
    save_metrics: bool = Field(True, description="Save per-round metrics to metrics.json.")
    save_model: bool = Field(False, description="Save the final global model state dict.")


# ---------------------------------------------------------------------------
# ExperimentConfig (root)
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Root configuration model for an FLP experiment.

    Maps 1-to-1 with the top-level keys in a YAML experiment file.
    All sections are optional and fall back to sensible defaults.

    Example YAML::

        name: my_experiment
        seed: 42
        dataset: mnist
        training:
          num_rounds: 20
          num_clients: 10
        simulation:
          partitioning: dirichlet
          alpha: 0.3
    """

    name: str = Field(
        "unnamed_experiment",
        min_length=1,
        description="Human-readable experiment identifier. Used as the output sub-directory name.",
    )
    seed: int = Field(
        42,
        ge=0,
        description="Global random seed for reproducibility. Must be >= 0.",
    )
    dataset: Literal["mnist"] = Field(
        "mnist",
        description="Dataset to use. Currently only 'mnist' is supported.",
    )
    data_dir: str = Field(
        "~/.flp/data",
        description="Directory where datasets are downloaded and cached.",
    )

    training: TrainingConfig = Field(default_factory=TrainingConfig)
    client: ClientConfig = Field(default_factory=ClientConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("name")
    @classmethod
    def name_no_spaces(cls, v: str) -> str:
        if " " in v:
            raise ValueError(
                f"experiment name must not contain spaces, got '{v}'. "
                "Use underscores or hyphens instead (e.g. 'my_experiment')."
            )
        return v

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an FLP experiment config from a YAML file.

    Produces clear, human-readable error messages for every validation failure,
    including the offending field path and the constraint that was violated.

    Args:
        path: Path to the ``.yaml`` experiment config file.

    Returns:
        Fully validated :class:`ExperimentConfig` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigValidationError: If the YAML fails schema validation.
        yaml.YAMLError: If the file is not valid YAML.

    Example::

        cfg = load_config("configs/baseline.yaml")
        print(cfg.training.num_rounds)
    """
    config_path = Path(path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: '{config_path}'\n"
            f"Hint: check the path or run `flp validate-config <path>` to diagnose."
        )

    with open(config_path) as f:
        try:
            raw: Any = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(
                f"Failed to parse YAML in '{config_path}':\n{exc}"
            ) from exc

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise ConfigValidationError(
            f"Config file '{config_path}' must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__}."
        )

    from pydantic import ValidationError

    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError.from_pydantic(exc, config_path) from exc


class ConfigValidationError(ValueError):
    """Raised when an experiment config fails Pydantic schema validation.

    Formats all field errors into a single human-readable message so that
    users know exactly which YAML keys to fix and why.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)

    @classmethod
    def from_pydantic(
        cls,
        exc: "ValidationError",  # type: ignore[name-defined]  # noqa: F821
        config_path: Path,
    ) -> "ConfigValidationError":
        from pydantic import ValidationError

        assert isinstance(exc, ValidationError)

        lines = [f"Config validation failed for '{config_path}':"]
        for error in exc.errors():
            loc = " -> ".join(str(p) for p in error["loc"]) if error["loc"] else "<root>"
            msg = error["msg"]
            val = error.get("input", "<unknown>")
            hint = _validation_hint(error)
            lines.append(f"\n  Field : {loc}")
            lines.append(f"  Error : {msg}")
            if val != "<unknown>":
                lines.append(f"  Value : {val!r}")
            if hint:
                lines.append(f"  Hint  : {hint}")

        lines.append(
            "\nRun `flp validate-config <path>` for a structured summary."
        )
        return cls("\n".join(lines))


def _validation_hint(error: dict[str, Any]) -> str:
    """Return a context-specific hint string for a Pydantic error dict."""
    loc = ".".join(str(p) for p in error.get("loc", []))
    err_type = error.get("type", "")

    hints: dict[str, str] = {
        "training.num_rounds": "Try num_rounds: 10",
        "training.num_clients": "Must have at least 2 clients for federation to make sense.",
        "training.client_fraction": "Use a value like 0.5 (50% of clients selected per round).",
        "simulation.alpha": "Typical range: 0.05 (extreme non-IID) to 10.0 (near-IID).",
        "simulation.dropout_rate": "Use 0.0 to disable dropout, or a value like 0.1 for 10%.",
        "privacy.epsilon": "Common values: 0.1 (strong), 1.0 (moderate), 10.0 (weak).",
        "privacy.delta": "Recommended: 1e-5. Must be strictly between 0 and 1.",
        "client.lr": "Typical learning rates: 0.001 to 0.1.",
        "client.batch_size": "Common sizes: 16, 32, 64, 128.",
        "seed": "Use any non-negative integer, e.g. seed: 42.",
        "name": "Use snake_case or kebab-case, e.g. name: my_experiment.",
    }

    if loc in hints:
        return hints[loc]

    if err_type in ("literal_error", "enum"):
        return "Check the allowed values in the config schema."
    if err_type in ("missing",):
        return "This field is required and has no default."
    if "greater_than" in err_type or "less_than" in err_type:
        return "Check the numeric bounds for this field."

    return ""
