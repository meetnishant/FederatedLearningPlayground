"""Tests for YAML config loading and Pydantic validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from flp.experiments.config_loader import (
    ConfigValidationError,
    ExperimentConfig,
    load_config,
)


@pytest.fixture
def tmp_config(tmp_path: Path):
    """Write YAML content to a temp file and return its path."""

    def _write(content: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    return _write


# ---------------------------------------------------------------------------
# load_config — happy path
# ---------------------------------------------------------------------------


class TestLoadConfigHappyPath:
    def test_minimal_empty_config_uses_all_defaults(self, tmp_config) -> None:
        cfg = load_config(tmp_config("{}"))
        assert cfg.name == "unnamed_experiment"
        assert cfg.seed == 42
        assert cfg.dataset == "mnist"
        assert cfg.training.num_rounds == 10
        assert cfg.training.num_clients == 10
        assert cfg.training.client_fraction == 0.5
        assert cfg.client.batch_size == 32
        assert cfg.client.local_epochs == 1
        assert cfg.client.lr == 0.01
        assert cfg.client.momentum == 0.9
        assert cfg.client.weight_decay == 1e-4
        assert cfg.simulation.partitioning == "dirichlet"
        assert cfg.simulation.alpha == 0.5
        assert cfg.simulation.dropout_rate == 0.0
        assert cfg.privacy.enabled is False
        assert cfg.output.save_plots is True

    def test_explicit_values_override_defaults(self, tmp_config) -> None:
        path = tmp_config("""
            name: my_experiment
            seed: 7
            training:
              num_rounds: 5
              num_clients: 4
              client_fraction: 0.75
            client:
              batch_size: 64
              local_epochs: 3
              lr: 0.001
            simulation:
              partitioning: shard
              num_shards_per_client: 4
              dropout_rate: 0.2
            privacy:
              enabled: true
              epsilon: 2.0
              delta: 1.0e-6
              max_grad_norm: 0.5
            output:
              dir: /tmp/out
              save_plots: false
              save_model: true
        """)
        cfg = load_config(path)
        assert cfg.name == "my_experiment"
        assert cfg.seed == 7
        assert cfg.training.num_rounds == 5
        assert cfg.training.num_clients == 4
        assert cfg.training.client_fraction == 0.75
        assert cfg.client.batch_size == 64
        assert cfg.client.local_epochs == 3
        assert cfg.client.lr == 0.001
        assert cfg.simulation.partitioning == "shard"
        assert cfg.simulation.num_shards_per_client == 4
        assert cfg.simulation.dropout_rate == 0.2
        assert cfg.privacy.enabled is True
        assert cfg.privacy.epsilon == 2.0
        assert cfg.privacy.delta == pytest.approx(1e-6)
        assert cfg.privacy.max_grad_norm == 0.5
        assert cfg.output.dir == "/tmp/out"
        assert cfg.output.save_plots is False
        assert cfg.output.save_model is True

    def test_null_yaml_treated_as_empty_dict(self, tmp_config) -> None:
        cfg = load_config(tmp_config(""))
        assert isinstance(cfg, ExperimentConfig)

    def test_iid_partitioning_accepted(self, tmp_config) -> None:
        cfg = load_config(tmp_config("simulation:\n  partitioning: iid\n"))
        assert cfg.simulation.partitioning == "iid"

    def test_dirichlet_partitioning_accepted(self, tmp_config) -> None:
        cfg = load_config(tmp_config("simulation:\n  partitioning: dirichlet\n"))
        assert cfg.simulation.partitioning == "dirichlet"

    def test_shard_partitioning_accepted(self, tmp_config) -> None:
        cfg = load_config(tmp_config("simulation:\n  partitioning: shard\n"))
        assert cfg.simulation.partitioning == "shard"

    def test_loads_actual_baseline_yaml(self) -> None:
        baseline = Path(__file__).parent.parent / "configs" / "baseline.yaml"
        if not baseline.exists():
            pytest.skip("configs/baseline.yaml not found")
        cfg = load_config(baseline)
        assert cfg.name == "baseline_fedavg"
        assert cfg.seed == 42
        assert cfg.training.num_rounds == 20
        assert cfg.training.num_clients == 10
        assert cfg.simulation.partitioning == "dirichlet"
        assert cfg.privacy.enabled is False


# ---------------------------------------------------------------------------
# load_config — error handling
# ---------------------------------------------------------------------------


class TestLoadConfigErrors:
    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(tmp_path / "ghost.yaml")

    def test_raises_config_validation_error_on_bad_values(self, tmp_config) -> None:
        path = tmp_config("training:\n  num_rounds: -1\n")
        with pytest.raises(ConfigValidationError):
            load_config(path)

    def test_error_message_contains_field_path(self, tmp_config) -> None:
        path = tmp_config("training:\n  num_clients: 1\n")
        with pytest.raises(ConfigValidationError, match="num_clients"):
            load_config(path)

    def test_error_message_contains_bad_value(self, tmp_config) -> None:
        path = tmp_config("client:\n  lr: -0.5\n")
        with pytest.raises(ConfigValidationError, match="-0.5"):
            load_config(path)

    def test_unknown_top_level_key_rejected(self, tmp_config) -> None:
        path = tmp_config("totally_unknown_key: 999\n")
        with pytest.raises(ConfigValidationError):
            load_config(path)

    def test_invalid_yaml_raises_yaml_error(self, tmp_config) -> None:
        import yaml
        path = tmp_config("key: [unclosed bracket\n")
        with pytest.raises(yaml.YAMLError):
            load_config(path)

    def test_non_dict_yaml_raises_config_validation_error(self, tmp_config) -> None:
        path = tmp_config("- item1\n- item2\n")
        with pytest.raises(ConfigValidationError, match="mapping"):
            load_config(path)


# ---------------------------------------------------------------------------
# ClientConfig validation
# ---------------------------------------------------------------------------


class TestClientConfig:
    def test_rejects_zero_batch_size(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("client:\n  batch_size: 0\n"))

    def test_rejects_zero_local_epochs(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("client:\n  local_epochs: 0\n"))

    def test_rejects_negative_lr(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("client:\n  lr: -0.01\n"))

    def test_rejects_zero_lr(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("client:\n  lr: 0.0\n"))

    def test_rejects_momentum_above_one(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("client:\n  momentum: 1.5\n"))

    def test_rejects_negative_weight_decay(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("client:\n  weight_decay: -0.001\n"))


# ---------------------------------------------------------------------------
# TrainingConfig validation
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_rejects_zero_rounds(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("training:\n  num_rounds: 0\n"))

    def test_rejects_one_client(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("training:\n  num_clients: 1\n"))

    def test_rejects_zero_client_fraction(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("training:\n  client_fraction: 0.0\n"))

    def test_rejects_client_fraction_above_one(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("training:\n  client_fraction: 1.5\n"))


# ---------------------------------------------------------------------------
# SimulationConfig validation
# ---------------------------------------------------------------------------


class TestSimulationConfig:
    def test_rejects_unknown_partitioning(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("simulation:\n  partitioning: invalid\n"))

    def test_rejects_zero_alpha(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("simulation:\n  alpha: 0.0\n"))

    def test_rejects_negative_alpha(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("simulation:\n  alpha: -1.0\n"))

    def test_rejects_dropout_rate_of_one(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("simulation:\n  dropout_rate: 1.0\n"))

    def test_rejects_negative_dropout_rate(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("simulation:\n  dropout_rate: -0.1\n"))

    def test_rejects_zero_shards_per_client(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("simulation:\n  num_shards_per_client: 0\n"))


# ---------------------------------------------------------------------------
# PrivacyConfig validation
# ---------------------------------------------------------------------------


class TestPrivacyConfig:
    def test_rejects_zero_epsilon(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("privacy:\n  epsilon: 0.0\n"))

    def test_rejects_negative_epsilon(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("privacy:\n  epsilon: -1.0\n"))

    def test_rejects_delta_of_zero(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("privacy:\n  delta: 0.0\n"))

    def test_rejects_delta_of_one(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("privacy:\n  delta: 1.0\n"))

    def test_rejects_zero_max_grad_norm(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("privacy:\n  max_grad_norm: 0.0\n"))


# ---------------------------------------------------------------------------
# ExperimentConfig validation
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_rejects_name_with_spaces(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError, match="spaces"):
            load_config(tmp_config("name: has spaces\n"))

    def test_rejects_negative_seed(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("seed: -1\n"))

    def test_rejects_unknown_dataset(self, tmp_config) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(tmp_config("dataset: cifar10\n"))

    def test_direct_model_construction_also_validates(self) -> None:
        with pytest.raises(Exception):
            ExperimentConfig(seed=-5)
