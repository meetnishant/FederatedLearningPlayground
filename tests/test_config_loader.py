"""Tests for YAML config loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from flp.experiments.config_loader import ExperimentConfig, load_config


@pytest.fixture
def tmp_config(tmp_path: Path):
    """Write a minimal valid config to a temp file and return its path."""

    def _write(content: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    return _write


class TestLoadConfig:
    def test_loads_valid_minimal_config(self, tmp_config) -> None:
        path = tmp_config("name: test_exp\nseed: 7\n")
        cfg = load_config(path)
        assert cfg.name == "test_exp"
        assert cfg.seed == 7

    def test_defaults_are_applied(self, tmp_config) -> None:
        path = tmp_config("{}")
        cfg = load_config(path)
        assert cfg.training.num_rounds == 10
        assert cfg.training.num_clients == 10
        assert cfg.simulation.partitioning == "dirichlet"
        assert cfg.privacy.enabled is False

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_full_config(self, tmp_config) -> None:
        path = tmp_config(
            """
            name: full_test
            seed: 123
            dataset: mnist
            training:
              num_rounds: 5
              num_clients: 4
              client_fraction: 0.75
            client:
              batch_size: 64
              local_epochs: 2
              lr: 0.001
            simulation:
              partitioning: shard
              dropout_rate: 0.1
            privacy:
              enabled: true
              epsilon: 2.0
              delta: 1.0e-5
            output:
              dir: /tmp/flp
              save_plots: false
            """
        )
        cfg = load_config(path)
        assert cfg.name == "full_test"
        assert cfg.training.num_rounds == 5
        assert cfg.training.num_clients == 4
        assert cfg.client.batch_size == 64
        assert cfg.simulation.partitioning == "shard"
        assert cfg.simulation.dropout_rate == 0.1
        assert cfg.privacy.enabled is True
        assert cfg.privacy.epsilon == 2.0
        assert cfg.output.save_plots is False


class TestConfigValidation:
    def test_invalid_partitioning_strategy(self, tmp_config) -> None:
        path = tmp_config("simulation:\n  partitioning: invalid_strategy\n")
        with pytest.raises(Exception):
            load_config(path)

    def test_negative_seed_raises(self) -> None:
        with pytest.raises(Exception):
            ExperimentConfig(seed=-1)

    def test_dropout_rate_out_of_range(self, tmp_config) -> None:
        path = tmp_config("simulation:\n  dropout_rate: 1.5\n")
        with pytest.raises(Exception):
            load_config(path)
