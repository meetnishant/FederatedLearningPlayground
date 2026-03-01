# Federated Learning Playground (FLP)

A simulation-first experimentation framework for exploring distributed AI training trade-offs under real-world constraints.

---

## What is FLP?

FLP is a **research and engineering sandbox** for federated learning — not a production framework. It exists to help engineers understand the behaviour of federated systems when subject to realistic constraints:

| Constraint | FLP Capability |
|---|---|
| Non-IID data distributions | Dirichlet, shard, and IID partitioning |
| Client dropout & stragglers | Configurable per-round dropout simulation |
| Communication bottlenecks | Byte-level communication cost tracking |
| Differential privacy | Gaussian mechanism with L2 gradient clipping |
| Aggregation strategies | FedAvg with weighted sample averaging |
| Reproducibility | Deterministic seeds + YAML-driven configs |

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Run the baseline experiment

```bash
flp run --config configs/baseline.yaml
```

### 3. Validate a config without running

```bash
flp validate-config configs/baseline.yaml
```

### 4. Run with verbose logging

```bash
flp run --config configs/baseline.yaml -v
```

---

## Project Structure

```
FederatedLearningPlayground/
├── pyproject.toml                  # Package metadata and dependencies
├── configs/
│   ├── baseline.yaml               # Standard FedAvg experiment
│   ├── noniid_strict.yaml          # Extreme non-IID with dropout
│   └── dp_fedavg.yaml              # Differentially private FedAvg
├── src/flp/
│   ├── cli.py                      # CLI entrypoint (flp run / flp validate-config)
│   ├── core/
│   │   ├── client.py               # FLClient: local training + weight sync
│   │   ├── server.py               # FLServer: round orchestration
│   │   ├── aggregator.py           # FedAvg weighted aggregation
│   │   └── trainer.py              # LocalTrainer: SGD loop + evaluation
│   ├── simulation/
│   │   ├── partitioning.py         # IID / Dirichlet / Shard data splits
│   │   ├── dropout.py              # Per-round client dropout
│   │   └── delay.py                # Communication delay + straggler filtering
│   ├── privacy/
│   │   ├── dp.py                   # Gaussian mechanism for (ε, δ)-DP
│   │   └── clipping.py             # L2 gradient and update clipping
│   ├── metrics/
│   │   ├── tracker.py              # Per-round accuracy, loss, client stats
│   │   └── communication.py        # Upload/download byte tracking
│   ├── experiments/
│   │   ├── config_loader.py        # Pydantic-validated YAML config models
│   │   └── runner.py               # End-to-end experiment orchestration
│   └── visualization/
│       └── plots.py                # Matplotlib plots (accuracy, loss, heatmap)
└── tests/
    ├── test_aggregator.py
    ├── test_partitioning.py
    ├── test_dropout.py
    ├── test_metrics.py
    └── test_config_loader.py
```

---

## Configuration Reference

All experiments are driven by YAML files. The full schema:

```yaml
name: my_experiment        # Experiment name (used for output directory)
seed: 42                   # Global random seed (reproducibility)
dataset: mnist             # Dataset: mnist (only option in v0.1)
data_dir: ~/.flp/data      # Where to cache downloaded data

training:
  num_rounds: 20           # Total federated communication rounds
  num_clients: 10          # Number of simulated clients
  client_fraction: 0.5     # Fraction selected per round

client:
  batch_size: 32           # Local mini-batch size
  local_epochs: 2          # Local SGD epochs per round
  lr: 0.01                 # Learning rate
  momentum: 0.9            # SGD momentum
  weight_decay: 0.0001     # L2 regularisation

simulation:
  partitioning: dirichlet  # iid | dirichlet | shard
  alpha: 0.5               # Dirichlet alpha (lower = more non-IID)
  num_shards_per_client: 2 # For shard strategy only
  dropout_rate: 0.1        # Per-client dropout probability per round

privacy:
  enabled: false           # Enable DP-FedAvg
  epsilon: 1.0             # Privacy budget
  delta: 1.0e-5            # Failure probability
  max_grad_norm: 1.0       # L2 clipping norm

output:
  dir: outputs             # Root output directory
  save_plots: true         # Save PNG plots
  save_metrics: true       # Save metrics.json
  save_model: false        # Save global model checkpoint
```

---

## Outputs

After a run, outputs are written to `outputs/<experiment_name>/`:

```
outputs/baseline_fedavg/
├── metrics.json                    # Full round-by-round metrics
└── plots/
    ├── global_accuracy.png         # Global test accuracy per round
    ├── global_loss.png             # Global test loss per round
    ├── per_client_accuracy.png     # Per-client accuracy heatmap
    └── client_participation.png    # Active clients per round (bar chart)
```

---

## Running Tests

```bash
pytest
pytest -v --tb=short          # Verbose with short tracebacks
pytest tests/test_aggregator.py   # Single test module
```

---

## Pre-built Experiments

| Config | Description |
|---|---|
| `configs/baseline.yaml` | Standard FedAvg, 10 clients, Dirichlet α=0.5, 10% dropout |
| `configs/noniid_strict.yaml` | 20 clients, α=0.05 (extreme non-IID), 20% dropout |
| `configs/dp_fedavg.yaml` | DP-FedAvg with ε=1.0, δ=1e-5, L2 clip norm=1.0 |

---

## Design Principles

1. **Simulation-first** — everything runs in a single process; no distributed runtime required
2. **Deterministic** — fixed seeds produce identical results across runs
3. **Modular** — each layer (core, simulation, privacy, metrics) is independently replaceable
4. **Minimal dependencies** — PyTorch, Pydantic, PyYAML, matplotlib, Click, Rich
5. **Educational clarity** — code is written to be read and understood, not micro-optimised

---

## Technical Requirements

- Python 3.11+
- PyTorch 2.2+
- No GPU required (CPU training works for MNIST scale)

---

## What FLP is NOT

- Not a replacement for [Flower](https://flower.dev/) or [TFF](https://www.tensorflow.org/federated)
- Not production-grade federated infrastructure
- Not a multi-machine distributed runtime

It is a **research and engineering playground**.

---

## Roadmap (Post-MVP)

- [ ] Asynchronous federated learning
- [ ] Secure aggregation simulation
- [ ] Top-k gradient compression
- [ ] Fairness metrics (q-FedAvg)
- [ ] Governance mode: audit logs, deterministic replay, model lineage
- [ ] Additional datasets (CIFAR-10, Shakespeare)
