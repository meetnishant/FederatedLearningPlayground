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
| Auditability & governance | SHA-256 model hash chain, round audit log, replay manifest (schema 1.1) |
| Async federated learning | Virtual-time event loop, per-client delivery delays, staleness threshold |
| Staleness-aware aggregation | Uniform, inverse-staleness, and exponential-decay weighting strategies |
| Gradient compression | Top-k sparsification, float16/int8 quantization, error-feedback residuals |
| Research metrics | Weight divergence, cosine similarity, Gini fairness coefficient, q-FedAvg |

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
│   ├── dp_fedavg.yaml              # Differentially private FedAvg
│   └── async_fedavg.yaml           # Async FL with delivery delays
├── src/flp/
│   ├── cli.py                      # CLI entrypoint (flp run / flp validate-config)
│   ├── core/
│   │   ├── client.py               # FLClient: local training + weight sync
│   │   ├── server.py               # FLServer: round orchestration + compression hook
│   │   ├── aggregator.py           # FedAvg weighted aggregation (custom weights support)
│   │   ├── trainer.py              # LocalTrainer: SGD loop + evaluation
│   │   ├── models.py               # MNISTNet CNN definition
│   │   ├── event_loop.py           # FLEvent + FLEventLoop (virtual-time priority queue)
│   │   ├── async_server.py         # AsyncFLServer + AsyncRoundSummary
│   │   └── staleness.py            # StalenessWeighter (uniform/inverse/exponential)
│   ├── compression/
│   │   ├── topk.py                 # Top-k sparsification
│   │   ├── quantization.py         # float16 / int8 quantization
│   │   ├── error_feedback.py       # Per-client residual accumulation buffer
│   │   └── __init__.py             # GradientCompressor facade
│   ├── research/
│   │   ├── divergence.py           # Weight divergence + cosine similarity
│   │   ├── fairness.py             # Gini coefficient, fairness metrics, q-FedAvg
│   │   └── __init__.py
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
│   ├── governance/
│   │   ├── audit.py                # AuditEvent + AuditLog (JSON/JSONL)
│   │   ├── hashing.py              # SHA-256 model and config hashing
│   │   └── replay.py               # ReplayManifest schema 1.1 (git hash, features)
│   └── visualization/
│       └── plots.py                # Matplotlib plots (accuracy, loss, heatmap)
└── tests/
    ├── test_aggregator.py
    ├── test_async.py               # FLEventLoop, AsyncFLServer, AsyncConfig (39 tests)
    ├── test_staleness.py           # StalenessWeighter + aggregator weights (42 tests)
    ├── test_compression.py         # TopK, quantization, error feedback (45 tests)
    ├── test_research.py            # Divergence, fairness, q-FedAvg (43 tests)
    ├── test_partitioning.py
    ├── test_dropout.py
    ├── test_metrics.py
    ├── test_config_loader.py
    ├── test_governance.py          # Hashing, AuditLog, ReplayManifest
    └── test_regulatory.py          # 15 banking/finance compliance scenarios
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

governance:
  enabled: false           # Enable governance mode
  save_audit_log: true     # Write audit_log.json + audit_log.jsonl
  save_replay_manifest: true  # Write replay_manifest.json (schema 1.1)

# Async FL — replaces FLServer with AsyncFLServer
async_fl:
  enabled: false           # Enable async FL mode
  delay_min: 0.0           # Min delivery delay in virtual rounds
  delay_max: 3.0           # Max delivery delay in virtual rounds
  staleness_threshold: 3   # Discard updates older than N server versions
  staleness_strategy: uniform  # uniform | inverse_staleness | exponential_decay
  staleness_decay_factor: 0.9  # Base for exponential decay (when strategy=exponential_decay)

# Gradient compression — applied server-side before aggregation
compression:
  enabled: false           # Enable compression
  strategy: topk           # topk | quantization
  topk_ratio: 0.1          # Fraction of elements to keep (0.1 = top 10%)
  quantization_bits: 16    # 16 = float16 (2×), 8 = int8 (4×)
  error_feedback: false    # Accumulate residuals across rounds (topk only)

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
├── summary.json                    # Top-level experiment report (always written)
├── metrics.json                    # Full round-by-round metrics
├── communication.json              # Upload/download byte accounting
├── global_model.pt                 # Final model weights (if save_model: true)
├── plots/
│   ├── global_accuracy.png         # Global test accuracy per round
│   ├── global_loss.png             # Global test loss per round
│   ├── per_client_accuracy.png     # Per-client accuracy heatmap
│   └── client_participation.png    # Active clients per round (bar chart)
└── governance/                     # Written when governance.enabled: true
    ├── audit_log.json              # All round events as a JSON array
    ├── audit_log.jsonl             # Same events, one JSON object per line
    └── replay_manifest.json        # Full reproducibility manifest
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
| `configs/async_fedavg.yaml` | Async FL with delivery delays up to 3 rounds, inverse-staleness weighting |

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

- [x] Asynchronous federated learning (virtual-time event loop, staleness threshold)
- [x] Staleness-aware aggregation (uniform, inverse-staleness, exponential-decay)
- [x] Top-k gradient compression + float16/int8 quantization + error feedback
- [x] Fairness metrics (Gini coefficient, q-FedAvg loss reweighting)
- [x] Weight divergence + cosine similarity between client updates
- [x] Governance mode: audit logs, deterministic replay, model lineage
- [ ] Secure aggregation simulation
- [ ] Additional datasets (CIFAR-10, Shakespeare)
