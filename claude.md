# Project: Federated Learning Playground (FLP)

## Vision

Federated Learning Playground (FLP) is a systems-oriented sandbox for experimenting with real-world constraints in federated AI training.

This is NOT another production federated learning framework.

This is:
- A simulation-first experimentation framework
- Designed to explore distributed AI trade-offs
- Built with regulated enterprise constraints in mind
- Focused on observability, reproducibility, and engineering clarity

The goal is to help engineers understand:

- Non-IID data effects
- Client dropout and stragglers
- Communication bottlenecks
- Compression strategies
- Differential privacy trade-offs
- Aggregation strategy impacts
- Convergence behavior under constraints

---

## Design Principles

1. Clean modular architecture
2. Deterministic experiments (reproducible seeds)
3. YAML-driven experiments
4. Clear separation:
   - Core federated logic
   - Simulation layer
   - Privacy layer
   - Metrics layer
5. Educational clarity over performance optimization
6. Minimal dependencies
7. PyTorch-based models (initially MNIST)

---

## Initial MVP Scope (v0.1)

Must support:

- N simulated clients
- Local model training per client
- FedAvg aggregation
- Non-IID partitioning
- Client dropout simulation
- Metrics tracking:
  - global accuracy
  - per-client accuracy
  - communication cost
- Reproducible experiment config via YAML
- Console logging
- Simple matplotlib visualization

---

## Architecture Overview
src/flp/
core/
    client.py
    server.py
    aggregator.py
    trainer.py
    models.py
simulation/
    partitioning.py
    dropout.py
    delay.py

privacy/
    dp.py
    clipping.py

metrics/
    tracker.py
    communication.py

experiments/
    runner.py
    config_loader.py

visualization/
    plots.py

governance/
    audit.py
    hashing.py
    replay.py


---

## Future Extensions (Not MVP)

- Asynchronous FL
- Secure aggregation simulation
- Gradient compression (Top-k)
- Fairness metrics
- ~~Governance mode~~ — **Implemented**: audit logs, deterministic replay, model lineage metadata

---

## Technical Constraints

- Python 3.11+
- PyTorch
- pydantic for config models
- yaml config for experiments
- matplotlib for visualization
- No heavy distributed frameworks
- Everything simulated in-process

---

## What This Project Is NOT

- Not competing with Flower
- Not replacing TensorFlow Federated
- Not production-grade FL infra
- Not multi-machine distributed runtime

It is a research + engineering playground.

---

## Quality Expectations

- Type hints everywhere
- Docstrings on public APIs
- Tests for core logic
- Clear error messages
- Clean CLI interface

---

## Deliverable Expectation

End result should allow:

flp run --config configs/baseline.yaml

And produce:

- training logs
- metrics output
- saved plots
- summary report