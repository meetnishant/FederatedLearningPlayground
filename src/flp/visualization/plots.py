"""Matplotlib-based visualisations for federated learning experiments."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from flp.metrics.tracker import MetricsTracker

matplotlib.use("Agg")  # Non-interactive backend for server/CLI use


def plot_global_accuracy(metrics: MetricsTracker, output_path: str) -> None:
    """Plot global test accuracy across federated rounds.

    Args:
        metrics: Populated metrics tracker.
        output_path: File path to save the PNG (parent dirs created automatically).
    """
    rounds = [r.round_num for r in metrics.rounds]
    accuracies = [r.global_accuracy * 100 for r in metrics.rounds]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rounds, accuracies, marker="o", linewidth=2, color="#2196F3", markersize=5)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Global Accuracy (%)", fontsize=12)
    ax.set_title("Global Model Accuracy per Federated Round", fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    _save(fig, output_path)


def plot_global_loss(metrics: MetricsTracker, output_path: str) -> None:
    """Plot global test loss across federated rounds.

    Args:
        metrics: Populated metrics tracker.
        output_path: File path to save the PNG.
    """
    rounds = [r.round_num for r in metrics.rounds]
    losses = [r.global_loss for r in metrics.rounds]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rounds, losses, marker="s", linewidth=2, color="#F44336", markersize=5)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Global Loss", fontsize=12)
    ax.set_title("Global Model Loss per Federated Round", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    _save(fig, output_path)


def plot_per_client_accuracy(metrics: MetricsTracker, output_path: str) -> None:
    """Plot per-client accuracy distributions across rounds as a heatmap.

    Args:
        metrics: Populated metrics tracker.
        output_path: File path to save the PNG.
    """
    rounds = metrics.rounds
    if not rounds:
        return

    all_client_ids = sorted(
        {cid for r in rounds for cid in r.per_client_accuracy}
    )
    data = np.full((len(all_client_ids), len(rounds)), np.nan)

    for col, r in enumerate(rounds):
        for row, cid in enumerate(all_client_ids):
            if cid in r.per_client_accuracy:
                data[row, col] = r.per_client_accuracy[cid] * 100

    fig, ax = plt.subplots(figsize=(max(8, len(rounds) * 0.6), max(5, len(all_client_ids) * 0.4)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels([str(r.round_num) for r in rounds], fontsize=8)
    ax.set_yticks(range(len(all_client_ids)))
    ax.set_yticklabels([f"C{cid}" for cid in all_client_ids], fontsize=8)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Client", fontsize=12)
    ax.set_title("Per-Client Accuracy Heatmap (%)", fontsize=14)
    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    fig.tight_layout()
    _save(fig, output_path)


def plot_client_participation(metrics: MetricsTracker, output_path: str) -> None:
    """Bar chart showing the number of active clients per round.

    Args:
        metrics: Populated metrics tracker.
        output_path: File path to save the PNG.
    """
    rounds = [r.round_num for r in metrics.rounds]
    active = [r.num_active_clients for r in metrics.rounds]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(rounds, active, color="#4CAF50", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Active Clients", fontsize=12)
    ax.set_title("Client Participation per Round", fontsize=14)
    ax.set_xticks(rounds)
    ax.bar_label(bars, padding=2, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    _save(fig, output_path)


def save_all_plots(metrics: MetricsTracker, output_dir: str) -> None:
    """Generate and save all standard FLP plots to ``output_dir``.

    Args:
        metrics: Populated metrics tracker.
        output_dir: Directory where PNG files will be written.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_global_accuracy(metrics, f"{output_dir}/global_accuracy.png")
    plot_global_loss(metrics, f"{output_dir}/global_loss.png")
    plot_per_client_accuracy(metrics, f"{output_dir}/per_client_accuracy.png")
    plot_client_participation(metrics, f"{output_dir}/client_participation.png")


def _save(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
