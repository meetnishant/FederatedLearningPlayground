"""Matplotlib-based visualisations for federated learning experiments."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

if TYPE_CHECKING:
    from flp.metrics.communication import CommunicationTracker
    from flp.metrics.tracker import MetricsTracker

matplotlib.use("Agg")  # Non-interactive backend for server/CLI use

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

_STYLE: dict = {
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

_PALETTE = {
    "blue": "#2196F3",
    "red": "#F44336",
    "green": "#4CAF50",
    "orange": "#FF9800",
    "purple": "#9C27B0",
    "teal": "#009688",
    "grey": "#9E9E9E",
}


# ---------------------------------------------------------------------------
# Accuracy vs Rounds
# ---------------------------------------------------------------------------


def plot_global_accuracy(
    metrics: MetricsTracker,
    output_path: str,
) -> None:
    """Line chart of global test-set accuracy across federated rounds.

    Annotates the best-accuracy point with its value and round number, fills
    the area under the curve, and labels the first and final round values.

    Args:
        metrics: Populated metrics tracker.
        output_path: File path to save the PNG (parent dirs created automatically).
    """
    if not metrics.rounds:
        return

    rounds = [r.round_num for r in metrics.rounds]
    accs = [r.global_accuracy * 100 for r in metrics.rounds]
    best_idx = int(np.argmax(accs))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.fill_between(rounds, accs, alpha=0.12, color=_PALETTE["blue"])
        ax.plot(
            rounds, accs,
            marker="o", linewidth=2.0, color=_PALETTE["blue"],
            markersize=5, markerfacecolor="white", markeredgewidth=1.5,
            zorder=3,
        )

        # Highlight best point
        ax.scatter(
            [rounds[best_idx]], [accs[best_idx]],
            s=80, color=_PALETTE["orange"], zorder=4,
            label=f"Best: {accs[best_idx]:.2f}% (round {rounds[best_idx]})",
        )
        ax.annotate(
            f"{accs[best_idx]:.2f}%",
            xy=(rounds[best_idx], accs[best_idx]),
            xytext=(8, 6), textcoords="offset points",
            fontsize=9, color=_PALETTE["orange"], fontweight="bold",
        )

        # First / last labels
        ax.annotate(
            f"{accs[0]:.2f}%", xy=(rounds[0], accs[0]),
            xytext=(6, -14), textcoords="offset points",
            fontsize=8, color="#666666",
        )
        ax.annotate(
            f"{accs[-1]:.2f}%", xy=(rounds[-1], accs[-1]),
            xytext=(-28, 6), textcoords="offset points",
            fontsize=8, color="#666666",
        )

        ax.set_xlabel("Federated Round")
        ax.set_ylabel("Global Accuracy (%)")
        ax.set_title("Global Model Accuracy per Federated Round")
        ax.set_ylim(max(0, min(accs) - 5), min(100, max(accs) + 8))
        ax.set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.legend(loc="lower right", fontsize=9, framealpha=0.8)

        fig.tight_layout()
        _save(fig, output_path)


# ---------------------------------------------------------------------------
# Loss vs Rounds
# ---------------------------------------------------------------------------


def plot_global_loss(
    metrics: MetricsTracker,
    output_path: str,
) -> None:
    """Dual-line chart: global test-set loss and weighted mean client training loss.

    Overlaying both curves reveals the train/test gap across rounds.

    Args:
        metrics: Populated metrics tracker.
        output_path: File path to save the PNG.
    """
    if not metrics.rounds:
        return

    rounds = [r.round_num for r in metrics.rounds]
    global_losses = [r.global_loss for r in metrics.rounds]
    client_losses = [r.avg_client_loss for r in metrics.rounds]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.fill_between(rounds, global_losses, alpha=0.10, color=_PALETTE["red"])
        ax.plot(
            rounds, global_losses,
            marker="s", linewidth=2.0, color=_PALETTE["red"],
            markersize=5, markerfacecolor="white", markeredgewidth=1.5,
            label="Global (test) loss", zorder=3,
        )
        ax.plot(
            rounds, client_losses,
            marker="^", linewidth=1.4, color=_PALETTE["orange"],
            markersize=4, linestyle="--", alpha=0.85,
            label="Avg client (train) loss",
        )

        ax.set_xlabel("Federated Round")
        ax.set_ylabel("Loss")
        ax.set_title("Global & Client Loss per Federated Round")
        ax.set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

        fig.tight_layout()
        _save(fig, output_path)


# ---------------------------------------------------------------------------
# Communication cost vs Rounds
# ---------------------------------------------------------------------------


def plot_communication_cost(
    comm_tracker: CommunicationTracker,
    output_path: str,
) -> None:
    """Stacked bar chart (upload / download per round) with cumulative line.

    Uses a dual y-axis: left for per-round MB, right for cumulative total MB.

    Args:
        comm_tracker: Populated communication tracker.
        output_path: File path to save the PNG.
    """
    records = comm_tracker.records
    if not records:
        return

    round_nums = [r.round_num for r in records]
    upload_mb = [r.upload_mb for r in records]
    download_mb = [r.download_mb for r in records]
    cumulative_mb = [b / (1024 ** 2) for b in comm_tracker.cumulative_bytes]

    x = np.arange(len(round_nums))
    bar_w = 0.55

    with plt.rc_context(_STYLE):
        fig, ax1 = plt.subplots(figsize=(max(9, len(round_nums) * 0.55), 5))

        b1 = ax1.bar(x, upload_mb, bar_w, label="Upload (↑)",
                     color=_PALETTE["blue"], alpha=0.82)
        b2 = ax1.bar(x, download_mb, bar_w, bottom=upload_mb,
                     label="Download (↓)", color=_PALETTE["teal"], alpha=0.82)

        ax1.set_xlabel("Federated Round")
        ax1.set_ylabel("MB per Round")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(r) for r in round_nums], fontsize=8)
        ax1.set_title("Communication Cost per Federated Round")

        # Cumulative on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(
            x, cumulative_mb,
            color=_PALETTE["orange"], linewidth=2.0, marker="o",
            markersize=4, markerfacecolor="white", markeredgewidth=1.5,
            label="Cumulative (MB)", zorder=5,
        )
        ax2.set_ylabel("Cumulative MB", color=_PALETTE["orange"])
        ax2.tick_params(axis="y", labelcolor=_PALETTE["orange"])
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(_PALETTE["orange"])

        handles = [b1, b2] + ax2.get_lines()
        labels = ["Upload (↑)", "Download (↓)", "Cumulative (MB)"]
        ax1.legend(handles, labels, loc="upper left", fontsize=9, framealpha=0.8)

        fig.tight_layout()
        _save(fig, output_path)


# ---------------------------------------------------------------------------
# Per-client accuracy heatmap
# ---------------------------------------------------------------------------


def plot_per_client_accuracy(
    metrics: MetricsTracker,
    output_path: str,
) -> None:
    """Heatmap of per-client accuracy (rows = clients, columns = rounds).

    Cells are colour-coded green (high) → red (low).  Grey cells indicate
    the client was inactive (dropped or not selected) that round.
    Each active cell is annotated with its integer accuracy value.

    Args:
        metrics: Populated metrics tracker.
        output_path: File path to save the PNG.
    """
    rounds = metrics.rounds
    if not rounds:
        return

    all_client_ids = sorted({cid for r in rounds for cid in r.per_client_accuracy})
    if not all_client_ids:
        return

    n_clients = len(all_client_ids)
    n_rounds = len(rounds)
    data = np.full((n_clients, n_rounds), np.nan)

    for col, r in enumerate(rounds):
        for row, cid in enumerate(all_client_ids):
            if cid in r.per_client_accuracy:
                data[row, col] = r.per_client_accuracy[cid] * 100

    fig_w = max(9, n_rounds * 0.65)
    fig_h = max(4, n_clients * 0.38)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        cmap = plt.cm.RdYlGn.copy()  # type: ignore[attr-defined]
        cmap.set_bad(color="#E0E0E0")

        masked = np.ma.masked_invalid(data)
        im = ax.imshow(
            masked, aspect="auto", cmap=cmap,
            vmin=0, vmax=100, interpolation="nearest",
        )

        cell_fontsize = max(5, min(8, int(80 / max(n_rounds, n_clients))))
        for row in range(n_clients):
            for col in range(n_rounds):
                if not np.isnan(data[row, col]):
                    val = data[row, col]
                    text_color = "black" if 25 < val < 75 else "white"
                    ax.text(col, row, f"{val:.0f}",
                            ha="center", va="center",
                            fontsize=cell_fontsize, color=text_color)

        ax.set_xticks(range(n_rounds))
        ax.set_xticklabels([str(r.round_num) for r in rounds], fontsize=8)
        ax.set_yticks(range(n_clients))
        ax.set_yticklabels([f"C{cid}" for cid in all_client_ids], fontsize=8)
        ax.set_xlabel("Federated Round")
        ax.set_ylabel("Client")
        ax.set_title("Per-Client Accuracy Heatmap (%)  [grey = inactive/dropped]")

        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Accuracy (%)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        fig.tight_layout()
        _save(fig, output_path)


# ---------------------------------------------------------------------------
# Client participation bar chart
# ---------------------------------------------------------------------------


def plot_client_participation(
    metrics: MetricsTracker,
    output_path: str,
) -> None:
    """Bar chart of active client count per round with a mean reference line.

    Args:
        metrics: Populated metrics tracker.
        output_path: File path to save the PNG.
    """
    if not metrics.rounds:
        return

    rounds = [r.round_num for r in metrics.rounds]
    active = [r.num_active_clients for r in metrics.rounds]
    mean_active = float(np.mean(active))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(9, len(rounds) * 0.55), 4))

        bars = ax.bar(
            rounds, active,
            color=_PALETTE["green"], alpha=0.82,
            edgecolor="white", linewidth=0.5,
        )
        ax.bar_label(bars, padding=3, fontsize=9, fmt="%d")
        ax.axhline(
            mean_active, color=_PALETTE["orange"], linewidth=1.5,
            linestyle="--", label=f"Mean: {mean_active:.1f}",
        )

        ax.set_xlabel("Federated Round")
        ax.set_ylabel("Active Clients")
        ax.set_title("Client Participation per Round")
        ax.set_xticks(rounds)
        ax.set_xticklabels([str(r) for r in rounds], fontsize=9)
        ax.set_ylim(0, max(active) + max(1, int(max(active) * 0.2)))
        ax.legend(fontsize=9, framealpha=0.8)

        fig.tight_layout()
        _save(fig, output_path)


# ---------------------------------------------------------------------------
# Accuracy spread / fairness view
# ---------------------------------------------------------------------------


def plot_accuracy_spread(
    metrics: MetricsTracker,
    output_path: str,
) -> None:
    """Shaded band: min–max per-client accuracy range overlaid with global accuracy.

    A narrow band means all clients benefit similarly from the global model.
    A wide band signals fairness concerns or extreme data heterogeneity.

    Args:
        metrics: Populated metrics tracker.
        output_path: File path to save the PNG.
    """
    if not metrics.rounds:
        return

    rounds = [r.round_num for r in metrics.rounds]
    mins = [r.min_client_accuracy * 100 for r in metrics.rounds]
    maxs = [r.max_client_accuracy * 100 for r in metrics.rounds]
    globals_ = [r.global_accuracy * 100 for r in metrics.rounds]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.fill_between(rounds, mins, maxs, alpha=0.18,
                        color=_PALETTE["purple"], label="Client acc. range")
        ax.plot(rounds, mins, linewidth=1.0, color=_PALETTE["purple"],
                linestyle=":", alpha=0.7, label=f"Min: {mins[-1]:.1f}%")
        ax.plot(rounds, maxs, linewidth=1.0, color=_PALETTE["purple"],
                linestyle=":", alpha=0.7, label=f"Max: {maxs[-1]:.1f}%")
        ax.plot(rounds, globals_, linewidth=2.0, color=_PALETTE["blue"],
                marker="o", markersize=4, markerfacecolor="white",
                markeredgewidth=1.4, label="Global (test) acc.")

        ax.set_xlabel("Federated Round")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Global Accuracy & Per-Client Spread per Round")
        ax.set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)
        ax.set_ylim(max(0, min(mins) - 5), min(100, max(maxs) + 8))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.legend(loc="lower right", fontsize=9, framealpha=0.8)

        fig.tight_layout()
        _save(fig, output_path)


# ---------------------------------------------------------------------------
# save_all_plots
# ---------------------------------------------------------------------------


def save_all_plots(
    metrics: MetricsTracker,
    output_dir: str,
    comm_tracker: CommunicationTracker | None = None,
) -> None:
    """Generate and save all standard FLP plots to ``output_dir``.

    Plots produced:

    - ``global_accuracy.png``     — accuracy vs rounds (annotated best point)
    - ``global_loss.png``         — global + avg client loss vs rounds
    - ``communication_cost.png``  — per-round and cumulative MB (requires *comm_tracker*)
    - ``per_client_accuracy.png`` — accuracy heatmap (clients × rounds)
    - ``client_participation.png``— active client count bar chart with mean line
    - ``accuracy_spread.png``     — min/max/global accuracy fairness band

    Args:
        metrics: Populated metrics tracker.
        output_dir: Directory where PNG files will be written.
        comm_tracker: Optional communication tracker; if provided,
            ``communication_cost.png`` is also generated.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_global_accuracy(metrics, f"{output_dir}/global_accuracy.png")
    plot_global_loss(metrics, f"{output_dir}/global_loss.png")
    plot_per_client_accuracy(metrics, f"{output_dir}/per_client_accuracy.png")
    plot_client_participation(metrics, f"{output_dir}/client_participation.png")
    plot_accuracy_spread(metrics, f"{output_dir}/accuracy_spread.png")

    if comm_tracker is not None:
        plot_communication_cost(comm_tracker, f"{output_dir}/communication_cost.png")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: str) -> None:
    """Save *fig* to *path* at 150 dpi and close it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
