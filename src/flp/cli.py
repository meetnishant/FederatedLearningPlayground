"""CLI entrypoint for the Federated Learning Playground (FLP)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    )


@click.group()
@click.version_option(package_name="flp")
def main() -> None:
    """Federated Learning Playground (FLP) — simulation-first FL experimentation."""


# ---------------------------------------------------------------------------
# flp run
# ---------------------------------------------------------------------------


@main.command("run")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to experiment YAML config file.",
)
@click.option(
    "--output-dir",
    "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Override the output directory from config.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable DEBUG logging.",
)
@click.option(
    "--no-progress",
    is_flag=True,
    default=False,
    help="Disable the Rich progress bar (useful when piping output).",
)
def run_experiment(
    config: Path,
    output_dir: Path | None,
    verbose: bool,
    no_progress: bool,
) -> None:
    """Run a federated learning experiment from a YAML config file.

    \b
    Examples:
        flp run --config configs/baseline.yaml
        flp run --config configs/baseline.yaml --output-dir /tmp/flp_out -v
        flp run --config configs/noniid_strict.yaml --no-progress
    """
    _configure_logging(verbose)

    from flp.experiments.config_loader import ConfigValidationError, load_config
    from flp.experiments.runner import ExperimentRunner

    # ---- Header ----
    console.print(
        Panel.fit(
            "[bold cyan]Federated Learning Playground[/bold cyan]\n"
            f"[dim]Config: {config}[/dim]",
            border_style="cyan",
        )
    )

    # ---- Load & validate config ----
    try:
        exp_config = load_config(config)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
    except ConfigValidationError as exc:
        console.print(f"[bold red]Config error:[/bold red]\n{exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[bold red]Unexpected config error:[/bold red] {exc}")
        sys.exit(1)

    if output_dir is not None:
        exp_config.output.dir = str(output_dir)

    _print_config_table(exp_config)

    # ---- Progress bar setup ----
    num_rounds = exp_config.training.num_rounds
    round_callback = None

    progress: Progress | None = None
    if not no_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=38),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]eta"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )

        # State shared between the callback closure and the outer scope
        _best: dict[str, float] = {"acc": 0.0}

        def _on_round(summary: object) -> None:  # type: ignore[name-defined]
            from flp.core.server import RoundSummary as _RS
            assert isinstance(summary, _RS)
            _best["acc"] = max(_best["acc"], summary.global_accuracy)
            progress.advance(task_id)  # type: ignore[union-attr]
            progress.update(  # type: ignore[union-attr]
                task_id,
                description=(
                    f"[bold blue]Round {summary.round_num:>{len(str(num_rounds))}}/"
                    f"{num_rounds}[/bold blue] "
                    f"acc=[green]{summary.global_accuracy:.4f}[/green] "
                    f"best=[cyan]{_best['acc']:.4f}[/cyan]"
                ),
            )

        round_callback = _on_round

    # ---- Run experiment ----
    runner = ExperimentRunner(exp_config)

    try:
        if progress is not None:
            with progress:
                task_id = progress.add_task(
                    f"[bold blue]Round  1/{num_rounds}[/bold blue]",
                    total=num_rounds,
                )
                metrics = runner.run(round_callback=round_callback)
        else:
            metrics = runner.run()

    except KeyboardInterrupt:
        console.print("\n[yellow]Experiment interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[bold red]Experiment failed:[/bold red] {exc}")
        if verbose:
            console.print_exception()
        sys.exit(1)

    # ---- Final results table ----
    summary = metrics.summary()
    output_root = Path(exp_config.output.dir) / exp_config.name

    results_table = Table(
        title="[bold]Experiment Results[/bold]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    results_table.add_column("Metric", style="dim", no_wrap=True)
    results_table.add_column("Value", justify="right", style="bold")

    results_table.add_row("Rounds completed", str(summary["num_rounds"]))
    results_table.add_row(
        "Best accuracy",
        f"[green]{float(summary['best_accuracy']):.4f}[/green]  (round {summary['best_round']})",
    )
    results_table.add_row("Final accuracy", f"{float(summary['final_accuracy']):.4f}")
    results_table.add_row("Final loss", f"{float(summary['final_loss']):.4f}")
    results_table.add_row(
        "Accuracy improvement",
        f"{float(summary['accuracy_improvement']):+.4f}",
    )
    results_table.add_row("Avg active clients", f"{float(summary['avg_active_clients']):.1f}")
    results_table.add_row("Output dir", str(output_root))

    console.print()
    console.print(results_table)
    console.print(
        Panel.fit(
            f"[bold green]Done.[/bold green]  "
            f"Results → [cyan]{output_root}/summary.json[/cyan]",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# flp validate-config
# ---------------------------------------------------------------------------


@main.command("validate-config")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate_config(config_path: Path) -> None:
    """Validate a YAML experiment config without running anything.

    \b
    Example:
        flp validate-config configs/baseline.yaml
    """
    from flp.experiments.config_loader import load_config

    try:
        cfg = load_config(config_path)
        console.print(f"[bold green]Config is valid:[/bold green] {config_path}")
        _print_config_table(cfg)
    except Exception as exc:
        console.print(f"[bold red]Config invalid:[/bold red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# flp list-configs
# ---------------------------------------------------------------------------


@main.command("list-configs")
@click.option(
    "--dir",
    "config_dir",
    default="configs",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to search for YAML configs.",
)
def list_configs(config_dir: Path) -> None:
    """List all YAML experiment configs in a directory.

    \b
    Example:
        flp list-configs
        flp list-configs --dir /path/to/configs
    """
    from flp.experiments.config_loader import load_config

    yaml_files = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))

    if not yaml_files:
        console.print(f"[yellow]No YAML configs found in '{config_dir}'.[/yellow]")
        return

    table = Table(
        title=f"Configs in [cyan]{config_dir}[/cyan]",
        show_header=True,
        header_style="bold",
    )
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Rounds", justify="right")
    table.add_column("Clients", justify="right")
    table.add_column("Partition")
    table.add_column("Status")

    for path in yaml_files:
        try:
            cfg = load_config(path)
            table.add_row(
                path.name,
                cfg.name,
                str(cfg.training.num_rounds),
                str(cfg.training.num_clients),
                cfg.simulation.partitioning,
                "[green]OK[/green]",
            )
        except Exception as exc:
            table.add_row(
                path.name, "—", "—", "—", "—",
                f"[red]INVALID[/red]: {exc!s:.40}",
            )

    console.print(table)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_config_table(cfg: object) -> None:
    from flp.experiments.config_loader import ExperimentConfig

    assert isinstance(cfg, ExperimentConfig)

    table = Table(
        title="Experiment Configuration",
        show_header=False,
        border_style="dim",
    )
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value")

    table.add_row("Name", cfg.name)
    table.add_row("Dataset", cfg.dataset)
    table.add_row("Seed", str(cfg.seed))
    table.add_row("Device", "cuda" if __import__("torch").cuda.is_available() else "cpu")
    table.add_row("Rounds", str(cfg.training.num_rounds))
    table.add_row("Clients", str(cfg.training.num_clients))
    table.add_row("Client fraction", f"{cfg.training.client_fraction:.0%}")
    table.add_row("Partitioning", cfg.simulation.partitioning)
    table.add_row("Alpha (Dirichlet)", str(cfg.simulation.alpha))
    table.add_row("Dropout rate", f"{cfg.simulation.dropout_rate:.0%}")
    table.add_row("Local epochs", str(cfg.client.local_epochs))
    table.add_row("Batch size", str(cfg.client.batch_size))
    table.add_row("Learning rate", str(cfg.client.lr))
    table.add_row("DP enabled", "[yellow]yes[/yellow]" if cfg.privacy.enabled else "no")
    table.add_row("Output dir", f"{cfg.output.dir}/{cfg.name}/")

    console.print(table)
