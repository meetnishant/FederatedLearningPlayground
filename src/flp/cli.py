"""CLI entrypoint for the Federated Learning Playground (FLP)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
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
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable DEBUG logging.")
def run_experiment(config: Path, output_dir: Path | None, verbose: bool) -> None:
    """Run a federated learning experiment from a YAML config file.

    \b
    Example:
        flp run --config configs/baseline.yaml
        flp run --config configs/baseline.yaml --output-dir /tmp/flp_out -v
    """
    _configure_logging(verbose)

    from flp.experiments.config_loader import load_config
    from flp.experiments.runner import ExperimentRunner

    console.print(
        Panel.fit(
            "[bold cyan]Federated Learning Playground[/bold cyan]\n"
            f"[dim]Config: {config}[/dim]",
            border_style="cyan",
        )
    )

    try:
        exp_config = load_config(config)
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(1)

    if output_dir is not None:
        exp_config.output.dir = str(output_dir)

    # Print experiment overview
    _print_config_summary(exp_config)

    try:
        runner = ExperimentRunner(exp_config)
        metrics = runner.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Experiment interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"[bold red]Experiment failed:[/bold red] {exc}")
        if verbose:
            console.print_exception()
        sys.exit(1)

    # Final summary table
    summary = metrics.summary()
    table = Table(title="Experiment Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_row("Rounds completed", str(summary["num_rounds"]))
    table.add_row("Best accuracy", f"{float(summary['best_accuracy']):.4f}")
    table.add_row("Final accuracy", f"{float(summary['final_accuracy']):.4f}")
    table.add_row("Final loss", f"{float(summary['final_loss']):.4f}")
    console.print(table)
    console.print("[bold green]Done.[/bold green]")


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
        console.print(f"  Experiment name : {cfg.name}")
        console.print(f"  Dataset         : {cfg.dataset}")
        console.print(f"  Clients         : {cfg.training.num_clients}")
        console.print(f"  Rounds          : {cfg.training.num_rounds}")
        console.print(f"  Partitioning    : {cfg.simulation.partitioning}")
        console.print(f"  Dropout rate    : {cfg.simulation.dropout_rate}")
        console.print(f"  DP enabled      : {cfg.privacy.enabled}")
    except Exception as exc:
        console.print(f"[bold red]Config invalid:[/bold red] {exc}")
        sys.exit(1)


def _print_config_summary(cfg: object) -> None:
    from flp.experiments.config_loader import ExperimentConfig

    assert isinstance(cfg, ExperimentConfig)
    table = Table(title="Experiment Configuration", show_header=False)
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("Name", cfg.name)
    table.add_row("Dataset", cfg.dataset)
    table.add_row("Seed", str(cfg.seed))
    table.add_row("Clients", str(cfg.training.num_clients))
    table.add_row("Rounds", str(cfg.training.num_rounds))
    table.add_row("Client fraction", str(cfg.training.client_fraction))
    table.add_row("Partitioning", cfg.simulation.partitioning)
    table.add_row("Alpha (Dirichlet)", str(cfg.simulation.alpha))
    table.add_row("Dropout rate", str(cfg.simulation.dropout_rate))
    table.add_row("Local epochs", str(cfg.client.local_epochs))
    table.add_row("Batch size", str(cfg.client.batch_size))
    table.add_row("Learning rate", str(cfg.client.lr))
    table.add_row("DP enabled", str(cfg.privacy.enabled))
    table.add_row("Output dir", cfg.output.dir)
    console.print(table)
