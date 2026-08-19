"""Run YAML-defined benchmark experiments and render comparison plots."""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark import (
    BenchmarkConfig,
    add_benchmark_arguments,
)
from experiments.config import ExperimentGroup, resolve_config_groups
from experiments.plot import (
    PlotRecord,
    load_reports,
    plot_1d,
    plot_2d,
    validate_grid,
)
from experiments import runner as experiment_runner
from utils.reporter import Reporter


REPORT_DIR = Path(__file__).resolve().parent / "report"


def build_parser() -> argparse.ArgumentParser:
    """
    Provided arguments:
        --config <path>: YAML config path; may be specified more than once.
        --plot-only <path>: JSON reports to plot without benchmarking.
    """
    parser = argparse.ArgumentParser(
        description="Run and plot YAML benchmark experiments."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        dest="config_paths",
        action="append",
        type=Path,
        help="YAML experiment config path; may be specified more than once.",
    )
    source.add_argument(
        "--plot-only",
        dest="report_paths",
        nargs="+",
        type=Path,
        help="Explicit experiment JSON reports to plot without benchmarking.",
    )
    add_benchmark_arguments(parser, include_defaults=False)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.report_paths is not None:
            overrides = _benchmark_overrides(args)
            if overrides:
                raise ValueError(
                    "benchmark overrides cannot be used together with --plot-only"
                )
            args.records = load_reports(args.report_paths)
            args.dimensions = validate_grid(args.records)
        else:
            args.groups = resolve_config_groups(
                args.config_paths,
                _benchmark_overrides(args),
            )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return args


def plot_records(
    records: Sequence[PlotRecord],
    dimensions: Sequence[str],
    *,
    plot_name: str,
    output_dir: str | Path = REPORT_DIR,
) -> list[Path]:
    """Dispatch a validated result group to its dimension-specific plot API."""
    actual_dimensions = validate_grid(records, dimensions)
    if not actual_dimensions:
        return []
    if len(actual_dimensions) == 1:
        return plot_1d(
            records,
            actual_dimensions[0],
            output_dir=output_dir,
            plot_name=plot_name,
        )
    return plot_2d(
        records,
        actual_dimensions[0],
        actual_dimensions[1],
        output_dir=output_dir,
        plot_name=plot_name,
    )


def report_path(experiment_name: str, now: datetime | None = None) -> Path:
    timestamp = now or datetime.now()
    filename = (
        f"{timestamp:%Y-%m-%d}-{experiment_name}-"
        f"{timestamp:%H%M%S%f}.json"
    )
    return REPORT_DIR / filename


def run_groups(
    groups: Sequence[ExperimentGroup],
    reporter: Reporter,
) -> list[Path]:
    """Execute all groups and save their reports and comparison plots."""
    plot_paths = []
    for group in groups:
        records = []
        for config in group.configs:
            result, configuration = (
                experiment_runner.execute_benchmark_isolated(config)
            )
            reporter.show_result(result, configuration)
            reporter.save_result(
                result,
                configuration,
                config.experiment_name,
                output_path=report_path(config.experiment_name),
                experiment_config=config,
            )
            records.append(PlotRecord(config=config, result=result))
        group_paths = plot_records(
            records,
            group.dimensions,
            plot_name=group.name,
        )
        reporter.show_plot_paths(group_paths)
        plot_paths.extend(group_paths)
    return plot_paths


def main() -> None:
    args = parse_args()
    reporter = Reporter()
    if args.report_paths is not None:
        paths = plot_records(
            args.records,
            args.dimensions,
            plot_name="comparison",
        )
        reporter.show_plot_paths(paths)
        return
    run_groups(args.groups, reporter)


def _benchmark_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        field.name: value
        for field in fields(BenchmarkConfig)
        if (value := getattr(args, field.name, None)) is not None
    }


if __name__ == "__main__":
    main()
