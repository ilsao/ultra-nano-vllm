from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from benchmarks.metrics import BenchmarkResult, MetricSummary

if TYPE_CHECKING:
    from benchmarks.benchmark import BenchmarkConfig
    from benchmarks.runner import BenchmarkConfiguration


# Keep standalone benchmark reports in their established output directory.
REPORT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "report"


class Reporter:
    """Render benchmark progress/results and persist structured reports."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def warmup(self):
        return self.console.status("[bold cyan]Warming up...", spinner="dots")

    def warmup_complete(self) -> None:
        self.console.print("[bold green]✓[/] Warmup complete")

    def track_repeats(self, repeats: int) -> Iterator[int]:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("Benchmarking", total=repeats)
            for run_index in range(repeats):
                yield run_index
                progress.update(task_id, advance=1)

    def show_result(
        self,
        bench_result: BenchmarkResult,
        configuration: BenchmarkConfiguration,
    ) -> None:
        config = Table.grid(padding=(0, 2))
        config.title = "[bold cyan]Configuration[/bold cyan]"
        config.title_justify = "left"
        config.add_column(style="cyan")
        config.add_column(justify="right")
        config.add_row("Model", configuration.model)
        config.add_row("Device", configuration.device)
        config.add_row("Dtype", configuration.dtype)
        config.add_row(
            "Tensor parallel",
            f"{configuration.tensor_parallel_size:,}",
        )
        config.add_row("Enforce eager", str(configuration.enforce_eager))
        config.add_row("Block size", f"{configuration.block_size:,}")
        config.add_row("Max model length", f"{configuration.max_model_len:,}")
        config.add_row(
            "GPU memory utilization",
            f"{configuration.gpu_memory_utilization:.0%}",
        )
        config.add_row(
            "KV-cache blocks",
            f"{configuration.num_kvcache_blocks:,}",
        )

        workload = Table.grid(padding=(0, 2))
        workload.title = "[bold cyan]Workload[/bold cyan]"
        workload.title_justify = "left"
        workload.add_column(style="cyan")
        workload.add_column(justify="right")
        workload.add_row("Repeats", f"{bench_result.repeats:,}")
        workload.add_row("Requests / run", f"{bench_result.num_requests:,}")
        workload.add_row(
            "Input tokens / run",
            _format_tokens(bench_result.input_tokens),
        )
        workload.add_row(
            "Output tokens / run",
            _format_token_summary(bench_result.output_tokens),
        )
        workload.add_row(
            "Total tokens / run",
            _format_token_summary(bench_result.total_tokens),
        )

        performance = Table(
            title="[bold cyan]Performance[/bold cyan]",
            box=box.SIMPLE_HEAD,
            header_style="bold",
            title_justify="left",
            expand=False,
        )
        performance.add_column("Metric")
        performance.add_column("Median", justify="right")
        performance.add_column("Min", justify="right")
        performance.add_column("Max", justify="right")
        performance.add_column("Mean ± std", justify="right")
        _add_metric_row(performance, "Elapsed time", bench_result.elapsed_time, "s")
        _add_metric_row(performance, "Latency p50", bench_result.latency_p50, "s")
        _add_metric_row(performance, "Latency p90", bench_result.latency_p90, "s")
        _add_metric_row(performance, "Latency p99", bench_result.latency_p99, "s")
        _add_metric_row(
            performance,
            "Request throughput",
            bench_result.request_throughput,
            "req/s",
        )
        _add_metric_row(
            performance,
            "Output throughput",
            bench_result.output_throughput,
            "tok/s",
        )
        _add_metric_row(
            performance,
            "Total throughput",
            bench_result.total_throughput,
            "tok/s",
        )
        _add_metric_row(
            performance,
            "Prefill throughput",
            bench_result.prefill_throughput,
            "tok/s",
        )
        _add_metric_row(
            performance,
            "Decode throughput",
            bench_result.decode_throughput,
            "tok/s",
        )
        _add_metric_row(
            performance,
            "Prefill time",
            bench_result.prefill_time,
            "s",
        )
        _add_metric_row(
            performance,
            "Decode time",
            bench_result.decode_time,
            "s",
        )

        kvcache = Table.grid(padding=(0, 2))
        kvcache.title = "[bold cyan]KV Cache[/bold cyan]"
        kvcache.title_justify = "left"
        kvcache.add_column(style="cyan")
        kvcache.add_column(justify="right")
        kvcache.add_row(
            "Peak used blocks",
            f"{bench_result.peak_kvcache_blocks:,} blocks",
        )
        kvcache.add_row(
            "Peak utilization",
            f"{bench_result.peak_kvcache_utilization:.2%}",
        )

        self.console.print(
            Panel(
                Group(config, workload, performance, kvcache),
                title="[bold]Nano-vLLM Offline Benchmark[/bold]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def save_result(
        self,
        bench_result: BenchmarkResult,
        configuration: BenchmarkConfiguration,
        experiment_name: str,
        *,
        output_path: Path | None = None,
        experiment_config: BenchmarkConfig | None = None,
    ) -> Path:
        """Save a benchmark result as a structured JSON report."""
        report_path = output_path or REPORT_DIR / (
            f"{_report_timestamp()}-{experiment_name}-benchmark.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {"experiment_name": experiment_name}
        if experiment_config is not None:
            report["experiment_config"] = asdict(experiment_config)
        report["configuration"] = asdict(configuration)
        report["result"] = asdict(bench_result)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.console.print(
            f"[bold green]✓[/] Report saved to [cyan]{report_path}[/cyan]"
        )
        return report_path

    def show_plot_paths(self, paths: Sequence[Path]) -> None:
        """Display the output path for each generated experiment plot."""
        for path in paths:
            self.console.print(
                f"[bold green]✓[/] Plot saved to [cyan]{path}[/cyan]"
            )


def _report_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H%M%S")


def _format_tokens(value: float) -> str:
    return f"{value:,.0f} tokens"


def _format_token_summary(summary: MetricSummary) -> str:
    value = _format_tokens(summary.median)
    if summary.minimum == summary.maximum:
        return value
    return (
        f"{value} [dim](range {_format_tokens(summary.minimum)} – "
        f"{_format_tokens(summary.maximum)})[/dim]"
    )


def _add_metric_row(
    table: Table,
    label: str,
    summary: MetricSummary,
    unit: str,
) -> None:
    variability = (
        f"[dim]{summary.mean:,.2f} ± "
        f"{summary.standard_deviation:,.2f} {unit}[/dim]"
        if summary.mean is not None and summary.standard_deviation is not None
        else "[dim]n/a[/dim]"
    )
    table.add_row(
        label,
        f"[bold green]{summary.median:,.2f} {unit}[/bold green]",
        f"[dim]{summary.minimum:,.2f} {unit}[/dim]",
        f"[dim]{summary.maximum:,.2f} {unit}[/dim]",
        variability,
    )
