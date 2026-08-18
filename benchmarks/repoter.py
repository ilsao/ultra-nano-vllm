from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING

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

from metrics import BenchmarkResult, MetricSummary

if TYPE_CHECKING:
    from runner import BenchmarkConfiguration


REPORT_DIR = Path(__file__).resolve().parent.parent / "report"


class BenchmarkReporter:
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
        _add_metric_row(performance, "Elapsed time", bench_result.elapsed_time, "s")
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

        memory = Table.grid(padding=(0, 2))
        memory.title = "[bold cyan]Memory[/bold cyan]"
        memory.title_justify = "left"
        memory.add_column(style="cyan")
        memory.add_column(justify="right")
        memory.add_row(
            "Peak allocated",
            _format_memory(bench_result.peak_memory_allocated_mib),
        )
        memory.add_row(
            "Peak reserved",
            _format_memory(bench_result.peak_memory_reserved_mib),
        )

        self.console.print(
            Panel(
                Group(config, workload, performance, memory),
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
        serving_system_name: str,
    ) -> Path:
        """ 
        Save the benchmark result to a JSON file in the report directory.
        
        Args:
            bench_result (BenchmarkResult): The benchmark result to save.
            configuration (BenchmarkConfiguration): The benchmark configuration.
            serving_system_name (str): The name of the serving system, used in the filename.
            
        Returns:
            Path: The path to the saved JSON file.
        """
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / (
            f"{_report_timestamp()}-{serving_system_name}-benchmark.json"
        )
        report = {
            "serving_system_name": serving_system_name,
            "configuration": asdict(configuration),
            "result": asdict(bench_result),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.console.print(
            f"[bold green]✓[/] Report saved to [cyan]{report_path}[/cyan]"
        )
        return report_path


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


def _format_memory(mib: float) -> str:
    if mib < 1024:
        return f"{mib:,.2f} MiB"
    return f"{mib / 1024:,.2f} GiB [dim]({mib:,.2f} MiB)[/dim]"


def _add_metric_row(
    table: Table,
    label: str,
    summary: MetricSummary,
    unit: str,
) -> None:
    table.add_row(
        label,
        f"[bold green]{summary.median:,.2f} {unit}[/bold green]",
        f"[dim]{summary.minimum:,.2f} {unit}[/dim]",
        f"[dim]{summary.maximum:,.2f} {unit}[/dim]",
    )
