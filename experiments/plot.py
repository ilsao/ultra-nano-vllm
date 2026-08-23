"""Plot benchmark experiment results across one or two parameter dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import tempfile
import textwrap
from typing import Callable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nano-vllm-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.benchmark import BenchmarkConfig
from benchmarks.metrics import BenchmarkResult, MetricSummary
from benchmarks.runner import BatchRunResult
from experiments.config import (
    CONFIG_DIMENSIONS,
    EXPERIMENT_DIMENSIONS,
    config_dimension_value,
    validate_experiment_name,
)
from nanovllm.engine.component import EngineComponent
from nanovllm.engine.scheduler_metrics import SchedulerMetrics


REPORT_DIR = Path(__file__).resolve().parent / "report"

_SCHEDULER_SERIES_STYLES = {
    "scheduler": ("tab:blue", "o"),
    "scheduler-v1": ("tab:orange", "s"),
    "scheduler-v2": ("tab:green", "^"),
}


@dataclass(frozen=True, slots=True)
class PlotRecord:
    """One resolved benchmark input paired with its measured result."""

    config: BenchmarkConfig
    result: BenchmarkResult
    workload_name: str | None = None
    component_name: str | None = None


@dataclass(frozen=True, slots=True)
class _Metric:
    name: str
    title: str
    unit: str
    summary: bool
    value: Callable[[BenchmarkResult], MetricSummary | float]


_METRICS = (
    _Metric(
        "elapsed-time",
        "Elapsed time",
        "s",
        True,
        lambda result: result.elapsed_time,
    ),
    _Metric(
        "request-throughput",
        "Request throughput",
        "req/s",
        True,
        lambda result: result.request_throughput,
    ),
    _Metric(
        "latency-p50",
        "Latency p50",
        "s",
        True,
        lambda result: result.latency_p50,
    ),
    _Metric(
        "latency-p90",
        "Latency p90",
        "s",
        True,
        lambda result: result.latency_p90,
    ),
    _Metric(
        "latency-p99",
        "Latency p99",
        "s",
        True,
        lambda result: result.latency_p99,
    ),
    _Metric(
        "output-throughput",
        "Output throughput",
        "tok/s",
        True,
        lambda result: result.output_throughput,
    ),
    _Metric(
        "total-throughput",
        "Total throughput",
        "tok/s",
        True,
        lambda result: result.total_throughput,
    ),
    _Metric(
        "prefill-throughput",
        "Prefill throughput",
        "tok/s",
        True,
        lambda result: result.prefill_throughput,
    ),
    _Metric(
        "decode-throughput",
        "Decode throughput",
        "tok/s",
        True,
        lambda result: result.decode_throughput,
    ),
    _Metric(
        "prefill-time",
        "Prefill time",
        "s",
        True,
        lambda result: result.prefill_time,
    ),
    _Metric(
        "decode-time",
        "Decode time",
        "s",
        True,
        lambda result: result.decode_time,
    ),
    _Metric(
        "peak-kvcache-blocks",
        "Peak KV-cache blocks",
        "blocks",
        False,
        lambda result: result.peak_kvcache_blocks,
    ),
    _Metric(
        "peak-kvcache-utilization",
        "Peak KV-cache utilization",
        "%",
        False,
        lambda result: result.peak_kvcache_utilization * 100,
    ),
)


def infer_dimensions(records: Sequence[PlotRecord]) -> tuple[str, ...]:
    """Return config fields whose values vary, in BenchmarkConfig field order."""
    if not records:
        raise ValueError("at least one plot record is required")
    labeled = [
        record.workload_name is not None and record.component_name is not None
        for record in records
    ]
    partially_labeled = [
        (record.workload_name is None) != (record.component_name is None)
        for record in records
    ]
    if any(partially_labeled) or (any(labeled) and not all(labeled)):
        raise ValueError(
            "experiment results must either all include workload/component "
            "labels or all use the legacy unlabeled format"
        )
    if all(labeled):
        return tuple(
            dimension
            for dimension in EXPERIMENT_DIMENSIONS
            if len({_dimension_value(record, dimension) for record in records}) > 1
        )
    return tuple(
        dimension
        for dimension in CONFIG_DIMENSIONS
        if len({_dimension_value(record, dimension) for record in records}) > 1
    )


def validate_grid(
    records: Sequence[PlotRecord],
    dimensions: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Validate that records form a unique, complete grid of up to two dimensions."""
    inferred = infer_dimensions(records)
    selected = tuple(dimensions) if dimensions is not None else inferred
    _validate_dimensions(selected)
    if selected != inferred:
        raise ValueError(
            f"plot dimensions {selected!r} do not match varying fields {inferred!r}"
        )
    if len(selected) > 2:
        raise ValueError(
            "experiments support at most two varying parameters; got "
            + ", ".join(selected)
        )

    coordinates = [
        tuple(_dimension_value(record, dimension) for dimension in selected)
        for record in records
    ]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("experiment results contain duplicate parameter coordinates")

    expected = 1
    for dimension in selected:
        expected *= len(_ordered_values(records, dimension))
    if len(records) != expected:
        raise ValueError(
            f"experiment results do not form a complete parameter grid "
            f"(expected {expected}, got {len(records)})"
        )
    return selected


def load_reports(paths: Sequence[str | Path]) -> list[PlotRecord]:
    """Load explicit experiment JSON reports into plot records."""
    if not paths:
        raise ValueError("at least one report path is required")

    records = []
    for path_value in paths:
        path = Path(path_value)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise TypeError("top-level JSON value must be an object")
            raw_config = report["experiment_config"]
            raw_result = report["result"]
            if not isinstance(raw_config, dict) or not isinstance(raw_result, dict):
                raise TypeError("experiment_config and result must be objects")
            decoded_config = dict(raw_config)
            raw_components = decoded_config.get("engine_component")
            if raw_components is None:
                decoded_config["engine_component"] = EngineComponent()
            elif isinstance(raw_components, dict):
                decoded_config["engine_component"] = EngineComponent(
                    **raw_components
                )
            else:
                raise TypeError("experiment_config.engine_component must be an object")
            config = BenchmarkConfig(**decoded_config)
            workload_name = report.get("workload_name")
            component_name = report.get("component_name")
            if (workload_name is None) != (component_name is None):
                raise ValueError(
                    "workload_name and component_name must be provided together"
                )
            if workload_name is not None:
                if not isinstance(workload_name, str) or not isinstance(
                    component_name, str
                ):
                    raise TypeError(
                        "workload_name and component_name must be strings"
                    )
                validate_experiment_name(workload_name)
                validate_experiment_name(component_name)
            result = _decode_result(raw_result)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"could not load experiment report {path}: {exc}") from exc
        records.append(
            PlotRecord(
                config=config,
                result=result,
                workload_name=workload_name,
                component_name=component_name,
            )
        )
    return records


def plot_1d(
    records: Sequence[PlotRecord],
    dimension: str,
    *,
    output_dir: str | Path = REPORT_DIR,
    plot_name: str = "experiment",
) -> list[Path]:
    """Save one line chart per metric for a one-dimensional experiment."""
    validate_grid(records, (dimension,))
    values = _ordered_values(records, dimension)
    by_value = {_dimension_value(record, dimension): record for record in records}
    ordered = [by_value[value] for value in values]
    x_positions = np.arange(len(values))
    timestamp = _plot_timestamp()
    paths = []

    for metric in _METRICS:
        centers = []
        lower_bounds = []
        upper_bounds = []
        for record in ordered:
            value = metric.value(record.result)
            if metric.summary:
                assert isinstance(value, MetricSummary)
                if value.mean is not None and value.standard_deviation is not None:
                    centers.append(value.mean)
                    lower_bounds.append(value.mean - value.standard_deviation)
                    upper_bounds.append(value.mean + value.standard_deviation)
                else:
                    centers.append(value.median)
                    lower_bounds.append(value.minimum)
                    upper_bounds.append(value.maximum)
            else:
                scalar = float(value)
                centers.append(scalar)

        labels = [_format_dimension_value(value, multiline=True) for value in values]
        figure, axis = plt.subplots(
            figsize=_figure_size(values, labels),
            constrained_layout=True,
        )
        axis.plot(x_positions, centers, marker="o", linewidth=2)
        if metric.summary:
            axis.fill_between(x_positions, lower_bounds, upper_bounds, alpha=0.2)
        axis.set_xticks(
            x_positions,
            labels,
        )
        axis.tick_params(axis="x", labelrotation=0)
        axis.set_xlabel(_format_dimension_name(dimension))
        axis.set_ylabel(f"{metric.title} ({metric.unit})")
        axis.set_title(f"{plot_name}: {metric.title}")
        axis.grid(axis="y", alpha=0.3)
        path = _output_path(
            output_dir,
            timestamp,
            plot_name,
            f"1d-{dimension}",
            metric.name,
        )
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def plot_2d(
    records: Sequence[PlotRecord],
    x_dimension: str,
    series_dimension: str,
    *,
    output_dir: str | Path = REPORT_DIR,
    plot_name: str = "experiment",
) -> list[Path]:
    """Save one multi-series line chart per metric for a 2D experiment."""
    validate_grid(records, (x_dimension, series_dimension))
    x_values = _ordered_values(records, x_dimension)
    series_values = _ordered_values(records, series_dimension)
    by_coordinate = {
        (
            _dimension_value(record, x_dimension),
            _dimension_value(record, series_dimension),
        ): record
        for record in records
    }
    x_positions = np.arange(len(x_values))
    timestamp = _plot_timestamp()
    paths = []
    markers = ("o", "s", "^", "D", "v", "P", "X", "*")
    scheduler_selectors = _scheduler_series_selectors(
        records, series_dimension
    )

    for metric in _METRICS:
        labels = [
            _format_dimension_value(value, multiline=True) for value in x_values
        ]
        figure, axis = plt.subplots(
            figsize=_figure_size(x_values, labels),
            constrained_layout=True,
        )
        for series_index, series_value in enumerate(series_values):
            scheduler_selector = scheduler_selectors.get(series_value)
            centers = []
            lower_bounds = []
            upper_bounds = []
            for x_value in x_values:
                value = metric.value(
                    by_coordinate[(x_value, series_value)].result
                )
                if metric.summary:
                    assert isinstance(value, MetricSummary)
                    if value.mean is not None and value.standard_deviation is not None:
                        centers.append(value.mean)
                        lower_bounds.append(value.mean - value.standard_deviation)
                        upper_bounds.append(value.mean + value.standard_deviation)
                    else:
                        centers.append(value.median)
                        lower_bounds.append(value.minimum)
                        upper_bounds.append(value.maximum)
                else:
                    centers.append(float(value))

            style = _SCHEDULER_SERIES_STYLES.get(scheduler_selector)
            line_options = {
                "marker": (
                    style[1] if style else markers[series_index % len(markers)]
                ),
                "linewidth": 2,
                "label": (
                    scheduler_selector
                    if scheduler_selector is not None
                    else _format_dimension_value(series_value, multiline=True)
                ),
            }
            if style:
                line_options["color"] = style[0]
            line = axis.plot(
                x_positions,
                centers,
                **line_options,
            )[0]
            if metric.summary:
                axis.fill_between(
                    x_positions,
                    lower_bounds,
                    upper_bounds,
                    color=line.get_color(),
                    alpha=0.16,
                )

        axis.set_xticks(
            x_positions,
            labels,
        )
        axis.tick_params(axis="x", labelrotation=0)
        axis.set_xlabel(_format_dimension_name(x_dimension))
        axis.set_ylabel(f"{metric.title} ({metric.unit})")
        axis.set_title(f"{plot_name}: {metric.title} ({metric.unit})")
        axis.legend(
            title=_format_dimension_name(series_dimension),
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
        )
        axis.grid(axis="y", alpha=0.3)
        path = _output_path(
            output_dir,
            timestamp,
            plot_name,
            f"2d-{x_dimension}-by-{series_dimension}",
            metric.name,
        )
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def _scheduler_series_selectors(
    records: Sequence[PlotRecord], series_dimension: str
) -> dict[object, str]:
    """Map scheduler-driven series values to their selector labels."""
    if series_dimension == "scheduler":
        return {
            _dimension_value(record, series_dimension): str(
                _dimension_value(record, series_dimension)
            )
            for record in records
        }
    if series_dimension != "component":
        return {}

    selectors: dict[object, str] = {}
    for record in records:
        series_value = _dimension_value(record, series_dimension)
        selector = record.config.engine_component.scheduler
        previous = selectors.setdefault(series_value, selector)
        if previous != selector:
            return {}
    if len(set(selectors.values())) != len(selectors):
        return {}
    return selectors


def select_2d_dimensions(
    records: Sequence[PlotRecord],
    dimensions: Sequence[str],
) -> tuple[str, str]:
    """Choose an x and series dimension without special-casing field names."""
    if len(dimensions) != 2:
        raise ValueError("two dimensions are required for a 2D plot")
    first, second = dimensions
    first_is_numeric = _is_numeric_dimension(records, first)
    second_is_numeric = _is_numeric_dimension(records, second)
    if first_is_numeric != second_is_numeric:
        return (first, second) if first_is_numeric else (second, first)
    return first, second


def _decode_result(raw: dict[str, object]) -> BenchmarkResult:
    decoded = dict(raw)
    raw_samples = decoded.pop("run_samples", [])
    if not isinstance(raw_samples, list):
        raise TypeError("result.run_samples must be an array")
    if (
        "peak_kvcache_blocks" not in decoded
        or "peak_kvcache_utilization" not in decoded
    ):
        if {
            "peak_memory_allocated_mib",
            "peak_memory_reserved_mib",
        } & decoded.keys():
            raise ValueError(
                "report uses legacy GPU-memory metrics; rerun the experiment "
                "to collect KV-cache block metrics"
            )
        raise ValueError("result is missing KV-cache block metrics")
    phase_fields = {
        "latency_p50",
        "latency_p90",
        "latency_p99",
        "prefill_throughput",
        "decode_throughput",
        "prefill_time",
        "decode_time",
    }
    if missing := phase_fields - decoded.keys():
        raise ValueError(
            "report is missing latency/phase metrics "
            f"({', '.join(sorted(missing))}); rerun the experiment"
        )
    for field_name in (
        "output_tokens",
        "total_tokens",
        "elapsed_time",
        "latency_p50",
        "latency_p90",
        "latency_p99",
        "request_throughput",
        "output_throughput",
        "total_throughput",
        "prefill_throughput",
        "decode_throughput",
        "prefill_time",
        "decode_time",
    ):
        value = decoded.get(field_name)
        if not isinstance(value, dict):
            raise TypeError(f"result.{field_name} must be an object")
        decoded[field_name] = MetricSummary(**value)
    samples = []
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, dict):
            raise TypeError("every result.run_samples entry must be an object")
        sample = dict(raw_sample)
        raw_scheduler_metrics = sample.get("scheduler_metrics", {})
        if not isinstance(raw_scheduler_metrics, dict):
            raise TypeError("run sample scheduler_metrics must be an object")
        sample["scheduler_metrics"] = SchedulerMetrics(**raw_scheduler_metrics)
        for tuple_field in ("request_latencies", "request_input_lengths"):
            value = sample.get(tuple_field, [])
            if not isinstance(value, list):
                raise TypeError(f"run sample {tuple_field} must be an array")
            sample[tuple_field] = tuple(value)
        samples.append(BatchRunResult(**sample))
    decoded["run_samples"] = tuple(samples)
    return BenchmarkResult(**decoded)


def _validate_dimensions(dimensions: Sequence[str]) -> None:
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("plot dimensions must be distinct")
    unknown = [
        dimension
        for dimension in dimensions
        if dimension not in CONFIG_DIMENSIONS + EXPERIMENT_DIMENSIONS
    ]
    if unknown:
        raise ValueError("unknown plot dimensions: " + ", ".join(unknown))


def _dimension_value(record: PlotRecord, dimension: str):
    if dimension == "workload":
        if record.workload_name is None:
            raise ValueError("workload dimension requires labeled experiment results")
        return record.workload_name
    if dimension == "component":
        if record.component_name is None:
            raise ValueError("component dimension requires labeled experiment results")
        return record.component_name
    return config_dimension_value(record.config, dimension)


def _ordered_values(records: Sequence[PlotRecord], dimension: str) -> list[object]:
    values = []
    for record in records:
        value = _dimension_value(record, dimension)
        if value not in values:
            values.append(value)
    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        values.sort()
    return values


def _is_numeric_dimension(
    records: Sequence[PlotRecord],
    dimension: str,
) -> bool:
    return all(
        isinstance(_dimension_value(record, dimension), (int, float))
        and not isinstance(_dimension_value(record, dimension), bool)
        for record in records
    )


def _format_dimension_name(name: str) -> str:
    return name.replace("_", " ").title()


def _format_dimension_value(value: object, *, multiline: bool = False) -> str:
    formatted = str(value)
    if not multiline or not isinstance(value, str):
        return formatted
    return "\n".join(
        textwrap.wrap(
            formatted,
            width=14,
            break_long_words=True,
            break_on_hyphens=True,
        )
    )


def _figure_size(
    values: Sequence[object],
    labels: Sequence[str],
) -> tuple[float, float]:
    width = max(8.0, min(24.0, 1.5 * len(values)))
    max_lines = max((label.count("\n") + 1 for label in labels), default=1)
    height = 5.0 + 0.3 * (max_lines - 1)
    return width, height


def _plot_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H%M%S%f")


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return slug or "experiment"


def _output_path(
    output_dir: str | Path,
    timestamp: str,
    plot_name: str,
    dimensions: str,
    metric: str,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = "-".join(
        (_slug(timestamp), _slug(plot_name), _slug(dimensions), _slug(metric))
    )
    candidate = directory / f"{stem}.png"
    suffix = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{suffix}.png"
        suffix += 1
    return candidate
