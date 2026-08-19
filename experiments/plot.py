"""Plot benchmark experiment results across one or two parameter dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import tempfile
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
from experiments.config import CONFIG_DIMENSIONS, config_dimension_value
from nanovllm.engine.component import EngineComponent


REPORT_DIR = Path(__file__).resolve().parent / "report"


@dataclass(frozen=True, slots=True)
class PlotRecord:
    """One resolved benchmark input paired with its measured result."""

    config: BenchmarkConfig
    result: BenchmarkResult


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
        "peak-memory-allocated",
        "Peak memory allocated",
        "MiB",
        False,
        lambda result: result.peak_memory_allocated_mib,
    ),
    _Metric(
        "peak-memory-reserved",
        "Peak memory reserved",
        "MiB",
        False,
        lambda result: result.peak_memory_reserved_mib,
    ),
)


def infer_dimensions(records: Sequence[PlotRecord]) -> tuple[str, ...]:
    """Return config fields whose values vary, in BenchmarkConfig field order."""
    if not records:
        raise ValueError("at least one plot record is required")
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
            result = _decode_result(raw_result)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"could not load experiment report {path}: {exc}") from exc
        records.append(PlotRecord(config=config, result=result))
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
        medians = []
        minimums = []
        maximums = []
        for record in ordered:
            value = metric.value(record.result)
            if metric.summary:
                assert isinstance(value, MetricSummary)
                medians.append(value.median)
                minimums.append(value.minimum)
                maximums.append(value.maximum)
            else:
                scalar = float(value)
                medians.append(scalar)

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(x_positions, medians, marker="o", linewidth=2)
        if metric.summary:
            axis.fill_between(x_positions, minimums, maximums, alpha=0.2)
        axis.set_xticks(
            x_positions,
            [_format_dimension_value(value) for value in values],
        )
        axis.set_xlabel(_format_dimension_name(dimension))
        axis.set_ylabel(f"{metric.title} ({metric.unit})")
        axis.set_title(f"{plot_name}: {metric.title}")
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        path = _output_path(
            output_dir,
            timestamp,
            plot_name,
            f"1d-{dimension}",
            metric.name,
        )
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    return paths


def plot_2d(
    records: Sequence[PlotRecord],
    x_dimension: str,
    y_dimension: str,
    *,
    output_dir: str | Path = REPORT_DIR,
    plot_name: str = "experiment",
) -> list[Path]:
    """Save one annotated heatmap per metric for a two-dimensional experiment."""
    validate_grid(records, (x_dimension, y_dimension))
    x_values = _ordered_values(records, x_dimension)
    y_values = _ordered_values(records, y_dimension)
    by_coordinate = {
        (
            _dimension_value(record, x_dimension),
            _dimension_value(record, y_dimension),
        ): record
        for record in records
    }
    timestamp = _plot_timestamp()
    paths = []

    for metric in _METRICS:
        matrix = np.empty((len(y_values), len(x_values)), dtype=float)
        annotations: list[list[str]] = []
        for y_index, y_value in enumerate(y_values):
            annotation_row = []
            for x_index, x_value in enumerate(x_values):
                value = metric.value(by_coordinate[(x_value, y_value)].result)
                if metric.summary:
                    assert isinstance(value, MetricSummary)
                    matrix[y_index, x_index] = value.median
                    annotation_row.append(
                        f"{value.median:.2f}\n"
                        f"[{value.minimum:.2f}, {value.maximum:.2f}]"
                    )
                else:
                    scalar = float(value)
                    matrix[y_index, x_index] = scalar
                    annotation_row.append(f"{scalar:.2f}")
            annotations.append(annotation_row)

        width = max(8.0, 1.7 * len(x_values))
        height = max(5.0, 1.2 * len(y_values))
        figure, axis = plt.subplots(figsize=(width, height))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(
            range(len(x_values)),
            [_format_dimension_value(value) for value in x_values],
        )
        axis.set_yticks(
            range(len(y_values)),
            [_format_dimension_value(value) for value in y_values],
        )
        axis.set_xlabel(_format_dimension_name(x_dimension))
        axis.set_ylabel(_format_dimension_name(y_dimension))
        axis.set_title(f"{plot_name}: {metric.title} ({metric.unit})")
        figure.colorbar(image, ax=axis, label=metric.unit)
        threshold = (float(matrix.min()) + float(matrix.max())) / 2
        for y_index, row in enumerate(annotations):
            for x_index, annotation in enumerate(row):
                color = "white" if matrix[y_index, x_index] <= threshold else "black"
                axis.text(
                    x_index,
                    y_index,
                    annotation,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
        figure.tight_layout()
        path = _output_path(
            output_dir,
            timestamp,
            plot_name,
            f"2d-{x_dimension}-{y_dimension}",
            metric.name,
        )
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    return paths


def _decode_result(raw: dict[str, object]) -> BenchmarkResult:
    decoded = dict(raw)
    for field_name in (
        "output_tokens",
        "total_tokens",
        "elapsed_time",
        "request_throughput",
        "output_throughput",
        "total_throughput",
    ):
        value = decoded.get(field_name)
        if not isinstance(value, dict):
            raise TypeError(f"result.{field_name} must be an object")
        decoded[field_name] = MetricSummary(**value)
    return BenchmarkResult(**decoded)


def _validate_dimensions(dimensions: Sequence[str]) -> None:
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("plot dimensions must be distinct")
    unknown = [
        dimension for dimension in dimensions if dimension not in CONFIG_DIMENSIONS
    ]
    if unknown:
        raise ValueError("unknown plot dimensions: " + ", ".join(unknown))


def _dimension_value(record: PlotRecord, dimension: str):
    return config_dimension_value(record.config, dimension)


def _ordered_values(records: Sequence[PlotRecord], dimension: str) -> list[object]:
    values = []
    for record in records:
        value = _dimension_value(record, dimension)
        if value not in values:
            values.append(value)
    return values


def _format_dimension_name(name: str) -> str:
    return name.replace("_", " ").title()


def _format_dimension_value(value: object) -> str:
    return str(value)


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
