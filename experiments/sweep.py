"""Expand positional workload and component columns into benchmark runs."""

from dataclasses import fields

from benchmarks.benchmark import BenchmarkConfig
from experiments.config import Config, ExperimentRun
from nanovllm.engine.component import EngineComponent


def expand_config(config: Config) -> list[ExperimentRun]:
    """Zip columns within each axis, then cross workloads with components."""
    benchmark_fields = [
        field
        for field in fields(BenchmarkConfig)
        if field.name not in {"engine_component", "experiment_name"}
    ]
    component_fields = list(fields(EngineComponent))

    workload_count = _column_count(
        config,
        "workload",
        [field.name for field in benchmark_fields],
        config.workload_name,
    )
    component_count = _column_count(
        config,
        "component",
        [field.name for field in component_fields],
        config.component_name,
    )
    _validate_labels("workload_name", config.workload_name, workload_count)
    _validate_labels("component_name", config.component_name, component_count)

    workload_columns = [
        {
            field.name: _column_value(getattr(config, field.name), index)
            for field in benchmark_fields
        }
        for index in range(workload_count)
    ]
    component_columns = [
        EngineComponent(
            **{
                field.name: _column_value(getattr(config, field.name), index)
                for field in component_fields
            }
        )
        for index in range(component_count)
    ]
    if len(set(tuple(column.values()) for column in workload_columns)) != len(
        workload_columns
    ):
        raise ValueError("workload columns contain duplicate configurations")
    if len(set(component_columns)) != len(component_columns):
        raise ValueError("component columns contain duplicate configurations")

    runs = []
    for workload_index, workload in enumerate(workload_columns):
        workload_name = config.workload_name[workload_index]
        for component_index, engine_component in enumerate(component_columns):
            component_name = config.component_name[component_index]
            experiment_name = f"{component_name}-{workload_name}"
            runs.append(
                ExperimentRun(
                    config=BenchmarkConfig(
                        **workload,
                        engine_component=engine_component,
                        experiment_name=experiment_name,
                    ),
                    workload_name=workload_name,
                    component_name=component_name,
                )
            )
    return runs


def _column_count(
    config: Config,
    group_name: str,
    field_names: list[str],
    labels: tuple[str, ...],
) -> int:
    count = max([len(getattr(config, name)) for name in field_names] + [len(labels)])
    invalid = [
        name
        for name in field_names
        if len(getattr(config, name)) not in {1, count}
    ]
    if invalid:
        raise ValueError(
            f"{group_name} column fields must contain either one value or "
            f"exactly {count}; invalid fields: {', '.join(invalid)}"
        )
    return count


def _validate_labels(field_name: str, labels: tuple[str, ...], count: int) -> None:
    if len(labels) != count:
        raise ValueError(
            f"{field_name} must contain exactly one label per column "
            f"(expected {count}, got {len(labels)})"
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"{field_name} values must be unique")


def _column_value(values: tuple[object, ...], index: int) -> object:
    return values[0] if len(values) == 1 else values[index]
