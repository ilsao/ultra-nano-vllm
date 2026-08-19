"""Expand normalized experiment configurations into scalar benchmark runs."""

from dataclasses import fields
from itertools import product

from benchmarks.benchmark import BenchmarkConfig
from experiments.config import Config
from nanovllm.engine.component import EngineComponent


def expand_config(config: Config) -> list[BenchmarkConfig]:
    """Expand one normalized YAML specification in deterministic field order."""
    benchmark_fields = [
        field
        for field in fields(BenchmarkConfig)
        if field.name not in {"engine_component", "experiment_name"}
    ]
    component_fields = list(fields(EngineComponent))
    sweep_fields = benchmark_fields + component_fields
    combinations = list(
        product(*(getattr(config, field.name) for field in sweep_fields))
    )
    if len(config.experiment_name) != len(combinations):
        raise ValueError(
            "experiment_name must contain exactly one name per expanded "
            f"configuration (expected {len(combinations)}, got "
            f"{len(config.experiment_name)})"
        )

    benchmark_names = [field.name for field in benchmark_fields]
    component_names = [field.name for field in component_fields]
    benchmark_count = len(benchmark_names)
    return [
        BenchmarkConfig(
            **dict(zip(benchmark_names, values[:benchmark_count])),
            engine_component=EngineComponent(
                **dict(zip(component_names, values[benchmark_count:]))
            ),
            experiment_name=experiment_name,
        )
        for values, experiment_name in zip(combinations, config.experiment_name)
    ]
