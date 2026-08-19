"""Expand normalized experiment configurations into scalar benchmark runs."""

from dataclasses import fields
from itertools import product

from benchmarks.benchmark import BenchmarkConfig
from experiments.config import Config


def expand_config(config: Config) -> list[BenchmarkConfig]:
    """Expand one normalized YAML specification in deterministic field order."""
    sweep_fields = [
        field for field in fields(BenchmarkConfig) if field.name != "experiment_name"
    ]
    combinations = list(
        product(*(getattr(config, field.name) for field in sweep_fields))
    )
    if len(config.experiment_name) != len(combinations):
        raise ValueError(
            "experiment_name must contain exactly one name per expanded "
            f"configuration (expected {len(combinations)}, got "
            f"{len(config.experiment_name)})"
        )

    field_names = [field.name for field in sweep_fields]
    return [
        BenchmarkConfig(
            **dict(zip(field_names, values)),
            experiment_name=experiment_name,
        )
        for values, experiment_name in zip(combinations, config.experiment_name)
    ]
