from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import yaml

from nanovllm.engine.component import (
    ENGINE_COMPONENT_DIMENSIONS,
    EngineComponent,
    validate_engine_component_selector,
)

if TYPE_CHECKING:
    from benchmarks.benchmark import BenchmarkConfig


DEFAULT_MODEL_PATH = "~/huggingface/Qwen3-0.6B/"
DEFAULT_EXPERIMENT_NAME = "nano-vllm"
DEFAULT_WORKLOAD_NAME = "workload"
DEFAULT_COMPONENT_NAME = "baseline"


@dataclass(frozen=True, slots=True)
class Config:
    """A normalized, unexpanded YAML experiment specification."""

    model: tuple[str, ...] = (DEFAULT_MODEL_PATH,)
    num_requests: tuple[int, ...] = (256,)
    input_len: tuple[int, ...] = (1024,)
    output_len: tuple[int, ...] = (1024,)
    max_model_len: tuple[int, ...] = (8192,)
    seed: tuple[int, ...] = (0,)
    temperature: tuple[float, ...] = (0.6,)
    repeats: tuple[int, ...] = (3,)
    enforce_eager: tuple[bool, ...] = (False,)
    scheduler: tuple[str, ...] = ("scheduler",)
    block_manager: tuple[str, ...] = ("block_manager",)
    attention: tuple[str, ...] = ("attention",)
    sampler: tuple[str, ...] = ("sampler",)
    store_kvcache: tuple[str, ...] = ("store_kvcache",)
    alternate_input_len: tuple[int | None, ...] = (None,)
    input_order: tuple[str, ...] = ("alternating",)
    workload_name: tuple[str, ...] = (DEFAULT_WORKLOAD_NAME,)
    component_name: tuple[str, ...] = (DEFAULT_COMPONENT_NAME,)

    def __post_init__(self) -> None:
        """Check that all fields are valid."""
        for field in fields(self):
            values = getattr(self, field.name)
            if not isinstance(values, tuple):
                raise TypeError(f"{field.name} must be a tuple")
            if not values:
                raise ValueError(f"{field.name} must not be empty")
            for value in values:
                validate_field_value(field.name, value)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Read and normalize one experiment YAML file without expanding it."""
        config_path = Path(path)
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"could not read config {config_path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError(f"config {config_path} must contain a top-level mapping")

        if "experiment_name" in raw:
            raise ValueError(
                f"config {config_path} uses the removed experiment_name field; "
                "use workload_name and component_name instead"
            )

        config_fields = {field.name: field for field in fields(cls)}
        unknown_fields = sorted(set(raw) - set(config_fields), key=str)
        if unknown_fields:
            names = ", ".join(map(str, unknown_fields))
            raise ValueError(f"config {config_path} has unknown fields: {names}")

        normalized: dict[str, tuple[Any, ...]] = {}
        defaults = cls()
        for field in fields(cls):
            value = raw.get(field.name, getattr(defaults, field.name))
            if isinstance(value, tuple):
                options = value
            elif isinstance(value, list):
                options = tuple(value)
            else:
                options = (value,)
            if not options:
                raise ValueError(f"{field.name} must not be empty")
            normalized[field.name] = tuple(
                normalize_field_value(field.name, option) for option in options
            )
        return cls(**normalized)


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """One resolved benchmark run and its conceptual grid coordinates."""

    config: BenchmarkConfig
    workload_name: str
    component_name: str


@dataclass(frozen=True, slots=True)
class ExperimentGroup:
    """Resolved benchmark runs originating from one YAML file."""

    name: str
    runs: tuple[ExperimentRun, ...]
    dimensions: tuple[str, ...]

    @property
    def configs(self) -> tuple[BenchmarkConfig, ...]:
        """Return scalar configs for callers that do not need column labels."""
        return tuple(run.config for run in self.runs)


_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "model": str,
    "num_requests": int,
    "input_len": int,
    "output_len": int,
    "max_model_len": int,
    "seed": int,
    "temperature": float,
    "repeats": int,
    "enforce_eager": bool,
    "scheduler": str,
    "block_manager": str,
    "attention": str,
    "sampler": str,
    "store_kvcache": str,
    "experiment_name": str,
    "alternate_input_len": (int, type(None)),
    "input_order": str,
    "workload_name": str,
    "component_name": str,
}

CONFIG_DIMENSIONS = tuple(
    field_name
    for field_name in _FIELD_TYPES
    if field_name not in {"experiment_name", "workload_name", "component_name"}
)
EXPERIMENT_DIMENSIONS = ("workload", "component")


def resolve_config_groups(
    config_paths: Sequence[str | Path],
    overrides: Mapping[str, object] | None = None,
) -> list[ExperimentGroup]:
    """Load, expand, override, and validate independent YAML config groups."""
    # Import lazily because benchmarks.benchmark imports this module while it
    # defines BenchmarkConfig, which experiments.sweep consumes.
    from experiments.sweep import expand_config

    replacements = dict(overrides or {})
    expanded_groups = [
        (Path(path), expand_config(Config.from_yaml(path))) for path in config_paths
    ]
    total_runs = sum(len(runs) for _, runs in expanded_groups)
    if "experiment_name" in replacements and total_runs > 1:
        raise ValueError(
            "--experiment-name cannot override an invocation with multiple runs"
        )

    groups = []
    all_configs = []
    for group_index, (path, runs) in enumerate(expanded_groups, start=1):
        resolved = tuple(
            replace(run, config=replace(run.config, **replacements))
            for run in runs
        )
        for run in resolved:
            config = run.config
            validate_workload_lengths(
                config.input_len,
                config.output_len,
                config.max_model_len,
                config.experiment_name,
                config.alternate_input_len,
            )
        dimensions = validate_experiment_runs(resolved)
        group_name = path.stem
        if sum(candidate.stem == path.stem for candidate, _ in expanded_groups) > 1:
            group_name = f"{group_name}-{group_index}"
        groups.append(ExperimentGroup(group_name, resolved, dimensions))
        all_configs.extend(run.config for run in resolved)

    duplicate_names = sorted(
        name
        for name, count in Counter(
            config.experiment_name for config in all_configs
        ).items()
        if count > 1
    )
    if duplicate_names:
        raise ValueError(
            "experiment names must be unique per invocation: "
            + ", ".join(duplicate_names)
        )
    return groups


def validate_experiment_runs(
    runs: Sequence[ExperimentRun],
) -> tuple[str, ...]:
    """Validate a complete labeled workload-by-component experiment grid."""
    if not runs:
        raise ValueError("an experiment group must contain at least one run")

    workload_names = _ordered_unique(run.workload_name for run in runs)
    component_names = _ordered_unique(run.component_name for run in runs)
    coordinates = [
        (run.workload_name, run.component_name)
        for run in runs
    ]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("experiment runs contain duplicate labeled coordinates")
    expected = len(workload_names) * len(component_names)
    if len(runs) != expected:
        raise ValueError(
            "experiment runs do not form a complete workload-by-component grid "
            f"(expected {expected}, got {len(runs)})"
        )

    workload_fields = tuple(
        field.name
        for field in fields(runs[0].config)
        if field.name not in {"engine_component", "experiment_name"}
    )
    workload_values: dict[str, tuple[object, ...]] = {}
    component_values: dict[str, EngineComponent] = {}
    for run in runs:
        values = tuple(getattr(run.config, name) for name in workload_fields)
        previous_workload = workload_values.setdefault(run.workload_name, values)
        if previous_workload != values:
            raise ValueError(
                f"workload {run.workload_name!r} resolves inconsistently across "
                "component columns"
            )
        previous_component = component_values.setdefault(
            run.component_name, run.config.engine_component
        )
        if previous_component != run.config.engine_component:
            raise ValueError(
                f"component {run.component_name!r} resolves inconsistently across "
                "workload columns"
            )

    if len(set(workload_values.values())) != len(workload_values):
        raise ValueError("workload columns resolve to duplicate configurations")
    if len(set(component_values.values())) != len(component_values):
        raise ValueError("component columns resolve to duplicate configurations")

    dimensions = []
    if len(workload_names) > 1:
        dimensions.append("workload")
    if len(component_names) > 1:
        dimensions.append("component")
    return tuple(dimensions)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    ordered = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return ordered


def infer_config_dimensions(
    configs: Sequence[BenchmarkConfig],
) -> tuple[str, ...]:
    """Return scalar benchmark fields that vary, in config field order."""
    if not configs:
        raise ValueError("an experiment group must contain at least one configuration")
    return tuple(
        dimension
        for dimension in CONFIG_DIMENSIONS
        if len({config_dimension_value(config, dimension) for config in configs})
        > 1
    )


def validate_config_grid(
    configs: Sequence[BenchmarkConfig],
) -> tuple[str, ...]:
    """Reject high-dimensional, duplicate, or incomplete parameter grids."""
    dimensions = infer_config_dimensions(configs)
    if len(dimensions) > 2:
        raise ValueError(
            "experiments support at most two varying parameters; got "
            + ", ".join(dimensions)
        )
    coordinates = [
        tuple(config_dimension_value(config, dimension) for dimension in dimensions)
        for config in configs
    ]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError(
            "experiment configurations contain duplicate parameter coordinates"
        )
    expected = 1
    for dimension in dimensions:
        expected *= len(
            {config_dimension_value(config, dimension) for config in configs}
        )
    if len(configs) != expected:
        raise ValueError(
            "experiment configurations do not form a complete parameter grid "
            f"(expected {expected}, got {len(configs)})"
        )
    return dimensions


def config_dimension_value(config: BenchmarkConfig, dimension: str) -> object:
    """Return a scalar or nested engine-component dimension value."""
    if dimension in ENGINE_COMPONENT_DIMENSIONS:
        return getattr(config.engine_component, dimension)
    return getattr(config, dimension)


def validate_experiment_name(value: str) -> str:
    """Return a safe report filename component or raise ``ValueError``."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value) or ".." in value:
        raise ValueError(
            "experiment name may contain only letters, numbers, '.', '_', "
            "and '-', without '..'"
        )
    return value


def validate_workload_lengths(
    input_len: int,
    output_len: int,
    max_model_len: int,
    experiment_name: str,
    alternate_input_len: int | None = None,
    *,
    requested_max_model_len: int | None = None,
) -> None:
    """Reject workloads that exceed their requested model context length."""
    if input_len <= 0:
        raise ValueError("input_len must be greater than zero")
    if alternate_input_len is not None and alternate_input_len <= 0:
        raise ValueError("alternate_input_len must be greater than zero")
    if output_len <= 0:
        raise ValueError("output_len must be greater than zero")
    if max_model_len <= 0:
        raise ValueError("max_model_len must be greater than zero")
    longest_input_len = max(input_len, alternate_input_len or input_len)
    total_len = longest_input_len + output_len
    if total_len > max_model_len:
        limit_description = f"max_model_len ({max_model_len})"
        if requested_max_model_len is not None:
            limit_description = (
                f"effective max_model_len ({max_model_len}); requested "
                f"max_model_len was {requested_max_model_len}"
            )
        raise ValueError(
            f"experiment {experiment_name!r} has longest input length + "
            f"output_len ({longest_input_len} + {output_len} = {total_len}) "
            f"greater than {limit_description}; increase max_model_len or "
            "reduce the workload lengths"
        )


def normalize_field_value(field_name: str, value: Any) -> Any:
    """
    Return a normalized field value or raise ``TypeError`` or ``ValueError``.

    Args:
        field_name: The name of the field to normalize.
        value: The value to normalize.
    
    Returns:
        The normalized value.
    """
    expected_type = _FIELD_TYPES[field_name]
    if (
        expected_type is float
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        value = float(value)
    validate_field_value(field_name, value)
    return value


def validate_field_value(field_name: str, value: Any) -> None:
    """ 
    Validate a field value.

    Args:
        field_name: The name of the field to validate.
        value: The value to validate.
    """
    expected_type = _FIELD_TYPES[field_name]
    _require_type(field_name, value, expected_type)
    if field_name in {
        "num_requests",
        "input_len",
        "output_len",
        "max_model_len",
        "repeats",
    }:
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than zero")
    elif field_name == "alternate_input_len":
        if value is not None and value <= 0:
            raise ValueError("alternate_input_len must be greater than zero")
    elif field_name == "input_order":
        if value not in {"alternating", "primary-first", "alternate-first"}:
            raise ValueError(
                "input_order must be alternating, primary-first, or alternate-first"
            )
    elif (
        field_name == "temperature"
        and value != 0
        and value <= 1e-10
    ):
        raise ValueError("temperature must be zero or greater than 1e-10")
    elif field_name in {"experiment_name", "workload_name", "component_name"}:
        validate_experiment_name(value)
    elif field_name in ENGINE_COMPONENT_DIMENSIONS:
        validate_engine_component_selector(field_name, value)


def _require_type(
    field_name: str,
    value: Any,
    expected_type: type | tuple[type, ...],
) -> None:
    """ 
    Require a value to be of a specific type.

    Args:
        field_name: The name of the field to check.
        value: The value to check.
        expected_type: The expected type of the value.
    """
    accepts_int = expected_type is int or (
        isinstance(expected_type, tuple) and int in expected_type
    )
    if not isinstance(value, expected_type) or (
        accepts_int and isinstance(value, bool)
    ):
        if isinstance(expected_type, tuple):
            expected_name = " or ".join(
                "None" if candidate is type(None) else candidate.__name__
                for candidate in expected_type
            )
        else:
            expected_name = expected_type.__name__
        raise TypeError(
            f"{field_name} must be {expected_name}, "
            f"got {type(value).__name__}"
        )
