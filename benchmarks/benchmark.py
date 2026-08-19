import argparse
from dataclasses import dataclass, field, fields
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import metrics, workloads
from benchmarks.runner import BenchmarkConfiguration, BenchmarkRunner
from experiments.config import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_MODEL_PATH,
    validate_experiment_name,
    validate_field_value,
)
from nanovllm import EngineComponent
from utils.reporter import Reporter


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """One fully resolved benchmark run."""

    model: str = DEFAULT_MODEL_PATH
    num_requests: int = 256
    input_len: int = 1024
    output_len: int = 1024
    seed: int = 0
    temperature: float = 0.6
    repeats: int = 3
    enforce_eager: bool = False
    engine_component: EngineComponent = field(default_factory=EngineComponent)
    experiment_name: str = DEFAULT_EXPERIMENT_NAME

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name == "engine_component":
                if not isinstance(self.engine_component, EngineComponent):
                    raise TypeError("engine_component must be EngineComponent")
                continue
            validate_field_value(field.name, getattr(self, field.name))


def experiment_name(value: str) -> str:
    """Validate the experiment name for argparse."""
    try:
        return validate_experiment_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def add_benchmark_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_defaults: bool,
) -> None:
    """
    Add the scalar benchmark options shared by benchmark and sweep CLIs.

    Supported options:
        - model
        - num-requests
        - input-len
        - output-len
        - seed
        - temperature
        - repeats
        - enforce-eager
    """
    defaults = BenchmarkConfig()

    def default(field_name: str):
        return getattr(defaults, field_name) if include_defaults else None

    parser.add_argument(
        "--model",
        default=default("model"),
        help=f"Local model path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=default("num_requests"),
        help="Number of requests in each offline batch.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=default("input_len"),
        help="Exact prompt length in tokens for every request.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=default("output_len"),
        help="Exact maximum output length in tokens for every request.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default("seed"),
        help="Workload random seed.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=default("temperature"),
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=default("repeats"),
        help="Number of measured offline batches.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=default("enforce_eager"),
        help="Enforce eager execution mode.",
    )
    parser.add_argument(
        "--experiment-name",
        type=experiment_name,
        default=default("experiment_name"),
        help="Experiment name used in the JSON report filename.",
    )


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Benchmark offline LLM throughput.")
    add_benchmark_arguments(parser, include_defaults=True)
    namespace = parser.parse_args(argv)
    try:
        return BenchmarkConfig(**vars(namespace))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def make_workload(
    config: BenchmarkConfig,
    runner: BenchmarkRunner,
    seed: int,
    num_requests: int,
):
    return workloads.synthetic_workload(
        num_requests=num_requests,
        input_len=config.input_len,
        output_len=config.output_len,
        seed=seed,
        temperature=config.temperature,
        vocab_size=runner.vocab_size,
        max_model_len=runner.max_model_len,
    )


def execute_benchmark(
    config: BenchmarkConfig,
    reporter: Reporter,
    *,
    close_runner: bool = False,
) -> tuple[metrics.BenchmarkResult, BenchmarkConfiguration]:
    """Measure one scalar benchmark configuration without rendering or saving."""
    runner = BenchmarkRunner(
        model_path=os.path.expanduser(config.model),
        enforce_eager=config.enforce_eager,
        engine_component=config.engine_component,
    )
    try:
        with reporter.warmup():
            runner.warmup(
                make_workload(config, runner, config.seed, num_requests=10)
            )
        reporter.warmup_complete()

        run_results = []
        for run_index in reporter.track_repeats(config.repeats):
            benchmark_requests = make_workload(
                config,
                runner,
                config.seed + 1 + run_index,
                num_requests=config.num_requests,
            )
            run_results.append(runner.run_benchmark(benchmark_requests))

        result = metrics.compute_benchmark_result(run_results)
        return result, runner.configuration
    finally:
        if close_runner:
            runner.close()


def main() -> None:
    config = parse_args()
    reporter = Reporter()
    result, configuration = execute_benchmark(config, reporter)
    reporter.show_result(result, configuration)
    reporter.save_result(result, configuration, config.experiment_name)


if __name__ == "__main__":
    main()
