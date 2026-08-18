import argparse
import os
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import metrics, workloads
from benchmarks.reporter import BenchmarkReporter
from benchmarks.runner import BenchmarkRunner


DEFAULT_MODEL_PATH = "~/huggingface/Qwen3-0.6B/"
DEFAULT_SERVING_SYSTEM_NAME = "nano-vllm"


def serving_system_name(value: str) -> str:
    """Validate the serving system name."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value) or ".." in value:
        raise argparse.ArgumentTypeError(
            "serving system name may contain only letters, numbers, '.', '_', "
            "and '-', without '..'"
        )
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark offline LLM throughput.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Local model path (default: %(default)s).",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=256,
        help="Number of requests in each offline batch.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=1024,
        help="Exact prompt length in tokens for every request.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1024,
        help="Exact maximum output length in tokens for every request.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Workload random seed.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of measured offline batches.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager execution mode.",
    )
    parser.add_argument(
        "--serving-system-name",
        type=serving_system_name,
        default=DEFAULT_SERVING_SYSTEM_NAME,
        help="Serving system name used in the JSON report filename.",
    )
    args = parser.parse_args()
    if args.num_requests <= 0:
        parser.error("--num-requests must be greater than zero")
    if args.input_len <= 0:
        parser.error("--input-len must be greater than zero")
    if args.output_len <= 0:
        parser.error("--output-len must be greater than zero")
    if args.temperature <= 1e-10:
        parser.error("--temperature must be greater than 1e-10")
    if args.repeats <= 0:
        parser.error("--repeats must be greater than zero")
    return args


def make_workload(args, runner: BenchmarkRunner, seed: int, num_requests: int):
    return workloads.synthetic_workload(
        num_requests=num_requests,
        input_len=args.input_len,
        output_len=args.output_len,
        seed=seed,
        temperature=args.temperature,
        vocab_size=runner.vocab_size,
        max_model_len=runner.max_model_len,
    )


def main():
    args = parse_args()
    reporter = BenchmarkReporter()
    model_path = os.path.expanduser(args.model)
    runner = BenchmarkRunner(
        model_path=model_path,
        enforce_eager=args.enforce_eager,
    )

    with reporter.warmup():
        runner.warmup(make_workload(args, runner, args.seed, num_requests=10))
    reporter.warmup_complete()

    run_results = []
    for run_index in reporter.track_repeats(args.repeats):
        benchmark_requests = make_workload(
            args,
            runner,
            args.seed + 1 + run_index,
            num_requests=args.num_requests,
        )
        run_results.append(runner.run_benchmark(benchmark_requests))

    result = metrics.compute_benchmark_result(run_results)
    reporter.show_result(result, runner.configuration)
    reporter.save_result(result, runner.configuration, args.serving_system_name)


if __name__ == "__main__":
    main()
