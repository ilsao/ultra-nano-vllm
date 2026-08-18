from dataclasses import dataclass
from statistics import median

from .runner import BatchRunResult


@dataclass
class MetricSummary:
    median: float
    minimum: float
    maximum: float


@dataclass
class BenchmarkResult:
    repeats: int
    num_requests: int
    input_tokens: int
    output_tokens: MetricSummary
    total_tokens: MetricSummary
    elapsed_time: MetricSummary
    request_throughput: MetricSummary
    output_throughput: MetricSummary
    total_throughput: MetricSummary
    peak_memory_allocated_mib: float
    peak_memory_reserved_mib: float


def _summarize(values: list[float]) -> MetricSummary:
    return MetricSummary(
        median=float(median(values)),
        minimum=float(min(values)),
        maximum=float(max(values)),
    )


def compute_benchmark_result(run_results: list[BatchRunResult]) -> BenchmarkResult:
    if not run_results:
        raise ValueError("run_results must not be empty")
    if not all(isinstance(run, BatchRunResult) for run in run_results):
        raise TypeError("every run result must be a BatchRunResult")

    first = run_results[0]
    if any(run.num_requests != first.num_requests for run in run_results):
        raise ValueError("all runs must contain the same number of requests")
    if any(run.input_tokens != first.input_tokens for run in run_results):
        raise ValueError("all runs must contain the same number of input tokens")
    if any(run.elapsed_time <= 0 for run in run_results):
        raise ValueError("all elapsed times must be greater than zero")

    return BenchmarkResult(
        repeats=len(run_results),
        num_requests=first.num_requests,
        input_tokens=first.input_tokens,
        output_tokens=_summarize([run.output_tokens for run in run_results]),
        total_tokens=_summarize([run.total_tokens for run in run_results]),
        elapsed_time=_summarize([run.elapsed_time for run in run_results]),
        request_throughput=_summarize([
            run.num_requests / run.elapsed_time for run in run_results
        ]),
        output_throughput=_summarize([
            run.output_tokens / run.elapsed_time for run in run_results
        ]),
        total_throughput=_summarize([
            run.total_tokens / run.elapsed_time for run in run_results
        ]),
        peak_memory_allocated_mib=max(
            run.peak_memory_allocated_mib for run in run_results
        ),
        peak_memory_reserved_mib=max(
            run.peak_memory_reserved_mib for run in run_results
        ),
    )
