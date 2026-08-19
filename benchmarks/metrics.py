from dataclasses import dataclass
from math import ceil, floor
from statistics import fmean, median, pstdev

from .runner import BatchRunResult


@dataclass
class MetricSummary:
    median: float
    minimum: float
    maximum: float
    mean: float | None = None
    standard_deviation: float | None = None


@dataclass
class BenchmarkResult:
    repeats: int
    num_requests: int
    input_tokens: int
    output_tokens: MetricSummary
    total_tokens: MetricSummary
    elapsed_time: MetricSummary
    latency_p50: MetricSummary
    latency_p90: MetricSummary
    latency_p99: MetricSummary
    request_throughput: MetricSummary
    output_throughput: MetricSummary
    total_throughput: MetricSummary
    prefill_throughput: MetricSummary
    decode_throughput: MetricSummary
    prefill_time: MetricSummary
    decode_time: MetricSummary
    peak_kvcache_blocks: int
    peak_kvcache_utilization: float


def _summarize(values: list[float]) -> MetricSummary:
    return MetricSummary(
        median=float(median(values)),
        minimum=float(min(values)),
        maximum=float(max(values)),
        mean=float(fmean(values)),
        standard_deviation=float(pstdev(values)),
    )


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _phase_throughput(tokens: int, elapsed_time: float) -> float:
    return tokens / elapsed_time if tokens else 0.0


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
    if any(
        len(run.request_latencies) != run.num_requests
        or any(latency <= 0 for latency in run.request_latencies)
        for run in run_results
    ):
        raise ValueError("every request must have a positive latency")
    if any(
        run.prefill_tokens <= 0
        or run.prefill_time <= 0
        or run.decode_tokens < 0
        or (run.decode_tokens > 0 and run.decode_time <= 0)
        or (run.decode_tokens == 0 and run.decode_time < 0)
        for run in run_results
    ):
        raise ValueError("prefill/decode token and time measurements are invalid")
    if any(run.kvcache_capacity_blocks <= 0 for run in run_results):
        raise ValueError("KV-cache block capacity must be greater than zero")
    if any(
        run.kvcache_capacity_blocks != first.kvcache_capacity_blocks
        for run in run_results
    ):
        raise ValueError("all runs must have the same KV-cache block capacity")
    if any(
        run.peak_kvcache_blocks < 0
        or run.peak_kvcache_blocks > run.kvcache_capacity_blocks
        for run in run_results
    ):
        raise ValueError("peak KV-cache blocks must be within block capacity")

    peak_kvcache_blocks = max(run.peak_kvcache_blocks for run in run_results)

    return BenchmarkResult(
        repeats=len(run_results),
        num_requests=first.num_requests,
        input_tokens=first.input_tokens,
        output_tokens=_summarize([run.output_tokens for run in run_results]),
        total_tokens=_summarize([run.total_tokens for run in run_results]),
        elapsed_time=_summarize([run.elapsed_time for run in run_results]),
        latency_p50=_summarize([
            _percentile(run.request_latencies, 0.50) for run in run_results
        ]),
        latency_p90=_summarize([
            _percentile(run.request_latencies, 0.90) for run in run_results
        ]),
        latency_p99=_summarize([
            _percentile(run.request_latencies, 0.99) for run in run_results
        ]),
        request_throughput=_summarize([
            run.num_requests / run.elapsed_time for run in run_results
        ]),
        output_throughput=_summarize([
            run.output_tokens / run.elapsed_time for run in run_results
        ]),
        total_throughput=_summarize([
            run.total_tokens / run.elapsed_time for run in run_results
        ]),
        prefill_throughput=_summarize([
            _phase_throughput(run.prefill_tokens, run.prefill_time)
            for run in run_results
        ]),
        decode_throughput=_summarize([
            _phase_throughput(run.decode_tokens, run.decode_time)
            for run in run_results
        ]),
        prefill_time=_summarize([run.prefill_time for run in run_results]),
        decode_time=_summarize([run.decode_time for run in run_results]),
        peak_kvcache_blocks=peak_kvcache_blocks,
        peak_kvcache_utilization=(
            peak_kvcache_blocks / first.kvcache_capacity_blocks
        ),
    )
