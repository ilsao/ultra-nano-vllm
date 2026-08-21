"""Analyze paired baseline/optimized scheduler reports and their Pareto tradeoff."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from math import exp, log
import os
from pathlib import Path
import random
from statistics import fmean
import tempfile
from typing import Callable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nano-vllm-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASELINE = "scheduler"
OPTIMIZED = "scheduler-optimized"


@dataclass(frozen=True, slots=True)
class Sample:
    condition: str
    scheduler: str
    seed: int
    total_throughput: float
    short_latency_p50: float
    long_latency_p99: float
    output_digest: str
    num_requests: int
    input_tokens: int
    output_tokens: int
    scheduler_time: float
    prefill_utilization: float
    candidate_inspections: int
    backfill_selections: int
    preemptions: int
    preempted_blocks: int
    extra_prefill_tokens: int


@dataclass(frozen=True, slots=True)
class Effect:
    delta: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class ConditionResult:
    condition: str
    pairs: int
    throughput: Effect
    short_p50: Effect
    long_p99: Effect
    verdict: str
    baseline_backfills: int
    optimized_backfills: int
    baseline_preemptions: int
    optimized_preemptions: int
    baseline_candidate_inspections: float
    optimized_candidate_inspections: float
    baseline_prefill_utilization: float
    optimized_prefill_utilization: float
    baseline_scheduler_time: float
    optimized_scheduler_time: float
    baseline_extra_prefill_tokens: float
    optimized_extra_prefill_tokens: float
    baseline_preempted_blocks: int
    optimized_preempted_blocks: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze paired scheduler Pareto experiment reports."
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        type=Path,
        required=True,
        help="Explicit AB and BA JSON report paths.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def load_samples(paths: Sequence[Path]) -> tuple[list[Sample], dict[str, object]]:
    samples = []
    reference_provenance = None
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            config = report["experiment_config"]
            result = report["result"]
            provenance = report["provenance"]
            workload_name = report["workload_name"]
            scheduler = config["engine_component"]["scheduler"]
            raw_samples = result["run_samples"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"could not load scheduler report {path}: {exc}") from exc
        if scheduler not in {BASELINE, OPTIMIZED}:
            raise ValueError(f"unsupported scheduler {scheduler!r} in {path}")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise ValueError(f"report {path} has no raw run samples")
        comparable_provenance = {
            key: provenance.get(key)
            for key in ("git_commit", "python", "torch", "cuda", "triton", "gpu")
        }
        if reference_provenance is None:
            reference_provenance = comparable_provenance
        elif comparable_provenance != reference_provenance:
            raise ValueError("scheduler reports have incompatible provenance")

        condition = _condition_key(workload_name, config)
        for raw_sample in raw_samples:
            samples.append(_decode_sample(condition, scheduler, raw_sample))
    return samples, reference_provenance or {}


def analyze(
    samples: Sequence[Sample],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> tuple[list[ConditionResult], list[str]]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be greater than zero")
    indexed: dict[tuple[str, str, int], Sample] = {}
    for sample in samples:
        key = (sample.condition, sample.scheduler, sample.seed)
        if key in indexed:
            raise ValueError(f"duplicate sample coordinate {key!r}")
        indexed[key] = sample

    conditions = sorted({sample.condition for sample in samples})
    results = []
    correctness_failures = []
    rng = random.Random(bootstrap_seed)
    for condition in conditions:
        baseline = {
            sample.seed: sample
            for sample in samples
            if sample.condition == condition and sample.scheduler == BASELINE
        }
        optimized = {
            sample.seed: sample
            for sample in samples
            if sample.condition == condition and sample.scheduler == OPTIMIZED
        }
        if set(baseline) != set(optimized):
            raise ValueError(f"condition {condition!r} has incomplete seed pairs")
        valid_pairs = []
        for seed in sorted(baseline):
            base = baseline[seed]
            opt = optimized[seed]
            if (
                base.output_digest != opt.output_digest
                or base.num_requests != opt.num_requests
                or base.input_tokens != opt.input_tokens
                or base.output_tokens != opt.output_tokens
            ):
                correctness_failures.append(f"{condition} seed {seed}")
                continue
            valid_pairs.append((base, opt))
        if not valid_pairs:
            continue

        throughput = paired_effect(
            valid_pairs,
            lambda sample: sample.total_throughput,
            bootstrap_samples,
            rng,
        )
        short_p50 = paired_effect(
            valid_pairs,
            lambda sample: sample.short_latency_p50,
            bootstrap_samples,
            rng,
        )
        long_p99 = paired_effect(
            valid_pairs,
            lambda sample: sample.long_latency_p99,
            bootstrap_samples,
            rng,
        )
        base_samples = [pair[0] for pair in valid_pairs]
        opt_samples = [pair[1] for pair in valid_pairs]
        results.append(
            ConditionResult(
                condition=condition,
                pairs=len(valid_pairs),
                throughput=throughput,
                short_p50=short_p50,
                long_p99=long_p99,
                verdict=classify(throughput, short_p50, long_p99),
                baseline_backfills=sum(s.backfill_selections for s in base_samples),
                optimized_backfills=sum(s.backfill_selections for s in opt_samples),
                baseline_preemptions=sum(s.preemptions for s in base_samples),
                optimized_preemptions=sum(s.preemptions for s in opt_samples),
                baseline_candidate_inspections=fmean(
                    s.candidate_inspections for s in base_samples
                ),
                optimized_candidate_inspections=fmean(
                    s.candidate_inspections for s in opt_samples
                ),
                baseline_prefill_utilization=fmean(
                    s.prefill_utilization for s in base_samples
                ),
                optimized_prefill_utilization=fmean(
                    s.prefill_utilization for s in opt_samples
                ),
                baseline_scheduler_time=fmean(s.scheduler_time for s in base_samples),
                optimized_scheduler_time=fmean(s.scheduler_time for s in opt_samples),
                baseline_extra_prefill_tokens=fmean(
                    s.extra_prefill_tokens for s in base_samples
                ),
                optimized_extra_prefill_tokens=fmean(
                    s.extra_prefill_tokens for s in opt_samples
                ),
                baseline_preempted_blocks=sum(s.preempted_blocks for s in base_samples),
                optimized_preempted_blocks=sum(s.preempted_blocks for s in opt_samples),
            )
        )
    return results, correctness_failures


def paired_effect(
    pairs: Sequence[tuple[Sample, Sample]],
    value: Callable[[Sample], float],
    bootstrap_samples: int,
    rng: random.Random,
) -> Effect:
    log_ratios = [log(value(opt) / value(base)) for base, opt in pairs]
    center = _ratio_delta(log_ratios)
    bootstrapped = []
    for _ in range(bootstrap_samples):
        resampled = [log_ratios[rng.randrange(len(log_ratios))] for _ in log_ratios]
        bootstrapped.append(_ratio_delta(resampled))
    bootstrapped.sort()
    return Effect(
        delta=center,
        lower=_quantile(bootstrapped, 0.025),
        upper=_quantile(bootstrapped, 0.975),
    )


def classify(throughput: Effect, short_p50: Effect, long_p99: Effect) -> str:
    effects = (throughput, short_p50, long_p99)
    optimized_dominates = (
        throughput.lower >= 0
        and short_p50.upper <= 0
        and long_p99.upper <= 0
        and (
            throughput.lower > 0
            or short_p50.upper < 0
            or long_p99.upper < 0
        )
    )
    baseline_dominates = (
        throughput.upper <= 0
        and short_p50.lower >= 0
        and long_p99.lower >= 0
        and (
            throughput.upper < 0
            or short_p50.lower > 0
            or long_p99.lower > 0
        )
    )
    if optimized_dominates:
        return "optimized-dominates"
    if baseline_dominates:
        return "baseline-dominates"
    confident_directions = set()
    for index, effect in enumerate(effects):
        if effect.lower > 0:
            confident_directions.add("optimized" if index == 0 else "baseline")
        elif effect.upper < 0:
            confident_directions.add("baseline" if index == 0 else "optimized")
    if confident_directions == {"baseline", "optimized"}:
        return "tradeoff"
    return "inconclusive"


def overall_verdict(
    results: Sequence[ConditionResult],
    correctness_failures: Sequence[str],
) -> str:
    if correctness_failures:
        return "correctness-failure"
    mixed = [
        result
        for result in results
        if result.condition.split("|", 1)[0].startswith("mixed-")
    ]
    controls = [
        result
        for result in results
        if result.condition.split("|", 1)[0].startswith("homogeneous-")
    ]
    if (
        mixed
        and all(result.verdict == "optimized-dominates" for result in mixed)
        and not any(result.verdict == "baseline-dominates" for result in controls)
    ):
        return "supports-unconditional-improvement"
    if any(
        result.verdict in {"tradeoff", "baseline-dominates"}
        for result in results
    ):
        return "refutes-unconditional-improvement"
    return "inconclusive"


def write_outputs(
    results: Sequence[ConditionResult],
    correctness_failures: Sequence[str],
    provenance: dict[str, object],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scheduler-pareto.csv"
    markdown_path = output_dir / "analyze.md"
    pareto_path = output_dir / "scheduler-pareto.png"
    mechanism_path = output_dir / "scheduler-mechanisms.png"
    _write_csv(results, csv_path)
    _write_markdown(results, correctness_failures, provenance, markdown_path)
    _plot_pareto(results, pareto_path)
    _plot_mechanisms(results, mechanism_path)
    return [markdown_path, csv_path, pareto_path, mechanism_path]


def _condition_key(workload_name: str, config: dict[str, object]) -> str:
    normalized = {
        key: value
        for key, value in config.items()
        if key not in {"seed", "repeats", "engine_component", "experiment_name"}
    }
    engine_component = dict(config["engine_component"])
    engine_component.pop("scheduler", None)
    normalized["engine_component"] = engine_component
    return f"{workload_name}|{json.dumps(normalized, sort_keys=True, separators=(',', ':'))}"


def _decode_sample(condition: str, scheduler: str, raw: dict[str, object]) -> Sample:
    latencies = tuple(float(value) for value in raw["request_latencies"])
    input_lengths = tuple(int(value) for value in raw["request_input_lengths"])
    if not latencies or len(latencies) != len(input_lengths):
        raise ValueError("run sample request latencies and input lengths must align")
    by_length: dict[int, list[float]] = {}
    for input_length, latency in zip(input_lengths, latencies):
        by_length.setdefault(input_length, []).append(latency)
    short_latencies = by_length[min(by_length)]
    long_latencies = by_length[max(by_length)]
    scheduler_metrics = raw["scheduler_metrics"]
    prefill_capacity = int(scheduler_metrics["prefill_capacity_tokens"])
    return Sample(
        condition=condition,
        scheduler=scheduler,
        seed=int(raw["seed"]),
        total_throughput=float(raw["total_tokens"]) / float(raw["elapsed_time"]),
        short_latency_p50=_percentile(short_latencies, 0.50),
        long_latency_p99=_percentile(long_latencies, 0.99),
        output_digest=str(raw["output_digest"]),
        num_requests=int(raw["num_requests"]),
        input_tokens=int(raw["input_tokens"]),
        output_tokens=int(raw["output_tokens"]),
        scheduler_time=float(scheduler_metrics["scheduler_time"]),
        prefill_utilization=(
            int(scheduler_metrics["prefill_scheduled_tokens"]) / prefill_capacity
            if prefill_capacity
            else 0.0
        ),
        candidate_inspections=int(scheduler_metrics["candidate_inspections"]),
        backfill_selections=int(scheduler_metrics["backfill_selections"]),
        preemptions=int(scheduler_metrics["preemptions"]),
        preempted_blocks=int(scheduler_metrics["preempted_blocks"]),
        extra_prefill_tokens=int(scheduler_metrics["extra_prefill_tokens"]),
    )


def _ratio_delta(log_ratios: Sequence[float]) -> float:
    return (exp(fmean(log_ratios)) - 1) * 100


def _quantile(values: Sequence[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _percentile(values: Sequence[float], quantile: float) -> float:
    return _quantile(sorted(values), quantile)


def _write_csv(results: Sequence[ConditionResult], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "condition",
                "pairs",
                "verdict",
                "throughput_delta_pct",
                "throughput_ci_low",
                "throughput_ci_high",
                "short_p50_delta_pct",
                "short_p50_ci_low",
                "short_p50_ci_high",
                "long_p99_delta_pct",
                "long_p99_ci_low",
                "long_p99_ci_high",
                "baseline_backfills",
                "optimized_backfills",
                "baseline_preemptions",
                "optimized_preemptions",
                "baseline_preempted_blocks",
                "optimized_preempted_blocks",
                "baseline_candidate_inspections_mean",
                "optimized_candidate_inspections_mean",
                "baseline_prefill_utilization_mean",
                "optimized_prefill_utilization_mean",
                "baseline_scheduler_time_mean",
                "optimized_scheduler_time_mean",
                "baseline_extra_prefill_tokens_mean",
                "optimized_extra_prefill_tokens_mean",
            )
        )
        for result in results:
            writer.writerow(
                (
                    result.condition,
                    result.pairs,
                    result.verdict,
                    result.throughput.delta,
                    result.throughput.lower,
                    result.throughput.upper,
                    result.short_p50.delta,
                    result.short_p50.lower,
                    result.short_p50.upper,
                    result.long_p99.delta,
                    result.long_p99.lower,
                    result.long_p99.upper,
                    result.baseline_backfills,
                    result.optimized_backfills,
                    result.baseline_preemptions,
                    result.optimized_preemptions,
                    result.baseline_preempted_blocks,
                    result.optimized_preempted_blocks,
                    result.baseline_candidate_inspections,
                    result.optimized_candidate_inspections,
                    result.baseline_prefill_utilization,
                    result.optimized_prefill_utilization,
                    result.baseline_scheduler_time,
                    result.optimized_scheduler_time,
                    result.baseline_extra_prefill_tokens,
                    result.optimized_extra_prefill_tokens,
                )
            )


def _write_markdown(
    results: Sequence[ConditionResult],
    failures: Sequence[str],
    provenance: dict[str, object],
    path: Path,
) -> None:
    lines = [
        "# Scheduler Pareto Analysis",
        "",
        "Effects are optimized / baseline paired geometric-mean deltas with "
        "95% bootstrap confidence intervals. Positive throughput and negative "
        "latency are favorable to the optimized scheduler.",
        "",
        "## Overall verdict",
        "",
        f"`{overall_verdict(results, failures)}`",
        "",
        "## Provenance",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in provenance.items())
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Workload | Pairs | Verdict | Throughput Δ | Short P50 Δ | Long P99 Δ | Backfills B/O | Preemptions B/O |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        name = result.condition.split("|", 1)[0]
        lines.append(
            f"| {name} | {result.pairs} | {result.verdict} | "
            f"{_format_effect(result.throughput)} | "
            f"{_format_effect(result.short_p50)} | "
            f"{_format_effect(result.long_p99)} | "
            f"{result.baseline_backfills}/{result.optimized_backfills} | "
            f"{result.baseline_preemptions}/{result.optimized_preemptions} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism diagnostics",
            "",
            "| Workload | Candidate inspections B/O | Prefill utilization B/O | Scheduler time B/O (s) | Extra prefill tokens B/O | Preempted blocks B/O |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        name = result.condition.split("|", 1)[0]
        lines.append(
            f"| {name} | {result.baseline_candidate_inspections:.1f}/"
            f"{result.optimized_candidate_inspections:.1f} | "
            f"{result.baseline_prefill_utilization:.3f}/"
            f"{result.optimized_prefill_utilization:.3f} | "
            f"{result.baseline_scheduler_time:.6f}/"
            f"{result.optimized_scheduler_time:.6f} | "
            f"{result.baseline_extra_prefill_tokens:.1f}/"
            f"{result.optimized_extra_prefill_tokens:.1f} | "
            f"{result.baseline_preempted_blocks}/"
            f"{result.optimized_preempted_blocks} |"
        )
    lines.extend(["", "## Correctness", ""])
    if failures:
        lines.append(
            "The following pairs had output/token mismatches and were excluded:"
        )
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("All paired samples matched output digests and token totals.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_effect(effect: Effect) -> str:
    return f"{effect.delta:+.2f}% [{effect.lower:+.2f}, {effect.upper:+.2f}]"


def _plot_pareto(results: Sequence[ConditionResult], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 7))
    for result in results:
        axis.errorbar(
            result.throughput.delta,
            result.short_p50.delta,
            xerr=[
                [result.throughput.delta - result.throughput.lower],
                [result.throughput.upper - result.throughput.delta],
            ],
            yerr=[
                [result.short_p50.delta - result.short_p50.lower],
                [result.short_p50.upper - result.short_p50.delta],
            ],
            marker="o",
            linestyle="none",
        )
        axis.annotate(
            result.condition.split("|", 1)[0],
            (result.throughput.delta, result.short_p50.delta),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axvline(0, color="black", linewidth=1)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("Total throughput delta (%) — higher is better")
    axis.set_ylabel("Short/overall P50 latency delta (%) — lower is better")
    axis.set_title("scheduler-optimized vs baseline Pareto effects")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_mechanisms(results: Sequence[ConditionResult], path: Path) -> None:
    names = [result.condition.split("|", 1)[0] for result in results]
    positions = list(range(len(results)))
    width = 0.38
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].bar(
        [position - width / 2 for position in positions],
        [result.baseline_backfills for result in results],
        width,
        label="baseline",
    )
    axes[0].bar(
        [position + width / 2 for position in positions],
        [result.optimized_backfills for result in results],
        width,
        label="optimized",
    )
    axes[0].set_ylabel("Backfill selections")
    axes[0].legend()
    axes[1].bar(
        [position - width / 2 for position in positions],
        [result.baseline_preemptions for result in results],
        width,
        label="baseline",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        [result.optimized_preemptions for result in results],
        width,
        label="optimized",
    )
    axes[1].set_ylabel("Preemptions")
    axes[1].set_xticks(positions, names, rotation=35, ha="right")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    samples, provenance = load_samples(args.reports)
    results, failures = analyze(
        samples,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    for path in write_outputs(results, failures, provenance, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
