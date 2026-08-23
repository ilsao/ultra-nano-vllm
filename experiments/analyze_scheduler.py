"""Analyze paired reference/candidate scheduler reports and Pareto tradeoffs."""

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
import textwrap
from typing import Callable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nano-vllm-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REFERENCE = "scheduler"
CANDIDATES = ("scheduler-v1", "scheduler-v2")

_SCHEDULER_COLORS = {
    "scheduler": "tab:blue",
    "scheduler-v1": "tab:orange",
    "scheduler-v2": "tab:green",
}


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
    candidate_selector: str
    pairs: int
    throughput: Effect
    short_p50: Effect
    long_p99: Effect
    verdict: str
    reference_backfills: int
    candidate_backfills: int
    reference_preemptions: int
    candidate_preemptions: int
    reference_candidate_inspections: float
    candidate_candidate_inspections: float
    reference_prefill_utilization: float
    candidate_prefill_utilization: float
    reference_scheduler_time: float
    candidate_scheduler_time: float
    reference_extra_prefill_tokens: float
    candidate_extra_prefill_tokens: float
    reference_preempted_blocks: int
    candidate_preempted_blocks: int


@dataclass(frozen=True, slots=True)
class CorrectnessFailure:
    condition: str
    candidate_selector: str
    seed: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze paired scheduler Pareto experiment reports."
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        type=Path,
        required=True,
        help="Explicit three-way scheduler comparison JSON report paths.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-selector", default=REFERENCE)
    parser.add_argument(
        "--candidate-selectors",
        nargs="+",
        default=list(CANDIDATES),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def _validate_selectors(
    reference_selector: str,
    candidate_selectors: Sequence[str],
) -> tuple[str, ...]:
    candidates = tuple(candidate_selectors)
    if (
        not reference_selector
        or not candidates
        or any(not value for value in candidates)
    ):
        raise ValueError("scheduler selectors must be non-empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate scheduler selectors must be unique")
    if reference_selector in candidates:
        raise ValueError("reference and candidate scheduler selectors must differ")
    return candidates


def load_samples(
    paths: Sequence[Path],
    *,
    reference_selector: str = REFERENCE,
    candidate_selectors: Sequence[str] = CANDIDATES,
) -> tuple[list[Sample], dict[str, object]]:
    candidates = _validate_selectors(reference_selector, candidate_selectors)
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
        if scheduler not in {reference_selector, *candidates}:
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
    reference_selector: str = REFERENCE,
    candidate_selectors: Sequence[str] = CANDIDATES,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> tuple[list[ConditionResult], list[CorrectnessFailure]]:
    candidates = _validate_selectors(reference_selector, candidate_selectors)
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be greater than zero")
    indexed: dict[tuple[str, str, int], Sample] = {}
    allowed_selectors = {reference_selector, *candidates}
    for sample in samples:
        if sample.scheduler not in allowed_selectors:
            raise ValueError(f"unsupported scheduler {sample.scheduler!r}")
        key = (sample.condition, sample.scheduler, sample.seed)
        if key in indexed:
            raise ValueError(f"duplicate sample coordinate {key!r}")
        indexed[key] = sample

    conditions = sorted({sample.condition for sample in samples})
    results = []
    correctness_failures = []
    rng = random.Random(bootstrap_seed)
    for condition in conditions:
        reference = {
            sample.seed: sample
            for sample in samples
            if sample.condition == condition
            and sample.scheduler == reference_selector
        }
        for candidate_selector in candidates:
            candidate = {
                sample.seed: sample
                for sample in samples
                if sample.condition == condition
                and sample.scheduler == candidate_selector
            }
            if set(reference) != set(candidate):
                raise ValueError(
                    f"condition {condition!r} candidate "
                    f"{candidate_selector!r} has incomplete seed pairs"
                )
            valid_pairs = []
            for seed in sorted(reference):
                reference_sample = reference[seed]
                candidate_sample = candidate[seed]
                if (
                    reference_sample.output_digest != candidate_sample.output_digest
                    or reference_sample.num_requests != candidate_sample.num_requests
                    or reference_sample.input_tokens != candidate_sample.input_tokens
                    or reference_sample.output_tokens != candidate_sample.output_tokens
                ):
                    correctness_failures.append(
                        CorrectnessFailure(
                            condition=condition,
                            candidate_selector=candidate_selector,
                            seed=seed,
                        )
                    )
                    continue
                valid_pairs.append((reference_sample, candidate_sample))
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
            reference_samples = [pair[0] for pair in valid_pairs]
            candidate_samples = [pair[1] for pair in valid_pairs]
            results.append(
                ConditionResult(
                    condition=condition,
                    candidate_selector=candidate_selector,
                    pairs=len(valid_pairs),
                    throughput=throughput,
                    short_p50=short_p50,
                    long_p99=long_p99,
                    verdict=classify(throughput, short_p50, long_p99),
                    reference_backfills=sum(
                        sample.backfill_selections for sample in reference_samples
                    ),
                    candidate_backfills=sum(
                        sample.backfill_selections for sample in candidate_samples
                    ),
                    reference_preemptions=sum(
                        sample.preemptions for sample in reference_samples
                    ),
                    candidate_preemptions=sum(
                        sample.preemptions for sample in candidate_samples
                    ),
                    reference_candidate_inspections=fmean(
                        sample.candidate_inspections
                        for sample in reference_samples
                    ),
                    candidate_candidate_inspections=fmean(
                        sample.candidate_inspections
                        for sample in candidate_samples
                    ),
                    reference_prefill_utilization=fmean(
                        sample.prefill_utilization for sample in reference_samples
                    ),
                    candidate_prefill_utilization=fmean(
                        sample.prefill_utilization for sample in candidate_samples
                    ),
                    reference_scheduler_time=fmean(
                        sample.scheduler_time for sample in reference_samples
                    ),
                    candidate_scheduler_time=fmean(
                        sample.scheduler_time for sample in candidate_samples
                    ),
                    reference_extra_prefill_tokens=fmean(
                        sample.extra_prefill_tokens for sample in reference_samples
                    ),
                    candidate_extra_prefill_tokens=fmean(
                        sample.extra_prefill_tokens for sample in candidate_samples
                    ),
                    reference_preempted_blocks=sum(
                        sample.preempted_blocks for sample in reference_samples
                    ),
                    candidate_preempted_blocks=sum(
                        sample.preempted_blocks for sample in candidate_samples
                    ),
                )
            )
    return results, correctness_failures


def paired_effect(
    pairs: Sequence[tuple[Sample, Sample]],
    value: Callable[[Sample], float],
    bootstrap_samples: int,
    rng: random.Random,
) -> Effect:
    log_ratios = [
        log(value(candidate) / value(reference))
        for reference, candidate in pairs
    ]
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
    candidate_dominates = (
        throughput.lower >= 0
        and short_p50.upper <= 0
        and long_p99.upper <= 0
        and (
            throughput.lower > 0
            or short_p50.upper < 0
            or long_p99.upper < 0
        )
    )
    reference_dominates = (
        throughput.upper <= 0
        and short_p50.lower >= 0
        and long_p99.lower >= 0
        and (
            throughput.upper < 0
            or short_p50.lower > 0
            or long_p99.lower > 0
        )
    )
    if candidate_dominates:
        return "candidate-dominates"
    if reference_dominates:
        return "reference-dominates"
    confident_directions = set()
    for index, effect in enumerate(effects):
        if effect.lower > 0:
            confident_directions.add("candidate" if index == 0 else "reference")
        elif effect.upper < 0:
            confident_directions.add("reference" if index == 0 else "candidate")
    if confident_directions == {"reference", "candidate"}:
        return "tradeoff"
    return "inconclusive"


def overall_verdict(
    results: Sequence[ConditionResult],
    correctness_failures: Sequence[CorrectnessFailure],
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
        and all(result.verdict == "candidate-dominates" for result in mixed)
        and not any(result.verdict == "reference-dominates" for result in controls)
    ):
        return "supports-unconditional-improvement"
    if any(
        result.verdict in {"tradeoff", "reference-dominates"}
        for result in results
    ):
        return "refutes-unconditional-improvement"
    return "inconclusive"


def write_outputs(
    results: Sequence[ConditionResult],
    correctness_failures: Sequence[CorrectnessFailure],
    provenance: dict[str, object],
    output_dir: Path,
    *,
    reference_selector: str = REFERENCE,
    candidate_selectors: Sequence[str] = CANDIDATES,
) -> list[Path]:
    candidates = _validate_selectors(reference_selector, candidate_selectors)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scheduler-pareto.csv"
    markdown_path = output_dir / "analyze.md"
    pareto_path = output_dir / "scheduler-pareto.png"
    mechanism_path = output_dir / "scheduler-mechanisms.png"
    _write_csv(results, csv_path, reference_selector)
    _write_markdown(
        results,
        correctness_failures,
        provenance,
        markdown_path,
        reference_selector,
        candidates,
    )
    _plot_pareto(results, pareto_path, reference_selector, candidates)
    _plot_mechanisms(
        results, mechanism_path, reference_selector, candidates
    )
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


def _write_csv(
    results: Sequence[ConditionResult],
    path: Path,
    reference_selector: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "condition",
                "reference_selector",
                "candidate_selector",
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
                "reference_backfills",
                "candidate_backfills",
                "reference_preemptions",
                "candidate_preemptions",
                "reference_preempted_blocks",
                "candidate_preempted_blocks",
                "reference_candidate_inspections_mean",
                "candidate_candidate_inspections_mean",
                "reference_prefill_utilization_mean",
                "candidate_prefill_utilization_mean",
                "reference_scheduler_time_mean",
                "candidate_scheduler_time_mean",
                "reference_extra_prefill_tokens_mean",
                "candidate_extra_prefill_tokens_mean",
            )
        )
        for result in results:
            writer.writerow(
                (
                    result.condition,
                    reference_selector,
                    result.candidate_selector,
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
                    result.reference_backfills,
                    result.candidate_backfills,
                    result.reference_preemptions,
                    result.candidate_preemptions,
                    result.reference_preempted_blocks,
                    result.candidate_preempted_blocks,
                    result.reference_candidate_inspections,
                    result.candidate_candidate_inspections,
                    result.reference_prefill_utilization,
                    result.candidate_prefill_utilization,
                    result.reference_scheduler_time,
                    result.candidate_scheduler_time,
                    result.reference_extra_prefill_tokens,
                    result.candidate_extra_prefill_tokens,
                )
            )


def _write_markdown(
    results: Sequence[ConditionResult],
    failures: Sequence[CorrectnessFailure],
    provenance: dict[str, object],
    path: Path,
    reference_selector: str,
    candidate_selectors: Sequence[str],
) -> None:
    lines = [
        "# Scheduler Pareto Analysis",
        "",
        f"Effects are paired against reference `{reference_selector}` using "
        "geometric-mean deltas with "
        "95% bootstrap confidence intervals. Positive throughput and negative "
        "latency are favorable to the candidate scheduler.",
        "",
        "## Overall verdicts",
        "",
    ]
    for candidate_selector in candidate_selectors:
        candidate_results = [
            result
            for result in results
            if result.candidate_selector == candidate_selector
        ]
        candidate_failures = [
            failure
            for failure in failures
            if failure.candidate_selector == candidate_selector
        ]
        lines.append(
            f"- `{candidate_selector}` vs `{reference_selector}`: "
            f"`{overall_verdict(candidate_results, candidate_failures)}`"
        )
    lines.extend(["", "## Provenance", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in provenance.items())
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Workload | Candidate | Pairs | Verdict | Throughput Δ | Short P50 Δ | Long P99 Δ | Backfills R/C | Preemptions R/C |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        name = result.condition.split("|", 1)[0]
        lines.append(
            f"| {name} | {result.candidate_selector} | {result.pairs} | "
            f"{result.verdict} | "
            f"{_format_effect(result.throughput)} | "
            f"{_format_effect(result.short_p50)} | "
            f"{_format_effect(result.long_p99)} | "
            f"{result.reference_backfills}/{result.candidate_backfills} | "
            f"{result.reference_preemptions}/{result.candidate_preemptions} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism diagnostics",
            "",
            "| Workload | Candidate | Candidate inspections R/C | Prefill utilization R/C | Scheduler time R/C (s) | Extra prefill tokens R/C | Preempted blocks R/C |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        name = result.condition.split("|", 1)[0]
        lines.append(
            f"| {name} | {result.candidate_selector} | "
            f"{result.reference_candidate_inspections:.1f}/"
            f"{result.candidate_candidate_inspections:.1f} | "
            f"{result.reference_prefill_utilization:.3f}/"
            f"{result.candidate_prefill_utilization:.3f} | "
            f"{result.reference_scheduler_time:.6f}/"
            f"{result.candidate_scheduler_time:.6f} | "
            f"{result.reference_extra_prefill_tokens:.1f}/"
            f"{result.candidate_extra_prefill_tokens:.1f} | "
            f"{result.reference_preempted_blocks}/"
            f"{result.candidate_preempted_blocks} |"
        )
    lines.extend(["", "## Correctness", ""])
    if failures:
        lines.append(
            "The following pairs had output/token mismatches and were excluded:"
        )
        lines.extend(
            f"- {failure.condition} candidate "
            f"{failure.candidate_selector} seed {failure.seed}"
            for failure in failures
        )
    else:
        lines.append("All paired samples matched output digests and token totals.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_effect(effect: Effect) -> str:
    return f"{effect.delta:+.2f}% [{effect.lower:+.2f}, {effect.upper:+.2f}]"


def _format_condition_label(condition: str) -> str:
    name = condition.split("|", 1)[0]
    return "\n".join(
        textwrap.wrap(
            name,
            width=18,
            break_long_words=True,
            break_on_hyphens=True,
        )
    )


def _plot_pareto(
    results: Sequence[ConditionResult],
    path: Path,
    reference_selector: str,
    candidate_selectors: Sequence[str],
) -> None:
    figure_width = max(11.0, min(18.0, 7.0 + 0.35 * len(results)))
    figure, axis = plt.subplots(
        figsize=(figure_width, 7),
        constrained_layout=True,
    )
    markers = (
        "o",
        "s",
        "^",
        "D",
        "v",
        "P",
        "X",
        "*",
        "<",
        ">",
        "h",
        "p",
    )
    colors = plt.get_cmap("tab20").colors
    conditions = list(dict.fromkeys(result.condition for result in results))
    workload_colors = {
        condition: colors[index % len(colors)]
        for index, condition in enumerate(conditions)
    }
    candidate_markers = {
        selector: markers[index % len(markers)]
        for index, selector in enumerate(candidate_selectors)
    }
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
            color=workload_colors[result.condition],
            marker=candidate_markers[result.candidate_selector],
            linestyle="none",
            label=(
                f"{result.candidate_selector}: "
                f"{_format_condition_label(result.condition)}"
            ),
        )
    axis.axvline(0, color="black", linewidth=1)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("Total throughput delta (%) — higher is better")
    axis.set_ylabel("Short/overall P50 latency delta (%) — lower is better")
    axis.set_title(
        f"{', '.join(candidate_selectors)} vs {reference_selector} Pareto effects"
    )
    axis.grid(alpha=0.25)
    axis.legend(
        title="Candidate / workload",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        fontsize=8,
    )
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_mechanisms(
    results: Sequence[ConditionResult],
    path: Path,
    reference_selector: str,
    candidate_selectors: Sequence[str],
) -> None:
    conditions = list(dict.fromkeys(result.condition for result in results))
    names = [_format_condition_label(condition) for condition in conditions]
    positions = list(range(len(conditions)))
    selectors = (reference_selector, *candidate_selectors)
    width = 0.8 / len(selectors)
    max_lines = max((name.count("\n") + 1 for name in names), default=1)
    figure_width = max(12.0, min(24.0, 1.5 * len(conditions)))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(figure_width, 9.0 + 0.3 * (max_lines - 1)),
        sharex=True,
        constrained_layout=True,
    )
    for selector_index, selector in enumerate(selectors):
        offset = (selector_index - (len(selectors) - 1) / 2) * width
        axes[0].bar(
            [position + offset for position in positions],
            [
                _mechanism_value(
                    results,
                    condition,
                    selector,
                    reference_selector,
                    "backfills",
                )
                for condition in conditions
            ],
            width,
            label=selector,
            color=_SCHEDULER_COLORS.get(selector),
        )
    axes[0].set_ylabel("Backfill selections")
    axes[0].legend()
    for selector_index, selector in enumerate(selectors):
        offset = (selector_index - (len(selectors) - 1) / 2) * width
        axes[1].bar(
            [position + offset for position in positions],
            [
                _mechanism_value(
                    results,
                    condition,
                    selector,
                    reference_selector,
                    "preemptions",
                )
                for condition in conditions
            ],
            width,
            label=selector,
            color=_SCHEDULER_COLORS.get(selector),
        )
    axes[1].set_ylabel("Preemptions")
    axes[1].set_xticks(positions, names)
    axes[1].tick_params(axis="x", labelrotation=0, labelsize=9)
    axes[1].legend()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _mechanism_value(
    results: Sequence[ConditionResult],
    condition: str,
    selector: str,
    reference_selector: str,
    metric: str,
) -> float:
    condition_results = [
        result for result in results if result.condition == condition
    ]
    if selector == reference_selector:
        if not condition_results:
            return float("nan")
        return float(getattr(condition_results[0], f"reference_{metric}"))
    for result in condition_results:
        if result.candidate_selector == selector:
            return float(getattr(result, f"candidate_{metric}"))
    return float("nan")


def main() -> None:
    args = build_parser().parse_args()
    samples, provenance = load_samples(
        args.reports,
        reference_selector=args.reference_selector,
        candidate_selectors=args.candidate_selectors,
    )
    results, failures = analyze(
        samples,
        reference_selector=args.reference_selector,
        candidate_selectors=args.candidate_selectors,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    for path in write_outputs(
        results,
        failures,
        provenance,
        args.output_dir,
        reference_selector=args.reference_selector,
        candidate_selectors=args.candidate_selectors,
    ):
        print(path)


if __name__ == "__main__":
    main()
