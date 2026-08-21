from pathlib import Path
from tempfile import TemporaryDirectory
import random
import unittest

from experiments.analyze_scheduler import (
    BASELINE,
    OPTIMIZED,
    Effect,
    Sample,
    analyze,
    classify,
    overall_verdict,
    paired_effect,
    write_outputs,
)


def make_sample(
    condition: str,
    scheduler: str,
    seed: int,
    *,
    throughput: float = 100.0,
    short_p50: float = 10.0,
    long_p99: float = 20.0,
    digest: str = "same",
) -> Sample:
    return Sample(
        condition=condition,
        scheduler=scheduler,
        seed=seed,
        total_throughput=throughput,
        short_latency_p50=short_p50,
        long_latency_p99=long_p99,
        output_digest=digest,
        num_requests=2,
        input_tokens=4,
        output_tokens=2,
        scheduler_time=0.01,
        prefill_utilization=0.5,
        candidate_inspections=3,
        backfill_selections=int(scheduler == OPTIMIZED),
        preemptions=0,
        preempted_blocks=0,
        extra_prefill_tokens=0,
    )


class SchedulerAnalysisTests(unittest.TestCase):
    def test_classifies_dominance_reverse_tradeoff_and_inconclusive(self):
        positive = Effect(10, 8, 12)
        negative = Effect(-10, -12, -8)
        zero = Effect(0, 0, 0)

        self.assertEqual(
            classify(positive, negative, negative), "optimized-dominates"
        )
        self.assertEqual(
            classify(negative, positive, positive), "baseline-dominates"
        )
        self.assertEqual(classify(negative, negative, positive), "tradeoff")
        self.assertEqual(classify(zero, zero, zero), "inconclusive")

    def test_overall_verdict_refutes_a_reliable_tradeoff(self):
        samples = [
            make_sample("mixed-one", BASELINE, 1),
            make_sample(
                "mixed-one",
                OPTIMIZED,
                1,
                throughput=90,
                short_p50=8,
                long_p99=20,
            ),
        ]
        results, failures = analyze(samples, bootstrap_samples=10)

        self.assertEqual(
            overall_verdict(results, failures),
            "refutes-unconditional-improvement",
        )

    def test_computes_paired_geometric_effect_and_deterministic_ci(self):
        pairs = [
            (
                make_sample("one", BASELINE, seed, throughput=100),
                make_sample("one", OPTIMIZED, seed, throughput=110),
            )
            for seed in range(1, 6)
        ]

        effect = paired_effect(
            pairs,
            lambda sample: sample.total_throughput,
            200,
            random.Random(0),
        )

        self.assertAlmostEqual(effect.delta, 10.0)
        self.assertAlmostEqual(effect.lower, 10.0)
        self.assertAlmostEqual(effect.upper, 10.0)

    def test_excludes_correctness_failures_and_rejects_missing_pairs(self):
        samples = []
        for seed in (1, 2):
            samples.extend(
                (
                    make_sample("one", BASELINE, seed),
                    make_sample(
                        "one",
                        OPTIMIZED,
                        seed,
                        throughput=110,
                        short_p50=8,
                        long_p99=18,
                        digest="different" if seed == 2 else "same",
                    ),
                )
            )

        results, failures = analyze(samples, bootstrap_samples=100)

        self.assertEqual(results[0].pairs, 1)
        self.assertEqual(results[0].verdict, "optimized-dominates")
        self.assertEqual(failures, ["one seed 2"])

        with self.assertRaisesRegex(ValueError, "incomplete seed pairs"):
            analyze(samples[:-1], bootstrap_samples=10)

    def test_writes_markdown_csv_and_diagnostic_figures(self):
        samples = [
            make_sample("one", BASELINE, 1),
            make_sample(
                "one",
                OPTIMIZED,
                1,
                throughput=110,
                short_p50=8,
                long_p99=18,
            ),
        ]
        results, failures = analyze(samples, bootstrap_samples=10)

        with TemporaryDirectory() as temp_dir:
            paths = write_outputs(
                results,
                failures,
                {"git_commit": "abc"},
                Path(temp_dir),
            )
            contents = paths[0].read_text(encoding="utf-8")

        self.assertEqual(len(paths), 4)
        self.assertIn("optimized-dominates", contents)


if __name__ == "__main__":
    unittest.main()
