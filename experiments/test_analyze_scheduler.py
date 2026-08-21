from pathlib import Path
from tempfile import TemporaryDirectory
import json
import random
import unittest
from unittest.mock import patch

from experiments import analyze_scheduler as scheduler_analysis
from experiments.analyze_scheduler import (
    CANDIDATES,
    REFERENCE,
    CorrectnessFailure,
    Effect,
    Sample,
    analyze,
    classify,
    overall_verdict,
    paired_effect,
    load_samples,
    write_outputs,
)

CANDIDATE = CANDIDATES[0]
SECOND_CANDIDATE = CANDIDATES[1]


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
        backfill_selections=int(scheduler != REFERENCE),
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
            classify(positive, negative, negative), "candidate-dominates"
        )
        self.assertEqual(
            classify(negative, positive, positive), "reference-dominates"
        )
        self.assertEqual(classify(negative, negative, positive), "tradeoff")
        self.assertEqual(classify(zero, zero, zero), "inconclusive")

    def test_overall_verdict_refutes_a_reliable_tradeoff(self):
        samples = [
            make_sample("mixed-one", REFERENCE, 1),
            make_sample(
                "mixed-one",
                CANDIDATE,
                1,
                throughput=90,
                short_p50=8,
                long_p99=20,
            ),
        ]
        results, failures = analyze(
            samples,
            candidate_selectors=(CANDIDATE,),
            bootstrap_samples=10,
        )

        self.assertEqual(
            overall_verdict(results, failures),
            "refutes-unconditional-improvement",
        )

    def test_computes_paired_geometric_effect_and_deterministic_ci(self):
        pairs = [
            (
                make_sample("one", REFERENCE, seed, throughput=100),
                make_sample("one", CANDIDATE, seed, throughput=110),
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
                    make_sample("one", REFERENCE, seed),
                    make_sample(
                        "one",
                        CANDIDATE,
                        seed,
                        throughput=110,
                        short_p50=8,
                        long_p99=18,
                        digest="different" if seed == 2 else "same",
                    ),
                )
            )

        results, failures = analyze(
            samples,
            candidate_selectors=(CANDIDATE,),
            bootstrap_samples=100,
        )

        self.assertEqual(results[0].pairs, 1)
        self.assertEqual(results[0].verdict, "candidate-dominates")
        self.assertEqual(
            failures,
            [CorrectnessFailure("one", CANDIDATE, 2)],
        )

        with self.assertRaisesRegex(ValueError, "incomplete seed pairs"):
            analyze(
                samples[:-1],
                candidate_selectors=(CANDIDATE,),
                bootstrap_samples=10,
            )

    def test_correctness_failure_only_excludes_affected_candidate_pair(self):
        samples = []
        for seed in (1, 2):
            samples.extend(
                (
                    make_sample("one", REFERENCE, seed),
                    make_sample("one", CANDIDATE, seed, throughput=110),
                    make_sample(
                        "one",
                        SECOND_CANDIDATE,
                        seed,
                        throughput=115,
                        digest="different" if seed == 2 else "same",
                    ),
                )
            )

        results, failures = analyze(samples, bootstrap_samples=10)

        self.assertEqual(
            {
                result.candidate_selector: result.pairs
                for result in results
            },
            {CANDIDATE: 2, SECOND_CANDIDATE: 1},
        )
        self.assertEqual(
            failures,
            [CorrectnessFailure("one", SECOND_CANDIDATE, 2)],
        )

    def test_writes_markdown_csv_and_diagnostic_figures(self):
        samples = [
            make_sample("one", REFERENCE, 1),
            make_sample(
                "one",
                CANDIDATE,
                1,
                throughput=110,
                short_p50=8,
                long_p99=18,
            ),
            make_sample(
                "one",
                SECOND_CANDIDATE,
                1,
                throughput=115,
                short_p50=7,
                long_p99=17,
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
            csv_contents = paths[1].read_text(encoding="utf-8")

        self.assertEqual(len(paths), 4)
        self.assertIn("candidate-dominates", contents)
        self.assertIn(f"`{CANDIDATE}` vs `{REFERENCE}`", contents)
        self.assertIn(f"`{SECOND_CANDIDATE}` vs `{REFERENCE}`", contents)
        self.assertIn(f"{REFERENCE},{CANDIDATE}", csv_contents)
        self.assertIn(f"{REFERENCE},{SECOND_CANDIDATE}", csv_contents)

    def test_diagnostic_plots_keep_long_names_out_of_the_data_area(self):
        conditions = ("mixed-4096-alternating", "mixed-8191-alternating")
        samples = []
        for condition in conditions:
            samples.extend(
                (
                    make_sample(condition, REFERENCE, 1),
                    make_sample(
                        condition,
                        CANDIDATE,
                        1,
                        throughput=110,
                        short_p50=8,
                        long_p99=18,
                    ),
                    make_sample(
                        condition,
                        SECOND_CANDIDATE,
                        1,
                        throughput=115,
                        short_p50=7,
                        long_p99=17,
                    ),
                )
            )
        results, _ = analyze(samples, bootstrap_samples=10)

        with TemporaryDirectory() as temp_dir:
            pareto_path = Path(temp_dir) / "pareto.png"
            with patch.object(scheduler_analysis.plt, "close"):
                scheduler_analysis._plot_pareto(
                    results, pareto_path, REFERENCE, CANDIDATES
                )
            pareto_figure = scheduler_analysis.plt.gcf()
            pareto_axis = pareto_figure.axes[0]
            legend_labels = [
                text.get_text() for text in pareto_axis.get_legend().get_texts()
            ]

            self.assertTrue(pareto_path.exists())
            self.assertEqual(len(pareto_axis.texts), 0)
            self.assertEqual(len(legend_labels), len(conditions) * 2)
            self.assertTrue(
                all(
                    any(condition in label.replace("\n", "") for condition in conditions)
                    for label in legend_labels
                )
            )
            self.assertTrue(
                all(
                    any(candidate in label for candidate in CANDIDATES)
                    for label in legend_labels
                )
            )
            self.assertTrue(all("\n" in label for label in legend_labels))
            scheduler_analysis.plt.close(pareto_figure)

            mechanism_path = Path(temp_dir) / "mechanisms.png"
            with patch.object(scheduler_analysis.plt, "close"):
                scheduler_analysis._plot_mechanisms(
                    results, mechanism_path, REFERENCE, CANDIDATES
                )
            mechanism_figure = scheduler_analysis.plt.gcf()
            tick_labels = [
                tick.get_text() for tick in mechanism_figure.axes[-1].get_xticklabels()
            ]

            self.assertTrue(mechanism_path.exists())
            self.assertEqual(
                [label.replace("\n", "") for label in tick_labels],
                list(conditions),
            )
            self.assertTrue(all("\n" in label for label in tick_labels))
            self.assertTrue(
                all(
                    tick.get_rotation() == 0
                    for tick in mechanism_figure.axes[-1].get_xticklabels()
                )
            )
            self.assertEqual(
                [text.get_text() for text in mechanism_figure.axes[0].get_legend().get_texts()],
                [REFERENCE, *CANDIDATES],
            )
            self.assertEqual(len(mechanism_figure.axes[0].containers), 3)
            self.assertEqual(len(mechanism_figure.axes[1].containers), 3)
            scheduler_analysis.plt.close(mechanism_figure)

    def test_custom_reference_and_candidate_selectors(self):
        samples = [
            make_sample("one", "scheduler-v1", 1),
            make_sample(
                "one",
                "scheduler-v2",
                1,
                throughput=110,
                short_p50=8,
                long_p99=18,
            ),
        ]

        results, failures = analyze(
            samples,
            reference_selector="scheduler-v1",
            candidate_selectors=("scheduler-v2",),
            bootstrap_samples=10,
        )

        self.assertFalse(failures)
        self.assertEqual(results[0].verdict, "candidate-dominates")
        with TemporaryDirectory() as temp_dir:
            paths = write_outputs(
                results,
                failures,
                {},
                Path(temp_dir),
                reference_selector="scheduler-v1",
                candidate_selectors=("scheduler-v2",),
            )
            markdown = paths[0].read_text(encoding="utf-8")
            csv_contents = paths[1].read_text(encoding="utf-8")

        self.assertIn("`scheduler-v2` vs `scheduler-v1`", markdown)
        self.assertIn("reference_selector,candidate_selector", csv_contents)
        self.assertIn("scheduler-v1,scheduler-v2", csv_contents)

    def test_rejects_identical_reference_and_candidate_selectors(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            analyze(
                [],
                reference_selector="scheduler-v1",
                candidate_selectors=("scheduler-v1",),
                bootstrap_samples=10,
            )

        with self.assertRaisesRegex(ValueError, "must be unique"):
            analyze(
                [],
                candidate_selectors=(CANDIDATE, CANDIDATE),
                bootstrap_samples=10,
            )

        with self.assertRaisesRegex(ValueError, "unsupported scheduler"):
            analyze(
                [make_sample("one", "unknown-scheduler", 1)],
                bootstrap_samples=10,
            )

    def test_load_samples_rejects_removed_scheduler_selector(self):
        removed_selector = "scheduler-" + "optimized"
        report = {
            "workload_name": "one",
            "experiment_config": {
                "seed": 0,
                "repeats": 1,
                "engine_component": {"scheduler": removed_selector},
            },
            "provenance": {},
            "result": {"run_samples": [{}]},
        }
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unsupported scheduler"):
                load_samples([path])


if __name__ == "__main__":
    unittest.main()
