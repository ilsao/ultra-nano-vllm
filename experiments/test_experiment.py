from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from benchmarks.benchmark import BenchmarkConfig
from benchmarks.runner import BenchmarkConfiguration
from experiments.plot import PlotRecord
from experiments import experiment
from experiments.test_plot import make_result


def make_configuration(enforce_eager: bool = False) -> BenchmarkConfiguration:
    return BenchmarkConfiguration(
        model="Qwen3-0.6B",
        device="GPU",
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=enforce_eager,
        block_size=256,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        num_kvcache_blocks=100,
    )


class ExperimentTests(unittest.TestCase):
    def test_supports_direct_script_invocation_from_repo_root(self):
        repo_root = Path(__file__).resolve().parent.parent
        completed = subprocess.run(
            [sys.executable, "experiments/experiment.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Run and plot YAML benchmark experiments", completed.stdout)

    def test_loads_repeated_configs_and_applies_cli_overrides(self):
        with TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.yaml"
            second = Path(temp_dir) / "second.yaml"
            first.write_text(
                "num_requests: [1, 2]\n"
                "experiment_name: [first-one, first-two]\n",
                encoding="utf-8",
            )
            second.write_text("experiment_name: second\n", encoding="utf-8")
            args = experiment.parse_args(
                [
                    "--config",
                    str(first),
                    "--config",
                    str(second),
                    "--enforce-eager",
                    "--temperature",
                    "0",
                    "--max-model-len",
                    "8192",
                ]
            )

        self.assertEqual(len(args.groups), 2)
        self.assertEqual(args.groups[0].dimensions, ("num_requests",))
        self.assertEqual(
            [
                config.experiment_name
                for group in args.groups
                for config in group.configs
            ],
            ["first-one", "first-two", "second"],
        )
        self.assertTrue(
            all(
                config.enforce_eager
                for group in args.groups
                for config in group.configs
            )
        )
        self.assertTrue(
            all(
                config.temperature == 0
                for group in args.groups
                for config in group.configs
            )
        )
        self.assertTrue(
            all(
                config.max_model_len == 8192
                for group in args.groups
                for config in group.configs
            )
        )

    def test_rejects_duplicate_names_name_override_and_high_dimensions(self):
        with TemporaryDirectory() as temp_dir:
            duplicate = Path(temp_dir) / "duplicate.yaml"
            duplicate.write_text(
                "num_requests: [1, 2]\nexperiment_name: [same, same]\n",
                encoding="utf-8",
            )
            high = Path(temp_dir) / "high.yaml"
            high.write_text(
                "num_requests: [1, 2]\n"
                "input_len: [10, 20]\n"
                "seed: [0, 1]\n"
                "experiment_name: [a, b, c, d, e, f, g, h]\n",
                encoding="utf-8",
            )
            unique = Path(temp_dir) / "unique.yaml"
            unique.write_text(
                "num_requests: [1, 2]\nexperiment_name: [one, two]\n",
                encoding="utf-8",
            )
            cases = [
                (["--config", str(duplicate)], "unique"),
                (
                    ["--config", str(unique), "--experiment-name", "override"],
                    "multiple runs",
                ),
                (["--config", str(high)], "at most two"),
            ]
            for argv, message in cases:
                with (
                    self.subTest(argv=argv),
                    patch("sys.stderr"),
                    self.assertRaises(SystemExit),
                ):
                    experiment.parse_args(argv)

    def test_builds_microsecond_report_path(self):
        path = experiment.report_path(
            "experiment-one", datetime(2026, 8, 18, 20, 30, 45, 123456)
        )
        self.assertEqual(
            path,
            experiment.REPORT_DIR / "2026-08-18-experiment-one-203045123456.json",
        )

    def test_run_groups_executes_displays_saves_and_plots_each_result(self):
        configs = (
            BenchmarkConfig(num_requests=1, experiment_name="first"),
            BenchmarkConfig(num_requests=2, experiment_name="second"),
        )
        group = experiment.ExperimentGroup("group", configs, ("num_requests",))
        reporter = Mock()
        results = [
            (make_result(1.0), make_configuration()),
            (make_result(2.0), make_configuration()),
        ]
        report_paths = [Path("first.json"), Path("second.json")]
        chart_paths = [Path("chart.png")]

        with (
            patch(
                "experiments.experiment.experiment_runner.execute_benchmark_isolated",
                side_effect=results,
            ) as run,
            patch("experiments.experiment.report_path", side_effect=report_paths),
            patch(
                "experiments.experiment.plot_records", return_value=chart_paths
            ) as render,
        ):
            returned = experiment.run_groups([group], reporter)

        self.assertEqual(
            run.call_args_list,
            [call(configs[0]), call(configs[1])],
        )
        self.assertEqual(
            reporter.show_result.call_args_list,
            [call(results[0][0], results[0][1]), call(results[1][0], results[1][1])],
        )
        self.assertEqual(reporter.save_result.call_count, 2)
        plotted_records = render.call_args.args[0]
        self.assertEqual([record.config for record in plotted_records], list(configs))
        reporter.show_plot_paths.assert_called_once_with(chart_paths)
        self.assertEqual(returned, chart_paths)

    def test_plot_records_dispatches_zero_one_and_two_dimensions(self):
        one = [
            PlotRecord(
                BenchmarkConfig(num_requests=value, experiment_name=f"one-{value}"),
                make_result(float(value)),
            )
            for value in (1, 2)
        ]
        two = [
            PlotRecord(
                BenchmarkConfig(
                    num_requests=requests,
                    input_len=length,
                    experiment_name=f"two-{index}",
                ),
                make_result(float(index + 1)),
            )
            for index, (requests, length) in enumerate(
                [(1, 10), (1, 20), (2, 10), (2, 20)]
            )
        ]
        scalar = [PlotRecord(BenchmarkConfig(), make_result(1.0))]

        with (
            patch(
                "experiments.experiment.plot_1d", return_value=[Path("1d.png")]
            ) as p1,
            patch(
                "experiments.experiment.plot_2d", return_value=[Path("2d.png")]
            ) as p2,
        ):
            self.assertEqual(
                experiment.plot_records(scalar, (), plot_name="scalar"), []
            )
            self.assertEqual(
                experiment.plot_records(one, ("num_requests",), plot_name="one"),
                [Path("1d.png")],
            )
            self.assertEqual(
                experiment.plot_records(
                    two, ("num_requests", "input_len"), plot_name="two"
                ),
                [Path("2d.png")],
            )
        p1.assert_called_once()
        p2.assert_called_once()

    def test_plot_only_main_never_executes_benchmark(self):
        record = PlotRecord(BenchmarkConfig(), make_result(1.0))
        args = SimpleNamespace(
            report_paths=[Path("saved.json")], records=[record], dimensions=()
        )
        reporter = Mock()
        with (
            patch("experiments.experiment.parse_args", return_value=args),
            patch("experiments.experiment.Reporter", return_value=reporter),
            patch("experiments.experiment.plot_records", return_value=[]) as render,
            patch(
                "experiments.experiment.experiment_runner.execute_benchmark_isolated"
            ) as execute,
        ):
            experiment.main()
        render.assert_called_once()
        reporter.show_plot_paths.assert_called_once_with([])
        execute.assert_not_called()

    def test_plot_only_parses_explicit_reports_and_rejects_overrides(self):
        config = BenchmarkConfig(experiment_name="saved")
        result = make_result(1.0)
        with TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "saved.json"
            report.write_text(
                json.dumps(
                    {
                        "experiment_config": asdict(config),
                        "result": asdict(result),
                    }
                ),
                encoding="utf-8",
            )
            args = experiment.parse_args(["--plot-only", str(report)])
            self.assertEqual(args.dimensions, ())
            self.assertEqual(args.records, [PlotRecord(config, result)])

            with patch("sys.stderr"), self.assertRaises(SystemExit):
                experiment.parse_args(
                    ["--plot-only", str(report), "--num-requests", "2"]
                )


if __name__ == "__main__":
    unittest.main()
