from io import StringIO
import json
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, call, patch

from nanovllm import EngineComponent, SamplingParams
from nanovllm.engine.llm_engine import LLMEngine
from rich.console import Console

from benchmarks import benchmark as benchmark_cli
from benchmarks import metrics, workloads
from benchmarks.runner import (
    BatchRunResult,
    BenchmarkConfiguration,
    BenchmarkRunner,
)
from utils import reporter


class BenchmarkCliArgumentTests(unittest.TestCase):
    def test_supports_direct_script_invocation_from_repo_root(self):
        repo_root = Path(__file__).resolve().parent.parent

        completed = subprocess.run(
            [sys.executable, "benchmarks/benchmark.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Benchmark offline LLM throughput.", completed.stdout)

    def test_experiment_name_defaults_to_nano_vllm(self):
        with patch("sys.argv", ["benchmark.py"]):
            args = benchmark_cli.parse_args()

        self.assertEqual(args.experiment_name, "nano-vllm")
        self.assertEqual(args.max_model_len, 8192)

    def test_accepts_custom_experiment_name(self):
        with patch(
            "sys.argv",
            [
                "benchmark.py",
                "--experiment-name",
                "custom.vllm_2",
                "--max-model-len",
                "8192",
            ],
        ):
            args = benchmark_cli.parse_args()

        self.assertEqual(args.experiment_name, "custom.vllm_2")
        self.assertEqual(args.max_model_len, 8192)

    def test_rejects_unsafe_experiment_name(self):
        for name in ["", "../vllm", "vllm/name", r"vllm\name", "vllm name"]:
            with (
                self.subTest(name=name),
                patch(
                    "sys.argv",
                    ["benchmark.py", "--experiment-name", name],
                ),
                patch("sys.stderr", new=StringIO()),
                self.assertRaises(SystemExit),
            ):
                benchmark_cli.parse_args()

    def test_rejects_cli_workload_longer_than_max_model_len(self):
        with (
            patch(
                "sys.argv",
                [
                    "benchmark.py",
                    "--input-len",
                    "4096",
                    "--output-len",
                    "1024",
                    "--max-model-len",
                    "4096",
                ],
            ),
            patch("sys.stderr", new=StringIO()),
            self.assertRaises(SystemExit),
        ):
            benchmark_cli.parse_args()

class SyntheticWorkloadTests(unittest.TestCase):
    def test_fixed_lengths_vocab_range_and_seed(self):
        kwargs = dict(
            num_requests=3,
            input_len=4,
            output_len=5,
            seed=7,
            temperature=0.6,
            vocab_size=11,
            max_model_len=9,
        )
        first = workloads.synthetic_workload(**kwargs)
        second = workloads.synthetic_workload(**kwargs)
        different = workloads.synthetic_workload(**{**kwargs, "seed": 8})

        self.assertEqual(
            [request.prompt_token_ids for request in first],
            [request.prompt_token_ids for request in second],
        )
        self.assertNotEqual(
            [request.prompt_token_ids for request in first],
            [request.prompt_token_ids for request in different],
        )
        self.assertTrue(all(len(request.prompt_token_ids) == 4 for request in first))
        self.assertTrue(
            all(0 <= token_id < 11 for request in first for token_id in request.prompt_token_ids)
        )
        self.assertTrue(all(request.sampling_params.max_tokens == 5 for request in first))
        self.assertTrue(all(request.sampling_params.ignore_eos for request in first))

    def test_does_not_change_global_random_state(self):
        random.seed(123)
        state = random.getstate()
        workloads.synthetic_workload(1, 1, 1, 0, 0.6, 10, 2)
        self.assertEqual(random.getstate(), state)

    def test_supports_zero_temperature_for_greedy_decoding(self):
        requests = workloads.synthetic_workload(2, 1, 1, 0, 0.0, 10, 2)

        self.assertTrue(
            all(request.sampling_params.temperature == 0 for request in requests)
        )

    def test_rejects_invalid_arguments(self):
        valid = dict(
            num_requests=1,
            input_len=1,
            output_len=1,
            seed=0,
            temperature=0.6,
            vocab_size=10,
            max_model_len=2,
        )
        invalid_values = {
            "num_requests": 0,
            "input_len": 0,
            "output_len": 0,
            "temperature": -1.0,
            "vocab_size": 0,
            "max_model_len": 0,
        }
        for name, value in invalid_values.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                workloads.synthetic_workload(**{**valid, name: value})

        with self.assertRaisesRegex(ValueError, "exceeds max_model_len"):
            workloads.synthetic_workload(**{**valid, "output_len": 2})


class BenchmarkRunnerTests(unittest.TestCase):
    def test_exposes_effective_configuration(self):
        config = SimpleNamespace(
            model="/models/Qwen3-0.6B/",
            hf_config=SimpleNamespace(vocab_size=151936, dtype="torch.bfloat16"),
            tensor_parallel_size=2,
            enforce_eager=False,
            kvcache_block_size=256,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            num_kvcache_blocks=12345,
        )
        fake_llm = Mock()
        fake_llm.model_runner.config = config

        components = EngineComponent()
        with (
            patch("benchmarks.runner.LLM", return_value=fake_llm) as llm,
            patch(
                "benchmarks.runner.torch.cuda.get_device_name",
                return_value="NVIDIA RTX 3080",
            ),
        ):
            runner = BenchmarkRunner(
                model_path="/models/Qwen3-0.6B/",
                enforce_eager=False,
                engine_component=components,
            )

        llm.assert_called_once_with(
            "/models/Qwen3-0.6B/",
            enforce_eager=False,
            max_model_len=8192,
            engine_component=components,
        )

        self.assertEqual(
            runner.configuration,
            BenchmarkConfiguration(
                model="Qwen3-0.6B",
                device="NVIDIA RTX 3080",
                dtype="bfloat16",
                tensor_parallel_size=2,
                enforce_eager=False,
                block_size=256,
                max_model_len=4096,
                gpu_memory_utilization=0.9,
                num_kvcache_blocks=12345,
            ),
        )

    def test_runs_all_requests_in_one_generate_call(self):
        requests = [
            workloads.BenchmarkRequest([1, 2], SamplingParams(max_tokens=2)),
            workloads.BenchmarkRequest([3], SamplingParams(max_tokens=3)),
        ]
        fake_llm = Mock()
        fake_llm.generate.return_value = [
            {"token_ids": [4, 5]},
            {"token_ids": [6, 7, 8]},
        ]
        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.llm = fake_llm
        runner.configuration = BenchmarkConfiguration(
            model="model",
            device="GPU",
            dtype="bfloat16",
            tensor_parallel_size=1,
            enforce_eager=False,
            block_size=256,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            num_kvcache_blocks=10,
        )
        fake_llm.get_peak_kvcache_blocks.return_value = 6
        fake_llm.last_generation_metrics = SimpleNamespace(
            request_latencies=(1.0, 1.5),
            prefill_time=0.5,
            decode_time=1.0,
            prefill_tokens=3,
            decode_tokens=3,
        )

        with (
            patch("benchmarks.runner.perf_counter", side_effect=[10.0, 12.0]),
            patch("benchmarks.runner.torch.cuda.synchronize") as synchronize,
        ):
            result = runner.run_benchmark(requests)

        fake_llm.generate.assert_called_once_with(
            [[1, 2], [3]],
            [requests[0].sampling_params, requests[1].sampling_params],
            use_tqdm=False,
        )
        self.assertEqual(synchronize.call_args_list, [call(), call()])
        fake_llm.reset_kvcache_metrics.assert_called_once_with()
        fake_llm.get_peak_kvcache_blocks.assert_called_once_with()
        self.assertEqual(result.elapsed_time, 2.0)
        self.assertEqual(result.num_requests, 2)
        self.assertEqual(result.input_tokens, 3)
        self.assertEqual(result.output_tokens, 5)
        self.assertEqual(result.total_tokens, 8)
        self.assertEqual(result.peak_kvcache_blocks, 6)
        self.assertEqual(result.kvcache_capacity_blocks, 10)
        self.assertEqual(result.request_latencies, (1.0, 1.5))
        self.assertEqual(result.prefill_time, 0.5)
        self.assertEqual(result.decode_time, 1.0)
        self.assertEqual(result.prefill_tokens, 3)
        self.assertEqual(result.decode_tokens, 3)

    def test_close_shuts_down_underlying_engine(self):
        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.llm = Mock()

        runner.close()

        runner.llm.exit.assert_called_once_with()


class LLMEngineLifecycleTests(unittest.TestCase):
    def test_generate_records_phase_times_tokens_and_request_latencies(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine.add_request = Mock()
        engine.is_finished = Mock(side_effect=[False, False, True])
        engine.step = Mock(
            side_effect=[
                ([(0, [10])], 2),
                ([(1, [20, 21])], -1),
            ]
        )
        engine.tokenizer = SimpleNamespace(decode=lambda tokens: str(tokens))

        with patch(
            "nanovllm.engine.llm_engine.perf_counter",
            side_effect=[0.0, 0.0, 2.0, 2.0, 5.0],
        ):
            outputs = engine.generate(
                [[1], [2]],
                [SamplingParams(), SamplingParams()],
                use_tqdm=False,
            )

        self.assertEqual(
            outputs,
            [
                {"text": "[10]", "token_ids": [10]},
                {"text": "[20, 21]", "token_ids": [20, 21]},
            ],
        )
        self.assertEqual(
            engine.last_generation_metrics.request_latencies,
            (2.0, 5.0),
        )
        self.assertEqual(engine.last_generation_metrics.prefill_time, 2.0)
        self.assertEqual(engine.last_generation_metrics.decode_time, 3.0)
        self.assertEqual(engine.last_generation_metrics.prefill_tokens, 2)
        self.assertEqual(engine.last_generation_metrics.decode_tokens, 1)

    def test_exit_is_idempotent_and_releases_workers(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine._closed = False
        model_runner = Mock()
        engine.model_runner = model_runner
        process = Mock()
        engine.ps = [process]
        engine.events = [Mock()]

        with patch("nanovllm.engine.llm_engine.atexit.unregister") as unregister:
            engine.exit()
            engine.exit()

        unregister.assert_called_once_with(engine.exit)
        model_runner.call.assert_called_once_with("exit")
        process.join.assert_called_once_with()
        self.assertFalse(hasattr(engine, "model_runner"))
        self.assertEqual(engine.ps, [])
        self.assertEqual(engine.events, [])


class MetricsTests(unittest.TestCase):
    def test_aggregates_median_range_and_peak_kvcache_usage(self):
        runs = [
            BatchRunResult(
                t,
                4,
                12,
                8,
                20,
                peak,
                20,
                (0.5, 1.0, 1.5, 2.0),
                t / 4,
                t / 2,
                12,
                4,
            )
            for t, peak in [
                (1.0, 10),
                (4.0, 12),
                (2.0, 11),
            ]
        ]

        result = metrics.compute_benchmark_result(runs)

        self.assertEqual(result.repeats, 3)
        self.assertEqual(result.elapsed_time.median, 2.0)
        self.assertEqual(result.elapsed_time.minimum, 1.0)
        self.assertEqual(result.elapsed_time.maximum, 4.0)
        self.assertEqual(result.request_throughput.median, 2.0)
        self.assertEqual(result.request_throughput.minimum, 1.0)
        self.assertEqual(result.request_throughput.maximum, 4.0)
        self.assertAlmostEqual(result.elapsed_time.mean, 7 / 3)
        self.assertGreater(result.elapsed_time.standard_deviation, 0)
        self.assertEqual(result.latency_p50.median, 1.25)
        self.assertAlmostEqual(result.latency_p90.median, 1.85)
        self.assertAlmostEqual(result.latency_p99.median, 1.985)
        self.assertEqual(result.prefill_throughput.median, 24.0)
        self.assertEqual(result.decode_throughput.median, 4.0)
        self.assertEqual(result.prefill_time.median, 0.5)
        self.assertEqual(result.decode_time.median, 1.0)
        self.assertEqual(result.peak_kvcache_blocks, 12)
        self.assertEqual(result.peak_kvcache_utilization, 0.6)

    def test_rejects_invalid_kvcache_usage(self):
        cases = [
            (
                [
                    BatchRunResult(
                        1.0, 1, 1, 1, 2, 1, 0, (1.0,), .5, .5, 1, 1
                    )
                ],
                "greater than zero",
            ),
            (
                [
                    BatchRunResult(
                        1.0, 1, 1, 1, 2, 1, 2, (1.0,), .5, .5, 1, 1
                    ),
                    BatchRunResult(
                        1.0, 1, 1, 1, 2, 1, 3, (1.0,), .5, .5, 1, 1
                    ),
                ],
                "same KV-cache block capacity",
            ),
            (
                [
                    BatchRunResult(
                        1.0, 1, 1, 1, 2, 3, 2, (1.0,), .5, .5, 1, 1
                    )
                ],
                "within block capacity",
            ),
        ]
        for runs, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                metrics.compute_benchmark_result(runs)

    def test_allows_a_workload_without_decode_steps(self):
        result = metrics.compute_benchmark_result([
            BatchRunResult(
                1.0, 1, 1, 1, 2, 1, 2, (1.0,), .5, 0.0, 1, 0
            )
        ])

        self.assertEqual(result.decode_time.mean, 0.0)
        self.assertEqual(result.decode_throughput.mean, 0.0)

    def test_rejects_invalid_latency_and_phase_measurements(self):
        invalid_runs = [
            BatchRunResult(1.0, 1, 1, 1, 2, 1, 2, (), .5, .5, 1, 1),
            BatchRunResult(1.0, 1, 1, 1, 2, 1, 2, (1.0,), 0, .5, 1, 1),
            BatchRunResult(1.0, 1, 1, 1, 2, 1, 2, (1.0,), .5, 0, 1, 1),
        ]
        for run in invalid_runs:
            with self.subTest(run=run), self.assertRaises(ValueError):
                metrics.compute_benchmark_result([run])

    def test_rejects_empty_results(self):
        with self.assertRaises(ValueError):
            metrics.compute_benchmark_result([])


class ReporterTests(unittest.TestCase):
    def test_default_report_dir_is_under_benchmarks(self):
        self.assertEqual(
            reporter.REPORT_DIR,
            Path(reporter.__file__).resolve().parent.parent / "benchmarks" / "report",
        )

    def test_displays_generated_plot_paths(self):
        console = Mock()
        paths = [Path("first.png"), Path("second.png")]

        reporter.Reporter(console=console).show_plot_paths(paths)

        console.print.assert_has_calls(
            [
                call("[bold green]✓[/] Plot saved to [cyan]first.png[/cyan]"),
                call("[bold green]✓[/] Plot saved to [cyan]second.png[/cyan]"),
            ]
        )

    def test_renders_grouped_plain_text_without_ansi_codes(self):
        configuration = BenchmarkConfiguration(
            model="Qwen3-0.6B",
            device="NVIDIA RTX 3080",
            dtype="bfloat16",
            tensor_parallel_size=1,
            enforce_eager=False,
            block_size=256,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            num_kvcache_blocks=12345,
        )
        result = metrics.BenchmarkResult(
            repeats=3,
            num_requests=256,
            input_tokens=262144,
            output_tokens=metrics.MetricSummary(262144, 262144, 262144),
            total_tokens=metrics.MetricSummary(524288, 520000, 524288),
            elapsed_time=metrics.MetricSummary(202.01, 195.03, 204.53),
            latency_p50=metrics.MetricSummary(100, 90, 110, 100, 5),
            latency_p90=metrics.MetricSummary(180, 170, 190, 180, 5),
            latency_p99=metrics.MetricSummary(200, 190, 204, 199, 4),
            request_throughput=metrics.MetricSummary(1.27, 1.25, 1.31),
            output_throughput=metrics.MetricSummary(1297.68, 1281.69, 1344.10),
            total_throughput=metrics.MetricSummary(2595.37, 2563.38, 2688.20),
            prefill_throughput=metrics.MetricSummary(3000, 2900, 3100, 3000, 50),
            decode_throughput=metrics.MetricSummary(1200, 1100, 1300, 1200, 50),
            prefill_time=metrics.MetricSummary(80, 75, 85, 80, 3),
            decode_time=metrics.MetricSummary(120, 115, 125, 120, 3),
            peak_kvcache_blocks=9876,
            peak_kvcache_utilization=0.8,
        )
        output = StringIO()
        console = Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=110,
        )

        reporter.Reporter(console=console).show_result(result, configuration)
        rendered = output.getvalue()

        self.assertIn("Configuration", rendered)
        self.assertIn("Qwen3-0.6B", rendered)
        self.assertIn("NVIDIA RTX 3080", rendered)
        self.assertIn("bfloat16", rendered)
        self.assertIn("Tensor parallel", rendered)
        self.assertIn("False", rendered)
        self.assertIn("256", rendered)
        self.assertIn("4,096", rendered)
        self.assertIn("90%", rendered)
        self.assertIn("12,345", rendered)
        self.assertIn("Workload", rendered)
        self.assertIn("Performance", rendered)
        self.assertIn("KV Cache", rendered)
        self.assertIn("262,144 tokens", rendered)
        self.assertNotIn("262,144 tokens – 262,144 tokens", rendered)
        self.assertIn("1,297.68 tok/s", rendered)
        self.assertIn("Latency p99", rendered)
        self.assertIn("Prefill throughput", rendered)
        self.assertIn("199.00 ± 4.00 s", rendered)
        self.assertIn("9,876 blocks", rendered)
        self.assertIn("80.00%", rendered)
        self.assertNotIn("\x1b", rendered)

    def test_progress_advances_once_per_repeat(self):
        console = Mock()
        progress = MagicMock()
        active_progress = Mock()
        active_progress.add_task.return_value = 9
        progress.__enter__.return_value = active_progress

        with patch(
            "utils.reporter.Progress",
            return_value=progress,
        ) as progress_type:
            run_indexes = list(
                reporter.Reporter(console=console).track_repeats(3)
            )

        self.assertEqual(run_indexes, [0, 1, 2])
        active_progress.add_task.assert_called_once_with("Benchmarking", total=3)
        active_progress.update.assert_has_calls([
            call(9, advance=1),
            call(9, advance=1),
            call(9, advance=1),
        ])
        self.assertTrue(progress_type.call_args.kwargs["transient"])

    def test_saves_structured_json_report(self):
        configuration = BenchmarkConfiguration(
            model="Qwen3-0.6B",
            device="NVIDIA RTX 3080",
            dtype="bfloat16",
            tensor_parallel_size=1,
            enforce_eager=False,
            block_size=256,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            num_kvcache_blocks=12345,
        )
        result = metrics.BenchmarkResult(
            repeats=3,
            num_requests=256,
            input_tokens=262144,
            output_tokens=metrics.MetricSummary(262144, 262144, 262144),
            total_tokens=metrics.MetricSummary(524288, 520000, 524288),
            elapsed_time=metrics.MetricSummary(202.01, 195.03, 204.53),
            latency_p50=metrics.MetricSummary(100, 90, 110, 100, 5),
            latency_p90=metrics.MetricSummary(180, 170, 190, 180, 5),
            latency_p99=metrics.MetricSummary(200, 190, 204, 199, 4),
            request_throughput=metrics.MetricSummary(1.27, 1.25, 1.31),
            output_throughput=metrics.MetricSummary(1297.68, 1281.69, 1344.10),
            total_throughput=metrics.MetricSummary(2595.37, 2563.38, 2688.20),
            prefill_throughput=metrics.MetricSummary(3000, 2900, 3100, 3000, 50),
            decode_throughput=metrics.MetricSummary(1200, 1100, 1300, 1200, 50),
            prefill_time=metrics.MetricSummary(80, 75, 85, 80, 3),
            decode_time=metrics.MetricSummary(120, 115, 125, 120, 3),
            peak_kvcache_blocks=9876,
            peak_kvcache_utilization=0.8,
        )
        console = Mock()

        with TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir) / "report"
            with (
                patch("utils.reporter.REPORT_DIR", report_dir),
                patch(
                    "utils.reporter._report_timestamp",
                    return_value="2026-08-18-203045",
                ),
            ):
                report_path = reporter.Reporter(console=console).save_result(
                    result,
                    configuration,
                    "custom-vllm",
                )

            self.assertEqual(
                report_path,
                report_dir / "2026-08-18-203045-custom-vllm-benchmark.json",
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["experiment_name"], "custom-vllm")
        self.assertEqual(report["configuration"]["model"], "Qwen3-0.6B")
        self.assertEqual(report["configuration"]["gpu_memory_utilization"], 0.9)
        self.assertEqual(report["result"]["repeats"], 3)
        self.assertEqual(
            report["result"]["elapsed_time"],
            {
                "median": 202.01,
                "minimum": 195.03,
                "maximum": 204.53,
                "mean": None,
                "standard_deviation": None,
            },
        )
        self.assertEqual(report["result"]["peak_kvcache_blocks"], 9876)
        self.assertEqual(report["result"]["peak_kvcache_utilization"], 0.8)
        self.assertEqual(report["result"]["latency_p99"]["mean"], 199)
        self.assertEqual(
            report["result"]["latency_p99"]["standard_deviation"],
            4,
        )
        console.print.assert_called_once_with(
            f"[bold green]✓[/] Report saved to [cyan]{report_path}[/cyan]"
        )

    def test_saves_experiment_config_to_explicit_path(self):
        configuration = BenchmarkConfiguration(
            model="Qwen3-0.6B",
            device="GPU",
            dtype="bfloat16",
            tensor_parallel_size=1,
            enforce_eager=False,
            block_size=256,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            num_kvcache_blocks=100,
        )
        result = metrics.compute_benchmark_result([
            BatchRunResult(
                1.0, 1, 2, 3, 5, 4, 6, (1.0,), .4, .6, 2, 2
            )
        ])
        experiment_config = benchmark_cli.BenchmarkConfig(
            num_requests=1,
            input_len=2,
            output_len=3,
            seed=7,
            temperature=0.8,
            repeats=1,
            experiment_name="sweep-one",
        )

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "sweep.json"
            saved_path = reporter.Reporter(console=Mock()).save_result(
                result,
                configuration,
                "sweep-one",
                output_path=output_path,
                experiment_config=experiment_config,
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved_path, output_path)
        self.assertEqual(report["experiment_config"]["input_len"], 2)
        self.assertEqual(report["experiment_config"]["temperature"], 0.8)
        self.assertEqual(report["configuration"]["model"], "Qwen3-0.6B")


class BenchmarkCliTests(unittest.TestCase):
    def test_execute_rejects_invalid_lengths_before_constructing_runner(self):
        config = benchmark_cli.BenchmarkConfig(
            input_len=4096,
            output_len=1024,
            max_model_len=4096,
            experiment_name="too-long",
        )

        with (
            patch("benchmarks.benchmark.BenchmarkRunner") as runner_type,
            self.assertRaisesRegex(ValueError, "too-long.*5120"),
        ):
            benchmark_cli.execute_benchmark(config, Mock())

        runner_type.assert_not_called()

    def test_execute_benchmark_runs_warmup_repeats_and_metrics(self):
        config = benchmark_cli.BenchmarkConfig(
            model="~/model",
            enforce_eager=False,
            experiment_name="nano-vllm",
            seed=4,
            repeats=3,
            num_requests=2,
        )
        reporter = Mock()
        reporter.warmup.return_value = MagicMock()
        reporter.track_repeats.return_value = range(3)
        runner = Mock()
        runner.run_benchmark.side_effect = ["run-1", "run-2", "run-3"]
        aggregate = Mock()

        with (
            patch(
                "benchmarks.benchmark.BenchmarkRunner",
                return_value=runner,
            ) as runner_type,
            patch("benchmarks.benchmark.make_workload", return_value=["request"]),
            patch(
                "benchmarks.benchmark.metrics.compute_benchmark_result",
                return_value=aggregate,
            ),
        ):
            result, configuration = benchmark_cli.execute_benchmark(config, reporter)

        runner_type.assert_called_once_with(
            model_path=str(Path("~/model").expanduser()),
            enforce_eager=False,
            max_model_len=8192,
            engine_component=config.engine_component,
        )

        self.assertEqual(runner.run_benchmark.call_count, 3)
        reporter.warmup.assert_called_once_with()
        reporter.warmup_complete.assert_called_once_with()
        reporter.track_repeats.assert_called_once_with(3)
        self.assertIs(result, aggregate)
        self.assertIs(configuration, runner.configuration)
        runner.close.assert_not_called()

    def test_execute_benchmark_closes_opt_in_runner_after_success(self):
        config = benchmark_cli.BenchmarkConfig(repeats=1, num_requests=1)
        reporter = Mock()
        reporter.warmup.return_value = MagicMock()
        reporter.track_repeats.return_value = range(1)
        runner = Mock()
        runner.run_benchmark.return_value = "run"

        with (
            patch("benchmarks.benchmark.BenchmarkRunner", return_value=runner),
            patch("benchmarks.benchmark.make_workload", return_value=["request"]),
            patch(
                "benchmarks.benchmark.metrics.compute_benchmark_result",
                return_value="aggregate",
            ),
        ):
            result, configuration = benchmark_cli.execute_benchmark(
                config,
                reporter,
                close_runner=True,
            )

        self.assertEqual(result, "aggregate")
        self.assertIs(configuration, runner.configuration)
        runner.close.assert_called_once_with()

    def test_execute_benchmark_closes_opt_in_runner_after_failure(self):
        config = benchmark_cli.BenchmarkConfig()
        reporter = Mock()
        reporter.warmup.return_value = MagicMock()
        runner = Mock()
        runner.warmup.side_effect = RuntimeError("warmup failed")

        with (
            patch("benchmarks.benchmark.BenchmarkRunner", return_value=runner),
            patch("benchmarks.benchmark.make_workload", return_value=["request"]),
            self.assertRaisesRegex(RuntimeError, "warmup failed"),
        ):
            benchmark_cli.execute_benchmark(
                config,
                reporter,
                close_runner=True,
            )

        runner.close.assert_called_once_with()

    def test_main_delegates_rendering_and_saving_to_reporter(self):
        config = benchmark_cli.BenchmarkConfig(experiment_name="single")
        reporter = Mock()

        with (
            patch("benchmarks.benchmark.parse_args", return_value=config),
            patch("benchmarks.benchmark.Reporter", return_value=reporter),
            patch(
                "benchmarks.benchmark.execute_benchmark",
                return_value=("aggregate", "engine"),
            ) as execute,
        ):
            benchmark_cli.main()

        execute.assert_called_once_with(config, reporter)
        reporter.show_result.assert_called_once_with("aggregate", "engine")
        reporter.save_result.assert_called_once_with(
            "aggregate", "engine", "single"
        )


if __name__ == "__main__":
    unittest.main()
