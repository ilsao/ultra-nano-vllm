from io import StringIO
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, call, patch

from nanovllm import SamplingParams
from rich.console import Console

import benchmark as benchmark_cli
import metrics
import repoter
import workloads
from runner import BatchRunResult, BenchmarkConfiguration, BenchmarkRunner


class BenchmarkCliArgumentTests(unittest.TestCase):
    def test_serving_system_name_defaults_to_nano_vllm(self):
        with patch("sys.argv", ["benchmark.py"]):
            args = benchmark_cli.parse_args()

        self.assertEqual(args.serving_system_name, "nano-vllm")

    def test_accepts_custom_serving_system_name(self):
        with patch(
            "sys.argv",
            ["benchmark.py", "--serving-system-name", "custom.vllm_2"],
        ):
            args = benchmark_cli.parse_args()

        self.assertEqual(args.serving_system_name, "custom.vllm_2")

    def test_rejects_unsafe_serving_system_name(self):
        for name in ["", "../vllm", "vllm/name", r"vllm\name", "vllm name"]:
            with (
                self.subTest(name=name),
                patch(
                    "sys.argv",
                    ["benchmark.py", "--serving-system-name", name],
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
            "temperature": 0.0,
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

        with (
            patch("runner.LLM", return_value=fake_llm),
            patch("runner.torch.cuda.get_device_name", return_value="NVIDIA RTX 3080"),
        ):
            runner = BenchmarkRunner(
                model_path="/models/Qwen3-0.6B/",
                enforce_eager=False,
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

        with (
            patch("runner.perf_counter", side_effect=[10.0, 12.0]),
            patch("runner.torch.cuda.synchronize") as synchronize,
            patch("runner.torch.cuda.reset_peak_memory_stats") as reset_peak,
            patch("runner.torch.cuda.max_memory_allocated", return_value=4 * 1024**2),
            patch("runner.torch.cuda.max_memory_reserved", return_value=6 * 1024**2),
        ):
            result = runner.run_benchmark(requests)

        fake_llm.generate.assert_called_once_with(
            [[1, 2], [3]],
            [requests[0].sampling_params, requests[1].sampling_params],
            use_tqdm=False,
        )
        self.assertEqual(synchronize.call_args_list, [call(), call()])
        reset_peak.assert_called_once_with()
        self.assertEqual(result.elapsed_time, 2.0)
        self.assertEqual(result.num_requests, 2)
        self.assertEqual(result.input_tokens, 3)
        self.assertEqual(result.output_tokens, 5)
        self.assertEqual(result.total_tokens, 8)
        self.assertEqual(result.peak_memory_allocated_mib, 4.0)
        self.assertEqual(result.peak_memory_reserved_mib, 6.0)


class MetricsTests(unittest.TestCase):
    def test_aggregates_median_range_and_peak_memory(self):
        runs = [
            BatchRunResult(t, 4, 12, 8, 20, allocated, reserved)
            for t, allocated, reserved in [
                (1.0, 10.0, 20.0),
                (4.0, 12.0, 19.0),
                (2.0, 11.0, 21.0),
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
        self.assertEqual(result.peak_memory_allocated_mib, 12.0)
        self.assertEqual(result.peak_memory_reserved_mib, 21.0)

    def test_rejects_empty_results(self):
        with self.assertRaises(ValueError):
            metrics.compute_benchmark_result([])


class ReporterTests(unittest.TestCase):
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
            request_throughput=metrics.MetricSummary(1.27, 1.25, 1.31),
            output_throughput=metrics.MetricSummary(1297.68, 1281.69, 1344.10),
            total_throughput=metrics.MetricSummary(2595.37, 2563.38, 2688.20),
            peak_memory_allocated_mib=13494.50,
            peak_memory_reserved_mib=512.0,
        )
        output = StringIO()
        console = Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=110,
        )

        repoter.BenchmarkReporter(console=console).show_result(result, configuration)
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
        self.assertIn("Memory", rendered)
        self.assertIn("262,144 tokens", rendered)
        self.assertNotIn("262,144 tokens – 262,144 tokens", rendered)
        self.assertIn("1,297.68 tok/s", rendered)
        self.assertIn("13.18 GiB (13,494.50 MiB)", rendered)
        self.assertIn("512.00 MiB", rendered)
        self.assertNotIn("\x1b", rendered)

    def test_progress_advances_once_per_repeat(self):
        console = Mock()
        progress = MagicMock()
        active_progress = Mock()
        active_progress.add_task.return_value = 9
        progress.__enter__.return_value = active_progress

        with patch("repoter.Progress", return_value=progress) as progress_type:
            run_indexes = list(
                repoter.BenchmarkReporter(console=console).track_repeats(3)
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
            request_throughput=metrics.MetricSummary(1.27, 1.25, 1.31),
            output_throughput=metrics.MetricSummary(1297.68, 1281.69, 1344.10),
            total_throughput=metrics.MetricSummary(2595.37, 2563.38, 2688.20),
            peak_memory_allocated_mib=13494.50,
            peak_memory_reserved_mib=512.0,
        )
        console = Mock()

        with TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir) / "report"
            with (
                patch("repoter.REPORT_DIR", report_dir),
                patch("repoter._report_timestamp", return_value="2026-08-18-203045"),
            ):
                report_path = repoter.BenchmarkReporter(console=console).save_result(
                    result,
                    configuration,
                    "custom-vllm",
                )

            self.assertEqual(
                report_path,
                report_dir / "2026-08-18-203045-custom-vllm-benchmark.json",
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["serving_system_name"], "custom-vllm")
        self.assertEqual(report["configuration"]["model"], "Qwen3-0.6B")
        self.assertEqual(report["configuration"]["gpu_memory_utilization"], 0.9)
        self.assertEqual(report["result"]["repeats"], 3)
        self.assertEqual(
            report["result"]["elapsed_time"],
            {"median": 202.01, "minimum": 195.03, "maximum": 204.53},
        )
        self.assertEqual(report["result"]["peak_memory_reserved_mib"], 512.0)
        console.print.assert_called_once_with(
            f"[bold green]✓[/] Report saved to [cyan]{report_path}[/cyan]"
        )


class BenchmarkCliTests(unittest.TestCase):
    def test_delegates_terminal_output_to_reporter(self):
        args = SimpleNamespace(
            model="~/model",
            enforce_eager=False,
            serving_system_name="nano-vllm",
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
            patch("benchmark.parse_args", return_value=args),
            patch("benchmark.BenchmarkReporter", return_value=reporter),
            patch("benchmark.BenchmarkRunner", return_value=runner),
            patch("benchmark.make_workload", return_value=["request"]),
            patch("benchmark.metrics.compute_benchmark_result", return_value=aggregate),
        ):
            benchmark_cli.main()

        self.assertEqual(runner.run_benchmark.call_count, 3)
        reporter.warmup.assert_called_once_with()
        reporter.warmup_complete.assert_called_once_with()
        reporter.track_repeats.assert_called_once_with(3)
        reporter.show_result.assert_called_once_with(aggregate, runner.configuration)
        reporter.save_result.assert_called_once_with(
            aggregate,
            runner.configuration,
            "nano-vllm",
        )


if __name__ == "__main__":
    unittest.main()
