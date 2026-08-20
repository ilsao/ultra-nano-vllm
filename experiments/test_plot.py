from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.benchmark import BenchmarkConfig
from benchmarks.metrics import BenchmarkResult, MetricSummary
from experiments.config import CONFIG_DIMENSIONS
from experiments import plot
from experiments.plot import (
    PlotRecord,
    infer_dimensions,
    load_reports,
    plot_1d,
    plot_2d,
    validate_grid,
)
from nanovllm import EngineComponent


def make_result(value: float) -> BenchmarkResult:
    summary = MetricSummary(value, value - 0.25, value + 0.5, value + 0.1, 0.2)
    return BenchmarkResult(
        repeats=3,
        num_requests=1,
        input_tokens=2,
        output_tokens=summary,
        total_tokens=summary,
        elapsed_time=summary,
        latency_p50=summary,
        latency_p90=summary,
        latency_p99=summary,
        request_throughput=summary,
        output_throughput=summary,
        total_throughput=summary,
        prefill_throughput=summary,
        decode_throughput=summary,
        prefill_time=summary,
        decode_time=summary,
        peak_kvcache_blocks=int(value + 10),
        peak_kvcache_utilization=(value + 10) / 100,
    )


class PlotTests(unittest.TestCase):
    def test_uses_config_domain_dimension_order(self):
        self.assertIs(plot.CONFIG_DIMENSIONS, CONFIG_DIMENSIONS)

    def test_infers_dimensions_in_benchmark_field_order(self):
        records = [
            PlotRecord(
                BenchmarkConfig(
                    num_requests=requests,
                    enforce_eager=eager,
                    experiment_name=f"run-{index}",
                ),
                make_result(float(index)),
            )
            for index, (requests, eager) in enumerate(
                [(1, False), (1, True), (2, False), (2, True)]
            )
        ]

        self.assertEqual(
            infer_dimensions(records), ("num_requests", "enforce_eager")
        )
        self.assertEqual(
            validate_grid(records), ("num_requests", "enforce_eager")
        )

    def test_infers_component_selector_as_categorical_dimension(self):
        with patch("nanovllm.engine.component._load_factory") as load_factory:
            records = [
                PlotRecord(
                    BenchmarkConfig(
                        engine_component=EngineComponent(scheduler=selector),
                        experiment_name=f"run-{index}",
                    ),
                    make_result(float(index + 1)),
                )
                for index, selector in enumerate(
                    ("scheduler", "optimized-scheduler-v1")
                )
            ]
            load_factory.assert_not_called()

        self.assertEqual(infer_dimensions(records), ("scheduler",))
        self.assertEqual(validate_grid(records), ("scheduler",))

    def test_plot_1d_saves_all_metric_files(self):
        records = [
            PlotRecord(
                BenchmarkConfig(num_requests=value, experiment_name=f"run-{value}"),
                make_result(float(value)),
            )
            for value in (1, 2, 4)
        ]
        with TemporaryDirectory() as temp_dir:
            paths = plot_1d(
                records,
                "num_requests",
                output_dir=temp_dir,
                plot_name="one dimensional",
            )
            self.assertEqual(len(paths), 13)
            self.assertTrue(all(path.exists() for path in paths))
            self.assertTrue(all(path.suffix == ".png" for path in paths))
            self.assertTrue(all("1d-num_requests" in path.name for path in paths))
            self.assertTrue(
                any("peak-kvcache-blocks" in path.name for path in paths)
            )
            self.assertTrue(
                any("peak-kvcache-utilization" in path.name for path in paths)
            )
            self.assertTrue(any("latency-p99" in path.name for path in paths))
            self.assertTrue(
                any("prefill-throughput" in path.name for path in paths)
            )
            self.assertTrue(any("decode-time" in path.name for path in paths))

    def test_orders_numeric_axes_and_preserves_categorical_order(self):
        integer_records = [
            PlotRecord(
                BenchmarkConfig(
                    num_requests=value,
                    experiment_name=f"requests-{value}",
                ),
                make_result(float(value)),
            )
            for value in (128, 16, 64, 32)
        ]
        float_records = [
            PlotRecord(
                BenchmarkConfig(
                    temperature=value,
                    experiment_name=f"temperature-{index}",
                ),
                make_result(float(index + 1)),
            )
            for index, value in enumerate((0.6, 0.1, 0.3))
        ]
        string_records = [
            PlotRecord(
                BenchmarkConfig(model=value, experiment_name=f"model-{index}"),
                make_result(float(index + 1)),
            )
            for index, value in enumerate(("model-z", "model-a", "model-m"))
        ]
        boolean_records = [
            PlotRecord(
                BenchmarkConfig(
                    enforce_eager=value,
                    experiment_name=f"eager-{index}",
                ),
                make_result(float(index + 1)),
            )
            for index, value in enumerate((True, False))
        ]

        self.assertEqual(
            plot._ordered_values(integer_records, "num_requests"),
            [16, 32, 64, 128],
        )
        self.assertEqual(
            plot._ordered_values(float_records, "temperature"),
            [0.1, 0.3, 0.6],
        )
        self.assertEqual(
            plot._ordered_values(string_records, "model"),
            ["model-z", "model-a", "model-m"],
        )
        self.assertEqual(
            plot._ordered_values(boolean_records, "enforce_eager"),
            [True, False],
        )

    def test_plot_2d_sorts_numeric_axes_without_moving_results(self):
        from matplotlib.axes import Axes

        records = [
            PlotRecord(
                BenchmarkConfig(
                    num_requests=requests,
                    temperature=temperature,
                    experiment_name=f"run-{index}",
                ),
                make_result(value),
            )
            for index, (requests, temperature, value) in enumerate(
                (
                    (2, 0.2, 22.0),
                    (1, 0.2, 12.0),
                    (2, 0.1, 21.0),
                    (1, 0.1, 11.0),
                )
            )
        ]
        matrices = []
        original_imshow = Axes.imshow

        def capture_matrix(axis, matrix, *args, **kwargs):
            matrices.append(np.array(matrix, copy=True))
            return original_imshow(axis, matrix, *args, **kwargs)

        with TemporaryDirectory() as temp_dir, patch.object(
            plot, "_METRICS", (plot._METRICS[0],)
        ), patch.object(Axes, "imshow", autospec=True, side_effect=capture_matrix):
            paths = plot_2d(
                records,
                "num_requests",
                "temperature",
                output_dir=temp_dir,
                plot_name="numeric axes",
            )

        self.assertEqual(len(paths), 1)
        self.assertEqual(
            plot._ordered_values(records, "num_requests"),
            [1, 2],
        )
        self.assertEqual(
            plot._ordered_values(records, "temperature"),
            [0.1, 0.2],
        )
        np.testing.assert_array_equal(
            matrices[0],
            np.array([[11.1, 21.1], [12.1, 22.1]]),
        )

    def test_plot_2d_supports_categorical_axes_and_saves_all_files(self):
        records = []
        index = 0
        for model in ("model-a", "model-b"):
            for eager in (False, True):
                records.append(
                    PlotRecord(
                        BenchmarkConfig(
                            model=model,
                            enforce_eager=eager,
                            experiment_name=f"run-{index}",
                        ),
                        make_result(float(index + 1)),
                    )
                )
                index += 1

        with TemporaryDirectory() as temp_dir:
            paths = plot_2d(
                records,
                "model",
                "enforce_eager",
                output_dir=temp_dir,
                plot_name="categorical",
            )
            self.assertEqual(len(paths), 13)
            self.assertTrue(all(path.exists() for path in paths))
            self.assertTrue(
                all("2d-model-enforce_eager" in path.name for path in paths)
            )

    def test_rejects_duplicate_and_incomplete_grids(self):
        duplicate = PlotRecord(BenchmarkConfig(), make_result(1.0))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_grid([duplicate, duplicate])

        incomplete = [
            PlotRecord(
                BenchmarkConfig(
                    num_requests=requests,
                    input_len=input_len,
                    experiment_name=f"run-{index}",
                ),
                make_result(float(index + 1)),
            )
            for index, (requests, input_len) in enumerate(
                [(1, 10), (1, 20), (2, 10)]
            )
        ]
        with self.assertRaisesRegex(ValueError, "complete parameter grid"):
            validate_grid(incomplete)

    def test_loads_reports_and_rejects_incompatible_json(self):
        config = BenchmarkConfig(num_requests=1, experiment_name="saved")
        result = make_result(2.0)
        with TemporaryDirectory() as temp_dir:
            valid = Path(temp_dir) / "valid.json"
            valid.write_text(
                json.dumps(
                    {
                        "experiment_config": asdict(config),
                        "result": asdict(result),
                    }
                ),
                encoding="utf-8",
            )
            records = load_reports([valid])
            self.assertEqual(records, [PlotRecord(config, result)])

            old = Path(temp_dir) / "old.json"
            old_config = asdict(config)
            old_config.pop("engine_component")
            old_config.pop("max_model_len")
            old.write_text(
                json.dumps(
                    {
                        "experiment_config": old_config,
                        "result": asdict(result),
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_reports([old]), [PlotRecord(config, result)])

            missing_phase = Path(temp_dir) / "missing-phase.json"
            incomplete_result = asdict(result)
            incomplete_result.pop("latency_p99")
            missing_phase.write_text(
                json.dumps(
                    {
                        "experiment_config": asdict(config),
                        "result": incomplete_result,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "missing latency/phase metrics.*rerun",
            ):
                load_reports([missing_phase])

            legacy = Path(temp_dir) / "legacy-memory.json"
            legacy_result = asdict(result)
            legacy_result.pop("peak_kvcache_blocks")
            legacy_result.pop("peak_kvcache_utilization")
            legacy_result["peak_memory_allocated_mib"] = 1024.0
            legacy_result["peak_memory_reserved_mib"] = 2048.0
            legacy.write_text(
                json.dumps(
                    {
                        "experiment_config": asdict(config),
                        "result": legacy_result,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "legacy GPU-memory metrics.*rerun",
            ):
                load_reports([legacy])

            invalid = Path(temp_dir) / "invalid.json"
            invalid.write_text('{"result": {}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "could not load"):
                load_reports([invalid])


if __name__ == "__main__":
    unittest.main()
