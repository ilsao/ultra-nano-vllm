from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

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
    summary = MetricSummary(value, value - 0.25, value + 0.5)
    return BenchmarkResult(
        repeats=3,
        num_requests=1,
        input_tokens=2,
        output_tokens=summary,
        total_tokens=summary,
        elapsed_time=summary,
        request_throughput=summary,
        output_throughput=summary,
        total_throughput=summary,
        peak_memory_allocated_mib=value + 10,
        peak_memory_reserved_mib=value + 20,
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
        with patch("nanovllm.engine.component._load_factory"):
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

        self.assertEqual(infer_dimensions(records), ("scheduler",))
        self.assertEqual(validate_grid(records), ("scheduler",))

    def test_plot_1d_saves_six_metric_files(self):
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
            self.assertEqual(len(paths), 6)
            self.assertTrue(all(path.exists() for path in paths))
            self.assertTrue(all(path.suffix == ".png" for path in paths))
            self.assertTrue(all("1d-num_requests" in path.name for path in paths))

    def test_plot_2d_supports_categorical_axes_and_saves_six_files(self):
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
            self.assertEqual(len(paths), 6)
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

            invalid = Path(temp_dir) / "invalid.json"
            invalid.write_text('{"result": {}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "could not load"):
                load_reports([invalid])


if __name__ == "__main__":
    unittest.main()
