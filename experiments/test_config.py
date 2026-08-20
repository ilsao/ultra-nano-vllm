from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmarks.benchmark import BenchmarkConfig
from experiments.config import (
    CONFIG_DIMENSIONS,
    Config,
    infer_config_dimensions,
    resolve_config_groups,
    validate_config_grid,
)
from nanovllm import EngineComponent


class ConfigTests(unittest.TestCase):
    def test_defaults_match_benchmark_defaults_as_singleton_tuples(self):
        self.assertEqual(
            Config(),
            Config(
                model=("~/huggingface/Qwen3-0.6B/",),
                num_requests=(256,),
                input_len=(1024,),
                output_len=(1024,),
                max_model_len=(8192,),
                seed=(0,),
                temperature=(0.6,),
                repeats=(3,),
                enforce_eager=(False,),
                experiment_name=("nano-vllm",),
            ),
        )

    def test_normalizes_scalar_yaml_without_expanding(self):
        config = self._load(
            """
            model: /models/test
            num_requests: 8
            max_model_len: 8192
            temperature: 1
            enforce_eager: true
            experiment_name: scalar-run
            """
        )

        self.assertEqual(config.model, ("/models/test",))
        self.assertEqual(config.num_requests, (8,))
        self.assertEqual(config.temperature, (1.0,))
        self.assertEqual(config.max_model_len, (8192,))
        self.assertEqual(config.enforce_eager, (True,))
        self.assertEqual(config.input_len, (1024,))
        self.assertEqual(config.experiment_name, ("scalar-run",))

    def test_accepts_zero_temperature_for_greedy_decoding(self):
        config = self._load("temperature: 0\n")

        self.assertEqual(config.temperature, (0.0,))

    def test_preserves_list_order_without_computing_product(self):
        config = self._load(
            """
            num_requests: [1, 2]
            input_len: [10, 20]
            experiment_name: [one-ten, one-twenty, two-ten, two-twenty]
            """
        )

        self.assertEqual(config.num_requests, (1, 2))
        self.assertEqual(config.input_len, (10, 20))
        self.assertEqual(
            config.experiment_name,
            ("one-ten", "one-twenty", "two-ten", "two-twenty"),
        )

    def test_all_repository_configs_resolve_to_baseline_experiments(self):
        config_dir = Path(__file__).resolve().parent / "configs"
        paths = sorted(config_dir.glob("*.yaml"))
        groups = resolve_config_groups(paths)
        expected = {
            "baseline-input-sweeps": (
                ("input_len",),
                tuple(
                    f"baseline-input-{value}"
                    for value in (128, 256, 512, 1024, 2048)
                ),
            ),
            "baseline-output-sweeps": (
                ("output_len",),
                tuple(
                    f"baseline-output-{value}"
                    for value in (64, 128, 256, 512, 768)
                ),
            ),
            "baseline-req-sweeps": (
                ("num_requests",),
                tuple(
                    f"baseline-req-{value}"
                    for value in (16, 32, 64, 128, 256)
                ),
            ),
        }

        self.assertEqual(
            {
                group.name: (
                    group.dimensions,
                    tuple(config.experiment_name for config in group.configs),
                )
                for group in groups
            },
            expected,
        )
        by_name = {group.name: group.configs for group in groups}
        self.assertEqual(
            [
                (config.num_requests, config.input_len, config.output_len)
                for config in by_name["baseline-req-sweeps"]
            ],
            [(value, 256, 256) for value in (16, 32, 64, 128, 256)],
        )
        self.assertEqual(
            [
                (config.num_requests, config.input_len, config.output_len)
                for config in by_name["baseline-input-sweeps"]
            ],
            [(64, value, 256) for value in (128, 256, 512, 1024, 2048)],
        )
        self.assertEqual(
            [
                (config.num_requests, config.input_len, config.output_len)
                for config in by_name["baseline-output-sweeps"]
            ],
            [(128, 256, value) for value in (64, 128, 256, 512, 768)],
        )
        configs = [config for group in groups for config in group.configs]
        self.assertEqual(len(configs), 15)
        self.assertTrue(all(config.repeats == 7 for config in configs))
        self.assertTrue(all(not config.enforce_eager for config in configs))
        self.assertTrue(all(config.max_model_len == 8192 for config in configs))
        self.assertTrue(
            all(
                config.input_len + config.output_len <= config.max_model_len
                for config in configs
            )
        )
        self.assertTrue(all(config.temperature == 0.0 for config in configs))
        self.assertTrue(
            all(config.engine_component == EngineComponent() for config in configs)
        )
        names = [config.experiment_name for config in configs]
        self.assertEqual(len(names), len(set(names)))

    def test_rejects_invalid_document_shapes_and_fields(self):
        cases = {
            "": "top-level mapping",
            "- one\n- two\n": "top-level mapping",
            "unknown: value\n": "unknown fields",
            "num_requests: []\n": "must not be empty",
            "num_requests: nope\n": "num_requests must be int",
            "model: [\n": "invalid YAML",
        }
        for document, message in cases.items():
            with self.subTest(document=document), self.assertRaisesRegex(
                (TypeError, ValueError), message
            ):
                self._load(document)

    def test_rejects_invalid_normalized_values(self):
        cases = [
            ({"num_requests": (0,)}, "num_requests"),
            ({"input_len": (0,)}, "input_len"),
            ({"output_len": (-1,)}, "output_len"),
            ({"max_model_len": (0,)}, "max_model_len"),
            ({"temperature": (-1.0,)}, "temperature"),
            ({"temperature": (1e-12,)}, "temperature"),
            ({"repeats": (0,)}, "repeats"),
            ({"num_requests": (True,)}, "num_requests must be int"),
            ({"experiment_name": ("../unsafe",)}, "experiment name"),
            ({"seed": ()}, "seed must not be empty"),
            ({"model": ["/model"]}, "model must be a tuple"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                (TypeError, ValueError), message
            ):
                Config(**kwargs)

    def _load(self, document: str) -> Config:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text(document, encoding="utf-8")
            return Config.from_yaml(path)


class ConfigGroupTests(unittest.TestCase):
    def test_resolves_independent_groups_and_applies_overrides(self):
        with TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.yaml"
            second = Path(temp_dir) / "second.yaml"
            first.write_text(
                "num_requests: [1, 2]\n"
                "experiment_name: [first-one, first-two]\n",
                encoding="utf-8",
            )
            second.write_text("experiment_name: second\n", encoding="utf-8")

            groups = resolve_config_groups(
                [first, second],
                {"enforce_eager": True},
            )

        self.assertEqual([group.name for group in groups], ["first", "second"])
        self.assertEqual(groups[0].dimensions, ("num_requests",))
        self.assertEqual(groups[1].dimensions, ())
        self.assertEqual(
            [config.experiment_name for group in groups for config in group.configs],
            ["first-one", "first-two", "second"],
        )
        self.assertTrue(
            all(
                config.enforce_eager
                for group in groups
                for config in group.configs
            )
        )

    def test_disambiguates_groups_with_the_same_file_stem(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "one" / "same.yaml"
            second = root / "two" / "same.yaml"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("experiment_name: first\n", encoding="utf-8")
            second.write_text("experiment_name: second\n", encoding="utf-8")

            groups = resolve_config_groups([first, second])

        self.assertEqual([group.name for group in groups], ["same-1", "same-2"])

    def test_resolves_component_sweep_as_categorical_dimension(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "components.yaml"
            path.write_text(
                "scheduler: [scheduler, optimized-scheduler-v1]\n"
                "experiment_name: [baseline, optimized]\n",
                encoding="utf-8",
            )
            with patch("experiments.config.validate_engine_component_selector"):
                groups = resolve_config_groups([path])

        self.assertEqual(groups[0].dimensions, ("scheduler",))
        self.assertEqual(
            [config.engine_component.scheduler for config in groups[0].configs],
            ["scheduler", "optimized-scheduler-v1"],
        )

    def test_rejects_invalid_group_names_overrides_and_dimensions(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            duplicate = root / "duplicate.yaml"
            duplicate.write_text(
                "num_requests: [1, 2]\nexperiment_name: [same, same]\n",
                encoding="utf-8",
            )
            unique = root / "unique.yaml"
            unique.write_text(
                "num_requests: [1, 2]\nexperiment_name: [one, two]\n",
                encoding="utf-8",
            )
            high = root / "high.yaml"
            high.write_text(
                "num_requests: [1, 2]\n"
                "input_len: [10, 20]\n"
                "seed: [0, 1]\n"
                "experiment_name: [a, b, c, d, e, f, g, h]\n",
                encoding="utf-8",
            )

            cases = [
                ([duplicate], None, "unique"),
                ([unique], {"experiment_name": "override"}, "multiple runs"),
                ([high], None, "at most two"),
            ]
            for paths, overrides, message in cases:
                with self.subTest(paths=paths), self.assertRaisesRegex(
                    ValueError, message
                ):
                    resolve_config_groups(paths, overrides)

    def test_rejects_workload_longer_than_max_model_len_before_execution(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "too-long.yaml"
            path.write_text(
                "input_len: 4096\n"
                "output_len: 1024\n"
                "max_model_len: 4096\n"
                "experiment_name: too-long\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"experiment 'too-long'.*4096 \+ 1024 = 5120.*increase",
            ):
                resolve_config_groups([path])

            groups = resolve_config_groups([path], {"max_model_len": 8192})

        self.assertEqual(groups[0].configs[0].max_model_len, 8192)

    def test_infers_dimensions_in_domain_order_and_validates_grid(self):
        configs = [
            BenchmarkConfig(
                num_requests=requests,
                enforce_eager=eager,
                experiment_name=f"run-{index}",
            )
            for index, (requests, eager) in enumerate(
                [(1, False), (1, True), (2, False), (2, True)]
            )
        ]

        self.assertEqual(
            CONFIG_DIMENSIONS,
            (
                "model",
                "num_requests",
                "input_len",
                "output_len",
                "max_model_len",
                "seed",
                "temperature",
                "repeats",
                "enforce_eager",
                "scheduler",
                "block_manager",
                "attention",
                "sampler",
                "store_kvcache",
            ),
        )
        self.assertEqual(
            infer_config_dimensions(configs),
            ("num_requests", "enforce_eager"),
        )
        self.assertEqual(
            validate_config_grid(configs),
            ("num_requests", "enforce_eager"),
        )
        context_configs = [
            BenchmarkConfig(
                max_model_len=value,
                experiment_name=f"context-{value}",
            )
            for value in (4096, 8192)
        ]
        self.assertEqual(
            infer_config_dimensions(context_configs),
            ("max_model_len",),
        )

    def test_rejects_empty_duplicate_and_incomplete_grids(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_config_grid([])

        duplicate = [
            BenchmarkConfig(experiment_name="first"),
            BenchmarkConfig(experiment_name="second"),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_config_grid(duplicate)

        incomplete = [
            BenchmarkConfig(
                num_requests=requests,
                input_len=input_len,
                experiment_name=f"run-{index}",
            )
            for index, (requests, input_len) in enumerate(
                [(1, 10), (1, 20), (2, 10)]
            )
        ]
        with self.assertRaisesRegex(ValueError, "complete parameter grid"):
            validate_config_grid(incomplete)


if __name__ == "__main__":
    unittest.main()
