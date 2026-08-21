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
class ConfigTests(unittest.TestCase):
    def test_defaults_match_benchmark_defaults_as_singleton_tuples(self):
        self.assertEqual(
            Config(),
            Config(
                model=("~/huggingface/Qwen3-0.6B/",),
                num_requests=(256,),
                input_len=(1024,),
                alternate_input_len=(None,),
                output_len=(1024,),
                max_model_len=(8192,),
                seed=(0,),
                temperature=(0.6,),
                repeats=(3,),
                enforce_eager=(False,),
                workload_name=("workload",),
                component_name=("baseline",),
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
            workload_name: scalar-run
            """
        )

        self.assertEqual(config.model, ("/models/test",))
        self.assertEqual(config.num_requests, (8,))
        self.assertEqual(config.temperature, (1.0,))
        self.assertEqual(config.max_model_len, (8192,))
        self.assertEqual(config.enforce_eager, (True,))
        self.assertEqual(config.input_len, (1024,))
        self.assertEqual(config.alternate_input_len, (None,))
        self.assertEqual(config.workload_name, ("scalar-run",))
        self.assertEqual(config.component_name, ("baseline",))

    def test_accepts_zero_temperature_for_greedy_decoding(self):
        config = self._load("temperature: 0\n")

        self.assertEqual(config.temperature, (0.0,))

    def test_preserves_list_order_without_computing_product(self):
        config = self._load(
            """
            num_requests: [1, 2]
            input_len: [10, 20]
            workload_name: [one-ten, two-twenty]
            """
        )

        self.assertEqual(config.num_requests, (1, 2))
        self.assertEqual(config.input_len, (10, 20))
        self.assertEqual(
            config.workload_name,
            ("one-ten", "two-twenty"),
        )

    def test_rejects_invalid_document_shapes_and_fields(self):
        cases = {
            "": "top-level mapping",
            "- one\n- two\n": "top-level mapping",
            "unknown: value\n": "unknown fields",
            "num_requests: []\n": "must not be empty",
            "num_requests: nope\n": "num_requests must be int",
            "experiment_name: old\n": "removed experiment_name",
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
            ({"alternate_input_len": (0,)}, "alternate_input_len"),
            ({"output_len": (-1,)}, "output_len"),
            ({"max_model_len": (0,)}, "max_model_len"),
            ({"temperature": (-1.0,)}, "temperature"),
            ({"temperature": (1e-12,)}, "temperature"),
            ({"repeats": (0,)}, "repeats"),
            ({"num_requests": (True,)}, "num_requests must be int"),
            ({"workload_name": ("../unsafe",)}, "experiment name"),
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
    def test_scheduler_optimized_configs_form_expected_safe_grids(self):
        config_dir = Path(__file__).resolve().parent / "configs"
        groups = resolve_config_groups(
            [
                config_dir / "scheduler-optimized-req-sweeps.yaml",
                config_dir / "scheduler-optimized-input-sweeps.yaml",
                config_dir / "scheduler-optimized-output-sweeps.yaml",
            ]
        )

        self.assertEqual(
            [(len(group.configs), group.dimensions) for group in groups],
            [
                (20, ("workload", "component")),
                (8, ("workload", "component")),
                (14, ("workload", "component")),
            ],
        )
        self.assertEqual(
            [
                max(
                    max(
                        config.input_len,
                        config.alternate_input_len or config.input_len,
                    )
                    + config.output_len
                    for config in group.configs
                )
                for group in groups
            ],
            [512, 8192, 3072],
        )

    def test_resolves_independent_groups_and_applies_overrides(self):
        with TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.yaml"
            second = Path(temp_dir) / "second.yaml"
            first.write_text(
                "num_requests: [1, 2]\n"
                "workload_name: [first-one, first-two]\n",
                encoding="utf-8",
            )
            second.write_text("workload_name: second\n", encoding="utf-8")

            groups = resolve_config_groups(
                [first, second],
                {"enforce_eager": True},
            )

        self.assertEqual([group.name for group in groups], ["first", "second"])
        self.assertEqual(groups[0].dimensions, ("workload",))
        self.assertEqual(groups[1].dimensions, ())
        self.assertEqual(
            [config.experiment_name for group in groups for config in group.configs],
            ["baseline-first-one", "baseline-first-two", "baseline-second"],
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
            first.write_text("workload_name: first\n", encoding="utf-8")
            second.write_text("workload_name: second\n", encoding="utf-8")

            groups = resolve_config_groups([first, second])

        self.assertEqual([group.name for group in groups], ["same-1", "same-2"])

    def test_resolves_component_sweep_as_categorical_dimension(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "components.yaml"
            path.write_text(
                "scheduler: [scheduler, optimized-scheduler-v1]\n"
                "component_name: [baseline, optimized]\n",
                encoding="utf-8",
            )
            with patch("experiments.config.validate_engine_component_selector"):
                groups = resolve_config_groups([path])

        self.assertEqual(groups[0].dimensions, ("component",))
        self.assertEqual(
            [config.engine_component.scheduler for config in groups[0].configs],
            ["scheduler", "optimized-scheduler-v1"],
        )

    def test_zips_multiple_workload_fields_into_one_conceptual_dimension(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "paired.yaml"
            path.write_text(
                "workload_name: [small, large]\n"
                "num_requests: [1, 2]\n"
                "input_len: [10, 20]\n"
                "seed: [3, 4]\n",
                encoding="utf-8",
            )

            group = resolve_config_groups([path])[0]

        self.assertEqual(group.dimensions, ("workload",))
        self.assertEqual(
            [
                (config.num_requests, config.input_len, config.seed)
                for config in group.configs
            ],
            [(1, 10, 3), (2, 20, 4)],
        )

    def test_rejects_cli_override_that_collapses_workload_columns(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "requests.yaml"
            path.write_text(
                "workload_name: [one, two]\nnum_requests: [1, 2]\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate configurations"):
                resolve_config_groups([path], {"num_requests": 1})

    def test_rejects_invalid_group_names_and_name_override(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            duplicate = root / "duplicate.yaml"
            duplicate.write_text(
                "num_requests: [1, 2]\nworkload_name: [same, same]\n",
                encoding="utf-8",
            )
            unique = root / "unique.yaml"
            unique.write_text(
                "num_requests: [1, 2]\nworkload_name: [one, two]\n",
                encoding="utf-8",
            )

            cases = [
                ([duplicate], None, "unique"),
                ([unique], {"experiment_name": "override"}, "multiple runs"),
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
                "workload_name: too-long\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"experiment 'baseline-too-long'.*4096 \+ 1024 = 5120.*increase",
            ):
                resolve_config_groups([path])

            groups = resolve_config_groups([path], {"max_model_len": 8192})

        self.assertEqual(groups[0].configs[0].max_model_len, 8192)

    def test_rejects_alternate_workload_longer_than_max_model_len(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "alternate-too-long.yaml"
            path.write_text(
                "input_len: 2\n"
                "alternate_input_len: 8\n"
                "output_len: 1\n"
                "max_model_len: 8\n"
                "workload_name: alternate-too-long\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"alternate-too-long.*8 \+ 1 = 9",
            ):
                resolve_config_groups([path])

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
                "alternate_input_len",
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
