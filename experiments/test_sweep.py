import unittest
from unittest.mock import patch

from experiments.config import Config
from experiments import sweep


class SweepTests(unittest.TestCase):
    def test_expands_cartesian_product_and_pairs_names(self):
        config = Config(
            num_requests=(1, 2),
            input_len=(10, 20),
            experiment_name=("one-ten", "one-twenty", "two-ten", "two-twenty"),
        )

        expanded = sweep.expand_config(config)

        self.assertEqual(
            [
                (config.num_requests, config.input_len, config.experiment_name)
                for config in expanded
            ],
            [
                (1, 10, "one-ten"),
                (1, 20, "one-twenty"),
                (2, 10, "two-ten"),
                (2, 20, "two-twenty"),
            ],
        )

    def test_rejects_incorrect_name_count(self):
        with self.assertRaisesRegex(ValueError, "expected 2, got 1"):
            sweep.expand_config(
                Config(num_requests=(1, 2), experiment_name=("only-one",))
            )

    def test_expands_component_selectors_after_scalar_fields(self):
        with patch("experiments.config.validate_engine_component_selector"):
            config = Config(
                num_requests=(1, 2),
                scheduler=("scheduler", "optimized-scheduler-v1"),
                experiment_name=("a", "b", "c", "d"),
            )

            expanded = sweep.expand_config(config)

        self.assertEqual(
            [
                (run.num_requests, run.engine_component.scheduler)
                for run in expanded
            ],
            [
                (1, "scheduler"),
                (1, "optimized-scheduler-v1"),
                (2, "scheduler"),
                (2, "optimized-scheduler-v1"),
            ],
        )

    def test_expands_max_model_len_in_scalar_field_order(self):
        config = Config(
            input_len=(10,),
            output_len=(10,),
            max_model_len=(20, 40),
            experiment_name=("context-20", "context-40"),
        )

        expanded = sweep.expand_config(config)

        self.assertEqual(
            [
                (run.max_model_len, run.experiment_name)
                for run in expanded
            ],
            [(20, "context-20"), (40, "context-40")],
        )

    def test_has_no_cli_or_benchmark_execution_responsibilities(self):
        self.assertFalse(hasattr(sweep, "parse_args"))
        self.assertFalse(hasattr(sweep, "main"))
        self.assertFalse(hasattr(sweep, "execute_benchmark"))


if __name__ == "__main__":
    unittest.main()
