import unittest
from unittest.mock import patch

from experiments.config import Config
from experiments import sweep


class SweepTests(unittest.TestCase):
    def test_zips_workloads_then_crosses_component_columns(self):
        with patch("experiments.config.validate_engine_component_selector"):
            config = Config(
                workload_name=("small", "large"),
                num_requests=(1, 2),
                input_len=(10, 20),
                component_name=("baseline", "optimized"),
                scheduler=("scheduler", "optimized-scheduler-v1"),
                block_manager=("block_manager", "optimized-block-manager-v1"),
            )

            expanded = sweep.expand_config(config)

        self.assertEqual(
            [
                (
                    run.config.num_requests,
                    run.config.input_len,
                    run.config.engine_component.scheduler,
                    run.config.engine_component.block_manager,
                    run.config.experiment_name,
                )
                for run in expanded
            ],
            [
                (1, 10, "scheduler", "block_manager", "baseline-small"),
                (
                    1,
                    10,
                    "optimized-scheduler-v1",
                    "optimized-block-manager-v1",
                    "optimized-small",
                ),
                (2, 20, "scheduler", "block_manager", "baseline-large"),
                (
                    2,
                    20,
                    "optimized-scheduler-v1",
                    "optimized-block-manager-v1",
                    "optimized-large",
                ),
            ],
        )

    def test_broadcasts_single_values_within_each_column_group(self):
        config = Config(
            workload_name=("one", "two"),
            num_requests=(1, 2),
            input_len=(10,),
        )

        expanded = sweep.expand_config(config)

        self.assertEqual(
            [
                (run.config.num_requests, run.config.input_len, run.config.experiment_name)
                for run in expanded
            ],
            [
                (1, 10, "baseline-one"),
                (2, 10, "baseline-two"),
            ],
        )

    def test_rejects_incorrect_column_and_label_counts(self):
        with self.assertRaisesRegex(ValueError, "invalid fields: input_len"):
            sweep.expand_config(
                Config(
                    workload_name=("one", "two", "three"),
                    num_requests=(1, 2, 3),
                    input_len=(10, 20),
                )
            )

        with self.assertRaisesRegex(ValueError, "workload_name.*expected 2, got 1"):
            sweep.expand_config(Config(num_requests=(1, 2)))

        with self.assertRaisesRegex(ValueError, "invalid fields: block_manager"):
            sweep.expand_config(
                Config(
                    component_name=("one", "two", "three"),
                    scheduler=("scheduler", "scheduler", "scheduler"),
                    block_manager=("block_manager", "block_manager"),
                )
            )

    def test_rejects_duplicate_labels_and_resolved_columns(self):
        with self.assertRaisesRegex(ValueError, "workload_name values must be unique"):
            sweep.expand_config(
                Config(
                    workload_name=("same", "same"),
                    num_requests=(1, 2),
                )
            )

        with self.assertRaisesRegex(ValueError, "duplicate configurations"):
            sweep.expand_config(
                Config(
                    workload_name=("one", "two"),
                    num_requests=(1, 1),
                )
            )

    def test_expands_max_model_len_in_scalar_field_order(self):
        config = Config(
            workload_name=("context-20", "context-40"),
            input_len=(10,),
            output_len=(10,),
            max_model_len=(20, 40),
        )

        expanded = sweep.expand_config(config)

        self.assertEqual(
            [
                (run.config.max_model_len, run.config.experiment_name)
                for run in expanded
            ],
            [(20, "baseline-context-20"), (40, "baseline-context-40")],
        )

    def test_has_no_cli_or_benchmark_execution_responsibilities(self):
        self.assertFalse(hasattr(sweep, "parse_args"))
        self.assertFalse(hasattr(sweep, "main"))
        self.assertFalse(hasattr(sweep, "execute_benchmark"))


if __name__ == "__main__":
    unittest.main()
