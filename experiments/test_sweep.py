import unittest

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

    def test_has_no_cli_or_benchmark_execution_responsibilities(self):
        self.assertFalse(hasattr(sweep, "parse_args"))
        self.assertFalse(hasattr(sweep, "main"))
        self.assertFalse(hasattr(sweep, "execute_benchmark"))


if __name__ == "__main__":
    unittest.main()
