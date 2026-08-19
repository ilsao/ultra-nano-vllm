import unittest
from unittest.mock import Mock, patch

from benchmarks.benchmark import BenchmarkConfig
from experiments import runner
from experiments.test_experiment import make_configuration
from experiments.test_plot import make_result


class IsolatedExecutionTests(unittest.TestCase):
    def test_worker_sends_successful_result_and_requests_cleanup(self):
        config = BenchmarkConfig(experiment_name="worker")
        result = make_result(1.0)
        configuration = make_configuration()
        connection = Mock()

        with (
            patch("experiments.runner.Reporter", return_value="reporter"),
            patch(
                "experiments.runner.execute_benchmark",
                return_value=(result, configuration),
            ) as execute,
        ):
            runner.run_benchmark_worker(config, connection)

        execute.assert_called_once_with(config, "reporter", close_runner=True)
        connection.send.assert_called_once_with(
            ("success", result, configuration)
        )
        connection.close.assert_called_once_with()

    def test_worker_serializes_failure_and_closes_connection(self):
        config = BenchmarkConfig(experiment_name="worker")
        connection = Mock()

        with patch(
            "experiments.runner.execute_benchmark",
            side_effect=RuntimeError("worker failed"),
        ):
            runner.run_benchmark_worker(config, connection)

        message = connection.send.call_args.args[0]
        self.assertEqual(message[0], "error")
        self.assertIn("RuntimeError: worker failed", message[1])
        connection.close.assert_called_once_with()

    def _process_mocks(self, message):
        receiving = Mock()
        receiving.recv.return_value = message
        sending = Mock()
        process = Mock()
        process.exitcode = 0
        process.is_alive.return_value = False
        context = Mock()
        context.Pipe.return_value = (receiving, sending)
        context.Process.return_value = process
        return context, receiving, sending, process

    def test_returns_successful_worker_payload_and_joins_process(self):
        config = BenchmarkConfig(experiment_name="isolated")
        result = make_result(1.0)
        configuration = make_configuration()
        context, receiving, sending, process = self._process_mocks(
            ("success", result, configuration)
        )

        with patch("experiments.runner.mp.get_context", return_value=context):
            returned = runner.execute_benchmark_isolated(config)

        self.assertEqual(returned, (result, configuration))
        context.Pipe.assert_called_once_with(duplex=False)
        self.assertIs(
            context.Process.call_args.kwargs["target"],
            runner.run_benchmark_worker,
        )
        self.assertEqual(context.Process.call_args.kwargs["args"], (config, sending))
        self.assertFalse(context.Process.call_args.kwargs["daemon"])
        process.start.assert_called_once_with()
        sending.close.assert_called_once_with()
        receiving.close.assert_called_once_with()
        process.join.assert_called_once_with()

    def test_surfaces_remote_traceback(self):
        config = BenchmarkConfig(experiment_name="failed")
        context, _, _, _ = self._process_mocks(
            ("error", "Traceback: worker exploded")
        )

        with (
            patch("experiments.runner.mp.get_context", return_value=context),
            self.assertRaisesRegex(RuntimeError, "worker exploded"),
        ):
            runner.execute_benchmark_isolated(config)

    def test_reports_worker_exit_without_payload(self):
        config = BenchmarkConfig(experiment_name="abrupt")
        context, receiving, _, process = self._process_mocks(None)
        receiving.recv.side_effect = EOFError
        process.exitcode = 9

        with (
            patch("experiments.runner.mp.get_context", return_value=context),
            self.assertRaisesRegex(RuntimeError, "exit code 9"),
        ):
            runner.execute_benchmark_isolated(config)

    def test_parent_interrupt_terminates_and_joins_live_worker(self):
        config = BenchmarkConfig(experiment_name="interrupted")
        context, receiving, _, process = self._process_mocks(None)
        receiving.recv.side_effect = KeyboardInterrupt
        process.is_alive.return_value = True

        with (
            patch("experiments.runner.mp.get_context", return_value=context),
            self.assertRaises(KeyboardInterrupt),
        ):
            runner.execute_benchmark_isolated(config)

        process.terminate.assert_called_once_with()
        process.join.assert_called_once_with()

    def test_process_start_failure_closes_both_pipe_ends(self):
        config = BenchmarkConfig(experiment_name="not-started")
        context, receiving, sending, process = self._process_mocks(None)
        process.start.side_effect = OSError("cannot spawn")

        with (
            patch("experiments.runner.mp.get_context", return_value=context),
            self.assertRaisesRegex(RuntimeError, "cannot spawn"),
        ):
            runner.execute_benchmark_isolated(config)

        receiving.close.assert_called_once_with()
        sending.close.assert_called_once_with()
        process.join.assert_not_called()


if __name__ == "__main__":
    unittest.main()
