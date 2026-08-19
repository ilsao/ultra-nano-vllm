"""Run one experiment benchmark in an isolated child process."""

from __future__ import annotations

import multiprocessing as mp
import traceback

from benchmarks.benchmark import BenchmarkConfig, execute_benchmark
from benchmarks.metrics import BenchmarkResult
from benchmarks.runner import BenchmarkConfiguration
from utils.reporter import Reporter


def run_benchmark_worker(config: BenchmarkConfig, connection) -> None:
    """Execute one benchmark and send a result envelope to the parent."""
    try:
        result, configuration = execute_benchmark(
            config,
            Reporter(),
            close_runner=True,
        )
        connection.send(("success", result, configuration))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def execute_benchmark_isolated(
    config: BenchmarkConfig,
) -> tuple[BenchmarkResult, BenchmarkConfiguration]:
    """
    Run one scalar benchmark in a fresh spawned process.
    
    Args:
        config: The benchmark configuration to run.
    
    Returns:
        A tuple containing the benchmark result and the configuration used.
    """
    context = mp.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(
        target=run_benchmark_worker,
        args=(config, sending),
        name=f"experiment-{config.experiment_name}",
        daemon=False,
    )

    try:
        process.start()
    except BaseException as exc:
        receiving.close()
        sending.close()
        raise RuntimeError(
            f"could not start benchmark worker for {config.experiment_name}: {exc}"
        ) from exc

    sending.close()
    message = None
    try:
        try:
            message = receiving.recv()
        except EOFError:
            pass
        process.join()
    except BaseException:
        if process.is_alive():
            process.terminate()
        process.join()
        raise
    finally:
        receiving.close()

    if message is None:
        raise RuntimeError(
            f"benchmark worker for {config.experiment_name} exited without "
            f"a result (exit code {process.exitcode})"
        )
    if not isinstance(message, tuple) or not message:
        raise RuntimeError(
            f"benchmark worker for {config.experiment_name} returned an "
            "invalid response"
        )
    if process.exitcode != 0:
        raise RuntimeError(
            f"benchmark worker for {config.experiment_name} exited with "
            f"code {process.exitcode}"
        )

    status, *payload = message
    if status == "error" and len(payload) == 1:
        raise RuntimeError(
            f"benchmark worker for {config.experiment_name} failed:\n{payload[0]}"
        )
    if status != "success" or len(payload) != 2:
        raise RuntimeError(
            f"benchmark worker for {config.experiment_name} returned an "
            "invalid response"
        )
    result, configuration = payload
    if not isinstance(result, BenchmarkResult) or not isinstance(
        configuration, BenchmarkConfiguration
    ):
        raise RuntimeError(
            f"benchmark worker for {config.experiment_name} returned invalid "
            "result types"
        )
    return result, configuration
