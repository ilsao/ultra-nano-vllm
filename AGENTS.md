# Repository Guidelines

## Project Overview

Ultra-Nano-vLLM is a performance research project built around nano-vLLM. Its
primary purpose is to benchmark nano-vLLM, run controlled experiments, profile
confirmed bottlenecks, and use the resulting evidence to optimize the inference
system. Nano-vLLM remains a lightweight offline LLM implementation built with
Python 3.10-3.12, PyTorch, Triton, and CUDA, but extending its implementation is
not the goal by itself.

The main areas are:

- `benchmarks/`: scalar benchmark configuration, workloads, metrics, and runner.
- `experiments/`: YAML sweeps, isolated execution, reports, and comparison plots.
- `nanovllm/`: the inference implementation being measured and optimized.
- `utils/`: shared reporting utilities.

Use the installation instructions in `README.md` as the source of truth for
CUDA and dependency setup.

## Research Workflow

Use benchmark and profiling evidence to guide optimization work:

1. Establish a reproducible benchmark measurement for the behavior of interest.
2. Use controlled experiments to isolate parameter effects and compare results.
3. Profile the confirmed workload to locate the actual bottleneck before
   changing inference code.
4. Make a focused optimization that preserves correctness and public behavior.
5. Re-run comparable benchmarks and experiments to check the improvement and
   guard against regressions.

Treat this as guidance rather than a requirement to produce a formal artifact
for every change. The repository does not yet prescribe a profiling framework,
command, or output directory; follow the tooling introduced by the task or
already present in the repository instead of inventing one implicitly.

## Development Workflow

Prefer the repository virtual environment when it exists:

```bash
.venv/bin/python -m unittest discover
.venv/bin/python -m compileall -q benchmarks experiments nanovllm utils
git diff --check
```

If `.venv/bin/python` is unavailable, use `python3` with the same module
commands. Add focused `unittest` coverage beside the subsystem being changed,
then run the complete suite when practical.

Do not run a real benchmark or experiment unless an NVIDIA GPU, a compatible
CUDA environment, and the configured local model are available. Unit tests
should mock model loading and expensive GPU work.

## Implementation Conventions

- Follow the existing dataclass, type-hint, and standard-library `unittest`
  patterns.
- Keep optimization changes focused and readable. Avoid new layers of
  abstraction unless they improve measured behavior, remove concrete
  duplication, or enforce an important boundary.
- Do not add dependencies, formatters, or linters unless the requested change
  requires them.
- Preserve unrelated changes in a dirty worktree. Do not overwrite or clean up
  files outside the requested scope.
- Reports under `benchmarks/report/` and `experiments/report/` are generated
  output and must remain untracked.

## Experiment Architecture

Keep experiment responsibilities separated:

- `experiments/sweep.py` only expands a normalized `Config` into deterministic
  scalar `BenchmarkConfig` values.
- `experiments/config.py` owns YAML parsing, normalization, overrides,
  dimension inference, and parameter-grid validation.
- `experiments/experiment.py` is the CLI and orchestration layer. It runs
  resolved groups, delegates reporting, and dispatches dimension-specific
  plotting.
- `experiments/plot.py` owns report loading, plot validation, and the public 1D
  and 2D plotting APIs.
- `experiments/runner.py` owns isolated benchmark process execution.

Engine implementations are selected by file name through
`nanovllm.engine.component.EngineComponent`. Keep the entrypoints engine
agnostic: benchmark and experiment code may transport this value but must not
branch on component roles or import concrete implementations.

The replaceable roles and their implementation directories are Scheduler,
BlockManager, Attention, Sampler, and the store-KV-cache kernel. Component files
must export `create_component(...)`; kernel files must export `create_kernel()`.
Return values must satisfy the corresponding Protocol. Add new implementations
inside the existing role directory and preserve the baseline package re-exports.
Selectors are trusted local file stems and may contain hyphens; never broaden
the loader to arbitrary paths or automatic repository-wide discovery.

Multiprocessing uses the `spawn` context. Any process target must remain an
importable module-level function; do not replace it with a nested function,
lambda, or non-picklable callable.

Sequential experiment configurations must run in fresh spawned processes so
CUDA graph pools, cached allocations, and NCCL state cannot leak between runs.
Experiment workers must call `execute_benchmark(..., close_runner=True)`, and
engine shutdown must remain idempotent because both explicit cleanup and
`atexit` may invoke it.

## Compatibility Requirements

Unless a task explicitly changes an interface, preserve:

- The standalone benchmark and experiment CLI syntax.
- The experiment YAML schema and deterministic Cartesian-product ordering.
- Default baseline component selections and selector ordering.
- The JSON report structure, including typed benchmark configuration, result,
  and `experiment_config` data required by plot-only mode.
- Independent plotting groups for each YAML file and support for zero, one, or
  two varying parameters.
- Six comparison plots per plotted group: elapsed time, request throughput,
  output throughput, total throughput, peak allocated memory, and peak reserved
  memory.

Validate all experiment groups before starting GPU work. Reject duplicate or
incomplete parameter grids and configurations with more than two varying
dimensions rather than producing ambiguous plots.
