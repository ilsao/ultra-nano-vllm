<div align="center">

<h1>Ultra-Nano-vLLM</h1>

<p>Benchmark and compare nano-vLLM inference components with reproducible workloads.</p>

<p>
  <a href="pyproject.toml"><img alt="Python 3.10–3.12" src="https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white"></a>
  <a href="#installation"><img alt="NVIDIA CUDA required" src="https://img.shields.io/badge/NVIDIA_CUDA-required-76B900?logo=nvidia&logoColor=white"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

<p>
  <a href="#installation">Installation</a> ·
  <a href="#benchmark">Benchmark</a> ·
  <a href="#experiments">Experiments</a> ·
  <a href="#profiling">Profiling</a> ·
  <a href="#pluggable-design">Pluggable Design</a> ·
  <a href="#relationship">nano-vLLM</a>
</p>

</div>

Ultra-Nano-vLLM extends
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) with structured
benchmarks, YAML experiment grids, isolated runs, comparison plots, scheduler
telemetry, and replaceable inference components.

## Project layout

| Area | Responsibility |
| --- | --- |
| `benchmarks/` | Scalar workloads, measurement, metrics, and JSON reports. |
| `experiments/` | YAML resolution, isolated execution, plots, and scheduler analysis. |
| `nanovllm/` | The inference engine and replaceable components being evaluated. |
| `utils/` | Shared reporting and optional profiling helpers. |

Experiment configurations are validated before GPU work. Each resolved run is
executed sequentially in a fresh spawned process, and generated reports remain
separate for every YAML file.

<a id="relationship"></a>

## Relationship to nano-vLLM

[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) provides the compact
offline inference engine, model execution path, and public `LLM` API used as
this repository's baseline. Ultra-Nano-vLLM keeps that implementation readable
while adding benchmark, experiment, reporting, plotting, process-isolation,
telemetry, and component-selection infrastructure for controlled comparisons.
Alternative schedulers and other optimized components are measured against the
retained baseline rather than changing benchmark behavior for one specific
implementation.

<a id="installation"></a>

## Installation

Ultra-Nano-vLLM supports Python 3.10–3.12 and requires an NVIDIA GPU with a
compatible CUDA environment. These commands target CUDA 12.8; select the
matching PyTorch wheel for a different CUDA version.

```bash
python3 -m venv .venv
source ./.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install ninja packaging
MAX_JOBS=2 python -m pip install -e . --no-build-isolation
```

The separate PyTorch step makes CUDA PyTorch available while FlashAttention is
built. `MAX_JOBS=2` limits build memory usage and may be adjusted for the host.

The included configurations expect Qwen3-0.6B at
`~/huggingface/Qwen3-0.6B/`:

```bash
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/
```

<a id="benchmark"></a>

## Run a benchmark

Run one scalar baseline workload with:

```bash
python -m benchmarks.benchmark \
  --model ~/huggingface/Qwen3-0.6B/ \
  --num-requests 33 \
  --input-len 256 \
  --alternate-input-len 4096 \
  --input-order alternate-first \
  --output-len 1 \
  --max-model-len 8192 \
  --temperature 0 \
  --repeats 10 \
  --experiment-name hol-prefill-baseline
```

`alternate-first` places the 4096-token cohort before the 256-token cohort.
The other supported orders are `alternating` and `primary-first`. Omit
`--alternate-input-len` for a homogeneous workload, and add `--enforce-eager`
to disable CUDA Graph execution.

The command prints a result table and writes JSON under `benchmarks/report/`.
Reports include latency p50/p90/p99, elapsed and phase time, request/token/phase
throughput, KV-cache occupancy, request-aligned measurements, and scheduler
telemetry.

<a id="experiments"></a>

## Run experiments

An experiment YAML zips workload fields into named workload columns, zips the
five component selectors into named component columns, then evaluates their
complete cross-product. Scalars are broadcast across their respective axis.

```yaml
model: ~/huggingface/Qwen3-0.6B/

workload_name: [short-only, long-first-2048, long-first-4096]
num_requests: 33
input_len: 256
alternate_input_len: [null, 2048, 4096]
input_order: [alternating, alternate-first, alternate-first]
output_len: 1
max_model_len: 8192
seed: 0
temperature: 0.0
repeats: 10
enforce_eager: false

component_name: [baseline, scheduler-v1, scheduler-v2]
scheduler: [scheduler, scheduler-v1, scheduler-v2]
block_manager: block_manager
attention: attention
sampler: sampler
store_kvcache: store_kvcache
```

This example resolves to nine runs in workload-first order. Workload and
component labels must be unique, and every generated report name follows
`<component_name>-<workload_name>`.

Run one or both included experiment groups with:

```bash
python -m experiments.experiment \
  --config experiments/configs/scheduler-comparison-minimal.yaml \
  --config experiments/configs/scheduler-comparison-hol-kv.yaml
```

Each `--config` remains an independent validation and plotting group. Scalar
CLI options such as `--repeats`, `--temperature`, and `--model` may override
the YAML values for the invocation.

### Included YAML configurations

| Config | Workloads | Purpose |
| --- | --- | --- |
| `scheduler-comparison-minimal.yaml` | Nine 33-request, one-token-output workloads | Compare short-only and mixed 256/1024–6144-token prompts across long-first, alternating, and short-first orders. |
| `scheduler-comparison-hol-kv.yaml` | Seven long-first 256/4096-token workloads with 45–57 requests | Cross decode-growth and initial-allocation KV-cache boundaries with 128-token outputs. |

Both configurations compare `scheduler`, `scheduler-v1`, and `scheduler-v2`
while holding the other component selectors constant. The baseline scheduler
uses FCFS waiting order, v1 performs a single deferred backfill scan, and v2
stably orders waiting requests by remaining prefill tokens before that scan.

Reports and thirteen comparison plots are written under `experiments/report/`.
The plots cover elapsed time, latency p50/p90/p99, request/output/total/prefill/
decode throughput, prefill/decode time, peak KV-cache blocks, and peak KV-cache
utilization.

Replot an existing compatible report group without running the model:

```bash
python -m experiments.experiment --plot-only \
  experiments/report/run-a.json \
  experiments/report/run-b.json
```

Analyze explicitly selected three-scheduler reports with paired seeds and
bootstrap confidence intervals:

```bash
python -m experiments.analyze_scheduler \
  --reports path/to/baseline-report.json path/to/v1-report.json \
    path/to/v2-report.json \
  --reference-selector scheduler \
  --candidate-selectors scheduler-v1 scheduler-v2 \
  --output-dir experiments/report/scheduler-comparison
```

Pairs with incompatible token totals or output digests are excluded. The
analyzer writes Markdown, CSV, a Pareto plot, and scheduler-mechanism diagnostics.

<a id="profiling"></a>

## Profiling

Keep the model, workload, sampling settings, execution mode, and component
selectors fixed when profiling a saved benchmark configuration. Separate
prefill from decode and correlate host scheduling with GPU kernels before
attributing a bottleneck to a component.

Set `ENABLE_NVTX=1` to emit optional ranges for scheduler operations, block
management, model input preparation, attention, KV-cache storage, sampling, and
logits computation:

```bash
ENABLE_NVTX=1 python -m benchmarks.benchmark \
  --model ~/huggingface/Qwen3-0.6B/ \
  --num-requests 33 \
  --input-len 256 \
  --alternate-input-len 4096 \
  --input-order alternate-first \
  --output-len 1 \
  --temperature 0
```

Profiling adds collection overhead and should not replace uninstrumented
benchmark reports used for comparisons.

<a id="pluggable-design"></a>

## Pluggable components and kernels

`EngineComponent` selects implementations by safe local filename stem. Paths
and `..` are rejected; hyphenated selectors are supported.

| YAML selector | Implementation location | Required factory |
| --- | --- | --- |
| `scheduler` | `nanovllm/engine/scheduler/<selector>.py` | `create_component(config=..., block_manager=...)` |
| `block_manager` | `nanovllm/engine/block_manager/<selector>.py` | `create_component(num_blocks=..., block_size=...)` |
| `attention` | `nanovllm/layers/attention/<selector>.py` | `create_component(num_heads=..., head_dim=..., scale=..., num_kv_heads=..., store_kvcache=...)` |
| `sampler` | `nanovllm/layers/sampler/<selector>.py` | `create_component()` |
| `store_kvcache` | `nanovllm/kernels/store_kvcache/<selector>.py` | `create_kernel()` |

Factory results must satisfy the runtime-checkable Protocol for their role in
`nanovllm.engine.component`. Omitted selectors use the baseline implementation.

The same selection API is available to Python callers:

```python
from nanovllm import EngineComponent, LLM

llm = LLM(
    "/YOUR/MODEL/PATH",
    engine_component=EngineComponent(scheduler="scheduler-v2"),
)
```
