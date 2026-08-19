import os
from dataclasses import dataclass
from time import perf_counter

import torch
from nanovllm import EngineComponent, LLM

from .workloads import BenchmarkRequest


@dataclass(frozen=True, slots=True)
class BenchmarkConfiguration:
    model: str
    device: str
    dtype: str
    tensor_parallel_size: int
    enforce_eager: bool
    block_size: int
    max_model_len: int
    gpu_memory_utilization: float
    num_kvcache_blocks: int


@dataclass
class BatchRunResult:
    elapsed_time: float
    num_requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    peak_kvcache_blocks: int
    kvcache_capacity_blocks: int
    request_latencies: tuple[float, ...]
    prefill_time: float
    decode_time: float
    prefill_tokens: int
    decode_tokens: int


class BenchmarkRunner:
    def __init__(
        self,
        model_path: str,
        enforce_eager: bool = False,
        max_model_len: int = 4096,
        engine_component: EngineComponent | None = None,
    ):
        self.llm = LLM(
            model_path,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            engine_component=engine_component,
        )
        config = self.llm.model_runner.config
        hf_config = config.hf_config
        if hf_config is None:
            raise RuntimeError("model configuration was not initialized")
        vocab_size = getattr(hf_config, "vocab_size", None)
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise RuntimeError("model configuration has no valid vocab_size")
        self.max_model_len = config.max_model_len
        self.vocab_size = vocab_size
        self.configuration = BenchmarkConfiguration(
            model=os.path.basename(os.path.normpath(config.model)),
            device=torch.cuda.get_device_name(0),
            dtype=str(hf_config.dtype).removeprefix("torch."),
            tensor_parallel_size=config.tensor_parallel_size,
            enforce_eager=config.enforce_eager,
            block_size=config.kvcache_block_size,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            num_kvcache_blocks=config.num_kvcache_blocks,
        )

    def close(self) -> None:
        """Release the underlying engine and its distributed CUDA resources."""
        self.llm.exit()

    def warmup(self, warmup_requests: list[BenchmarkRequest]) -> None:
        if not warmup_requests:
            raise ValueError("warmup_requests must not be empty")
        prompt_token_ids = [req.prompt_token_ids for req in warmup_requests]
        sampling_params = [req.sampling_params for req in warmup_requests]
        self.llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)

    def run_benchmark(self, benchmark_requests: list[BenchmarkRequest]) -> BatchRunResult:
        """Run one offline batch and measure its end-to-end wall-clock time."""
        if not benchmark_requests:
            raise ValueError("benchmark_requests must not be empty")

        prompt_token_ids = [req.prompt_token_ids for req in benchmark_requests]
        sampling_params = [req.sampling_params for req in benchmark_requests]

        torch.cuda.synchronize()
        self.llm.reset_kvcache_metrics()
        start_time = perf_counter()
        outputs = self.llm.generate(
            prompt_token_ids,
            sampling_params,
            use_tqdm=False,
        )
        torch.cuda.synchronize()
        elapsed_time = perf_counter() - start_time
        peak_kvcache_blocks = self.llm.get_peak_kvcache_blocks()
        generation_metrics = self.llm.last_generation_metrics

        if len(outputs) != len(benchmark_requests):
            raise RuntimeError(
                f"expected {len(benchmark_requests)} outputs, got {len(outputs)}"
            )

        input_tokens = sum(len(token_ids) for token_ids in prompt_token_ids)
        output_tokens = sum(len(output["token_ids"]) for output in outputs)
        return BatchRunResult(
            elapsed_time=elapsed_time,
            num_requests=len(benchmark_requests),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            peak_kvcache_blocks=peak_kvcache_blocks,
            kvcache_capacity_blocks=self.configuration.num_kvcache_blocks,
            request_latencies=generation_metrics.request_latencies,
            prefill_time=generation_metrics.prefill_time,
            decode_time=generation_metrics.decode_time,
            prefill_tokens=generation_metrics.prefill_tokens,
            decode_tokens=generation_metrics.decode_tokens,
        )
