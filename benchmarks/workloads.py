import random

from dataclasses import dataclass
from nanovllm import SamplingParams


@dataclass
class BenchmarkRequest:
    prompt_token_ids: list[int]
    sampling_params: SamplingParams


def synthetic_workload(
    num_requests: int,
    input_len: int,
    output_len: int,
    seed: int,
    temperature: float,
    vocab_size: int,
    max_model_len: int,
) -> list[BenchmarkRequest]:
    """Create a deterministic, fixed-length synthetic workload."""
    if num_requests <= 0:
        raise ValueError("num_requests must be greater than zero")
    if input_len <= 0:
        raise ValueError("input_len must be greater than zero")
    if output_len <= 0:
        raise ValueError("output_len must be greater than zero")
    if temperature != 0 and temperature <= 1e-10:
        raise ValueError("temperature must be zero or greater than 1e-10")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than zero")
    if max_model_len <= 0:
        raise ValueError("max_model_len must be greater than zero")
    if input_len + output_len > max_model_len:
        raise ValueError(
            f"input_len + output_len ({input_len + output_len}) exceeds "
            f"max_model_len ({max_model_len})"
        )

    rng = random.Random(seed)
    requests = []
    append_request = requests.append
    for _ in range(num_requests):
        prompt_token_ids = [rng.randrange(vocab_size) for _ in range(input_len)]
        sampling_params = SamplingParams(
            temperature=temperature,
            ignore_eos=True,
            max_tokens=output_len,
        )
        append_request(BenchmarkRequest(prompt_token_ids, sampling_params))

    return requests
