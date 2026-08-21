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
    alternate_input_len: int | None = None,
) -> list[BenchmarkRequest]:
    """Create a deterministic synthetic workload with optional alternating lengths."""
    if num_requests <= 0:
        raise ValueError("num_requests must be greater than zero")
    if input_len <= 0:
        raise ValueError("input_len must be greater than zero")
    if alternate_input_len is not None and alternate_input_len <= 0:
        raise ValueError("alternate_input_len must be greater than zero")
    if output_len <= 0:
        raise ValueError("output_len must be greater than zero")
    if temperature != 0 and temperature <= 1e-10:
        raise ValueError("temperature must be zero or greater than 1e-10")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than zero")
    if max_model_len <= 0:
        raise ValueError("max_model_len must be greater than zero")
    longest_input_len = max(input_len, alternate_input_len or input_len)
    if longest_input_len + output_len > max_model_len:
        raise ValueError(
            "longest input length + output_len "
            f"({longest_input_len + output_len}) exceeds max_model_len "
            f"({max_model_len})"
        )

    rng = random.Random(seed)
    requests = []
    append_request = requests.append
    for request_index in range(num_requests):
        request_input_len = (
            alternate_input_len
            if alternate_input_len is not None and request_index % 2 == 1
            else input_len
        )
        prompt_token_ids = [
            rng.randrange(vocab_size) for _ in range(request_input_len)
        ]
        sampling_params = SamplingParams(
            temperature=temperature,
            ignore_eos=True,
            max_tokens=output_len,
        )
        append_request(BenchmarkRequest(prompt_token_ids, sampling_params))

    return requests
