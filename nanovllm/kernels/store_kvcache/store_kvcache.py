import torch
import triton
import triton.language as tl

from utils.profiling import nvtx_annotate

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    # Each Triton program handles one token in [0, num_tokens).
    idx = tl.program_id(0)

    # slot_mapping[idx] identifies the token's physical KV-cache slot.
    slot = tl.load(slot_mapping_ptr + idx)

    # A slot value of -1 means this token should not be stored.
    if slot == -1:
        return

    # Flatten [num_heads, head_dim] into one contiguous width D.
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # Each cache slot stores D contiguous elements, so slot * D is the
    # destination offset for this token in the flattened KV cache.
    cache_offsets = slot * D + tl.arange(0, D)

    # Write this token's K/V vectors to its assigned physical cache slot.
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

@nvtx_annotate("store_kvcache")
def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    # key/value shape: [num_tokens, num_heads, head_dim].
    num_tokens, num_heads, head_dim = key.shape

    # Treat each token's heads and head dimensions as one contiguous vector.
    width = num_heads * head_dim

    # The kernel expects the head dimensions and adjacent heads to be packed.
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim

    # Each physical cache slot contains exactly one flattened token vector.
    assert k_cache.stride(1) == width and v_cache.stride(1) == width

    # Every input token must have one destination cache slot.
    assert slot_mapping.numel() == num_tokens
    store_kvcache_kernel[(num_tokens,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        width,
    )


def create_kernel():
    return store_kvcache
