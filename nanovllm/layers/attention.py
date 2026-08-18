import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


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
    # Each Triton program handles one token.
    # Grid size is (N,), so idx corresponds to the token index in [0, N).
    idx = tl.program_id(0)

    # slot_mapping[idx] tells us which physical KV-cache slot this token should be written into.
    slot = tl.load(slot_mapping_ptr + idx)

    # A slot value of -1 means this token should not be stored.
    if slot == -1:
        return

    # Flatten [num_heads, head_dim] into one contiguous dimension D.
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # Each physical cache slot stores D contiguous elements.
    # Therefore slot * D gives the starting address of this token's
    # destination in the flattened KV cache.
    cache_offsets = slot * D + tl.arange(0, D)

    # Write this token's K/V vectors into its assigned physical cache slot.
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    # key/value shape:
    #   [N, num_heads, head_dim]
    #
    # N is the number of tokens whose KV entries need to be stored.
    N, num_heads, head_dim = key.shape

    # Flatten the per-token KV representation:
    D = num_heads * head_dim

    # The kernel treats each token's [num_heads, head_dim] values
    # as one contiguous array of D elements, so the last dimension
    # must be contiguous.
    assert key.stride(-1) == 1 and value.stride(-1) == 1

    # Adjacent attention heads must also be packed contiguously. 
    assert key.stride(1) == head_dim and value.stride(1) == head_dim

    # Each KV-cache slot contains exactly D contiguous elements.
    # shape of the cache is: [num_blocks, block_size, num_kv_heads, head_dim]
    assert k_cache.stride(1) == D and v_cache.stride(1) == D

    # There must be exactly one destination cache slot for each token.
    assert slot_mapping.numel() == N

    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, 
                                       cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, 
                                       cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, 
                                       block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, 
                                        block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
