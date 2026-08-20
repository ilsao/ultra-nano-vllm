from collections.abc import Callable

import torch
from torch import nn
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.utils.context import get_context

from utils.profiling import nvtx_annotate

class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        store_kvcache: Callable,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.store_kvcache = store_kvcache
        self.k_cache = self.v_cache = torch.tensor([])

    @nvtx_annotate("Attention.forward")
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            return flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        return flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=self.scale,
            causal=True,
        )


def create_component(
    *,
    num_heads: int,
    head_dim: int,
    scale: float,
    num_kv_heads: int,
    store_kvcache: Callable,
) -> Attention:
    return Attention(
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        store_kvcache,
    )
