# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch

from .kernel_backend import BackendKernel
from .triton_kernels.flash_attn_v2 import triton_flash_attention_forward
from .triton_kernels.paged_attn_v1 import triton_paged_attention_v1
from .triton_kernels.paged_attn_v2 import triton_paged_attention_v2


def ref_masked_attention(
    query: torch.Tensor,  # [1, num_heads, head_size]
    key: torch.Tensor,  # [context_len, num_heads, head_size]
    value: torch.Tensor,  # [context_len, num_heads, head_size]
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query = query * scale
    attn = torch.einsum("qhd,khd->hqk", query, key)
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum("hqk,khd->qhd", attn, value)
    return out


def torch_paged_attention_forward(
    output: torch.Tensor,  # [num_tokens, num_heads, head_size]
    query: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_heads, block_size, head_size]
    value_cache: torch.Tensor,  # [num_blocks, num_heads, block_size, head_size]
    scale: float,
    block_tables: torch.Tensor,  # [num_tokens, max_num_blocks_per_seq]
    context_lens: torch.Tensor,  # [num_tokens]
) -> None:
    num_heads = value_cache.shape[1]
    block_size = value_cache.shape[2]
    head_size = value_cache.shape[3]

    num_input_tokens = query.shape[0]
    for i in range(num_input_tokens):
        q = query[i].unsqueeze(0)  # [1, num_heads, head_size]
        block_table = block_tables[i]  # [max_num_blocks_per_seq]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, block_offset, :]  # [num_heads, head_size]
            keys.append(k)

            v = value_cache[block_number, :, block_offset, :]  # [num_heads, head_size]
            values.append(v)

        keys = torch.stack(keys, dim=0)  # [context_len, num_heads, head_size]
        values = torch.stack(values, dim=0)

        out = ref_masked_attention(q, keys, values, scale)
        out = out.view(num_heads, head_size)
        output[i].copy_(out, non_blocking=True)


class PagedAttention(BackendKernel):
    """
    Paged attention implementation with online softmax used for single query attention

    Parameters:
      - query: Tensor, must be in the following format:
          [batch_size * 1, num_heads, embedding_size_per_head]
      - key/value_cache: Tensor, ,ust be in the following format:
          [num_blocks, num_heads, block_size, embedding_size_per_head]
      - head_mapping:
      - block_tables:
      - context_lens:
      - max_context_len:
      - scale: Optional[float], `1.0 / query.shape[-1] ** 0.5` by default
    """

    def __init__(self, version: int = 1):
        super().__init__()
        self.version = version

    def _torch_impl(
        self,
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        max_context_len=1,
    ):
        torch_paged_attention_forward(
            output, query, key_cache, value_cache, scale, block_tables, context_lens
        )

    def _triton_impl(
        self,
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        max_context_len=1,
    ):
        if self.version == 1:
            triton_paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                head_mapping,
                scale,
                block_tables,
                context_lens,
            )
        elif self.version == 2:
            triton_paged_attention_v2(
                output,
                query,
                key_cache,
                value_cache,
                head_mapping,
                scale,
                block_tables,
                context_lens,
                max_context_len,
            )

    def forward(
        self,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        max_context_len=1,
    ):
        output = torch.empty_like(query)
        BackendKernel.forward(
            self,
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            max_context_len,
        )
        return output


def torch_flash_attention_forward(q, k, v, causal, sm_scale=None):
    q_dim = q.dim()
    sm_scale = 1.0 / q.shape[-1] ** 0.5 if sm_scale is None else sm_scale
    # layout info
    batch_size = q.shape[0]
    seq_len = q.shape[1]
    groups = q.shape[2] if q_dim == 5 else 1
    num_heads = q.shape[-2]
    head_dims = q.shape[-1]

    M = torch.tril(torch.ones((seq_len, seq_len), device="cuda"))
    if q_dim == 4:
        p = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) * sm_scale
        if causal:
            p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).half()
        ref_out = torch.matmul(p, v.permute(0, 2, 1, 3))
        ref_out = ref_out.permute(0, 2, 1, 3)
    else:
        p = torch.matmul(q.permute(0, 2, 3, 1, 4), k.permute(0, 2, 3, 4, 1)) * sm_scale
        if causal:
            p[:, :, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).half()
        ref_out = torch.matmul(p, v.permute(0, 2, 3, 1, 4))
        ref_out = ref_out.permute(0, 3, 1, 2, 4)
    return ref_out


class FlashAttention(BackendKernel):
    """
    Parameters:
        - query: Tensor
        - key: Tensor
        - value: Tensor
        - causal: Optional[bool]
        - sm_scale: Optional[float]

    Input tensors must be in the following format:
        [batch_size, sequence_length, num_heads, embedding_size_per_head]
    Inputs can also be of dimension 5 with GQA in the following format:
        [batch_size, sequence_length, head_groups, heads_per_group, embedding_size_per_head]
    Inputs can be non-contiguous - we only require the last dimension's stride to be 1
    """

    def __init__(self):
        super().__init__()

    def _torch_impl(self, q, k, v, causal, sm_scale=None):
        return torch_flash_attention_forward(q, k, v, causal, sm_scale=sm_scale)

    def _triton_impl(self, q, k, v, causal, sm_scale=None):
        return triton_flash_attention_forward(q, k, v, causal, sm_scale=sm_scale)

    def forward(self, q, k, v, causal, sm_scale=None):
        # layout constraints
        q_dim, k_dim, v_dim = q.dim(), k.dim(), v.dim()
        # [batch_size, seq_len, num_heads, head_dims] or
        # [batch_size, seq_len, groups, kv_heads, head_dims]
        assert q_dim == 4 or q_dim == 5
        assert q_dim == k_dim and q_dim == v_dim
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        return BackendKernel.forward(self, q, k, v, causal, sm_scale)
