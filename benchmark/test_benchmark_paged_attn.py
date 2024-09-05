# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import unittest
from typing import Optional

import numpy as np
import torch
import triton
from parameterized import parameterized
import pytest

from flashnn.triton_kernels.paged_attn import (
    paged_attn_w_mma,
    paged_attn_w_mma_transv,
    paged_attn_wo_mma, 
    paged_attn_w_mma_unrolling2,
    paged_attn_w_mma_unrolling4,
)

try:
    from vllm import _custom_ops as ops

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    from vllm._custom_C import paged_attention_custom

    HAS_VLLM_CUSTOM_PAGED = True
except ImportError:
    HAS_VLLM_CUSTOM_PAGED = False

TEST_SEED = 0


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


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,  # [num_tokens, q_heads, head_size]
    query: torch.Tensor,  # [num_tokens, q_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, kv_heads, head_size/x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, kv_heads, head_size, block_size]
    block_tables: torch.Tensor,  # [num_tokens, max_num_blocks_per_seq]
    context_lens: torch.Tensor,  # [num_tokens]
) -> None:
    q_heads = query.shape[1]
    kv_heads = key_cache.shape[1]
    h_ratio = q_heads // kv_heads
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]

    num_input_tokens = query.shape[0]
    for i in range(num_input_tokens):
        q = query[i].unsqueeze(0)  # [1, q_heads, head_size]
        block_table = block_tables[i]  # [max_num_blocks_per_seq]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[
                block_number, :, :, block_offset, :
            ]  # [kv_heads, head_size/x, x]
            k = k.reshape(kv_heads, head_size)  # [kv_heads, head_size]
            if h_ratio != 1:
                k = k.repeat_interleave(h_ratio, 0)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]  # [kv_heads, head_size]
            if h_ratio != 1:
                v = v.repeat_interleave(h_ratio, 0)
            values.append(v)

        keys = torch.stack(keys, dim=0)  # [context_len, kv_heads, head_size]
        values = torch.stack(values, dim=0)

        scale = 1.0 / (head_size**0.5)
        out = ref_masked_attention(q, keys, values, scale)
        out = out.view(q_heads, head_size)
        output[i].copy_(out, non_blocking=True)


configs = []
HEAD_DIM = 128
def get_input_shapes():
    test_cases = [
    # (1, 16384, 32, 4),
    # (1, 16384, 16, 16),
    # (1, 4096, 16, 16),
    # (1, 8192, 16, 16),
    # (1, 16384, 16,16),
    
    (1, 1024, 52, 4),
    (1, 2048, 52, 4),
    (1, 4096, 52, 4),
    (1, 8192, 52, 4),
    (1, 16384, 52,4),
    (64,1024, 52, 4),
    (64,16384, 52, 4),
    (1, 1024, 8 ,1),
    (1, 2048, 8 ,1),
    (1, 4096, 8 ,1),
    (1, 8192, 8 ,1),
    (1, 16384, 8, 1),
    (64,1024, 8, 1),
    (64,16384, 8, 1),
    (64, 16384, 32, 4),
    ]
    return test_cases

total_slots = 16 * 10240
input_block_size = 16
input_block_num = total_slots // input_block_size
split_size = 256

configs.append(
    triton.testing.Benchmark(
        x_names=['num_seqs', 'seq_len', 'q_head', 'kv_head'],
        x_vals=get_input_shapes(),
        line_arg="provider",
        line_vals=(
            # ["triton_mma", "triton_mma_transv"]
            ["triton_fma", "triton_mma", "triton_mma_transv", "triton_mma_unrolling2", "triton_mma_unrolling4"]
            # + (["vllm_v1", "vllm_v2"] if HAS_VLLM else [])
            + (["vllm_custom"] if HAS_VLLM_CUSTOM_PAGED else [])
        ),
        line_names=(
            # ["TritonMMA", "TriMMATransV"]
            ["Triton FMA", "TritonMMA", "TriMMATransV", "MMA Unroll2", "MMA Unroll4"]
            # + (["vLLM_V1", "vLLM_V2"] if HAS_VLLM else [])
            + (["vLLM_CUSTOM"] if HAS_VLLM_CUSTOM_PAGED else [])
        ),
        styles=[
            ("red", "-"),
            ("yellow", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("orange", "-"),
            ("purple", "-"),
        ],
        ylabel="ms",
        plot_name=f"head_size=128, block_size={input_block_size},num_blocks={input_block_num},parititon_size={split_size}",
        args={
            "head_size": 128,
            "block_size": input_block_size,
            "num_blocks": input_block_num,
            "dtype": torch.float16,
        },
    )
)

@triton.testing.perf_report(configs)
def benchmark(
    num_seqs,
    seq_len,
    q_head,
    kv_head,
    head_size,
    block_size,
    num_blocks,
    dtype,
    provider,
    eps=1e-5,
    device="cuda",
):
    # vllm implementation needs to use block_size 16 and partition_size 256
    if provider.startswith("vllm") and provider != "vllm_v1":
        PARTITION_SIZE = 256
        block_size = 16
    
    query = torch.empty(num_seqs, q_head, head_size, dtype=dtype, device="cuda")

    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_block_shape = (kv_head, head_size // x, block_size, x)
    key_cache = torch.randn((num_blocks, *key_block_shape), dtype=dtype, device="cuda")
    value_block_shape = (kv_head, head_size, block_size)
    value_cache = torch.randn(
        (num_blocks, *value_block_shape), dtype=dtype, device="cuda"
    )

    context_lens = [seq_len for _ in range(num_seqs)]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for i in range(num_seqs):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")

    scale = float(1.0 / (head_size**0.5))

    assert q_head % kv_head == 0
    num_queries_per_kv = q_head // kv_head
    head_mapping = torch.repeat_interleave(
        torch.arange(kv_head, dtype=torch.int32, device="cuda"), num_queries_per_kv
    )

    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_seqs)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int, device="cuda")
    out = torch.empty_like(query)

    key_cache_tri = key_cache.permute(0, 1, 3, 2, 4).flatten(3, 4).contiguous().cuda()
    value_cache_tri = value_cache.permute(0, 1, 3, 2).contiguous().cuda()
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ref_single_query_cached_kv_attention(
                out,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
            ),
            warmup=20,
            rep=100,
            quantiles=quantiles,
        )

    if provider.startswith("triton"):
        device = torch.cuda.device_of(query)
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        if num_seqs * kv_head > 2 * num_sms:
            num_splits = 1
            partition_size = 0
            if max_context_len >= 8192:
                partition_size = max(split_size, block_size)
                num_splits = triton.cdiv(max_context_len, partition_size)
        else:
            partition_size = max(split_size, block_size)
            num_splits = triton.cdiv(max_context_len, partition_size)
            if max_context_len <= 1024 or block_size >= 256:
                num_splits = 1
                partition_size = 0
    if provider == "triton_fma":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: paged_attn_wo_mma(
                out,
                query,
                key_cache_tri,
                value_cache_tri,
                context_lens,
                block_tables,
                scale,
                max_context_len,
                num_splits,
                partition_size,
                device,
            ),
            warmup=20,
            rep=100,
            quantiles=quantiles,
        )

    if provider == "triton_mma":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: paged_attn_w_mma(
                out,
                query,
                key_cache_tri,
                value_cache_tri,
                context_lens,
                block_tables,
                scale,
                max_context_len,
                num_splits,
                partition_size,
                device,
            ),
            warmup=20,
            rep=100,
            quantiles=quantiles,
        )

    if provider == "triton_mma_transv":
        value_cache_tri_transv = value_cache.permute(0, 1, 3, 2)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: paged_attn_w_mma_transv(
                out,
                query,
                key_cache_tri,
                value_cache_tri_transv,
                context_lens,
                block_tables,
                scale,
                max_context_len,
                num_splits,
                partition_size,
                device,
            ),
            warmup=20,
            rep=100,
            quantiles=quantiles,
        )

    if provider == "triton_mma_unrolling4":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: paged_attn_w_mma_unrolling4(
                out,
                query,
                key_cache_tri,
                value_cache_tri,
                context_lens,
                block_tables,
                scale,
                max_context_len,
                num_splits,
                partition_size,
                device,
            ),
            warmup=20,
            rep=100,
            quantiles=quantiles,
        )

    if provider == "triton_mma_unrolling2":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: paged_attn_w_mma_unrolling2(
                out,
                query,
                key_cache_tri,
                value_cache_tri,
                context_lens,
                block_tables,
                scale,
                max_context_len,
                num_splits,
                partition_size,
                device,
            ),
            warmup=20,
            rep=100,
            quantiles=quantiles,
        )


    if provider == "vllm_v1":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ops.paged_attention_v1(
                out,
                query,
                key_cache,
                value_cache,
                kv_head,
                scale,
                block_tables,
                context_lens,
                block_size,
                max_context_len,
                None,
                "auto",
                1.0,
            ),
            warmup=20,
            rep=100,
            quantiles=quantiles,
        )

    if provider.startswith("vllm") and provider != "vllm_v1":
        PARTITION_SIZE = 256
        num_partitions = (max_context_len + PARTITION_SIZE - 1) // PARTITION_SIZE
        tmp_output = torch.empty(
            size=(num_seqs, q_head, num_partitions, head_size),
            dtype=out.dtype,
            device=out.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, q_head, num_partitions),
            dtype=torch.float32,
            device=out.device,
        )
        max_logits = torch.empty_like(exp_sums)
        # block size need to be hard-coded for customized HIP kernel
        block_size = 16
        if provider == "vllm_v2":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: ops.paged_attention_v2(
                    out,
                    exp_sums,
                    max_logits,
                    tmp_output,
                    query,
                    key_cache,
                    value_cache,
                    kv_head,
                    scale,
                    block_tables,
                    context_lens,
                    block_size,
                    max_context_len,
                    None,
                    "auto",
                    1.0,
                ),
                warmup=20,
                rep=100,
                quantiles=quantiles,
            )
        if provider == "vllm_custom":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: paged_attention_custom(
                    out,
                    exp_sums,
                    max_logits,
                    tmp_output,
                    query,
                    key_cache,
                    value_cache,
                    kv_head,
                    scale,
                    block_tables,
                    context_lens,
                    block_size,
                    max_context_len,
                    None,
                    "auto",
                ),
                warmup=20,
                rep=100,
                quantiles=quantiles,
            )

    def ms2us(ms):
        return round(ms * 1000, 2)

    return ms2us(ms), ms2us(min_ms), ms2us(max_ms)


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)


@pytest.mark.parametrize('B, ctx_n, num_q_heads, num_kv_heads',
                        get_input_shapes())
def test_paged_attn(B, ctx_n, num_q_heads, num_kv_heads, head_size=HEAD_DIM, block_size=16, num_blocks=10240, partition_size=256, dtype=torch.half):
    qkv = torch.ones(B, 3, num_q_heads, head_size, dtype=dtype, device="cuda")
    query, key, value = qkv.unbind(dim=1)

    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_block_shape = (num_kv_heads, head_size // x, block_size, x)
    key_cache = torch.ones((num_blocks, *key_block_shape), dtype=dtype, device="cuda")
    value_block_shape = (num_kv_heads, head_size, block_size)
    value_cache = torch.randn(
        (num_blocks, *value_block_shape), dtype=dtype, device="cuda"
    )

    context_lens = [ctx_n for _ in range(B)]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")
    # max_context_len=8192
    # context_lens = [max_context_len for _ in range(num_seqs)]
    # context_lens = torch.tensor(context_lens, dtype=torch.int, device='cuda')

    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for i in range(B):
        block_table = [
            i for i in range(max_num_blocks_per_seq)
            # random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")
    # head_mapping = torch.arange(num_q_heads, dtype=torch.int32, device="cuda")

    scale = float(1.0 / (head_size**0.5))

    assert num_q_heads % num_kv_heads == 0
    num_queries_per_kv = num_q_heads // num_kv_heads
    # head_mapping = torch.repeat_interleave(
    #     torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"), num_queries_per_kv
    # )

    num_slots = block_size * num_blocks
    # slot_mapping = random.sample(range(num_slots), num_seqs)
    # slot_mapping = [i for i in range(num_seqs)]
    # slot_mapping = torch.tensor(slot_mapping, dtype=torch.int, device="cuda")

    # reference torch implementation
    out_ref = torch.empty_like(query)
    ref_single_query_cached_kv_attention(
                out_ref,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
            ),
    
    # triton implementation needs to partition size and shuffle the input
    device = torch.cuda.device_of(query)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if B * num_kv_heads > 2 * num_sms:
        num_splits = 1
        partition_size = 0
        if max_context_len >= 8192:
            partition_size = max(split_size, block_size)
            num_splits = triton.cdiv(max_context_len, partition_size)
    else:
        partition_size = max(split_size, block_size)
        num_splits = triton.cdiv(max_context_len, partition_size)
        if max_context_len <= 1024 or block_size >= 256:
            num_splits = 1
            partition_size = 0

    key_cache_tri = key_cache.permute(0, 1, 3, 2, 4).flatten(3, 4).contiguous().cuda()
    value_cache_tri = value_cache.permute(0, 1, 3, 2).contiguous().cuda()

    # triton fma implementation
    out_fma = torch.empty_like(query)
    paged_attn_wo_mma(
                out_fma,
                query,
                key_cache_tri,
                value_cache_tri,
                context_lens,
                block_tables,
                scale,
                max_context_len,
                num_splits,
                partition_size,
                device,
            )

    # triton implementation without unrolling
    out_mma = torch.empty_like(query)
    paged_attn_w_mma(
                out_mma,
                query,
                key_cache_tri,
                value_cache_tri,
                context_lens,
                block_tables,
                scale,
                max_context_len,
                num_splits,
                partition_size,
                device,
            )

    # transposed V input
    value_cache_tri_trans = value_cache.permute(0, 1, 3, 2)
    out_mma_transv = torch.empty_like(query)
    paged_attn_w_mma_transv(
                out_mma_transv,
                query,
                key_cache_tri,
                value_cache_tri_trans,
                context_lens,
                block_tables,
                scale,
                max_context_len,
                num_splits,
                partition_size,
                device,
            )


    # triton implementation with unrolling4
    out_mma_unrolling4 = torch.empty_like(query)
    paged_attn_w_mma_unrolling4(
        out_mma_unrolling4,
        query,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        scale,
        max_context_len,
        num_splits,
        partition_size,
        device,
    )


    # hip customized implementation
    out_customized_hip = torch.empty_like(query)
    # fixed split_size 256, cannot change
    PARTITION_SIZE = 256
    num_partitions = (max_context_len + PARTITION_SIZE - 1) // PARTITION_SIZE
    tmp_output = torch.empty(
        size=(B, num_q_heads, num_partitions, head_size),
        dtype=out_customized_hip.dtype,
        device=out_customized_hip.device,
    )
    exp_sums = torch.empty(
        size=(B, num_q_heads, num_partitions),
        dtype=torch.float32,
        device=out_customized_hip.device,
    )
    max_logits = torch.empty_like(exp_sums)
    paged_attention_custom(
                    out_customized_hip,
                    exp_sums,
                    max_logits,
                    tmp_output,
                    query,
                    key_cache,
                    value_cache,
                    num_kv_heads,
                    scale,
                    block_tables,
                    context_lens,
                    block_size,
                    max_context_len,
                    None,
                    "auto",
                )    

    assert torch.allclose(out_ref, out_fma, atol=1e-2, rtol=1e-2)
    assert torch.allclose(out_ref, out_mma, atol=1e-2, rtol=1e-2)
    assert torch.allclose(out_ref, out_mma_transv, atol=1e-2, rtol=1e-2)
    assert torch.allclose(out_ref, out_mma_unrolling4, atol=1e-2, rtol=1e-2)
    assert torch.allclose(out_ref, out_customized_hip, atol=1e-2, rtol=1e-2)
