# Adapted from https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cu
#
# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import torch
import triton
import triton.language as tl
from flashnn.kernel_backend import THREADS_PER_WARP

PARTITION_SIZE = 512


@triton.jit
def _single_query_cached_kv_attention_v2(
    exp_sums,  # [num_seqs, num_heads, max_num_partitions]
    max_logits,  # [num_seqs, num_heads, max_num_partitions]
    out,  # [num_seqs, num_heads, max_num_partitions, head_size]
    q,  # [num_seqs, num_heads, head_size]
    k_cache,  # [num_blocks, num_heads, block_size, head_size]
    v_cache,  # [num_blocks, num_heads, block_size, head_size]
    head_mapping,  # [num_heads]
    scale,
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens,  # [num_seqs]
    partiton_size,
    max_num_blocks_per_seq,
    alibi_slopes,  # [num_heads]
    stride_qm,
    stride_qn,
    stride_om,
    stride_on,
    stride_ok,
    stride_km,
    stride_kn,
    stride_kk,
    stride_exp_m,
    stride_exp_n,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(axis=1)
    par_idx = tl.program_id(axis=2)
    seq_len = tl.load(seq_lens + seq_idx)

    if par_idx * partiton_size >= seq_len:
        return

    num_context_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    num_blocks_per_par = partiton_size // BLOCK_SIZE

    start_block_idx = par_idx * num_blocks_per_par
    end_block_idx = tl.minimum(start_block_idx + num_blocks_per_par, num_context_blocks)

    head_idx = tl.program_id(axis=0)
    kv_head_idx = tl.load(head_mapping + head_idx)

    if alibi_slopes is None:
        alibi_slope = 0.0
    else:
        alibi_slope = tl.load(alibi_slopes + head_idx)

    block_offs = tl.arange(0, BLOCK_SIZE)
    head_size_offs = tl.arange(0, HEAD_SIZE)
    q = tl.load(q + seq_idx * stride_qm + head_idx * stride_qn + head_size_offs)
    q = (q * scale).to(tl.float16)

    qkv = tl.zeros([BLOCK_SIZE, HEAD_SIZE], dtype=tl.float32)
    qk_max = float("-inf")
    exp_sum = 0.0
    fp16_0 = tl.zeros([1, 1], dtype=k_cache.dtype.element_ty)
    base_offs_kv = (
        kv_head_idx * stride_kn
        + block_offs[:, None] * stride_kk
        + head_size_offs[None, :]
    )

    for block_idx in range(start_block_idx, end_block_idx):
        physical_block_idx = tl.load(
            block_tables + seq_idx * max_num_blocks_per_seq + block_idx
        )
        mask = (block_offs[:, None] < (seq_len - block_idx * BLOCK_SIZE)) & (
            head_size_offs[None, :] < HEAD_SIZE
        )
        offs_kv = physical_block_idx * stride_km + base_offs_kv

        k = tl.load(k_cache + offs_kv, mask=mask, other=fp16_0)
        v = tl.load(v_cache + offs_kv, mask=mask, other=fp16_0)

        _qk = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        _qk += alibi_slope * (block_idx * BLOCK_SIZE + block_offs - seq_len + 1)
        _qk_max = tl.maximum(tl.max(_qk, axis=0), qk_max)
        qk = tl.where(
            block_offs[:, None] < (seq_len - block_idx * BLOCK_SIZE),
            _qk[:, None],
            float("-inf"),
        )

        _exp_sum = exp_sum * tl.exp(qk_max - _qk_max) + tl.sum(
            tl.exp(_qk - _qk_max), axis=0
        )
        qkv = (
            qkv * (exp_sum * tl.exp(qk_max - _qk_max) / _exp_sum)
            + (tl.exp(qk - _qk_max) / _exp_sum) * v
        )
        qk_max = _qk_max
        exp_sum = _exp_sum

    offs_exp = seq_idx * stride_exp_m + head_idx * stride_exp_n + par_idx
    tl.store(exp_sums + offs_exp, exp_sum)
    tl.store(max_logits + offs_exp, qk_max)

    offs_out = (
        seq_idx * stride_om
        + head_idx * stride_on
        + par_idx * stride_ok
        + head_size_offs
    )
    tl.store(out + offs_out, tl.sum(qkv, axis=0))


@triton.jit
def _single_query_cached_kv_attention_v2_unroll4(
    exp_sums,  # [num_seqs, num_heads, max_num_partitions]
    max_logits,  # [num_seqs, num_heads, max_num_partitions]
    out,  # [num_seqs, num_heads, max_num_partitions, head_size]
    q,  # [num_seqs, num_heads, head_size]
    k_cache,  # [num_blocks, num_heads, block_size, head_size]
    v_cache,  # [num_blocks, num_heads, block_size, head_size]
    head_mapping,  # [num_heads]
    scale,
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens,  # [num_seqs]
    partiton_size,
    max_num_blocks_per_seq,
    alibi_slopes,  # [num_heads]
    stride_qm,
    stride_qn,
    stride_om,
    stride_on,
    stride_ok,
    stride_km,
    stride_kn,
    stride_kk,
    stride_exp_m,
    stride_exp_n,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(axis=1)
    par_idx = tl.program_id(axis=2)
    seq_len = tl.load(seq_lens + seq_idx)

    if par_idx * partiton_size >= seq_len:
        return

    num_context_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    num_blocks_per_par = partiton_size // BLOCK_SIZE

    start_block_idx = par_idx * num_blocks_per_par
    end_block_idx = tl.minimum(start_block_idx + num_blocks_per_par, num_context_blocks)

    head_idx = tl.program_id(axis=0)
    kv_head_idx = tl.load(head_mapping + head_idx)

    if alibi_slopes is None:
        alibi_slope = 0.0
    else:
        alibi_slope = tl.load(alibi_slopes + head_idx)

    block_offs = tl.arange(0, BLOCK_SIZE)
    head_size_offs = tl.arange(0, HEAD_SIZE)
    q = tl.load(q + seq_idx * stride_qm + head_idx * stride_qn + head_size_offs)
    q = (q * scale).to(tl.float16)

    qkv = tl.zeros([BLOCK_SIZE, HEAD_SIZE], dtype=tl.float32)
    qk_max = float("-inf")
    exp_sum = 0.0
    fp16_0 = tl.zeros([1, 1], dtype=k_cache.dtype.element_ty)
    base_offs_kv = (
        kv_head_idx * stride_kn
        + block_offs[:, None] * stride_kk
        + head_size_offs[None, :]
    )
    block_base_ptrs = block_tables + seq_idx * max_num_blocks_per_seq

    for block_idx in range(start_block_idx, end_block_idx, 4):
        mask_0 = block_offs[:, None] < (seq_len - (block_idx + 0) * BLOCK_SIZE)
        mask_1 = block_offs[:, None] < (seq_len - (block_idx + 1) * BLOCK_SIZE)
        mask_2 = block_offs[:, None] < (seq_len - (block_idx + 2) * BLOCK_SIZE)
        mask_3 = block_offs[:, None] < (seq_len - (block_idx + 3) * BLOCK_SIZE)
        offs_kv_0 = tl.load(block_base_ptrs + block_idx + 0) * stride_km + base_offs_kv
        offs_kv_1 = tl.load(block_base_ptrs + block_idx + 1) * stride_km + base_offs_kv
        offs_kv_2 = tl.load(block_base_ptrs + block_idx + 2) * stride_km + base_offs_kv
        offs_kv_3 = tl.load(block_base_ptrs + block_idx + 3) * stride_km + base_offs_kv

        k_0 = tl.load(k_cache + offs_kv_0, mask=mask_0, other=fp16_0)
        k_1 = tl.load(k_cache + offs_kv_1, mask=mask_1, other=fp16_0)
        k_2 = tl.load(k_cache + offs_kv_2, mask=mask_2, other=fp16_0)
        k_3 = tl.load(k_cache + offs_kv_3, mask=mask_3, other=fp16_0)

        v_0 = tl.load(v_cache + offs_kv_0, mask=mask_0, other=fp16_0)
        v_1 = tl.load(v_cache + offs_kv_1, mask=mask_1, other=fp16_0)
        v_2 = tl.load(v_cache + offs_kv_2, mask=mask_2, other=fp16_0)
        v_3 = tl.load(v_cache + offs_kv_3, mask=mask_3, other=fp16_0)

        _qk_0 = tl.sum((q[None, :] * k_0).to(tl.float32), axis=1)
        _qk_1 = tl.sum((q[None, :] * k_1).to(tl.float32), axis=1)
        _qk_2 = tl.sum((q[None, :] * k_2).to(tl.float32), axis=1)
        _qk_3 = tl.sum((q[None, :] * k_3).to(tl.float32), axis=1)

        _qk_0 += alibi_slope * ((block_idx + 0) * BLOCK_SIZE + block_offs - seq_len + 1)
        _qk_1 += alibi_slope * ((block_idx + 1) * BLOCK_SIZE + block_offs - seq_len + 1)
        _qk_2 += alibi_slope * ((block_idx + 2) * BLOCK_SIZE + block_offs - seq_len + 1)
        _qk_3 += alibi_slope * ((block_idx + 3) * BLOCK_SIZE + block_offs - seq_len + 1)

        _qk_max = tl.maximum(tl.max(_qk_0, axis=0), qk_max)
        _qk_max = tl.maximum(tl.max(_qk_1, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_2, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_3, axis=0), _qk_max)

        qk_0 = tl.where(mask_0, _qk_0[:, None], float("-inf"))
        qk_1 = tl.where(mask_1, _qk_1[:, None], float("-inf"))
        qk_2 = tl.where(mask_2, _qk_2[:, None], float("-inf"))
        qk_3 = tl.where(mask_3, _qk_3[:, None], float("-inf"))

        _exp_sum = (
            exp_sum * tl.exp(qk_max - _qk_max)
            + tl.sum(tl.exp(_qk_0 - _qk_max), axis=0)
            + tl.sum(tl.exp(_qk_1 - _qk_max), axis=0)
            + tl.sum(tl.exp(_qk_2 - _qk_max), axis=0)
            + tl.sum(tl.exp(_qk_3 - _qk_max), axis=0)
        )
        qkv = (
            qkv * (exp_sum * tl.exp(qk_max - _qk_max) / _exp_sum)
            + (tl.exp(qk_0 - _qk_max) / _exp_sum) * v_0
            + (tl.exp(qk_1 - _qk_max) / _exp_sum) * v_1
            + (tl.exp(qk_2 - _qk_max) / _exp_sum) * v_2
            + (tl.exp(qk_3 - _qk_max) / _exp_sum) * v_3
        )
        qk_max = _qk_max
        exp_sum = _exp_sum

    offs_exp = seq_idx * stride_exp_m + head_idx * stride_exp_n + par_idx
    tl.store(exp_sums + offs_exp, exp_sum)
    tl.store(max_logits + offs_exp, qk_max)

    offs_out = (
        seq_idx * stride_om
        + head_idx * stride_on
        + par_idx * stride_ok
        + head_size_offs
    )
    tl.store(out + offs_out, tl.sum(qkv, axis=0))


@triton.jit
def _paged_attention_v2_reduce(
    out,  # [num_seqs, num_heads, head_size]
    exp_sums,  # [num_seqs, num_heads, max_num_partitions]
    max_logits,  # [num_seqs, num_heads, max_num_partitions]
    tmp_out,  # [num_seqs, num_heads, max_num_partitions, head_size]
    context_lens,  # [num_seqs]
    stride_exp_m,
    stride_exp_n,
    stride_out_m,
    stride_out_n,
    stride_tmp_m,
    stride_tmp_n,
    stride_tmp_k,
    HEAD_SIZE: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,
):
    seq_idx = tl.program_id(axis=1)
    head_idx = tl.program_id(axis=0)
    context_len = tl.load(context_lens + seq_idx)

    num_partitions = tl.cdiv(context_len, PARTITION_SIZE)

    exp_sum = 0.0
    max_logit = float("-inf")
    offs_logit = seq_idx * stride_exp_m + head_idx * stride_exp_n

    head_size_offs = tl.arange(0, HEAD_SIZE)
    tmp_out_ptr = seq_idx * stride_tmp_m + head_idx * stride_tmp_n
    out_ptr = seq_idx * stride_out_m + head_idx * stride_out_n + head_size_offs

    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)
    global_exp_sum = tl.zeros([1], dtype=tl.float32)

    logits = tl.load(
        max_logits + offs_logit + tl.arange(0, NUM_PARTITIONS),
        mask=tl.arange(0, NUM_PARTITIONS) < num_partitions,
        other=float("-inf"),
    )
    max_logit = tl.max(logits, axis=0)

    exp_sum = tl.load(
        exp_sums + offs_logit + tl.arange(0, NUM_PARTITIONS),
        mask=tl.arange(0, NUM_PARTITIONS) < num_partitions,
        other=0.0,
    )
    rescaled_exp_sum = exp_sum * tl.exp(logits - max_logit)
    global_exp_sum += tl.sum(rescaled_exp_sum, axis=0)

    tmp = tl.load(
        tmp_out
        + tmp_out_ptr
        + tl.arange(0, NUM_PARTITIONS)[:, None] * stride_tmp_k
        + head_size_offs
    )
    acc += tl.sum(tmp * rescaled_exp_sum[:, None], axis=0)

    inv_sum = 1.0 / (global_exp_sum + 1e-6)
    tl.store(out + out_ptr, acc * inv_sum)


def triton_paged_attention_v2(
    out,  # [num_seqs, num_heads, head_size]
    query,  # [num_seqs, num_heads, head_size]
    key_cache,  # [num_blocks, num_heads, block_size, head_size]
    value_cache,  # [num_blocks, num_heads, block_size, head_size]
    head_mapping,  # [num_heads]
    scale,
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    context_lens,  # [num_seqs]
    max_context_len,
    alibi_slopes=None,  # [num_heads]
):
    assert (
        key_cache.is_contiguous() and value_cache.is_contiguous()
    ), "kv cache must be contiguous"
    num_heads = value_cache.shape[1]
    head_size = value_cache.shape[-1]
    block_size = value_cache.shape[-2]
    num_seqs = query.shape[0]

    max_num_partitions = triton.cdiv(max_context_len, PARTITION_SIZE)

    exp_sums = torch.empty(
        (num_seqs, num_heads, max_num_partitions), dtype=torch.float32, device="cuda"
    )
    max_logits = torch.empty(
        (num_seqs, num_heads, max_num_partitions), dtype=torch.float16, device="cuda"
    )
    tmp_out = torch.empty(
        (num_seqs, num_heads, max_num_partitions, head_size),
        dtype=torch.float32,
        device="cuda",
    )

    # online softmax with unroll4
    method_name = (
        "single_query_cached_kv_attention_v2_unroll4_"
        + str(block_size)
        + "_"
        + str(head_size)
    )
    kwargs = [
        exp_sums,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        PARTITION_SIZE,
        block_tables.shape[1],
        alibi_slopes,
        query.stride(0),
        query.stride(1),
        tmp_out.stride(0),
        tmp_out.stride(1),
        tmp_out.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        exp_sums.stride(0),
        exp_sums.stride(1),
    ]
    grid = (num_heads, num_seqs, max_num_partitions)
    const_kwargs = {"BLOCK_SIZE": block_size, "HEAD_SIZE": head_size}
    _single_query_cached_kv_attention_v2_unroll4[grid](*kwargs, **const_kwargs)

    # reduction across partitions
    num_partitions = triton.next_power_of_2(max_num_partitions)
    kwargs = [
        out,
        exp_sums,
        max_logits,
        tmp_out,
        context_lens,
        exp_sums.stride(0),
        exp_sums.stride(1),
        out.stride(0),
        out.stride(1),
        tmp_out.stride(0),
        tmp_out.stride(1),
        tmp_out.stride(2),
    ]
    grid = (num_heads, num_seqs, 1)
    const_kwargs = {
        "HEAD_SIZE": head_size,
        "NUM_PARTITIONS": num_partitions,
        "num_warps": triton.cdiv(head_size, THREADS_PER_WARP),
    }
    _paged_attention_v2_reduce[grid](*kwargs, **const_kwargs)
