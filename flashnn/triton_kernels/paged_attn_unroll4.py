# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import torch
import triton
import triton.language as tl


# Requires triton >= 2.2.0
def paged_attention(
    out: torch.Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    query: torch.Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    key_cache: torch.Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    value_cache: torch.Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE], required same stride with key_cache
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    attn_scale: float,
    max_context_len: int,
    num_splits: int = 0,
    alibi_slope: torch.Tensor = None,
) -> None:
    num_seqs = query.shape[0]
    num_kv_heads = key_cache.shape[1]
    kv_block_size = key_cache.shape[2]
    head_size = key_cache.shape[3]
    query_group_size = query.shape[1] // num_kv_heads

    assert head_size in (16, 32, 64, 128, 256, 512), f"head_size={head_size}"
    assert (
        query_group_size == 1 or kv_block_size >= 16
    ), f"kv_block_size={kv_block_size}"
    # query_group_size in (1, 2, 4, 8, 16, 32, 64, 128, 256)
    # assert query_group_size > 0 and query_group_size & (query_group_size-1) == 0, f"query_group_size={query_group_size}"

    # config for A100
    # TODO: support more devices and optimize
    device = torch.cuda.device_of(query)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if num_splits == 0:
        if num_seqs * num_kv_heads > 2 * num_sms:
            num_splits = 1
            partition_size = 0
            if max_context_len >= 8192:
                partition_size = max(512, kv_block_size)
                num_splits = triton.cdiv(max_context_len, partition_size)
        else:
            partition_size = max(512, kv_block_size)
            num_splits = triton.cdiv(max_context_len, partition_size)
            if max_context_len <= 1024 or kv_block_size >= 256:
                num_splits = 1
                partition_size = 0
    # User hint
    elif num_splits > 1:
        partition_size = triton.cdiv(max_context_len, num_splits)
        partition_size = triton.next_power_of_2(partition_size)
    kwargs = [
        out,
        query,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
        num_splits,
        partition_size,
        device,
        alibi_slope,
    ]
    if query_group_size == 1:
        paged_attn_wo_mma(*kwargs)
    else:
        paged_attn_w_mma(*kwargs)


def paged_attn_wo_mma(
    out: torch.Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    query: torch.Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    key_cache: torch.Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    value_cache: torch.Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE], required same stride with key_cache
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    attn_scale: float,
    max_context_len: int,
    num_splits: int,
    partition_size: int,
    device,
    alibi_slope: torch.Tensor = None,
) -> None:
    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    kv_block_size = key_cache.shape[2]
    head_size = key_cache.shape[3]
    query_group_size = num_q_heads // num_kv_heads

    with torch.cuda.device(device):
        shape_info = (num_seqs, num_q_heads, num_splits)
        exp_sums = torch.empty(size=shape_info, dtype=torch.float32, device="cuda")
        max_logits = torch.empty(size=shape_info, dtype=torch.float32, device="cuda")
        if num_splits == 1:
            tmp_out = out
        else:
            tmp_out = torch.empty(
                (*shape_info, head_size), dtype=torch.float32, device="cuda"
            )
        kwargs = [
            exp_sums,
            max_logits,
            tmp_out,
            query,
            key_cache,
            value_cache,
            attn_scale,
            block_tables,
            context_lens,
            block_tables.shape[1],
            alibi_slope,
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
        grid = (num_q_heads, num_seqs, num_splits)
        const_kwargs = {
            "BLOCK_SIZE": kv_block_size,
            "HEAD_SIZE": head_size,
            "QUERY_GROUP_SIZE": query_group_size,
            "PARTITION_SIZE": 0 if num_splits == 1 else partition_size,
            "POWER_OF_2_MAX_SEQ_LEN": triton.next_power_of_2(max_context_len),
            "USE_PARTITIONING": False if num_splits == 1 else True,
        }
        _paged_attn_wo_mma_kernel[grid](*kwargs, **const_kwargs)

        if num_splits != 1:
            # reduction across partitions
            padded_num_splits = triton.next_power_of_2(num_splits)
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
            grid = (num_q_heads, num_seqs, 1)
            const_kwargs = {
                "HEAD_SIZE": head_size,
                "PADDED_NUM_SPLITS": padded_num_splits,
                "PARTITION_SIZE": partition_size,
            }
            _paged_attn_wo_mma_v2_reduce_kernel[grid](*kwargs, **const_kwargs)


def paged_attn_w_mma(
    out: torch.Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    query: torch.Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    key_cache: torch.Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    value_cache: torch.Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE], required same stride with key_cache
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    attn_scale: float,
    max_context_len: int,
    num_splits: int,
    partition_size: int,
    device,
    alibi_slope: torch.Tensor = None,
) -> None:
    num_seqs = query.shape[0]
    num_kv_heads = key_cache.shape[1]
    kv_block_size = key_cache.shape[2]
    head_size = key_cache.shape[3]
    query_group_size = query.shape[1] // num_kv_heads
    if query_group_size == 1:
        padded_group_size = 1
    elif query_group_size < 16:
        padded_group_size = 16
    else:
        padded_group_size = triton.next_power_of_2(query_group_size)

    with torch.cuda.device(device):
        assert alibi_slope is None
        grid = (num_seqs, num_kv_heads, num_splits)
        shape_info = (num_seqs, num_kv_heads, num_splits, query_group_size)
        m_i = torch.empty(size=shape_info, dtype=torch.float32, device=query.device)
        l_i = torch.empty(size=shape_info, dtype=torch.float32, device=query.device)
        tmp_out = torch.empty(
            size=(*shape_info, head_size), dtype=out.dtype, device=out.device
        )
        kwargs = [
            m_i,
            l_i,
            out if num_splits == 1 else tmp_out,
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            attn_scale,
            block_tables.stride(0),
            block_tables.stride(1),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
        ]
        if num_splits == 1:
            kwargs += [
                out.stride(0),
                out.stride(1),
                out.stride(1),
                out.stride(1),
                out.stride(2),
            ]
        else:
            kwargs += [
                tmp_out.stride(0),
                tmp_out.stride(1),
                tmp_out.stride(2),
                tmp_out.stride(3),
                tmp_out.stride(4),
            ]
        const_kwargs = {
            "HEAD_SIZE": head_size,
            "QUERY_GROUP_SIZE": query_group_size,
            "PADDED_QUERY_GROUP_SIZE": padded_group_size,
            "NUM_KV_HEADS": num_kv_heads,
            "KV_BLOCK_SIZE": kv_block_size,
            "PARTITION_SIZE": partition_size,
        }
        _paged_attn_w_mma_kernel[grid](*kwargs, **const_kwargs)

        if num_splits != 1:
            assert (partition_size >= kv_block_size) and (
                partition_size % kv_block_size == 0
            ), f"partition_size={partition_size}, kv_block_size={kv_block_size}"
            reduce_grid = (num_seqs, num_kv_heads, 1)
            kwargs = [
                out,
                m_i,
                l_i,
                tmp_out,
                context_lens,
                num_splits,
                out.stride(0),
                out.stride(1),
                out.stride(2),
            ]
            const_kwargs = {
                "HEAD_SIZE": head_size,
                "QUERY_GROUP_SIZE": query_group_size,
                "PADDED_QUERY_GROUP_SIZE": padded_group_size,
                "NUM_KV_HEADS": num_kv_heads,
                "PARTITION_SIZE": partition_size,
                "NUM_PARTITIONS": triton.next_power_of_2(num_splits),
            }
            _paged_attn_w_mma_v2_reduce_kernel[grid](*kwargs, **const_kwargs)


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=stages, num_warps=warps)
        for stages in [0, 1, 3, 4]
        for warps in [4, 8, 16]
    ],
    key=["QUERY_GROUP_SIZE", "HEAD_SIZE", "KV_BLOCK_SIZE"],
)
@triton.jit
def _paged_attn_w_mma_kernel(
    m_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    l_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    out_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE, HEAD_SIZE]
    q_ptr,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    k_cache_ptr,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    v_cache_ptr,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    context_lens_ptr,  # [num_seqs]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    attn_scale,
    stride_bt0,
    stride_bt1,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_kv0,
    stride_kv1,
    stride_kv2,
    stride_kv3,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    stride_o4,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PADDED_QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    part_idx = tl.program_id(2)
    max_num_partitions = tl.num_programs(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634

    USE_PARTITIONING = PARTITION_SIZE > 0
    context_len = tl.load(context_lens_ptr + seq_idx)
    if USE_PARTITIONING:
        context_start_idx = part_idx * PARTITION_SIZE
        if context_start_idx >= context_len:
            return
        context_end_idx = tl.minimum(context_start_idx + PARTITION_SIZE, context_len)
        num_blocks = tl.cdiv(context_end_idx - context_start_idx, KV_BLOCK_SIZE)
    else:
        num_blocks = tl.cdiv(context_len, KV_BLOCK_SIZE)

    block_offset = tl.arange(0, KV_BLOCK_SIZE)
    head_offset = tl.arange(0, HEAD_SIZE)
    padding_group_offset = tl.arange(0, PADDED_QUERY_GROUP_SIZE)

    kv_offset = (
        kv_head_idx * stride_kv1
        + block_offset[:, None] * stride_kv2
        + head_offset[None, :] * stride_kv3
    )

    # Load queries.
    q_offset = (
        seq_idx * stride_q0
        + (kv_head_idx * QUERY_GROUP_SIZE + padding_group_offset[:, None]) * stride_q1
        + head_offset[None, :] * stride_q2
    )
    group_mask = padding_group_offset[:, None] < QUERY_GROUP_SIZE
    # q: [PADDED_QUERY_GROUP_SIZE, HEAD_SIZE]
    q = tl.load(q_ptr + q_offset, mask=group_mask, other=0.0)
    q = (q * attn_scale).to(q_ptr.dtype.element_ty)

    m_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32)
    acc = tl.zeros([PADDED_QUERY_GROUP_SIZE, HEAD_SIZE], dtype=tl.float32)

    num_prev_blocks = part_idx * (PARTITION_SIZE // KV_BLOCK_SIZE)
    block_table_base = block_tables_ptr + seq_idx * stride_bt0
    for i in range(0,num_blocks,4):
        block_idx = num_prev_blocks + i
        block_number = tl.load(
            block_table_base + block_idx * stride_bt1
        )
        block_number1 = tl.load(
            block_table_base + (block_idx + 1)* stride_bt1
        )
        block_number2 = tl.load(
            block_table_base + (block_idx + 2)* stride_bt1
        )
        block_number3 = tl.load(
            block_table_base + (block_idx + 3)* stride_bt1
        )

        # Load a key block.
        kv_block_offset = block_number * stride_kv0 + kv_offset
        kv_block_offset1 = block_number1 * stride_kv0 + kv_offset
        kv_block_offset2 = block_number2 * stride_kv0 + kv_offset
        kv_block_offset3 = block_number3 * stride_kv0 + kv_offset
        mask_offset = block_idx * KV_BLOCK_SIZE + block_offset
        mask_offset1 = (block_idx + 1) * KV_BLOCK_SIZE + block_offset
        mask_offset2 = (block_idx + 2) * KV_BLOCK_SIZE + block_offset
        mask_offset3 = (block_idx + 3) * KV_BLOCK_SIZE + block_offset
        kv_mask = mask_offset[:, None] < context_len
        kv_mask1 = mask_offset1[:, None] < context_len
        kv_mask2 = mask_offset2[:, None] < context_len
        kv_mask3 = mask_offset3[:, None] < context_len

        # k: [KV_BLOCK_SIZE, HEAD_SIZE]
        k = tl.load(k_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)
        k1 = tl.load(k_cache_ptr + kv_block_offset1, mask=kv_mask1, other=0.0)
        k2 = tl.load(k_cache_ptr + kv_block_offset2, mask=kv_mask, other=0.0)
        k3 = tl.load(k_cache_ptr + kv_block_offset3, mask=kv_mask1, other=0.0)

        # qk: [PADDED_QUERY_GROUP_SIZE, KV_BLOCK_SIZE]
        if PADDED_QUERY_GROUP_SIZE == 1:
            qk = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
            qk1 = tl.sum(q[:, None, :] * k1[None, :, :], axis=2)
            qk2 = tl.sum(q[:, None, :] * k2[None, :, :], axis=2)
            qk3 = tl.sum(q[:, None, :] * k3[None, :, :], axis=2)
        else:
            qk = tl.dot(q, k.T, out_dtype=tl.float32)
            qk1 = tl.dot(q, k1.T, out_dtype=tl.float32)
            qk2 = tl.dot(q, k2.T, out_dtype=tl.float32)
            qk3 = tl.dot(q, k3.T, out_dtype=tl.float32)

        # qk *= attn_scale
        qk = tl.where(mask_offset < context_len, qk, float("-inf"))
        qk1 = tl.where(mask_offset1 < context_len, qk1, float("-inf"))
        qk2 = tl.where(mask_offset2 < context_len, qk2, float("-inf"))
        qk3 = tl.where(mask_offset3 < context_len, qk3, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_i_new = tl.maximum(m_i_new, tl.max(qk1, axis=1))
        m_i_new = tl.maximum(m_i_new, tl.max(qk2, axis=1))
        m_i_new = tl.maximum(m_i_new, tl.max(qk3, axis=1))

        # p: [PADDED_QUERY_GROUP_SIZE, KV_BLOCK_SIZE]
        p = tl.math.exp2((qk - m_i_new[:, None]) * log2e)
        p1 = tl.math.exp2((qk1 - m_i_new[:, None]) * log2e)
        p2 = tl.math.exp2((qk2 - m_i_new[:, None]) * log2e)
        p3 = tl.math.exp2((qk3 - m_i_new[:, None]) * log2e)
        alpha = tl.math.exp2((m_i - m_i_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLOCK_SIZE, HEAD_SIZE]
        v = tl.load(v_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)
        v1 = tl.load(v_cache_ptr + kv_block_offset1, mask=kv_mask1, other=0.0)
        v2 = tl.load(v_cache_ptr + kv_block_offset2, mask=kv_mask2, other=0.0)
        v3 = tl.load(v_cache_ptr + kv_block_offset3, mask=kv_mask3, other=0.0)

        if PADDED_QUERY_GROUP_SIZE == 1:
            acc += tl.sum(p.T[:, :, None] * v[:, None, :], axis=0)
            acc += tl.sum(p1.T[:, :, None] * v1[:, None, :], axis=0)
            acc += tl.sum(p2.T[:, :, None] * v2[:, None, :], axis=0)
            acc += tl.sum(p3.T[:, :, None] * v3[:, None, :], axis=0)
        else:
            p = p.to(v.dtype)
            p1 = p1.to(v.dtype)
            p2 = p2.to(v.dtype)
            p3 = p3.to(v.dtype)
            acc += tl.dot(p, v, out_dtype=tl.float32)
            acc += tl.dot(p1, v1, out_dtype=tl.float32)
            acc += tl.dot(p2, v2, out_dtype=tl.float32)
            acc += tl.dot(p3, v3, out_dtype=tl.float32)

        l_i = l_i * alpha + tl.sum(p, axis=1) +  tl.sum(p1, axis=1) + tl.sum(p2, axis=1) +  tl.sum(p3, axis=1)
        m_i = m_i_new
    acc = acc / l_i[:, None]

    if USE_PARTITIONING:
        part_offset = (
            (seq_idx * NUM_KV_HEADS + kv_head_idx)
            * max_num_partitions
            * QUERY_GROUP_SIZE
            + part_idx * QUERY_GROUP_SIZE
            + padding_group_offset
        )
        mask = padding_group_offset < QUERY_GROUP_SIZE
        tl.store(m_i_ptr + part_offset, m_i, mask=mask)
        tl.store(l_i_ptr + part_offset, l_i, mask=mask)

    out_offset = seq_idx * stride_o0
    if USE_PARTITIONING:
        out_offset += kv_head_idx * stride_o1
    else:
        out_offset += kv_head_idx * QUERY_GROUP_SIZE * stride_o1
    out_offset += (
        part_idx * stride_o2
        + padding_group_offset[:, None] * stride_o3
        + head_offset[None, :] * stride_o4
    )

    group_mask = padding_group_offset[:, None] < QUERY_GROUP_SIZE
    tl.store(out_ptr + out_offset, acc, mask=group_mask)


@triton.autotune(
    configs=[triton.Config({}, num_warps=warps) for warps in [4, 8, 16]],
    key=["QUERY_GROUP_SIZE", "HEAD_SIZE", "NUM_PARTITIONS", "PARTITION_SIZE"],
)
@triton.jit
def _paged_attn_w_mma_v2_reduce_kernel(
    out_ptr,  # [num_seqs, NUM_KV_HEADS, QUERY_GROUP_SIZE, HEAD_SIZE]
    m_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    l_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    tmp_out_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE, HEAD_SIZE]
    context_lens_ptr,  # [num_seqs]
    max_num_partitions,  # partition stride
    stride_o0,
    stride_o1,
    stride_o2,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PADDED_QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    context_len = tl.load(context_lens_ptr + seq_idx)

    num_partitions = tl.cdiv(context_len, PARTITION_SIZE)
    group_head_offset = (
        tl.arange(0, PADDED_QUERY_GROUP_SIZE)[:, None] * HEAD_SIZE
        + tl.arange(0, HEAD_SIZE)[None, :]
    )
    group_mask = tl.arange(0, PADDED_QUERY_GROUP_SIZE)[:, None] < QUERY_GROUP_SIZE
    if num_partitions == 1:
        tmp_out_offset = (
            seq_idx * NUM_KV_HEADS + kv_head_idx
        ) * max_num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE + group_head_offset
        tmp_out = tl.load(tmp_out_ptr + tmp_out_offset, mask=group_mask, other=0.0)

        out_offset = (
            seq_idx * stride_o0
            + kv_head_idx * QUERY_GROUP_SIZE * stride_o1
            + group_head_offset * stride_o2
        )
        tl.store(out_ptr + out_offset, tmp_out, mask=group_mask)
        return

    # Get the global max logit.
    ml_offset = (
        (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * QUERY_GROUP_SIZE
        + tl.arange(0, NUM_PARTITIONS)[:, None] * QUERY_GROUP_SIZE
        + tl.arange(0, PADDED_QUERY_GROUP_SIZE)[None, :]
    )

    mask = (tl.arange(0, NUM_PARTITIONS)[:, None] < num_partitions) & (
        tl.arange(0, PADDED_QUERY_GROUP_SIZE)[None, :] < QUERY_GROUP_SIZE
    )
    # m_i: [NUM_PARTITIONS, PADDED_QUERY_GROUP_SIZE]
    m_i = tl.load(m_i_ptr + ml_offset, mask=mask, other=float("-inf"))
    # m: [PADDED_QUERY_GROUP_SIZE]
    m = tl.max(m_i, axis=0)

    # Rescale the exp sums and compute the global sum.
    # l_i: [NUM_PARTITIONS, PADDED_QUERY_GROUP_SIZE]
    l_i = tl.load(l_i_ptr + ml_offset, mask=mask, other=0.0)
    l_i *= tl.exp(m_i - m[None, :])
    # l: [PADDED_QUERY_GROUP_SIZE]
    l = tl.sum(l_i, axis=0)
    # r: [NUM_PARTITIONS, PADDED_QUERY_GROUP_SIZE]
    r = l_i / l[None, :]
    r = tl.reshape(r, (NUM_PARTITIONS, PADDED_QUERY_GROUP_SIZE, 1))

    tmp_out_offset = (
        (seq_idx * NUM_KV_HEADS + kv_head_idx)
        * max_num_partitions
        * QUERY_GROUP_SIZE
        * HEAD_SIZE
        + tl.arange(0, NUM_PARTITIONS)[:, None, None] * QUERY_GROUP_SIZE * HEAD_SIZE
        + tl.arange(0, PADDED_QUERY_GROUP_SIZE)[None, :, None] * HEAD_SIZE
        + tl.arange(0, HEAD_SIZE)[None, None, :]
    )
    # tmp_out: [NUM_PARTITIONS, PADDED_QUERY_GROUP_SIZE, HEAD_SIZE]
    tmp_out = tl.load(tmp_out_ptr + tmp_out_offset, mask=mask[:, :, None], other=0.0)
    # out: [PADDED_QUERY_GROUP_SIZE, HEAD_SIZE]
    out = tl.sum((tmp_out * r).to(tl.float32), axis=0)

    out_offset = (
        seq_idx * stride_o0
        + kv_head_idx * QUERY_GROUP_SIZE * stride_o1
        + group_head_offset * stride_o2
    )
    tl.store(out_ptr + out_offset, out, mask=group_mask)


@triton.jit
def _inner_paged_attn_unroll_0_kernel(
    q,
    k_cache,
    v_cache,
    stride_km,
    block_base_ptrs,
    base_offs_kv,
    alibi_slope,
    block_offs,
    seq_len,
    qkv,
    qk_max,
    exp_sum,
    BLOCK_SIZE: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
):
    for block_idx in range(LO, HI, 1):
        offs_kv_0 = tl.load(block_base_ptrs + block_idx + 0) * stride_km + base_offs_kv
        k_0 = tl.load(k_cache + offs_kv_0)
        v_0 = tl.load(v_cache + offs_kv_0)
        _qk_0 = tl.sum((q[None, :] * k_0).to(tl.float32), axis=1)

        if alibi_slope is not None:
            _qk_0 += alibi_slope * (
                (block_idx + 0) * BLOCK_SIZE + block_offs - seq_len + 1
            )

        _qk_max = tl.maximum(tl.max(_qk_0, axis=0), qk_max)
        exp_tmp = tl.exp(_qk_0 - _qk_max)
        _exp_sum = exp_sum * tl.exp(qk_max - _qk_max) + tl.sum(exp_tmp, axis=0)
        qkv_sum_tmp = (tl.exp(_qk_0[:, None] - _qk_max)).to(
            v_cache.dtype.element_ty
        ) * v_0
        qkv = (qkv * (exp_sum * tl.exp(qk_max - _qk_max)) + qkv_sum_tmp) / _exp_sum
        qk_max = _qk_max
        exp_sum = _exp_sum
    return qkv, qk_max, exp_sum


@triton.jit
def _inner_paged_attn_unroll_2_kernel(
    q,
    k_cache,
    v_cache,
    stride_km,
    block_base_ptrs,
    base_offs_kv,
    alibi_slope,
    block_offs,
    seq_len,
    qkv,
    qk_max,
    exp_sum,
    BLOCK_SIZE: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
):
    for block_idx in range(LO, HI, 2):
        offs_kv_0 = tl.load(block_base_ptrs + block_idx + 0) * stride_km + base_offs_kv
        offs_kv_1 = tl.load(block_base_ptrs + block_idx + 1) * stride_km + base_offs_kv

        k_0 = tl.load(k_cache + offs_kv_0)
        k_1 = tl.load(k_cache + offs_kv_1)

        v_0 = tl.load(v_cache + offs_kv_0)
        v_1 = tl.load(v_cache + offs_kv_1)

        _qk_0 = tl.sum((q[None, :] * k_0).to(tl.float32), axis=1)
        _qk_1 = tl.sum((q[None, :] * k_1).to(tl.float32), axis=1)

        if alibi_slope is not None:
            _qk_0 += alibi_slope * (
                (block_idx + 0) * BLOCK_SIZE + block_offs - seq_len + 1
            )
            _qk_1 += alibi_slope * (
                (block_idx + 1) * BLOCK_SIZE + block_offs - seq_len + 1
            )

        _qk_max = tl.maximum(tl.max(_qk_0, axis=0), qk_max)
        _qk_max = tl.maximum(tl.max(_qk_1, axis=0), _qk_max)

        exp_tmp = tl.exp(_qk_0 - _qk_max) + tl.exp(_qk_1 - _qk_max)
        _exp_sum = exp_sum * tl.exp(qk_max - _qk_max) + tl.sum(exp_tmp, axis=0)
        qkv_sum_tmp = (tl.exp(_qk_0[:, None] - _qk_max)).to(
            v_cache.dtype.element_ty
        ) * v_0 + (tl.exp(_qk_1[:, None] - _qk_max)).to(v_cache.dtype.element_ty) * v_1
        qkv = (qkv * (exp_sum * tl.exp(qk_max - _qk_max)) + qkv_sum_tmp) / _exp_sum
        qk_max = _qk_max
        exp_sum = _exp_sum
    return qkv, qk_max, exp_sum


@triton.jit
def _inner_paged_attn_unroll_4_kernel(
    q,
    k_cache,
    v_cache,
    stride_km,
    block_base_ptrs,
    base_offs_kv,
    alibi_slope,
    block_offs,
    seq_len,
    qkv,
    qk_max,
    exp_sum,
    BLOCK_SIZE: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
):
    for block_idx in range(LO, HI, 4):
        offs_kv_0 = tl.load(block_base_ptrs + block_idx + 0) * stride_km + base_offs_kv
        offs_kv_1 = tl.load(block_base_ptrs + block_idx + 1) * stride_km + base_offs_kv
        offs_kv_2 = tl.load(block_base_ptrs + block_idx + 2) * stride_km + base_offs_kv
        offs_kv_3 = tl.load(block_base_ptrs + block_idx + 3) * stride_km + base_offs_kv

        k_0 = tl.load(k_cache + offs_kv_0)
        k_1 = tl.load(k_cache + offs_kv_1)
        k_2 = tl.load(k_cache + offs_kv_2)
        k_3 = tl.load(k_cache + offs_kv_3)

        v_0 = tl.load(v_cache + offs_kv_0)
        v_1 = tl.load(v_cache + offs_kv_1)
        v_2 = tl.load(v_cache + offs_kv_2)
        v_3 = tl.load(v_cache + offs_kv_3)

        _qk_0 = tl.sum((q[None, :] * k_0).to(tl.float32), axis=1)
        _qk_1 = tl.sum((q[None, :] * k_1).to(tl.float32), axis=1)
        _qk_2 = tl.sum((q[None, :] * k_2).to(tl.float32), axis=1)
        _qk_3 = tl.sum((q[None, :] * k_3).to(tl.float32), axis=1)

        if alibi_slope is not None:
            _qk_0 += alibi_slope * (
                (block_idx + 0) * BLOCK_SIZE + block_offs - seq_len + 1
            )
            _qk_1 += alibi_slope * (
                (block_idx + 1) * BLOCK_SIZE + block_offs - seq_len + 1
            )
            _qk_2 += alibi_slope * (
                (block_idx + 2) * BLOCK_SIZE + block_offs - seq_len + 1
            )
            _qk_3 += alibi_slope * (
                (block_idx + 3) * BLOCK_SIZE + block_offs - seq_len + 1
            )

        _qk_max = tl.maximum(tl.max(_qk_0, axis=0), qk_max)
        _qk_max = tl.maximum(tl.max(_qk_1, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_2, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_3, axis=0), _qk_max)

        exp_tmp = (
            tl.exp(_qk_0 - _qk_max)
            + tl.exp(_qk_1 - _qk_max)
            + tl.exp(_qk_2 - _qk_max)
            + tl.exp(_qk_3 - _qk_max)
        )
        _exp_sum = exp_sum * tl.exp(qk_max - _qk_max) + tl.sum(exp_tmp, axis=0)
        qkv_sum_tmp = (
            (tl.exp(_qk_0[:, None] - _qk_max)).to(v_cache.dtype.element_ty) * v_0
            + (tl.exp(_qk_1[:, None] - _qk_max)).to(v_cache.dtype.element_ty) * v_1
            + (tl.exp(_qk_2[:, None] - _qk_max)).to(v_cache.dtype.element_ty) * v_2
            + (tl.exp(_qk_3[:, None] - _qk_max)).to(v_cache.dtype.element_ty) * v_3
        )
        qkv = (qkv * (exp_sum * tl.exp(qk_max - _qk_max)) + qkv_sum_tmp) / _exp_sum
        qk_max = _qk_max
        exp_sum = _exp_sum
    return qkv, qk_max, exp_sum


@triton.jit
def _inner_paged_attn_unroll_8_kernel(
    q,
    k_cache,
    v_cache,
    stride_km,
    block_base_ptrs,
    base_offs_kv,
    alibi_slope,
    block_offs,
    seq_len,
    qkv,
    qk_max,
    exp_sum,
    BLOCK_SIZE: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
):
    for block_idx in range(LO, HI, 8):
        offs_kv_0 = tl.load(block_base_ptrs + block_idx + 0) * stride_km + base_offs_kv
        offs_kv_1 = tl.load(block_base_ptrs + block_idx + 1) * stride_km + base_offs_kv
        offs_kv_2 = tl.load(block_base_ptrs + block_idx + 2) * stride_km + base_offs_kv
        offs_kv_3 = tl.load(block_base_ptrs + block_idx + 3) * stride_km + base_offs_kv
        offs_kv_4 = tl.load(block_base_ptrs + block_idx + 4) * stride_km + base_offs_kv
        offs_kv_5 = tl.load(block_base_ptrs + block_idx + 5) * stride_km + base_offs_kv
        offs_kv_6 = tl.load(block_base_ptrs + block_idx + 6) * stride_km + base_offs_kv
        offs_kv_7 = tl.load(block_base_ptrs + block_idx + 7) * stride_km + base_offs_kv

        k_0 = tl.load(k_cache + offs_kv_0)
        k_1 = tl.load(k_cache + offs_kv_1)
        k_2 = tl.load(k_cache + offs_kv_2)
        k_3 = tl.load(k_cache + offs_kv_3)
        k_4 = tl.load(k_cache + offs_kv_4)
        k_5 = tl.load(k_cache + offs_kv_5)
        k_6 = tl.load(k_cache + offs_kv_6)
        k_7 = tl.load(k_cache + offs_kv_7)

        v_0 = tl.load(v_cache + offs_kv_0)
        v_1 = tl.load(v_cache + offs_kv_1)
        v_2 = tl.load(v_cache + offs_kv_2)
        v_3 = tl.load(v_cache + offs_kv_3)
        v_4 = tl.load(v_cache + offs_kv_4)
        v_5 = tl.load(v_cache + offs_kv_5)
        v_6 = tl.load(v_cache + offs_kv_6)
        v_7 = tl.load(v_cache + offs_kv_7)

        _qk_0 = tl.sum((q[None, :] * k_0).to(tl.float32), axis=1)
        _qk_1 = tl.sum((q[None, :] * k_1).to(tl.float32), axis=1)
        _qk_2 = tl.sum((q[None, :] * k_2).to(tl.float32), axis=1)
        _qk_3 = tl.sum((q[None, :] * k_3).to(tl.float32), axis=1)
        _qk_4 = tl.sum((q[None, :] * k_4).to(tl.float32), axis=1)
        _qk_5 = tl.sum((q[None, :] * k_5).to(tl.float32), axis=1)
        _qk_6 = tl.sum((q[None, :] * k_6).to(tl.float32), axis=1)
        _qk_7 = tl.sum((q[None, :] * k_7).to(tl.float32), axis=1)

        if alibi_slope is not None:
            _qk_0 += alibi_slope * (
                (block_idx + 0) * BLOCK_SIZE + block_offs - seq_len + 1
            )
            _qk_1 += alibi_slope * (
                (block_idx + 1) * BLOCK_SIZE + block_offs - seq_len + 1
            )
            _qk_2 += alibi_slope * (
                (block_idx + 2) * BLOCK_SIZE + block_offs - seq_len + 1
            )
            _qk_3 += alibi_slope * (
                (block_idx + 3) * BLOCK_SIZE + block_offs - seq_len + 1
            )
            _qk_4 += alibi_slope * (
                (block_idx + 4) * BLOCK_SIZE + block_offs - seq_len + 1
            )
            _qk_5 += alibi_slope * (
                (block_idx + 5) * BLOCK_SIZE + block_offs - seq_len + 1
            )
            _qk_6 += alibi_slope * (
                (block_idx + 6) * BLOCK_SIZE + block_offs - seq_len + 1
            )
            _qk_7 += alibi_slope * (
                (block_idx + 7) * BLOCK_SIZE + block_offs - seq_len + 1
            )

        _qk_max = tl.maximum(tl.max(_qk_0, axis=0), qk_max)
        _qk_max = tl.maximum(tl.max(_qk_1, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_2, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_3, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_4, axis=0), qk_max)
        _qk_max = tl.maximum(tl.max(_qk_5, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_6, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_7, axis=0), _qk_max)

        exp_tmp = (
            tl.exp(_qk_0 - _qk_max)
            + tl.exp(_qk_1 - _qk_max)
            + tl.exp(_qk_2 - _qk_max)
            + tl.exp(_qk_3 - _qk_max)
            + tl.exp(_qk_4 - _qk_max)
            + tl.exp(_qk_5 - _qk_max)
            + tl.exp(_qk_6 - _qk_max)
            + tl.exp(_qk_7 - _qk_max)
        )
        _exp_sum = exp_sum * tl.exp(qk_max - _qk_max) + tl.sum(exp_tmp, axis=0)
        qkv_sum_tmp = (
            (tl.exp(_qk_0[:, None] - _qk_max)).to(v_cache.dtype.element_ty) * v_0
            + (tl.exp(_qk_1[:, None] - _qk_max)).to(v_cache.dtype.element_ty) * v_1
            + (tl.exp(_qk_2[:, None] - _qk_max)).to(v_cache.dtype.element_ty) * v_2
            + (tl.exp(_qk_3[:, None] - _qk_max)).to(v_cache.dtype.element_ty) * v_3
            + (tl.exp(_qk_4[:, None] - _qk_max)).to(v_cache.dtype.element_ty) * v_4
            + (tl.exp(_qk_5[:, None] - _qk_max)).to(v_cache.dtype.element_ty) * v_5
            + (tl.exp(_qk_6[:, None] - _qk_max)).to(v_cache.dtype.element_ty) * v_6
            + (tl.exp(_qk_7[:, None] - _qk_max)).to(v_cache.dtype.element_ty) * v_7
        )
        qkv = (qkv * (exp_sum * tl.exp(qk_max - _qk_max)) + qkv_sum_tmp) / _exp_sum
        qk_max = _qk_max
        exp_sum = _exp_sum
    return qkv, qk_max, exp_sum


@triton.autotune(
    configs=[triton.Config({"UNROLL_FACTOR": uf}) for uf in [1, 2, 4, 8]],
    key=[
        "POWER_OF_2_MAX_SEQ_LEN",
        "QUERY_GROUP_SIZE",
        "USE_PARTITIONING",
        "BLOCK_SIZE",
        "HEAD_SIZE",
        "PARTITION_SIZE",
    ],
)
@triton.jit
def _paged_attn_wo_mma_kernel(
    exp_sums,  # [num_seqs, q_heads, max_num_partitions]
    max_logits,  # [num_seqs, q_heads, max_num_partitions]
    out,  # [num_seqs, q_heads, max_num_partitions, head_size]
    q,  # [num_seqs, q_heads, head_size]
    k_cache,  # [num_blocks, kv_heads, block_size, head_size]
    v_cache,  # [num_blocks, kv_heads, block_size, head_size]
    scale,
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens,  # [num_seqs]
    max_num_blocks_per_seq,
    alibi_slopes,  # [q_heads]
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
    QUERY_GROUP_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    POWER_OF_2_MAX_SEQ_LEN: tl.constexpr,
    USE_PARTITIONING: tl.constexpr,
    UNROLL_FACTOR: tl.constexpr,
):
    head_idx = tl.program_id(axis=0)
    kv_head_idx = head_idx // QUERY_GROUP_SIZE
    seq_idx = tl.program_id(axis=1)
    par_idx = tl.program_id(axis=2)
    seq_len = tl.load(seq_lens + seq_idx)

    if par_idx * PARTITION_SIZE >= seq_len:
        return

    num_context_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    if USE_PARTITIONING:
        num_blocks_per_par = PARTITION_SIZE // BLOCK_SIZE
        start_block_idx = par_idx * num_blocks_per_par
        end_block_idx = tl.minimum(
            start_block_idx + num_blocks_per_par, num_context_blocks
        )
    else:
        start_block_idx = 0
        end_block_idx = num_context_blocks

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

    hi_unroll = ((end_block_idx - 1) // UNROLL_FACTOR) * UNROLL_FACTOR
    if UNROLL_FACTOR == 1:
        qkv, qk_max, exp_sum = _inner_paged_attn_unroll_0_kernel(
            q,
            k_cache,
            v_cache,
            stride_km,
            block_base_ptrs,
            base_offs_kv,
            alibi_slope,
            block_offs,
            seq_len,
            qkv,
            qk_max,
            exp_sum,
            BLOCK_SIZE,
            start_block_idx,
            hi_unroll,
        )
    elif UNROLL_FACTOR == 2:
        qkv, qk_max, exp_sum = _inner_paged_attn_unroll_2_kernel(
            q,
            k_cache,
            v_cache,
            stride_km,
            block_base_ptrs,
            base_offs_kv,
            alibi_slope,
            block_offs,
            seq_len,
            qkv,
            qk_max,
            exp_sum,
            BLOCK_SIZE,
            start_block_idx,
            hi_unroll,
        )
    elif UNROLL_FACTOR == 4:
        qkv, qk_max, exp_sum = _inner_paged_attn_unroll_4_kernel(
            q,
            k_cache,
            v_cache,
            stride_km,
            block_base_ptrs,
            base_offs_kv,
            alibi_slope,
            block_offs,
            seq_len,
            qkv,
            qk_max,
            exp_sum,
            BLOCK_SIZE,
            start_block_idx,
            hi_unroll,
        )
    elif UNROLL_FACTOR == 8:
        qkv, qk_max, exp_sum = _inner_paged_attn_unroll_8_kernel(
            q,
            k_cache,
            v_cache,
            stride_km,
            block_base_ptrs,
            base_offs_kv,
            alibi_slope,
            block_offs,
            seq_len,
            qkv,
            qk_max,
            exp_sum,
            BLOCK_SIZE,
            start_block_idx,
            hi_unroll,
        )
    tl.debug_barrier()
    # last iterations must use mask
    for block_idx in range(hi_unroll, end_block_idx):
        physical_block_idx = tl.load(
            block_tables + seq_idx * max_num_blocks_per_seq + block_idx
        )
        mask = block_offs[:, None] < (seq_len - block_idx * BLOCK_SIZE)
        offs_kv = physical_block_idx * stride_km + base_offs_kv

        k = tl.load(k_cache + offs_kv, mask=mask, other=fp16_0)
        v = tl.load(v_cache + offs_kv, mask=mask, other=fp16_0)

        _qk = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        _qk = tl.where(
            block_offs < (seq_len - block_idx * BLOCK_SIZE), _qk, float("-inf")
        )
        _qk += alibi_slope * (block_idx * BLOCK_SIZE + block_offs - seq_len + 1)
        _qk_max = tl.maximum(tl.max(_qk, axis=0), qk_max)

        _exp_sum = exp_sum * tl.exp(qk_max - _qk_max) + tl.sum(
            tl.exp(_qk - _qk_max), axis=0
        )
        qkv = (
            qkv * (exp_sum * tl.exp(qk_max - _qk_max))
            + (tl.exp(_qk[:, None] - _qk_max)) * v
        )
        qkv = qkv / _exp_sum
        qk_max = _qk_max
        exp_sum = _exp_sum

    if USE_PARTITIONING:
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


@triton.autotune(
    configs=[triton.Config({}, num_warps=warps) for warps in [4, 8, 16]],
    key=["HEAD_SIZE", "PADDED_NUM_SPLITS", "PARTITION_SIZE"],
)
@triton.jit
def _paged_attn_wo_mma_v2_reduce_kernel(
    out,  # [num_seqs, num_q_heads, head_size]
    exp_sums,  # [num_seqs, num_q_heads, max_num_partitions]
    max_logits,  # [num_seqs, num_q_heads, max_num_partitions]
    tmp_out,  # [num_seqs, num_q_heads, max_num_partitions, head_size]
    context_lens,  # [num_seqs]
    stride_exp_m,
    stride_exp_n,
    stride_out_m,
    stride_out_n,
    stride_tmp_m,
    stride_tmp_n,
    stride_tmp_k,
    HEAD_SIZE: tl.constexpr,
    PADDED_NUM_SPLITS: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(axis=1)
    head_idx = tl.program_id(axis=0)
    context_len = tl.load(context_lens + seq_idx)

    num_partitions = tl.cdiv(context_len, PARTITION_SIZE)

    max_logit = float("-inf")
    offs_logit = seq_idx * stride_exp_m + head_idx * stride_exp_n

    head_size_offs = tl.arange(0, HEAD_SIZE)
    tmp_out_ptr = seq_idx * stride_tmp_m + head_idx * stride_tmp_n
    out_ptr = seq_idx * stride_out_m + head_idx * stride_out_n + head_size_offs

    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)
    global_exp_sum = tl.zeros([1], dtype=tl.float32)

    logits = tl.load(
        max_logits + offs_logit + tl.arange(0, PADDED_NUM_SPLITS),
        mask=tl.arange(0, PADDED_NUM_SPLITS) < num_partitions,
        other=float("-inf"),
    )
    max_logit = tl.max(logits, axis=0)

    exp_sum = tl.load(
        exp_sums + offs_logit + tl.arange(0, PADDED_NUM_SPLITS),
        mask=tl.arange(0, PADDED_NUM_SPLITS) < num_partitions,
        other=0.0,
    )
    rescaled_exp_sum = exp_sum * tl.exp(logits - max_logit)
    global_exp_sum += tl.sum(rescaled_exp_sum, axis=0)

    tmp = tl.load(
        tmp_out
        + tmp_out_ptr
        + tl.arange(0, PADDED_NUM_SPLITS)[:, None] * stride_tmp_k
        + head_size_offs
    )
    acc += tl.sum(tmp * rescaled_exp_sum[:, None], axis=0)

    inv_sum = 1.0 / (global_exp_sum + 1e-6)
    tl.store(out + out_ptr, acc * inv_sum)
