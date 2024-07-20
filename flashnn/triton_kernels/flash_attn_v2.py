"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import torch
import triton
import triton.language as tl

from flashnn.kernel_backend import get_autotune_triton_kernels


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if STAGE != 1:
            k = tl.load(K_block_ptr, boundary_check=(0, 1))
        else:
            k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if STAGE != 1:
            n_ctx_mask = tl.where(
                (offs_m[:, None] < N_CTX) & ((start_n + offs_n[None, :]) < N_CTX),
                0,
                float("-inf"),
            )
            qk += n_ctx_mask
        qk += tl.dot(q, k)
        if STAGE == 2:
            # TODO: support more kind of mask
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        if STAGE != 1:
            v = tl.load(V_block_ptr, boundary_check=(0, 1))
        else:
            v = tl.load(V_block_ptr)
        acc = tl.dot(p.to(tl.float16), v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


def _get_flash_attn_autotune_configs():
    configs = [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
        for BM in [64, 128]
        for BN in [64, 128]
        for s in ([1] if torch.version.hip is not None else [3, 4])
        for w in [4, 8]
    ]

    return configs


@triton.jit
def _triton_attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_km,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vm,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    POWER_OF_2_N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    GROUPS: tl.constexpr,
    ORDER_12: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_g = tl.program_id(2)
    q_offset = (
        off_z.to(tl.int64) * stride_qz
        + (off_h * GROUPS + off_g).to(tl.int64) * stride_qh
    )
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = (
        off_z.to(tl.int64) * stride_oz
        + (off_h * GROUPS + off_g).to(tl.int64) * stride_oh
    )
    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vm, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_km),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if ORDER_12:
        # stage 1: off-band
        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                start_m,
                qk_scale,
                BLOCK_M,
                BLOCK_DMODEL,
                BLOCK_N,
                4 - STAGE,
                offs_m,
                offs_n,
                N_CTX,
            )
        # stage 2: on-band
        if STAGE & 2:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                start_m,
                qk_scale,
                BLOCK_M,
                BLOCK_DMODEL,
                BLOCK_N,
                2,
                offs_m,
                offs_n,
                N_CTX,
            )
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            # tl.debug_barrier()
    else:
        # stage 2: on-band
        if STAGE & 2:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                start_m,
                qk_scale,
                BLOCK_M,
                BLOCK_DMODEL,
                BLOCK_N,
                2,
                offs_m,
                offs_n,
                N_CTX,
            )
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            # tl.debug_barrier()
        # stage 1: off-band
        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                start_m,
                qk_scale,
                BLOCK_M,
                BLOCK_DMODEL,
                BLOCK_N,
                4 - STAGE,
                offs_m,
                offs_n,
                N_CTX,
            )
    # epilogue
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))


def triton_flash_attention_forward(q, k, v, causal, sm_scale=None, ORDER_12=False):
    # layout constraints
    q_dim, k_dim, v_dim = q.dim(), k.dim(), v.dim()
    # [batch_size, seq_len, num_heads, head_dims] or
    assert q_dim == 4 and q_dim == k_dim and q_dim == v_dim
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    # num_heads constraints
    num_heads_q = q.shape[2]
    num_heads_k = k.shape[2]
    num_heads_v = v.shape[2]
    assert num_heads_k == num_heads_v
    assert num_heads_q % num_heads_k == 0
    groups = num_heads_q // num_heads_k

    o = torch.empty_like(q)
    BLOCK_M = 64
    BLOCK_N = 64
    num_stages = 2
    num_warps = 8
    stage = 3 if causal else 1

    # layout information
    batch_size = q.shape[0]
    seq_len = q.shape[1]
    head_dims = q.shape[-1]

    sm_scale = 1.0 / Lk**0.5 if sm_scale is None else sm_scale

    kwargs = [
        q,
        k,
        v,
        sm_scale,
        o,
        q.stride(0),
        q.stride(-2),
        q.stride(1),
        q.stride(-1),
        k.stride(0),
        k.stride(-2),
        k.stride(1),
        k.stride(-1),
        v.stride(0),
        v.stride(-2),
        v.stride(1),
        v.stride(-1),
        o.stride(0),
        o.stride(-2),
        o.stride(1),
        o.stride(-1),
        batch_size,
        num_heads_k,
        seq_len,
    ]
    POWER_OF_2_N_CTX = triton.next_power_of_2(seq_len)
    const_kwargs = {
        "POWER_OF_2_N_CTX": POWER_OF_2_N_CTX,
        "BLOCK_DMODEL": Lk,
        "STAGE": stage,
        "GROUPS": groups,
        "ORDER_12": ORDER_12,
    }

    method_name = "flash_attn_v2_" + "_".join(
        str(value) for value in const_kwargs.values()
    )

    if get_autotune_triton_kernels():

        def grid(META):
            return (
                triton.cdiv(seq_len, META["BLOCK_M"]),
                batch_size * num_heads_k,
                groups,
            )

        def keep(conf):
            BLOCK_M = conf.kwargs["BLOCK_M"]
            BLOCK_N = conf.kwargs["BLOCK_N"]
            if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
                return False
            return True

        flash_attn = triton.autotune(
            configs=list(filter(keep, _get_flash_attn_autotune_configs())),
            key=["POWER_OF_2_N_CTX"],
        )(_triton_attn_fwd)
    else:
        base_config = {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "num_stages": num_stages,
            "num_warps": num_warps,
        }
        grid = (
            triton.cdiv(seq_len, base_config["BLOCK_M"]),
            batch_size * num_heads_k,
            groups,
        )
        const_kwargs.update(base_config)
        flash_attn = _triton_attn_fwd
    flash_attn[grid](*kwargs, **const_kwargs)
    return o
