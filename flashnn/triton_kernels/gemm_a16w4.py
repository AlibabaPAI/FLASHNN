# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import torch
import triton
import triton.language as tl
from flashnn.kernel_backend import get_autotune_triton_kernels


def _init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def _get_a16w4_configs(is_perchannel: bool = True):
    configs = []
    for block_m in [16]:
        for block_n in [32, 64, 128]:
            for num_warps in [4, 8, 16]:
                for split_k in [4, 8]:
                    for group_m in [1, 4]:
                        if is_perchannel:
                            for block_k in [32, 64, 128]:
                                block_tile = {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                    "SPLIT_K": split_k,
                                    "GROUP_M": group_m,
                                }
                                configs.append(
                                    triton.Config(
                                        block_tile,
                                        num_stages=1,
                                        num_warps=num_warps,
                                        pre_hook=_init_to_zero("C"),
                                    )
                                )
                        else:
                            block_tile = {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "SPLIT_K": split_k,
                                "GROUP_M": group_m,
                            }
                            configs.append(
                                triton.Config(
                                    block_tile,
                                    num_stages=1,
                                    num_warps=num_warps,
                                    pre_hook=_init_to_zero("C"),
                                )
                            )
    return configs


def _get_autotune_configs(is_perchannel: bool = False):
    if get_autotune_triton_kernels():
        return _get_a16w4_configs(is_perchannel)
    else:
        if is_perchannel:
            block_tile = {
                "BLOCK_M": 16,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "SPLIT_K": 4,
                "GROUP_M": 1,
            }
        else:
            block_tile = {
                "BLOCK_M": 16,
                "BLOCK_N": 64,
                "SPLIT_K": 4,
                "GROUP_M": 1,
            }
        return [
            triton.Config(
                block_tile, num_stages=1, num_warps=4, pre_hook=_init_to_zero("C")
            )
        ]


@triton.jit
def _triton_gemm_a16w4_per_channel_kernel(
    A,
    B,
    C,
    scale_b,
    bias,
    zero_points,
    M,
    N,
    K,
    rescale_m,
    rescale_n,
    rescale_k,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_zpk,
    stride_zpn,
    stride_scalek,
    stride_scalen,
    add_bias: tl.constexpr,
    add_zero_points: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # for split k
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rbn[:, None] * stride_bn + rk[None, :] * stride_bk)

    acc_l = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    acc_h = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    _A0 = tl.zeros((1, 1), dtype=A.dtype.element_ty)
    _B0 = tl.zeros((1, 1), dtype=B.dtype.element_ty)

    if add_zero_points:
        offs_zero_points = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
        zero_points_ptrs = zero_points + offs_zero_points
        _ZERO_POINT0 = tl.zeros([1], dtype=zero_points.dtype.element_ty)
        zero_points_vals = tl.load(
            zero_points_ptrs, mask=offs_zero_points < 2 * N, other=_ZERO_POINT0
        )
        zero_points_vals = tl.reshape(zero_points_vals, (BLOCK_N, 2))
        (zp_l, zp_h) = tl.split(zero_points_vals)
    offs_scale = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    scale_ptrs = scale_b + offs_scale
    _SCALE0 = tl.zeros([1], dtype=scale_b.dtype.element_ty)
    scales = tl.load(scale_ptrs, mask=offs_scale < 2 * N, other=_SCALE0)
    # decompose
    #     A dot (B - zero_points) * scales
    # into
    #     ((A dot B) - (A dot zero_points)) * scales
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        k_remaining = K - k * (BLOCK_K * SPLIT_K)

        b_int4_two = tl.load(B, mask=rk[None, :] < k_remaining, other=_B0)

        b_int4_l = (
            b_int4_two.__lshift__(4).to(tl.int8).__rshift__(4).to(A.dtype.element_ty)
        )
        b_int4_h = b_int4_two.__rshift__(4).to(A.dtype.element_ty)

        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_A0)  # M x K
        a = tl.trans(a)

        if add_zero_points:
            b_int4_l -= zp_l[:, None]
            b_int4_h -= zp_h[:, None]

        acc_l += tl.dot(b_int4_l, a, out_dtype=tl.float32, allow_tf32=True)  # M x N
        acc_h += tl.dot(b_int4_h, a, out_dtype=tl.float32, allow_tf32=True)  # M x N

        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc_l = tl.trans(acc_l)
    acc_h = tl.trans(acc_h)
    acc = tl.interleave(acc_l, acc_h)  # M x 2N

    offs_scale = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    scale_ptrs = scale_b + offs_scale
    _SCALE0 = tl.zeros([1], dtype=scale_b.dtype.element_ty)
    scales = tl.load(scale_ptrs, mask=offs_scale < 2 * N, other=_SCALE0)
    acc *= scales[None, :]

    acc = acc.to(C.dtype.element_ty)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    mask = (rm < M)[:, None] & (rn < 2 * N)[None, :]
    if add_bias:
        offs_bias = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
        bias_ptrs = bias + offs_bias
        _BIAS0 = tl.zeros([1], dtype=bias.dtype.element_ty)
        bias_vals = tl.load(bias_ptrs, mask=offs_bias < 2 * N, other=_BIAS0)
        if pid_z == 0:
            acc += bias_vals[None, :]
    # Handles write-back with reduction-splitting.
    if SPLIT_K == 1:
        tl.store(C + rm[:, None] * stride_cm + rn[None, :], acc, mask=mask)
    else:
        tl.atomic_add(C + rm[:, None] * stride_cm + rn[None, :], acc, mask=mask)


@triton.jit
def _triton_gemm_a16w4_sub_channel_kernel(
    A,  # activation
    B,  # quant weight
    C,  # output
    scale_b,  # deqant scales
    bias,
    zero_points,
    M,
    N,
    K,
    rescale_m,
    rescale_n,
    rescale_k,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_zpk,
    stride_zpn,
    stride_scalek,
    stride_scalen,
    add_bias: tl.constexpr,
    add_zero_points: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # for split k
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rbn[:, None] * stride_bn + rk[None, :] * stride_bk)

    acc_l = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    acc_h = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    _A0 = tl.zeros((1, 1), dtype=A.dtype.element_ty)
    _B0 = tl.zeros((1, 1), dtype=B.dtype.element_ty)

    if add_zero_points:
        zero_points_offs = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
        _ZERO_POINT0 = tl.zeros([1], dtype=zero_points.dtype.element_ty)

    scale_offs = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    _SCALE0 = tl.zeros([1], dtype=scale_b.dtype.element_ty)

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        k_remaining = K - k * (BLOCK_K * SPLIT_K)

        b_int4_two = tl.load(B, mask=rk[None, :] < k_remaining, other=_B0)  # N x K

        b_int4_l = b_int4_two.__lshift__(4).to(tl.int8).__rshift__(4)
        b_int4_h = b_int4_two.__rshift__(4)

        # dequantize weight
        if add_zero_points:
            zero_points_ptrs = (
                zero_points
                + k * SPLIT_K * stride_zpk
                + pid_z * stride_zpk
                + zero_points_offs
            )
            zero_points_vals = tl.load(
                zero_points_ptrs, mask=zero_points_offs < 2 * N, other=_ZERO_POINT0
            )
            zero_points_vals = tl.reshape(zero_points_vals, (BLOCK_N, 2))
            (zp_l, zp_h) = tl.split(zero_points_vals)
            b_int4_l -= zp_l[:, None]
            b_int4_h -= zp_h[:, None]
        scales_val = tl.load(
            scale_b + k * SPLIT_K * stride_scalek + pid_z * stride_scalek + scale_offs,
            mask=scale_offs < 2 * N,
            other=_SCALE0,
        )
        scales_val = tl.reshape(scales_val, (BLOCK_N, 2))
        (scale_l, scale_h) = tl.split(scales_val)
        b_int4_l = b_int4_l * scale_l[:, None]
        b_int4_h = b_int4_h * scale_h[:, None]

        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_A0)
        a = tl.trans(a)

        acc_l += tl.dot(b_int4_l, a, out_dtype=tl.float32, allow_tf32=True)  # N x M
        acc_h += tl.dot(b_int4_h, a, out_dtype=tl.float32, allow_tf32=True)

        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    acc_l = tl.trans(acc_l)
    acc_h = tl.trans(acc_h)
    acc = tl.interleave(acc_l, acc_h)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    mask = (rm < M)[:, None] & (rn < 2 * N)[None, :]
    if add_bias:
        offs_bias = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
        bias_ptrs = bias + offs_bias
        _BIAS0 = tl.zeros([1], dtype=bias.dtype.element_ty)
        bias_vals = tl.load(bias_ptrs, mask=offs_bias < 2 * N, other=_BIAS0)
        if pid_z == 0:
            acc += bias_vals[None, :]
    # Handles write-back with reduction-splitting.
    if SPLIT_K == 1:
        tl.store(C + rm[:, None] * stride_cm + rn[None, :], acc, mask=mask)
    else:
        tl.atomic_add(C + rm[:, None] * stride_cm + rn[None, :], acc, mask=mask)


def triton_gemm_a16w4_forward(out, act, quant_w, scale_w, bias=None, zero_points=None):
    assert quant_w.dtype == torch.int8, "Weight must be int8 type"
    assert act.is_contiguous(), "Activation must be contiguous"
    assert quant_w.is_contiguous(), "Weight must be contiguous"
    assert act.shape[1] == quant_w.shape[1], "Matrix B must be transposed"

    scale_w = scale_w.squeeze()

    M, K = act.shape
    N, K = quant_w.shape

    add_bias = True if bias is not None else False
    add_zero_points = True if zero_points is not None else False
    is_perchannel = scale_w.dim() == 1

    rescale_m = M // 16
    rescale_n = N // 512
    rescale_k = K // 512

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            META["SPLIT_K"],
        )

    kwargs = {
        "A": act,
        "B": quant_w,
        "C": out,
        "scale_b": scale_w,
        "bias": bias,
        "zero_points": zero_points,
        "M": M,
        "N": N,
        "K": K,
        "rescale_m": rescale_m,
        "rescale_n": rescale_n,
        "rescale_k": rescale_k,
        "stride_am": act.stride(0),
        "stride_ak": act.stride(1),
        "stride_bn": quant_w.stride(0),
        "stride_bk": quant_w.stride(1),
        "stride_cm": out.stride(0),
        "stride_cn": out.stride(1),
        "stride_zpk": zero_points.stride(0) if add_zero_points else 0,
        "stride_zpn": zero_points.stride(1)
        if add_zero_points and not is_perchannel
        else 0,
        "stride_scalek": 0 if is_perchannel else scale_w.stride(0),
        "stride_scalen": 0 if is_perchannel else scale_w.stride(1),
        "add_bias": add_bias,
        "add_zero_points": add_zero_points,
    }
    # per channel a16w4
    if scale_w.dim() == 1:
        triton_gemm_a16w4_per_channel = triton.autotune(
            configs=_get_autotune_configs(is_perchannel),
            key=["M", "N", "K"],
        )(_triton_gemm_a16w4_per_channel_kernel)
        triton_gemm_a16w4_per_channel[grid](**kwargs)
    # sub channel a16w4
    else:
        k_per_scale = int(act.shape[1] / scale_w.shape[0])
        assert k_per_scale > 0, "k_per_scale should greater than 0"
        triton_gemm_a16w4_sub_channel = triton.autotune(
            configs=_get_autotune_configs(is_perchannel),
            key=["M", "N", "K"],
        )(_triton_gemm_a16w4_sub_channel_kernel)
        triton_gemm_a16w4_sub_channel[grid](BLOCK_K=k_per_scale, **kwargs)

    return out
