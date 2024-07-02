# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import torch
import triton
import triton.language as tl
from flashnn.kernel_backend import get_autotune_triton_kernels


def _get_autotune_configs():
    if get_autotune_triton_kernels():
        a8w8_configs = [
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_SIZE_M": 1},
                num_stages=2,
                num_warps=4,
            ),
        ]
        return a8w8_configs
    else:
        return [
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_SIZE_M": 1},
                num_stages=2,
                num_warps=4,
            )
        ]


@triton.jit
def _triton_gemm_a8w8_kernel(
    # Pointers to matrices
    A,
    B,
    C,
    alpha_row_ptr,
    alpha_col_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul
        out <- ((int8)A[m, k] * (int8)B[n, k]) *
               ((fp16)scale_row[m, 1] * (fp16)scale_col[1, n])
    A has shape (M, K), B has shape (N, K) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :])
    B = B + (rbn[:, None] * stride_bn + rk[None, :])
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    _0 = tl.zeros([1, 1], dtype=A.dtype.element_ty)
    accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        k_remaining = K - k * BLOCK_K
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
        b = tl.load(B, mask=rk[None, :] < k_remaining, other=_0)
        b = tl.trans(b)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # -----------------------------------------------------------
    # `alpha_row_ptrs` is a block of [BLOCK_M] pointers
    # `alpha_col_ptrs` is a block of [BLOCK_N] pointers
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    alpha_row_ptrs = alpha_row_ptr + offs_cm
    alpha_col_ptrs = alpha_col_ptr + offs_cn
    _ALPHA0 = tl.zeros([1], dtype=alpha_row_ptr.dtype.element_ty)
    alpha_row = tl.load(alpha_row_ptrs, mask=offs_cm < M, other=_ALPHA0).to(tl.float32)
    alpha_col = tl.load(alpha_col_ptrs, mask=offs_cn < N, other=_ALPHA0).to(tl.float32)
    accumulator = accumulator * alpha_row[:, None]
    accumulator = accumulator * alpha_col[None, :]
    c = accumulator.to(C.dtype.element_ty)

    # Write back the block of the output matrix C with masks.
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + offs_cn[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def triton_gemm_a8w8_forward(out, a, b, alpha_row, alpha_col):
    # Check constraints.
    assert (
        a.dtype == torch.int8 and b.dtype == torch.int8
    ), "Matrix A/B must be int8 type"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert (
        out.dtype == torch.float16 or out.dtype == torch.bfloat16
    ), "Output type must be float16 or bfloat16"
    assert (
        out.dtype == alpha_row.dtype and out.dtype == alpha_col.dtype
    ), "Output type must match scale type"
    assert a.shape[1] == b.shape[1], "Matrix B must be transposed"
    M, K = a.shape
    N, K = b.shape

    method_name = "gemm_a8w8_" + str(M) + "_" + str(N) + "_" + str(K)
    kwargs = [
        a,
        b,
        out,
        torch.squeeze(alpha_row),
        torch.squeeze(alpha_col),
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
    ]
    gemm_a8w8 = triton.autotune(configs=_get_autotune_configs(), key=["M", "N", "K"])(
        _triton_gemm_a8w8_kernel
    )

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), 1, 1)

    gemm_a8w8[grid](*kwargs)
