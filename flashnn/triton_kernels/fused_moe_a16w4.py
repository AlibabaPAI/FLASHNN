import torch
import triton
import triton.language as tl

from flashnn.triton_kernels.triton_utils import compile_and_cache_kernels


@triton.jit
def _fused_moe_kernel_a16w4_perchannel(
    # Pointers to matrices
    A,
    B,
    C,
    scale_b_ptr,
    zero_points_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `A`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_scale_be,
    stride_scale_bn,
    stride_scale_bk,
    stride_zero_points_e,
    stride_zero_points_n,
    stride_zero_points_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    add_zero_points: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = B + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if add_zero_points:
        offs_zero_points = pid_n * BLOCK_SIZE_N * 2 + tl.arange(0, 2 * BLOCK_SIZE_N)
        zero_points_ptrs = zero_points_ptr + off_experts * stride_zero_points_e + offs_zero_points
        _ZERO_POINT0 = tl.zeros([1], dtype=zero_points_ptr.dtype.element_ty)
        zero_points_vals = tl.load(zero_points_ptrs, mask=offs_zero_points < 2 * N, other=_ZERO_POINT0)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    _A0 = tl.zeros([1, 1], dtype=A.dtype.element_ty)
    _B0 = tl.zeros([1, 1], dtype=B.dtype.element_ty)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N * 2), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=_A0)
        b_int4_two = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=_B0)

        b_int4_l = b_int4_two.__lshift__(4).to(tl.int8).__rshift__(4)
        b_int4_h = b_int4_two.__rshift__(4)
        b = tl.interleave(b_int4_l, b_int4_h).to(A.dtype.element_ty)

        if add_zero_points:
            b -= zero_points_vals[None, :]

        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b, out_dtype=tl.float32)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_scale = pid_n * BLOCK_SIZE_N * 2 + tl.arange(0, BLOCK_SIZE_N * 2)
    scale_ptrs = scale_b_ptr + off_experts * stride_scale_be + offs_scale * stride_scale_bn
    _SCALE0 = tl.zeros([1], dtype=scale_b_ptr.dtype.element_ty)
    scales = tl.load(scale_ptrs, mask=offs_scale < 2 * N, other=_SCALE0)
    accumulator *= scales[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(A.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N * 2 + tl.arange(0, BLOCK_SIZE_N * 2)
    c_ptrs = C + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N * 2)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _fused_moe_kernel_a16w4_subchannel(
    # Pointers to matrices
    A,
    B,
    C,
    scale_b_ptr,
    zero_points_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `A`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_scale_be,
    stride_scale_bn,
    stride_scale_bk,
    stride_zero_points_e,
    stride_zero_points_n,
    stride_zero_points_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    add_zero_points: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = B + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    _A0 = tl.zeros([1, 1], dtype=A.dtype.element_ty)
    _B0 = tl.zeros([1, 1], dtype=B.dtype.element_ty)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N * 2), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=_A0)
        b_int4_two = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=_B0)  # [K x N]

        b_int4_l = b_int4_two.__lshift__(4).to(tl.int8).__rshift__(4)
        b_int4_h = b_int4_two.__rshift__(4)
        b = tl.interleave(b_int4_l, b_int4_h).to(A.dtype.element_ty)  # [K x 2N]

        # dequantize weight
        if add_zero_points:
            offs_zp_n = (pid_n * BLOCK_SIZE_N * 2 + tl.arange(0, 2 * BLOCK_SIZE_N)) % (2 * N)
            _ZERO_POINT0 = tl.zeros([1], dtype=zero_points_ptr.dtype.element_ty)
            offs_zp_k = tl.arange(0, 1)
            zp_ptrs = zero_points_ptr + off_experts * stride_zero_points_e + offs_zp_n * stride_zero_points_n + k
            zero_points_vals = tl.load(zp_ptrs)
            b = b - zero_points_vals

        offs_scale_n = pid_n * BLOCK_SIZE_N * 2 + tl.arange(0, 2 * BLOCK_SIZE_N)
        _SCALE0 = tl.zeros([1], dtype=scale_b_ptr.dtype.element_ty)
        scale_b_ptrs = scale_b_ptr + off_experts * stride_scale_be + offs_scale_n * stride_scale_bn + k
        scales_val = tl.load(scale_b_ptrs, mask=offs_scale_n < 2 * N, other=_SCALE0)
        b = b * scales_val[None, :]

        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b, out_dtype=tl.float32)
        # accumulator *= scales_val[None, :]
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(A.dtype.element_ty)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N * 2 + tl.arange(0, BLOCK_SIZE_N * 2)
    c_ptrs = C + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N * 2)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def fused_moe_a16w4_forward(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    scale_b: torch.Tensor,
    zero_points: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) with A16W4 using token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K),
        where '*' can be any shape representing batches and K is the feature dimension of each token.
    - B: The stacked MOE weight tensor with shape (E, N, K),
        where E is the number of experts, K is the input feature dimension, and N is the output feature dimension.
        It should pack alone N dimension
    - C: The output cache tensor with shape (M, topk, N),
        where M is the total number of tokens post padding, topk is the number of times each token is repeated,
        and N is the output feature dimension.
    - scale_b / zero_points: Tensors that used to dequant int4 B, where dequant_B = (B - zero_points) * scale_b,
        for perchannel case, the shape of scale_b and zero_points is (E, N),
        for subchannel case, the shape of scale_b and zero_points is (E, N, K // channel_size).
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are assigned to.
    - expert_ids: A tensor containing the indices of the expert for each block.
        It determines which expert matrix from B should be used for each block in A.
    This kernel performs the multiplication of a token by its corresponding expert matrix as determined by `expert_ids`.
    The sorting of `sorted_token_ids` by expert index and padding ensures divisibility by BLOCK_SIZE_M,
    which is necessary to maintain consistency in block matrix multiplication across different blocks processed by the same expert.
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert B.shape[1] % 16 == 0 and B.shape[2] % 16 == 0

    add_zero_points = True if zero_points is not None else False
    is_perchannel = scale_b.dim() == 2  # (E, N)

    grid = (
        triton.cdiv(sorted_token_ids.shape[0], config['BLOCK_SIZE_M'])
        * triton.cdiv(B.shape[1], config['BLOCK_SIZE_N']),
        1,
        1,
    )

    kwargs = [
        A,
        B,
        C,
        scale_b,
        zero_points,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),  # E
        B.stride(1),  # N
        B.stride(2),  # K
        C.stride(1),
        C.stride(2),
        scale_b.stride(0),  # E
        scale_b.stride(1),  # N
        scale_b.stride(-1),  # K
    ]

    kwargs += (
        [1, 1, 1] if not add_zero_points else [zero_points.stride(0), zero_points.stride(1), zero_points.stride(-1)]
    )

    const_kwargs = {"MUL_ROUTED_WEIGHT": mul_routed_weight, "top_k": top_k, "num_warps": 4}

    if add_zero_points:
        const_kwargs.update({"add_zero_points": True})
    else:
        const_kwargs.update({"add_zero_points": False})

    if not is_perchannel:
        k_per_scale = B.shape[-1] // scale_b.shape[-1]
        config['BLOCK_SIZE_K'] = k_per_scale

    const_kwargs.update(config)

    method_name = "fuse_moe_a16w4_" + '_'.join(str(value) for value in const_kwargs.values())

    if is_perchannel:
        fuse_moe_a16w4 = _fused_moe_kernel_a16w4_perchannel
    else:
        fuse_moe_a16w4 = _fused_moe_kernel_a16w4_subchannel

    compile_and_cache_kernels(fuse_moe_a16w4, method_name, grid, kwargs, const_kwargs)
