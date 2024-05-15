import triton
import triton.language as tl

from flashnn.module.triton.triton_utils import compile_and_cache_kernels, get_autotune_triton_kernel


@triton.jit
def _triton_gemm_a16w8_per_channel_kernel(
    A,
    B,
    C,
    scale_b,
    bias,
    zero_points,
    M,
    N,
    K,
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
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if add_zero_points:
        offs_zero_points = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        zero_points_ptrs = zero_points + offs_zero_points
        _ZERO_POINT0 = tl.zeros([1], dtype=zero_points.dtype.element_ty)
        zero_points_vals = tl.load(zero_points_ptrs, mask=offs_zero_points < N, other=_ZERO_POINT0)
    _A0 = tl.zeros((1, 1), dtype=A.dtype.element_ty)
    _B0 = tl.zeros((1, 1), dtype=B.dtype.element_ty)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        k_remaining = K - k * (BLOCK_K * SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_A0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=_B0)

        if add_zero_points:
            b = b - zero_points_vals[None, :]

        b_fp = b.to(A.dtype.element_ty)
        acc += tl.dot(a, b_fp, out_dtype=tl.float32, allow_tf32=True)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    offs_scale = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    scale_ptrs = scale_b + offs_scale
    _SCALE0 = tl.zeros([1], dtype=scale_b.dtype.element_ty)
    scales = tl.load(scale_ptrs, mask=offs_scale < N, other=_SCALE0)
    acc *= scales
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if add_bias:
        offs_bias = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_ptrs = bias + offs_bias
        _BIAS0 = tl.zeros([1], dtype=bias.dtype.element_ty)
        bias_vals = tl.load(bias_ptrs, mask=offs_bias < N, other=_BIAS0)
        if pid_z == 0:
            acc += bias_vals[None, :]
    # Handles write-back with reduction-splitting.
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


@triton.jit
def _triton_gemm_a16w8_sub_channel_kernel(
    A,
    B,
    C,
    scale_b,
    bias,
    zero_points,
    M,
    N,
    K,
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
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale_w_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    _SCALE0 = tl.zeros([1], dtype=scale_b.dtype.element_ty)
    _A0 = tl.zeros((1, 1), dtype=A.dtype.element_ty)
    _B0 = tl.zeros((1, 1), dtype=B.dtype.element_ty)
    if add_zero_points:
        _ZERO_POINT0 = tl.zeros([1], dtype=zero_points.dtype.element_ty)
        zero_points_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        k_remaining = K - k * (BLOCK_K * SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_A0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=_B0)
        if add_zero_points:
            zero_points_ptrs = zero_points + (k * SPLIT_K + pid_z) * stride_zpk + zero_points_offs
            zero_points_vals = tl.load(zero_points_ptrs, mask=zero_points_offs < N, other=_ZERO_POINT0)
            b = b - zero_points_vals[None, :]
        scale_ptrs = scale_b + k * SPLIT_K * stride_scalek + pid_z * stride_scalek + scale_w_offs
        scales = tl.load(scale_ptrs, mask=scale_w_offs < N, other=_SCALE0)
        b_fp = b * scales[None, :]
        acc += tl.dot(a, b_fp, out_dtype=tl.float32, allow_tf32=True)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if add_bias:
        offs_bias = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_ptrs = bias + offs_bias
        _BIAS0 = tl.zeros([1], dtype=bias.dtype.element_ty)
        bias_vals = tl.load(bias_ptrs, mask=offs_bias < N, other=_BIAS0)
        if pid_z == 0:
            acc += bias_vals[None, :]
    # Handles write-back with reduction-splitting.
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


def _init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def _construct_method_name(base_name, is_perchannel, add_zero_points, add_bias, shape_info):
    base_name += "per_channel_" if is_perchannel else "sub_channel_"
    base_name += "asysmmetric_" if add_zero_points else "symmetric_"
    base_name += "bias_" if add_bias else "nobias_"
    base_name += '_'.join(str(i) for i in shape_info)
    return base_name


def _get_a16w8_configs(is_perchannel: bool = True):
    configs = []
    if is_perchannel:
        configs.append(
            triton.Config(
                {'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK, 'SPLIT_K': SK, 'GROUP_M': GM},
                num_stages=stages,
                num_warps=warps,
                pre_hook=_init_to_zero('C'),
            )
            for BM in [16, 32]
            for BN in [32, 64, 128, 256]
            for BK in [32, 64, 128, 256]
            for SK in [3, 4, 6]
            for stages in [1, 3, 4, 5]
            for warps in [4, 8, 16]
        )
    else:
        configs.append(
            triton.Config(
                {'BLOCK_M': BM, 'BLOCK_N': BN, 'SPLIT_K': SK, 'GROUP_M': GM},
                num_stages=stages,
                num_warps=warps,
                pre_hook=_init_to_zero('C'),
            )
            for BM in [16, 32]
            for BN in [32, 64, 128, 256]
            for SK in [3, 4, 6]
            for stages in [1, 3, 4, 5]
            for warps in [4, 8, 16]
        )
    return configs


def gemm_a16w8_forward(out, act, quant_w, scale_w, zero_points=None, bias=None):
    scale_w = scale_w.squeeze()

    M, K = act.shape
    N, K = quant_w.shape

    add_bias = True if bias is not None else False
    add_zero_points = True if zero_points is not None else False
    is_perchannel = scale_w.dim() == 1

    kwargs = [
        act,
        quant_w,
        out,
        scale_w,
        bias,
        zero_points,
        M,
        N,
        K,
        act.stride(0),
        act.stride(1),
        quant_w.stride(0),
        quant_w.stride(1),
        out.stride(0),
        out.stride(1),
        zero_points.stride(0) if add_zero_points else 0,
        zero_points.stride(1) if add_zero_points and not is_perchannel else 0,
        0 if is_perchannel else scale_w.stride(0),
        0 if is_perchannel else scale_w.stride(1),
    ]

    const_kwargs = {"add_bias": add_bias, "add_zero_points": add_zero_points}
    if not is_perchannel:
        k_per_scale = int(act.shape[1] / scale_w.shape[0])
        assert k_per_scale > 0, "k_per_scale should greater than 0"
        const_kwargs.update({"BLOCK_K": k_per_scale})
    shape_info = [M, N, K]
    method_name = _construct_method_name("gemm_a16w8_", is_perchannel, add_zero_points, add_bias, shape_info)
    if get_autotune_triton_kernel():

        def grid(META):
            return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'], 1)

        if is_perchannel:
            a16w8_kernel = triton.autotune(configs=_get_a16w8_configs(is_perchannel=True), key=['M', 'N', 'K'])(
                _triton_gemm_a16w8_per_channel_kernel
            )
        else:
            a16w8_kernel = triton.autotune(configs=_get_a16w8_configs(is_perchannel=False), key=['M', 'N', 'K'])(
                _triton_gemm_a16w8_sub_channel_kernel
            )
    else:
        base_config = {'BLOCK_M': 16, 'BLOCK_N': 64, 'SPLIT_K': 4, 'GROUP_M': 1, "num_stages": 1, "num_warps": 4}
        if is_perchannel:
            base_config.update({"BLOCK_K": 64})
        grid = (
            triton.cdiv(M, base_config['BLOCK_M']) * triton.cdiv(N, base_config['BLOCK_N']),
            base_config['SPLIT_K'],
            1,
        )
        const_kwargs.update(base_config)
        if is_perchannel:
            a16w8_kernel = _triton_gemm_a16w8_per_channel_kernel
        else:
            a16w8_kernel = _triton_gemm_a16w8_sub_channel_kernel

    grid_kwargs = ["BLOCK_M", "BLOCK_N", "SPLIT_K"]
    compile_and_cache_kernels(
        a16w8_kernel, method_name, grid, kwargs, grid_kwargs=grid_kwargs, const_kwargs=const_kwargs
    )
