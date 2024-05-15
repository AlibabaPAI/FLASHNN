import torch
import triton
import triton.language as tl

from flashnn.triton.triton_utils import compile_and_cache_kernels, get_autotune_triton_kernel


@triton.jit
def triton_gemm_a16w4_weight_kn_to_k2n_packing_kernel(
    weight_ptr,
    new_weight_ptr,
    K,
    N,
    stride_k,
    stride_n,
    stride_ok,
    stride_on,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # [0:2:K, :]
    offs_k_l = pid_k * BLOCK_K + tl.arange(0, BLOCK_K // 2) * 2
    w_ptrs_l = weight_ptr + (offs_k_l[:, None] * stride_k + offs_n[None, :] * stride_n)
    w_vals_l = tl.load(
        w_ptrs_l,
        mask=(offs_k_l[:, None] < K) & (offs_n[None, :] < N),
        other=0,
    )
    w_vals_l = w_vals_l & 0x0F

    # [1:2:K, :]
    offs_k_h = pid_k * BLOCK_K + tl.arange(0, BLOCK_K // 2) * 2 + 1
    w_ptrs_h = weight_ptr + (offs_k_h[:, None] * stride_k + offs_n[None, :] * stride_n)
    w_vals_h = tl.load(
        w_ptrs_h,
        mask=(offs_k_h[:, None] < K) & (offs_n[None, :] < N),
        other=0,
    )
    w_vals_h = w_vals_h.__lshift__(4) & 0xF0

    new_w_vals = w_vals_l | w_vals_h

    offs_k = pid_k * BLOCK_K // 2 + tl.arange(0, BLOCK_K // 2)
    new_w_ptrs = new_weight_ptr + offs_k[:, None] * stride_ok + offs_n[None, :] * stride_on
    tl.store(new_w_ptrs, new_w_vals, mask=(offs_k[:, None] < K // 2) & (offs_n[None, :] < N))


def triton_gemm_a16w4_weight_kn_to_k2n_packing(weights):
    """
    packing `weights` with shape (K x N x int8) into (K//2 x N x int8)
    """
    k, n = weights.shape
    BLOCK_K = 512
    BLOCK_N = 512
    grid = (triton.cdiv(k, BLOCK_K), triton.cdiv(n, BLOCK_N), 1)
    new_weights = torch.empty(k // 2, n, dtype=torch.int8, device=weights.device)
    triton_gemm_a16w4_weight_packing_kernel[grid](
        weights,
        new_weights,
        k,
        n,
        weights.stride(0),
        weights.stride(1),
        new_weights.stride(0),
        new_weights.stride(1),
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
    )
    return new_weights


@triton.jit
def triton_gemm_a16w4_weight_kn_to_n2k_packing_kernel(
    weight_ptr,
    new_weight_ptr,
    K,
    N,
    stride_k,
    stride_n,
    stride_on,
    stride_ok,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    # [:, 0:2:N]
    offs_n_l = pid_n * BLOCK_N + tl.arange(0, BLOCK_N // 2) * 2
    w_ptrs_l = weight_ptr + (offs_k[:, None] * stride_k + offs_n_l[None, :] * stride_n)
    w_vals_l = tl.load(
        w_ptrs_l,
        mask=(offs_k[:, None] < K) & (offs_n_l[None, :] < N),
        other=0,
    )
    w_vals_l = w_vals_l & 0x0F  # K x N/2

    # [:, 1:2:N]
    offs_n_h = pid_n * BLOCK_N + tl.arange(0, BLOCK_N // 2) * 2 + 1
    w_ptrs_h = weight_ptr + (offs_k[:, None] * stride_k + offs_n_h[None, :] * stride_n)
    w_vals_h = tl.load(
        w_ptrs_h,
        mask=(offs_k[:, None] < K) & (offs_n_h[None, :] < N),
        other=0,
    )
    w_vals_h = w_vals_h.__lshift__(4) & 0xF0  # K x N/2

    new_w_vals = w_vals_l | w_vals_h  # K x N/2
    new_w_vals = tl.trans(new_w_vals)  # N/2 x K

    offs_n = pid_n * BLOCK_N // 2 + tl.arange(0, BLOCK_N // 2)
    new_w_ptrs = new_weight_ptr + offs_n[:, None] * stride_on + offs_k[None, :] * stride_ok
    tl.store(new_w_ptrs, new_w_vals, mask=(offs_n[:, None] < N // 2) & (offs_k[None, :] < K))


def triton_gemm_a16w4_weight_kn_to_n2k_packing(weights):
    """
    packing `weights` with shape (K x N x int8) into (N//2 x K x int8)
    """
    k, n = weights.shape
    BLOCK_K = 512
    BLOCK_N = 512
    grid = (triton.cdiv(k, BLOCK_K), triton.cdiv(n, BLOCK_N), 1)
    new_weights = torch.empty(n // 2, k, dtype=torch.int8, device=weights.device)
    triton_gemm_a16w4_weight_kn_to_n2k_packing_kernel[grid](
        weights,
        new_weights,
        k,
        n,
        weights.stride(0),
        weights.stride(1),
        new_weights.stride(0),
        new_weights.stride(1),
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
    )
    return new_weights


@triton.jit
def _swizzle_tile(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


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
    pid_k = tl.program_id(1)  # split_k

    # re-order program ID for better L2 performance
    pid_m, pid_n = _swizzle_tile(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N * 2) // 2
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rbn[:, None] * stride_bn + rk[None, :] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N * 2), dtype=tl.float32)
    _A0 = tl.zeros((1, 1), dtype=A.dtype.element_ty)
    _B0 = tl.zeros((1, 1), dtype=B.dtype.element_ty)
    if add_zero_points:
        offs_zero_points = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
        zero_points_ptrs = zero_points + offs_zero_points
        _ZERO_POINT0 = tl.zeros([1], dtype=zero_points.dtype.element_ty)
        zero_points_vals = tl.load(zero_points_ptrs, mask=offs_zero_points < 2 * N, other=_ZERO_POINT0)
    l_shifter = (1 - tl.arange(0, BLOCK_N * 2) % 2) * 4

    lo, hi = 0, tl.cdiv(K, BLOCK_K * SPLIT_K)
    for k in range(lo, hi):
        k_remaining = K - k * (BLOCK_K * SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_A0)  # M x K
        b = tl.load(B, mask=rk[None, :] < k_remaining, other=_B0)  # 2N x K
        b = (b << l_shifter[:, None]).to(tl.int8).__rshift__(4)
        if add_zero_points:
            b -= zero_points_vals[:, None]
        b = tl.trans(b)  # K x 2N
        b = b.to(A.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)  # M x 2N
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    offs_scale = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    scale_ptrs = scale_b + offs_scale
    _SCALE0 = tl.zeros([1], dtype=scale_b.dtype.element_ty)
    scales = tl.load(scale_ptrs, mask=offs_scale < 2 * N, other=_SCALE0)
    acc *= scales[None, :]
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    mask = (rm < M)[:, None] & (rn < 2 * N)[None, :]
    if add_bias:
        if pid_k == 0:
            offs_bias = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
            bias_ptrs = bias + offs_bias
            _BIAS0 = tl.zeros([1], dtype=bias.dtype.element_ty)
            bias_vals = tl.load(bias_ptrs, mask=offs_bias < 2 * N, other=_BIAS0)
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
    pid_k = tl.program_id(1)  # split_k

    # re-order program ID for better L2 performance
    pid_m, pid_n = _swizzle_tile(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N * 2) // 2
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rbn[None, :] * stride_bn + rk[:, None] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N * 2), dtype=tl.float32)
    _A0 = tl.zeros((1, 1), dtype=A.dtype.element_ty)
    _B0 = tl.zeros((1, 1), dtype=B.dtype.element_ty)
    if add_zero_points:
        zero_points_offs = (pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)) % (2 * N)
        zero_points += zero_points_offs
        _ZERO_POINT0 = tl.zeros([1], dtype=zero_points.dtype.element_ty)

    scale_offs = (pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)) % (2 * N)
    _SCALE0 = tl.zeros([1], dtype=scale_b.dtype.element_ty)
    scale_b += scale_offs

    l_shifter = (1 - tl.arange(0, BLOCK_N * 2) % 2) * 4

    lo, hi = 0, tl.cdiv(K, BLOCK_K * SPLIT_K)
    for k in range(0, hi):
        k_remaining = K - k * (BLOCK_K * SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_A0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=_B0)
        b = (b << l_shifter[None, :]).to(tl.int8).__rshift__(4)
        # dequantize weight
        g_id = k * SPLIT_K + pid_k
        if add_zero_points:
            zero_points_vals = tl.load(zero_points + g_id * stride_zpk)
            b -= zero_points_vals[None, :]
        scales_val = tl.load(scale_b + g_id * stride_scalek)
        b *= scales_val[None, :]
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)  # M x 2N
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc.to(C.dtype.element_ty)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    mask = (rm < M)[:, None] & (rn < 2 * N)[None, :]
    if add_bias:
        if pid_k == 0:
            _BIAS0 = tl.zeros([1], dtype=bias.dtype.element_ty)
            bias_vals = tl.load(bias + rn, mask=rn < 2 * N, other=_BIAS0)
            acc += bias_vals[None, :]
    # Handles write-back with reduction-splitting.
    if SPLIT_K == 1:
        tl.store(C + rm[:, None] * stride_cm + rn[None, :], acc, mask=mask)
    else:
        tl.atomic_add(C + rm[:, None] * stride_cm + rn[None, :], acc, mask=mask, sem='release')


def _init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def _construct_method_name(base_name, is_perchannel, add_zero_points, add_bias, shape_info):
    base_name += "per_channel_" if is_perchannel else "sub_channel_"
    base_name += "asysmmetric_" if add_zero_points else "symmetric_"
    base_name += "bias_" if add_bias else "nobias_"
    base_name += '_'.join(str(i) for i in shape_info)
    return base_name


def _get_a16w4_configs(is_perchannel: bool = True):
    if is_perchannel:
        configs = [
            triton.Config(
                {'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK, 'SPLIT_K': sk, 'GROUP_M': gm},
                num_stages=stages,
                num_warps=warps,
                pre_hook=lambda nargs: nargs['C'].zero_(),
            )
            for BM in [16]
            for BN in [32, 64]
            for BK in [32, 64, 128]
            for stages in [1, 3, 4]
            for warps in [4, 8]
            for sk in [1, 4, 8]
            for gm in [1, 4, 8]
        ]
    else:
        configs = [
            triton.Config(
                {'BLOCK_M': BM, 'BLOCK_N': BN, 'SPLIT_K': sk, 'GROUP_M': gm},
                num_stages=stages,
                num_warps=warps,
                pre_hook=lambda nargs: nargs['C'].zero_(),
            )
            for BM in [16]
            for BN in [32, 64]
            for stages in [1, 3, 4]
            for warps in [4, 8]
            for sk in [4, 8]
            for gm in [1, 4, 8]
        ]
    return configs


def gemm_a16w4_forward(act, quant_w, scale_w, zero_points=None, bias=None, out_ty=torch.float16):
    """
    act: M x K x fp16
    quant_w: N x K x quint4x2
    scale_w: N x fp16 for perchannel;
        S x N x fp16 for subchannel, where S is subchannel size
    zero_points: N x int8 for perchannel;
        S x N x int8 for subchannel
    bias: N x fp16
    """
    scale_w = scale_w.squeeze()

    M, K = act.shape
    N, K = quant_w.shape

    add_bias = True if bias is not None else False
    add_zero_points = True if zero_points is not None else False
    is_perchannel = scale_w.dim() == 1
    out = torch.zeros((M, N * 2), dtype=out_ty, device=act.device)

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
    method_name = _construct_method_name("gemm_a16w4_", is_perchannel, add_zero_points, add_bias, shape_info)
    if get_autotune_triton_kernel():

        def grid(META):
            return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'], 1)

        if is_perchannel:
            a16w4_kernel = triton.autotune(configs=_get_a16w4_configs(is_perchannel=True), key=['M', 'N', 'K'])(
                _triton_gemm_a16w4_per_channel_kernel
            )
        else:
            a16w4_kernel = triton.autotune(configs=_get_a16w4_configs(is_perchannel=False), key=['M', 'N', 'K'])(
                _triton_gemm_a16w4_sub_channel_kernel
            )
    else:
        base_config = {'BLOCK_M': 16, 'BLOCK_N': 32, 'SPLIT_K': 4, 'GROUP_M': 1, "num_stages": 3, "num_warps": 4}
        if is_perchannel:
            base_config.update({"BLOCK_K": 64})
        grid = (
            triton.cdiv(M, base_config['BLOCK_M']) * triton.cdiv(N, base_config['BLOCK_N']),
            base_config['SPLIT_K'],
            1,
        )
        const_kwargs.update(base_config)
        if is_perchannel:
            a16w4_kernel = _triton_gemm_a16w4_per_channel_kernel
        else:
            a16w4_kernel = _triton_gemm_a16w4_sub_channel_kernel

    grid_kwargs = ["BLOCK_M", "BLOCK_N", "SPLIT_K"]
    compile_and_cache_kernels(
        a16w4_kernel, method_name, grid, kwargs, grid_kwargs=grid_kwargs, const_kwargs=const_kwargs
    )

    return out
