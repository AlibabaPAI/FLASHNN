import torch
import triton
import triton.language as tl

from flashnn.triton.triton_utils import compile_and_cache_kernels, get_autotune_triton_kernel


def _get_layer_norm_autotune_configs():
    configs = [
        triton.Config({'BLOCK_SIZE': size}, num_warps=warps)
        for size in [64, 128, 256, 512, 1024]
        for warps in [2, 4, 8, 16]
    ]
    return configs


@triton.jit
def _layer_norm_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weight
    B,  # pointer to the bias
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


def layernorm_forward(x, weight, bias, eps):
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.view(-1, x.shape[-1])
    M, N = x_arg.shape
    # construct mean and rstd
    mean = torch.empty(M, dtype=torch.float32, device='cuda')
    rstd = torch.empty(M, dtype=torch.float32, device='cuda')
    # launch kernel
    method_name = "layer_norm_" + str(N)
    kwargs = [x_arg, y, weight, bias, mean, rstd, x_arg.stride(0), N, eps]
    const_kwargs = {}
    if get_autotune_triton_kernel():
        layer_norm = triton.autotune(configs=_get_layer_norm_autotune_configs(), key=['N'])(_layer_norm_kernel)
    else:
        const_kwargs.update({'BLOCK_SIZE': 512})
        layer_norm = _layer_norm_kernel
    grid = (M, 1, 1)
    compile_and_cache_kernels(layer_norm, method_name, grid, kwargs, const_kwargs)

    return y
