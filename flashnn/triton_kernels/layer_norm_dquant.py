# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import torch
import triton
import triton.language as tl
from flashnn.kernel_backend import get_autotune_triton_kernels


def _get_autotune_configs(enable_auto_tune: bool = False):
    if get_autotune_triton_kernels():
        configs = [
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=16),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=16),
        ]
    else:
        configs = [
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        ]
    return configs


@triton.jit
def _layer_norm_dquant_kernel(
    X,  # pointer to the input
    Y,  # pointer to the normed output
    W,  # pointer to the weight
    B,  # pointer to the bias
    out,  # pointer to the output
    scale,  # pointer to the scale
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    out += row * stride

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

    _max_x = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _norm = (x - mean) * rstd * w + b
        tl.store(out + cols, _norm, mask=mask)
        _max_x = tl.maximum(_max_x, tl.max(tl.abs(_norm), axis=0))
    scale_x = _max_x / 127.0
    tl.store(scale + row, scale_x)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        _norm = tl.load(out + cols, mask=mask, other=0.0)
        _norm = _norm / scale_x + 0.5
        tl.store(Y + cols, _norm.to(tl.int8), mask=mask)


def triton_layer_norm_dquant_forward(x, weight, bias, eps):
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    scale = torch.empty((M,), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    y = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    # launch kernel
    kwargs = [x_arg, y, weight, bias, out, scale, x_arg.stride(0), N, eps]
    layer_norm_dquant = triton.autotune(configs=_get_autotune_configs(), key=["N"])(
        _layer_norm_dquant_kernel
    )
    grid = (M, 1, 1)
    layer_norm_dquant[(M,)](*kwargs)

    return out, y, scale
