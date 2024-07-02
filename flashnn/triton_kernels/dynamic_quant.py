# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import triton
import triton.language as tl
from flashnn.kernel_backend import THREADS_PER_WARP


@triton.jit
def _abs_max(val1, val2):
    val1_abs = tl.abs(val1)
    val2_abs = tl.abs(val2)
    if val1_abs >= val2_abs:
        return val1_abs
    else:
        return val2_abs


@triton.jit
def _triton_dynamic_quantize_kernel(
    output_ptr,
    input_ptr,
    scale_ptr,
    stride_outputm,
    stride_outputn,
    stride_inputm,
    stride_inputn,
    n_elements,
    M: tl.constexpr,
    N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, N)
    mask = offsets < n_elements
    input_ptrs = input_ptr + pid * stride_inputm + offsets
    input_vals = tl.load(input_ptrs, mask=mask, other=1e-6)
    abs_max_f = tl.reduce(input_vals, 0, _abs_max)
    dynamic_per_token_scale = 127.0 / abs_max_f
    precison_mask = tl.where(input_vals > 0, 0.5, -0.5)
    output_vals = (input_vals * dynamic_per_token_scale + precison_mask).to(tl.int8)
    output_ptrs = output_ptr + pid * stride_outputm + offsets
    tl.store(output_ptrs, output_vals, mask=mask)
    tl.store(scale_ptr + pid, abs_max_f / 127.0)


def triton_dynamic_quantize(out, input, scale):
    assert input.is_contiguous(), "input must be contiguous"
    num_tokens = input.size(0)
    hidden_size = input.size(1)
    block_size = 1024
    # tl.reduce requires the number of elements
    # must be power-of-two
    if hidden_size & (hidden_size - 1) == 0 and hidden_size > 0:
        block_size = min(hidden_size / 2, block_size)
    else:
        hidden_size = triton.next_power_of_2(int(hidden_size))
        block_size = min(hidden_size / 2, block_size)
    num_warps = int(max(block_size / THREADS_PER_WARP, 1))
    _triton_dynamic_quantize_kernel[(num_tokens,)](
        out,
        input,
        scale,
        out.stride(0),
        out.stride(1),
        input.stride(0),
        input.stride(1),
        n_elements=input.size(1),
        M=num_tokens,
        N=hidden_size,
        num_warps=num_warps,
    )
