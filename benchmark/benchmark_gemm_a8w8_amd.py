# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import triton
import triton.language as tl
import flashnn
import pytest

def num_tensors(M, N, K):
    size = M * N + M * K + N * K + M + N
    total_size = 512 * 1024 * 1024
    num = triton.cdiv(total_size, size)
    return num 


NK_shapes = []
for hidden_size, intermediate_size, tp, num_attention_heads, num_key_value_heads in [
    (13312, 71168, 8, 104, 8),
    (13312, 71168, 4, 104, 4),
    (8192, 29568, 4, 64, 8),
    (8192, 29568, 2, 64, 8),
    (3584, 18944, 1, 28, 4),
]:
    GQA = num_attention_heads // num_key_value_heads
    NK_shapes += [
        (hidden_size, intermediate_size // tp), # FFN Layer1
        ((intermediate_size // tp) * 2, hidden_size), # FFN Layer2
        (hidden_size, hidden_size // tp), # Attn output
        ((hidden_size // tp) + ((hidden_size // tp) // GQA) * 2, hidden_size), # QKV Projection
    ]


configs = []
for N, K in NK_shapes:
    configs.append(
        triton.testing.Benchmark(
            x_names=["M"],
            x_vals=[1, 10, 20, 30, 40, 764, 1024, 2048, 4096, 4096 * 2],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="TFLOPS",
            args={"N": N, "K": K},
            plot_name=f"gemm-a8w8-N_{N}-K{K}_TFLOPS",
        )
    )


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):

    tensor_num = num_tensors(M, N, K)
    a = []
    b = []
    alpha_row = []
    alpha_col = []
    for _ in range(tensor_num):
        a_tmp = torch.randint(-128, 127, (M, K), dtype=torch.int8).cuda()
        b_tmp = torch.randint(-128, 127, (N, K), dtype=torch.int8).cuda()
        alpha_row_tmp = torch.rand([M, 1], dtype=torch.half).cuda()
        alpha_col_tmp = torch.rand([1, N], dtype=torch.half).cuda()

        a.append(a_tmp)
        b.append(b_tmp)
        alpha_row.append(alpha_row_tmp)
        alpha_col.append(alpha_col_tmp)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'triton':
        flashnn.set_use_triton(True)
        flashnn.set_autotune_triton_kernels(True)
        triton_gemm_a8w8 = flashnn.GemmA8W8(out_ty=torch.half)
        ms, min_ms, max_ms = triton.testing.do_bench_rotating_tensor(
            lambda i: triton_gemm_a8w8(a[i % tensor_num], b[i % tensor_num].T, alpha_row[i % tensor_num], alpha_col[i % tensor_num]), rep=100, quantiles=quantiles
        )
    if provider == 'torch':
        flashnn.set_use_triton(False)
        torch_gemm_a8w8 = flashnn.GemmA8W8(out_ty=torch.half)
        ms, min_ms, max_ms = triton.testing.do_bench_rotating_tensor(
            lambda i: torch_gemm_a8w8(a[i % tensor_num], b[i % tensor_num], alpha_row[i % tensor_num], alpha_col[i % tensor_num]), rep=100, quantiles=quantiles
        )

    tflops = lambda x: round(2 * M * N * K / x * 1e-9, 2)
    # tflops = lambda x : x
    return tflops(ms), tflops(min_ms), tflops(max_ms)


if __name__ == "__main__":
    benchmark.run(show_plots=False, print_data=True)

def get_shapes():
    shapes = []
    for n, k, in NK_shapes:
        for m in [1, 10, 20, 30, 40, 764, 1024, 2048, 4096, 4096 * 2]:
            shapes.append((m, n, k))
    return shapes

@pytest.mark.parametrize('m, n, k', get_shapes())
def test_gemm_a8w8(m, n, k):
    torch.random.manual_seed(0)
    with torch.no_grad():
        a = torch.randint(-12, 12, (m, k), dtype=torch.int8).cuda()
        b = torch.randint(-12, 12, (n, k), dtype=torch.int8).cuda()

        alpha_row = torch.rand([m, 1], dtype=torch.half).cuda()
        alpha_col = torch.rand([1, n], dtype=torch.half).cuda()

        torch_gemm_a8w8 = flashnn.GemmA8W8(out_ty=torch.half)
        out_torch = torch_gemm_a8w8(a, b, alpha_row=alpha_row, alpha_col=alpha_col)

        flashnn.set_use_triton(True)
        flashnn.set_autotune_triton_kernels(True)
        triton_gemm_a8w8 = flashnn.GemmA8W8(out_ty=torch.half)
        out_triton = triton_gemm_a8w8(a, b, alpha_row, alpha_col)

        diff = ~np.isclose(out_triton.half().cpu().numpy(), out_torch.half().cpu().numpy(), rtol=1e-2)
        assert diff.sum() < 10, f"m={m}, n={n}, k={k}"