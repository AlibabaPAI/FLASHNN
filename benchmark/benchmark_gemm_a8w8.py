import numpy as np
import torch
import triton
import triton.language as tl
import flashnn

 
configs = []
for N, K in [
    (13312, 8896),
    (17792, 13312),
    (1920, 13312),
    (13312, 1664),
    (35584, 13312),
    (3584, 10240),
    (13312, 3328),
]:
    configs.append(
        triton.testing.Benchmark(
            x_names=['M'],
            x_vals=[1, 10, 20, 30, 40, 764, 1024, 2048, 4096, 4096 * 2],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['Triton', "Torch"],
            styles=[('blue', '-'), ('red', '-')],
            ylabel='TFLOPS',
            args={"N": N, "K": K},
            plot_name=f'gemm-a8w8-N{N}-K{K}_TFLOPS',
        )
    )
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randint(-128, 127, (M, K), dtype=torch.int8).cuda()
    b = torch.randint(-128, 127, (N, K), dtype=torch.int8).cuda()
    alpha_row = torch.rand([M, 1], dtype=torch.half).cuda()
    alpha_col = torch.rand([1, N], dtype=torch.half).cuda()
    quantiles = [0.5, 0.2, 0.8]
    M, K = a.shape
    N, K = b.shape

    if provider == 'triton':
        flashnn.set_use_triton(True)
        # flashnn.set_autotune_triton_kernels(True)
        triton_gemm_a8w8 = flashnn.GemmA8W8(out_ty=torch.half)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_gemm_a8w8(a, b, alpha_row, alpha_col), rep=100, quantiles=quantiles
        )
    if provider == 'torch':
        flashnn.set_use_triton(False)
        torch_gemm_a8w8 = flashnn.GemmA8W8(out_ty=torch.half)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_gemm_a8w8(a, b, alpha_row, alpha_col), rep=100, quantiles=quantiles
        )

    tflops = lambda x: round(2 * M * N * K / x * 1e-9, 2)
    return tflops(ms), tflops(min_ms), tflops(max_ms)


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
