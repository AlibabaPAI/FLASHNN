# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import torch
import triton

import flashnn

configs = []
for K in [1664, 3328, 8896, 13312, 17792, 35584]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["M"],
            x_vals=[1, 10, 20, 30, 40, 764, 1024, 2048, 4096, 4096 * 2],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="Latency",
            args={"K": K},
            plot_name=f"dynamic_quant-K_{K}_latency(us)",
        )
    )


@triton.testing.perf_report(configs)
def benchmark(M, K, provider, input_dtype=torch.half, device="cuda"):
    x = torch.rand([M, K], dtype=input_dtype, device=device)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        flashnn.set_use_triton(True)
        triton_dynamic_quant = flashnn.DynamicQuantize()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_dynamic_quant(x), rep=100, quantiles=quantiles
        )
    if provider == "torch":
        flashnn.set_use_triton(False)
        torch_dynamic_quant = flashnn.DynamicQuantize()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_dynamic_quant(x), rep=100, quantiles=quantiles
        )

    def ms2us(ms):
        return ms * 1000

    return ms2us(ms), ms2us(min_ms), ms2us(max_ms)


if __name__ == "__main__":
    benchmark.run(show_plots=False, print_data=True)
