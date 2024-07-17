# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import torch
import triton

from flashnn.triton_kernels.flash_attn_v2 import triton_flash_attention_forward

try:
    from flash_attn import flash_attn_func

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False


configs = []
BATCH, D_HEAD = 1, 128
QK_HEADS = []
for Q_HEAD in [64, 32, 16]:
    for KV_HEAD in [32, 16, 8, 2]:
        if Q_HEAD % KV_HEAD != 0:
            continue
        QK_HEADS.append((Q_HEAD, KV_HEAD))
for Q_HEAD, KV_HEAD in QK_HEADS:
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[2**i for i in range(9, 16)],
            line_arg="provider",
            line_vals=["triton"] + (["flash"] if HAS_FLASH else []),
            line_names=["Triton"] + (["Flash-2"] if HAS_FLASH else []),
            styles=[("red", "-"), ("blue", "-"), ("black", "-")],
            ylabel="ms",
            plot_name=f"batch={BATCH},num_heads_q={Q_HEAD},num_heads_kv={KV_HEAD},head_dim={D_HEAD},latency(ms)",
            args={
                "BATCH": BATCH,
                "Q_HEAD": Q_HEAD,
                "KV_HEAD": KV_HEAD,
                "D_HEAD": D_HEAD,
                "dtype": torch.float16,
                "causal": True,
            },
        )
    )


@triton.testing.perf_report(configs)
def bench_flash_attention(
    BATCH,
    Q_HEAD,
    KV_HEAD,
    N_CTX,
    D_HEAD,
    causal,
    provider,
    dtype=torch.float16,
    device="cuda",
):
    warmup = 25
    rep = 100
    quantiles = [0.5, 0.2, 0.8]
    q = torch.randn((BATCH, N_CTX, Q_HEAD, D_HEAD), dtype=dtype, device="cuda")
    k = torch.randn((BATCH, N_CTX, KV_HEAD, D_HEAD), dtype=dtype, device="cuda")
    v = torch.randn((BATCH, N_CTX, KV_HEAD, D_HEAD), dtype=dtype, device="cuda")
    if provider == "triton":
        # import flashnn
        # flashnn.set_autotune_triton_kernels(True)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_flash_attention_forward(q, k, v, causal),
            warmup=warmup,
            rep=rep,
            quantiles=quantiles,
        )
    if provider == "flash":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attn_func(q, k, v, causal=causal),
            warmup=warmup,
            rep=rep,
            quantiles=quantiles,
        )

    def ms2us(ms):
        return ms * 1000

    return ms, min_ms, max_ms


if __name__ == "__main__":
    bench_flash_attention.run(show_plots=False, print_data=True)
