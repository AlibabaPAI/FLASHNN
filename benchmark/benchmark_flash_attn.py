import itertools
import os
import unittest

import numpy as np
import torch
import triton
from parameterized import parameterized

from flashnn.triton_kernels.flash_attn_v2 import triton_flash_attention_forward


class FlashAttnTest(unittest.TestCase):
    def setUp(self):
        os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"

    shape_params = [
        (2, 20, 6, 16),
        (2, 200, 6, 16),
        (2, 200, 6, 16),  # test triton kernel cache
        (2, 500, 6, 128),
    ]
    causal_params = [True, False]
    expand_params = [
        (*shape, causal)
        for shape, causal in list(itertools.product(shape_params, causal_params))
    ]

    @parameterized.expand(expand_params)
    def test_flash_attention(self, Z, N_CTX, H, D_HEAD, causal, dtype=torch.float16):
        torch.manual_seed(20)
        q = torch.randn((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
        k = torch.randn((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
        v = torch.randn((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
        sm_scale = 1 / q.shape[-1] ** 0.5
        dout = torch.randn_like(q)
        # reference implementation
        M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
        p = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) * sm_scale
        if causal:
            p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).half()
        # p = torch.exp(p)
        ref_out = torch.matmul(p, v.permute(0, 2, 1, 3))
        ref_out = ref_out.permute(0, 2, 1, 3)
        # triton implementation
        tri_out = triton_flash_attention_forward(q, k, v, causal).half()
        # compare
        diff = ~np.isclose(
            ref_out.cpu().numpy(), tri_out.cpu().numpy(), rtol=1e-3, atol=1e-3
        )
        self.assertTrue(diff.sum() < 10, f"diff.sum={diff.sum()}")

    @parameterized.expand(expand_params)
    def test_flash_attention_with_GQA(
        self, Z, N_CTX, H, D_HEAD, causal, dtype=torch.float16
    ):
        torch.manual_seed(20)
        q = torch.randn((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
        k = torch.randn((Z, N_CTX, H // 3, D_HEAD), dtype=dtype, device="cuda")
        v = torch.randn((Z, N_CTX, H // 3, D_HEAD), dtype=dtype, device="cuda")
        sm_scale = 1 / q.shape[-1] ** 0.5
        dout = torch.randn_like(q)
        # reference implementation
        M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
        p = (
            torch.matmul(
                q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1).repeat_interleave(3, dim=1)
            )
            * sm_scale
        )
        if causal:
            p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).half()
        ref_out = torch.matmul(p, v.permute(0, 2, 1, 3).repeat_interleave(3, dim=1))
        ref_out = ref_out.permute(0, 2, 1, 3)
        # triton implementation
        tri_out = triton_flash_attention_forward(q, k, v, causal).half()
        # compare
        diff = ~np.isclose(
            ref_out.cpu().numpy(), tri_out.cpu().numpy(), rtol=1e-3, atol=1e-3
        )
        self.assertTrue(diff.sum() < 10, f"diff.sum={diff.sum()}")


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
