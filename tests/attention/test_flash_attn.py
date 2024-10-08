# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import os
import unittest

import flashnn
import numpy as np
import torch
from parameterized import parameterized


class FlashAttnTest(unittest.TestCase):
    def setUp(self):
        os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"

    shape_params = [
        (2, 20, 4, 16),
        (2, 200, 4, 16),
        (2, 500, 4, 128),
    ]
    causal_params = [True, False]
    expand_params = [
        (*shape, causal)
        for shape, causal in list(itertools.product(shape_params, causal_params))
    ]

    @parameterized.expand(expand_params)
    def test_flash_attention(self, Z, N_CTX, H, D_HEAD, causal, dtype=torch.float16):
        torch.manual_seed(0)
        q = torch.randn((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
        k = torch.randn((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
        v = torch.randn((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda")

        flashnn.set_use_triton(False)
        torch_attn = flashnn.FlashAttention()
        torch_out = torch_attn(q, k, v, causal)

        flashnn.set_use_triton(True)
        triton_attn = flashnn.FlashAttention()
        triton_out = triton_attn(q, k, v, causal)

        diff = ~np.isclose(
            torch_out.cpu().numpy(), triton_out.cpu().numpy(), rtol=1e-3, atol=1e-3
        )
        self.assertTrue(diff.sum() < 10, f"diff.sum={diff.sum()}")

    @parameterized.expand(expand_params)
    def test_flash_attention_GQA(
        self, Z, N_CTX, H, D_HEAD, causal, dtype=torch.float16
    ):
        torch.manual_seed(0)
        q = torch.randn((Z, N_CTX, 2, H, D_HEAD), dtype=dtype, device="cuda")
        k = torch.randn((Z, N_CTX, 2, H, D_HEAD), dtype=dtype, device="cuda")
        v = torch.randn((Z, N_CTX, 2, H, D_HEAD), dtype=dtype, device="cuda")

        flashnn.set_use_triton(False)
        torch_attn = flashnn.FlashAttention()
        torch_out = torch_attn(q, k, v, causal)

        flashnn.set_use_triton(True)
        triton_attn = flashnn.FlashAttention()
        triton_out = triton_attn(q, k, v, causal)

        diff = ~np.isclose(
            torch_out.cpu().numpy(), triton_out.cpu().numpy(), rtol=1e-3, atol=1e-3
        )
        self.assertTrue(diff.sum() < 10, f"diff.sum={diff.sum()}")


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    unittest.main()
