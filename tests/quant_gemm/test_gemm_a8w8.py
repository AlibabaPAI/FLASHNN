# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest

import flashnn
import torch

torch.random.manual_seed(0)


class TorchGemmA8W8(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, alpha_row, alpha_col):
        b = b.transpose(0, 1)
        x = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        scale = torch.matmul(alpha_row, alpha_col)
        out = torch.mul(x, scale)
        return out.to(torch.half)


class GemmA8W8Test(unittest.TestCase):
    def setUp(self):
        os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"

    def test_gemm_a8w8(self):
        sizes = [
            (128, 128, 128),
        ]
        with torch.no_grad():
            # Test TritonGemmA8W8 with random int8 inputs
            for m, n, k in sizes:
                a = torch.randint(-128, 127, (m, k), dtype=torch.int8).cuda()
                b = torch.randint(-128, 127, (n, k), dtype=torch.int8).cuda()
                alpha_row = torch.rand([m, 1], dtype=torch.half).cuda()
                alpha_col = torch.rand([1, n], dtype=torch.half).cuda()

                torch_gemm_a8w8 = TorchGemmA8W8()
                out_torch = torch_gemm_a8w8(a, b, alpha_row, alpha_col)

                tirton_gemm_a8w8 = flashnn.GemmA8W8()
                out_triton = tirton_gemm_a8w8(a, b, alpha_row, alpha_col)
                torch.testing.assert_close(
                    out_triton, out_torch, rtol=0.001, atol=0.002, check_dtype=False
                )


if __name__ == "__main__":
    unittest.main()
