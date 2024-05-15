import os
import unittest

import torch

import flashnn

torch.random.manual_seed(0)


class TorchGemmA8W8(torch.nn.Module):
    def __init__(self, m, n, alpha_row=None, alpha_col=None, interleave=False):
        super().__init__()
        if alpha_row is None:
            self.alpha_row = torch.nn.Parameter(torch.ones(m, 1)).half().cuda()
        else:
            self.alpha_row = alpha_row
        if alpha_col is None:
            self.alpha_col = torch.nn.Parameter(torch.ones(1, n)).half().cuda()
        else:
            self.alpha_col = alpha_col
        self.interleave = interleave

    def forward(self, a, b):
        b = b.transpose(0, 1)
        x = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        scale = torch.matmul(self.alpha_row, self.alpha_col)
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

                torch_gemm_a8w8 = TorchGemmA8W8(
                    a.shape[0], b.shape[0], alpha_row=alpha_row, alpha_col=alpha_col
                )
                out_torch = torch_gemm_a8w8(a, b)

                tirton_gemm_a8w8 = tri_blade.GemmA8W8(
                    a.shape[0], b.shape[0], alpha_row=alpha_row, alpha_col=alpha_col
                )
                out_triton = tirton_gemm_a8w8(a, b)
                torch.testing.assert_close(
                    out_triton, out_torch, rtol=0.001, atol=0.002, check_dtype=False
                )


if __name__ == "__main__":
    unittest.main()
