# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest
from typing import List

import flashnn
import torch
import torch.nn as nn
from parameterized import parameterized

torch.manual_seed(0)


class RMSNormRef(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        nn.Module.__init__(self)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TestRMSNorm(unittest.TestCase):
    @parameterized.expand([([4, 128],), ([3, 64, 64],)])
    def test_rms_norm(self, input_shape: List[int]):
        hidden_size = input_shape[-1]
        inp = torch.randn(input_shape).half().cuda()

        # torch reference
        mod = RMSNormRef(hidden_size, eps=1e-5).half().cuda()
        ref_output = mod(inp)

        # test triton kernel
        mod = flashnn.RMSNorm(hidden_size, eps=1e-5).half().cuda()
        os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"
        output = mod(inp)
        torch.testing.assert_close(output, ref_output, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
