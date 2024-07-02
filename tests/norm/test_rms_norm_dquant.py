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

seed = 0
torch.manual_seed(seed)


class RMSNormDquantRef(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        nn.Module.__init__(self)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm_dquant(
        self,
        x,
    ):
        norm_out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        norm_out *= self.weight
        max_norm_out = torch.max(torch.abs(norm_out), dim=-1, keepdim=True)[0]
        scale = max_norm_out / 127.0
        quantized_norm_out = torch.round(norm_out / scale)
        quantized_norm_out = torch.clamp(quantized_norm_out, -128.0, 127.0)
        scale = scale.reshape(x.shape[:-1])
        return quantized_norm_out.to(torch.int8), scale

    def forward(self, x):
        torch_out, torch_scale = self._norm_dquant(x.float())
        return torch_out, torch_scale.type_as(x)


class TestRMSNormDquant(unittest.TestCase):
    @parameterized.expand([([4, 128],), ([3, 32, 1024],)])
    def test_rms_norm_dquant(self, input_shape: List[int]):
        hidden_size = input_shape[-1]
        inp = torch.randn(input_shape).half().cuda()

        # torch reference
        mod = RMSNormDquantRef(hidden_size, eps=1e-5).half().cuda()
        ref_output, ref_scale = mod(inp)

        # test triton kernel
        mod = flashnn.RMSNormDquant(hidden_size, eps=1e-5).half().cuda()
        os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"
        tri_output, tri_scale = mod(inp)
        torch.testing.assert_close(tri_output, ref_output, rtol=1e-2, atol=1)
        torch.testing.assert_close(tri_scale, ref_scale, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
