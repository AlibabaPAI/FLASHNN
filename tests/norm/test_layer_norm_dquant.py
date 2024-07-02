# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest
from typing import List

import flashnn
import torch
from parameterized import parameterized

torch.random.manual_seed(0)


class TestLayernormDquant(unittest.TestCase):
    def _dquant(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().max(dim=-1, keepdim=True)[0] / 127
        q_x = torch.round(x / scale)
        q_x = torch.clamp(q_x, -128, 127)
        return q_x.to(torch.int8), scale

    @parameterized.expand([([4, 128],), ([2, 4, 128],)])
    def test_layernorm_dquant(self, input_shape: List[int]):
        hidden_size = input_shape[-1]
        mod = torch.nn.LayerNorm(hidden_size, eps=1e-5).half().cuda()
        inp = torch.randn(input_shape).half().cuda()

        # run reference
        ref_output, ref_scale = self._dquant(mod(inp))
        ref_scale = ref_scale.reshape(-1, 1)

        # test triton backend
        flashnn.set_use_triton(True)
        qmod = flashnn.LayernormDquant(hidden_size, eps=1e-5).half().cuda()
        os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"
        output_tri, scale_tri = qmod(inp)
        torch.testing.assert_close(output_tri, ref_output, rtol=0.01, atol=2.0)
        torch.testing.assert_close(scale_tri, ref_scale, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
