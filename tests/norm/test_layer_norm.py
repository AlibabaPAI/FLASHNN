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


class TestLayerNorm(unittest.TestCase):
    @parameterized.expand([([4, 128],), ([2, 64, 32],)])
    def test_layer_norm(self, input_shape: List[int]):
        hidden_size = input_shape[-1]
        inp = torch.randn(input_shape).half().cuda()

        # run reference
        torch_mod = torch.nn.LayerNorm(hidden_size, eps=1e-5).half().cuda()
        ref_output = torch_mod(inp)

        # test triton backend
        triton_mod = flashnn.LayerNorm(hidden_size, eps=1e-5).half().cuda()
        os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"
        tri_output = triton_mod(inp)
        torch.testing.assert_close(tri_output, ref_output, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
