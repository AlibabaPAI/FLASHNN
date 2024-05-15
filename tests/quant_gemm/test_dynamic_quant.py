import os
import unittest

import torch

import flashnn


class TorchDynamicQuantize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        max_x = x.abs().max(-1, keepdim=True)[0]
        scale = max_x / 127.0
        out = torch.round(x / scale)
        return out, scale.squeeze(-1)


class DynamicQuantTest(unittest.TestCase):
    def setUp(self):
        os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"

    def test_dynamic_quant(self):
        torch.random.manual_seed(0)
        sizes = [
            (32, 32),
            (64, 64),
            (128, 128),
            (117, 987),
            (2560, 8000),
        ]
        with torch.no_grad():
            for num_tokens, hidden_size in sizes:
                input = (
                    torch.rand(num_tokens, hidden_size, dtype=torch.half, device="cuda")
                ) * 2 - 1
                # run reference
                dq = TorchDynamicQuantize()
                out_torch, scale_torch = dq(input)

                # run triton kernel
                out_triton, scale_triton = tri_blade.DynamicQuantize()(input)

                diff = ~torch.isclose(
                    out_triton.int().cpu(), out_torch.int().cpu(), atol=1
                )
                self.assertTrue(
                    diff.sum() < 10,
                    f"num_tokens={num_tokens}, hidden_size={hidden_size}",
                )

                diff = ~torch.isclose(
                    scale_triton.half().cpu(), scale_torch.half().cpu(), rtol=1e-2
                )
                self.assertTrue(
                    diff.sum() < 10,
                    f"num_tokens={num_tokens}, hidden_size={hidden_size}",
                )


if __name__ == "__main__":
    unittest.main()
