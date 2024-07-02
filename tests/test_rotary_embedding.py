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


class RotaryEmbeddingRef(nn.Module):
    seq_len_cached = None
    cos_cached = None
    sin_cached = None

    def __init__(self, dim, base=10000, precision=torch.half):
        nn.Module.__init__(self)
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.precision = precision

    def _prepare_cos_sin(self, k, offset=0, max_seq_len=None, seq_dim=0):
        if max_seq_len is None:
            max_seq_len = k.shape[seq_dim]
            max_seq_len += offset
        if max_seq_len != RotaryEmbeddingRef.seq_len_cached:
            self.seq_len_cached = max_seq_len
            t = torch.arange(max_seq_len, device=k.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(k.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            RotaryEmbeddingRef.cos_cached = emb.cos()[:, None, None, :]
            RotaryEmbeddingRef.sin_cached = emb.sin()[:, None, None, :]
            if self.precision == torch.bfloat16:
                RotaryEmbeddingRef.cos_cached = RotaryEmbeddingRef.cos_cached.bfloat16()
                RotaryEmbRotaryEmbeddingRefedding.sin_cached = (
                    RotaryEmbeddingRef.sin_cached.bfloat16()
                )
            if self.precision == torch.half:
                RotaryEmbeddingRef.cos_cached = RotaryEmbeddingRef.cos_cached.half()
                RotaryEmbeddingRef.sin_cached = RotaryEmbeddingRef.sin_cached.half()
            RotaryEmbeddingRef.seq_len_cached = max_seq_len

        cos, sin = RotaryEmbeddingRef.cos_cached, RotaryEmbeddingRef.sin_cached
        return cos, sin, max_seq_len

    def _rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat(
            (-x2, x1), dim=x1.ndim - 1
        )  # dim=-1 triggers a bug in earlier torch versions

    def _apply_rotary_pos_emb(self, q, k, cos, sin, offset: int = 0):
        cos, sin = (
            cos[offset : q.shape[0] + offset, ...],
            sin[offset : q.shape[0] + offset, ...],
        )
        return (q * cos) + (self._rotate_half(q) * sin), (k * cos) + (
            self._rotate_half(k) * sin
        )

    def forward(self, q, k, offset=0, max_seq_len=None, seq_dim=0):
        cos, sin, _ = self._prepare_cos_sin(k, offset, max_seq_len, seq_dim)
        query_torch, key_torch = self._apply_rotary_pos_emb(
            q, k, cos, sin, offset=offset
        )
        return query_torch, key_torch


class TestRotaryEmbedding(unittest.TestCase):
    @parameterized.expand(
        [
            # GQA
            ([1, 0, 1, 40, 8, 40, torch.half],),
        ]
    )
    def test_rotary_embedding(self, shape_parameters: List):
        seq_len, offset, bs, h, h_k, d, dtype = shape_parameters
        base = 1000
        query = torch.rand((seq_len, bs, h, d), dtype=dtype, device="cuda")
        key = torch.rand((seq_len, bs, h_k, d), dtype=dtype, device="cuda")

        # run torch reference
        rotary_embd = RotaryEmbeddingRef(d, base, precision=dtype).cuda()
        torch_query_out, torch_key_out = rotary_embd(query, key, offset)

        # run triton backend
        os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"
        rotary_embd = flashnn.RotaryEmbedding(d, base, precision=dtype).cuda()
        triton_query_out, triton_key_out = rotary_embd(query, key, offset)
        diff = ~torch.isclose(
            triton_query_out.half().cpu(),
            torch_query_out.half().cpu(),
            rtol=1e-2,
            atol=1e-2,
        )
        self.assertTrue(
            diff.sum() < 10,
            f"triton backend: query_out mismatch with diff.sum()={diff.sum()}, bs={bs}, seq_len={seq_len}, offset={offset}, h={h}, h_k={h_k}, d={d}, dtype={dtype}",
        )
        diff = ~torch.isclose(
            triton_key_out.half().cpu(),
            torch_key_out.half().cpu(),
            rtol=1e-2,
            atol=1e-2,
        )
        self.assertTrue(
            diff.sum() < 10,
            f"triton backend: key_out mismatch with diff.sum()={diff.sum()}, bs={bs}, seq_len={seq_len}, offset={offset}, h={h}, h_k={h_k}, d={d}, dtype={dtype}",
        )


if __name__ == "__main__":
    unittest.main()
