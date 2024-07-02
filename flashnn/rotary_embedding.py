# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import torch

from flashnn.kernel_backend import BackendKernel

from .triton_kernels.rotary_embedding import triton_rotary_embd_forward


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in earlier torch versions


class RotaryEmbedding(BackendKernel):
    """
    TODO: add documentation
    """

    seq_len_cached = None
    cos_cached = None
    sin_cached = None

    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.precision = precision

    def _prepare_cos_sin(self, k, offset=0, max_seq_len=None, seq_dim=0):
        if max_seq_len is None:
            max_seq_len = k.shape[seq_dim]
            max_seq_len += offset
        if max_seq_len != RotaryEmbedding.seq_len_cached:
            self.seq_len_cached = max_seq_len
            t = torch.arange(max_seq_len, device=k.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(k.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            RotaryEmbedding.cos_cached = emb.cos()[:, None, None, :]
            RotaryEmbedding.sin_cached = emb.sin()[:, None, None, :]
            if self.precision == torch.bfloat16:
                RotaryEmbedding.cos_cached = RotaryEmbedding.cos_cached.bfloat16()
                RotaryEmbedding.sin_cached = RotaryEmbedding.sin_cached.bfloat16()
            if self.precision == torch.half:
                RotaryEmbedding.cos_cached = RotaryEmbedding.cos_cached.half()
                RotaryEmbedding.sin_cached = RotaryEmbedding.sin_cached.half()
            RotaryEmbedding.seq_len_cached = max_seq_len

        cos, sin = RotaryEmbedding.cos_cached, RotaryEmbedding.sin_cached
        return cos, sin, max_seq_len

    def _triton_impl(self, q, k, offset, max_seq_len, seq_dim):
        cos, sin, seq_len = self._prepare_cos_sin(k, offset, max_seq_len, seq_dim)
        query_rot, key_rot = triton_rotary_embd_forward(
            q, k, cos, sin, offset, seq_len, seq_dim
        )
        return query_rot, key_rot

    def _torch_impl(self, q, k, offset, max_seq_len, seq_dim):
        cos, sin, _ = self._prepare_cos_sin(k, offset, max_seq_len, seq_dim)
        cos, sin = (
            cos[offset : q.shape[0] + offset, ...],
            sin[offset : q.shape[0] + offset, ...],
        )
        query = (q * cos) + (_rotate_half(q) * sin)
        key = (k * cos) + (_rotate_half(k) * sin)
        return query, key

    def forward(self, q, k, offset=0, max_seq_len=None, seq_dim=0):
        return BackendKernel.forward(self, q, k, offset, max_seq_len, seq_dim)
