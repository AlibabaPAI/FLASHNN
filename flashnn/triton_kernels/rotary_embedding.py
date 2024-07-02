# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import copy

import torch
import triton
import triton.language as tl
from flashnn.kernel_backend import get_autotune_triton_kernels


def _get_autotune_configs():
    if get_autotune_triton_kernels():
        configs = [
            triton.Config(
                {"BLOCK_SIZE_SEQ": 8, "BLOCK_SIZE_BH": 16, "BLOCK_SIZE_D": 64}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 8, "BLOCK_SIZE_BH": 8, "BLOCK_SIZE_D": 64}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 8, "BLOCK_SIZE_BH": 4, "BLOCK_SIZE_D": 64}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 8, "BLOCK_SIZE_BH": 16, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 8, "BLOCK_SIZE_BH": 8, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 8, "BLOCK_SIZE_BH": 4, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 8, "BLOCK_SIZE_BH": 16, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 8, "BLOCK_SIZE_BH": 8, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 8, "BLOCK_SIZE_BH": 4, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 16, "BLOCK_SIZE_D": 64}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 8, "BLOCK_SIZE_D": 64}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 4, "BLOCK_SIZE_D": 64}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 128, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 64, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 32, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 16, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 8, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 4, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 128, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 64, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 32, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 16, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 8, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 4, "BLOCK_SIZE_BH": 4, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 1, "BLOCK_SIZE_BH": 128, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 1, "BLOCK_SIZE_BH": 64, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 1, "BLOCK_SIZE_BH": 32, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 1, "BLOCK_SIZE_BH": 16, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 1, "BLOCK_SIZE_BH": 8, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 1, "BLOCK_SIZE_BH": 4, "BLOCK_SIZE_D": 32}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 1, "BLOCK_SIZE_BH": 128, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 1, "BLOCK_SIZE_BH": 64, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 1, "BLOCK_SIZE_BH": 32, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 1, "BLOCK_SIZE_BH": 16, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 1, "BLOCK_SIZE_BH": 8, "BLOCK_SIZE_D": 16}
            ),
            triton.Config(
                {"BLOCK_SIZE_SEQ": 1, "BLOCK_SIZE_BH": 4, "BLOCK_SIZE_D": 16}
            ),
        ]
        new_configs = []
        for cfg in configs:
            cp_cfg = copy.deepcopy(cfg)
            cp_cfg.num_warps = 8
            new_configs.append(cp_cfg)

            cp_cfg = copy.deepcopy(cfg)
            cp_cfg.num_stages = 4
            new_configs.append(cp_cfg)

            cp_cfg = copy.deepcopy(cfg)
            cp_cfg.num_stages = 4
            cp_cfg.num_warps = 8
            new_configs.append(cp_cfg)

            cp_cfg = copy.deepcopy(cfg)
            cp_cfg.num_stages = 1
            cp_cfg.num_warps = 8
            new_configs.append(cp_cfg)
        configs.extend(new_configs)
        return new_configs
    else:
        return [
            triton.Config(
                {"BLOCK_SIZE_SEQ": 1, "BLOCK_SIZE_BH": 16, "BLOCK_SIZE_D": 64},
                num_warps=4,
                num_stages=2,
            )
        ]


@triton.jit
def _rotary_embedding_kernel(
    q_rot_ptr,
    k_rot_ptr,
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    batch_size,
    num_heads,
    num_kv,
    hidden_size,
    q_strides,
    q_strideb,
    q_strideh,
    q_strided,
    k_strides,
    k_strideb,
    k_stridekv,
    k_strided,
    seq_offset,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_BH: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_bh_blocks = tl.cdiv(batch_size * num_heads, BLOCK_SIZE_BH)
    num_d_blocks = tl.cdiv(hidden_size // 2, BLOCK_SIZE_D)
    bh_id = pid % num_bh_blocks
    d_id = pid // num_bh_blocks % num_d_blocks
    seq_block_id = pid // num_bh_blocks // num_d_blocks

    seq_offs = seq_offset + seq_block_id * BLOCK_SIZE_SEQ + tl.arange(0, BLOCK_SIZE_SEQ)

    bh_offs = bh_id * BLOCK_SIZE_BH + tl.arange(0, BLOCK_SIZE_BH)
    q_common_offs = (
        seq_offs[:, None, None] * q_strides + bh_offs[None, :, None] * q_strideh
    )
    k_common_offs = (
        seq_offs[:, None, None] * k_strides
        + bh_offs[None, :, None] // (num_heads // num_kv) * k_stridekv
    )
    q_base_offs, qo_base_offs = q_ptr + q_common_offs, q_rot_ptr + q_common_offs
    k_base_offs, ko_base_offs = k_ptr + k_common_offs, k_rot_ptr + k_common_offs
    c_base_offs = cos_ptr + seq_offs[:, None] * hidden_size
    s_base_offs = sin_ptr + seq_offs[:, None] * hidden_size

    hidden_block_range = tl.arange(0, BLOCK_SIZE_D)

    hidden_offs_l = d_id * BLOCK_SIZE_D + hidden_block_range
    hidden_offs_r = hidden_size // 2 + hidden_offs_l
    mask_l, mask_r = hidden_offs_l < hidden_size // 2, hidden_offs_r < hidden_size
    mask_bh = bh_offs < batch_size * num_heads
    mask_seq = seq_offs < seq_len
    mask_bh_seq = mask_bh[None, :, None] & mask_seq[:, None, None]

    q_l, k_l = tl.load(
        q_base_offs + hidden_offs_l[None, None, :] * q_strided,
        mask=mask_l[None, None, :] & mask_bh_seq,
        other=0,
    ), tl.load(
        k_base_offs + hidden_offs_l[None, None, :] * k_strided,
        mask=mask_l[None, None, :] & mask_bh_seq,
        other=0,
    )
    q_r, k_r = tl.load(
        q_base_offs + hidden_offs_r[None, None, :] * q_strided,
        mask=mask_r[None, None, :] & mask_bh_seq,
        other=0,
    ), tl.load(
        k_base_offs + hidden_offs_r[None, None, :] * k_strided,
        mask=mask_r[None, None, :] & mask_bh_seq,
        other=0,
    )
    cos_l, cos_r = (
        tl.load(c_base_offs + hidden_offs_l[None, :], mask=mask_l[None, :], other=0)[
            :, None, :
        ],
        tl.load(c_base_offs + hidden_offs_r[None, :], mask=mask_r[None, :], other=0)[
            :, None, :
        ],
    )
    sin_l, sin_r = (
        tl.load(s_base_offs + hidden_offs_l[None, :], mask=mask_l[None, :], other=0)[
            :, None, :
        ],
        tl.load(s_base_offs + hidden_offs_r[None, :], mask=mask_r[None, :], other=0)[
            :, None, :
        ],
    )

    qo_l = q_l * cos_l - q_r * sin_l
    tl.store(
        qo_base_offs + hidden_offs_l, qo_l, mask=mask_l[None, None, :] & mask_bh_seq
    )
    qo_r = q_r * cos_r + q_l * sin_r
    tl.store(
        qo_base_offs + hidden_offs_r, qo_r, mask=mask_r[None, None, :] & mask_bh_seq
    )
    ko_l = k_l * cos_l - k_r * sin_l
    tl.store(
        ko_base_offs + hidden_offs_l, ko_l, mask=mask_l[None, None, :] & mask_bh_seq
    )
    ko_r = k_r * cos_r + k_l * sin_r
    tl.store(
        ko_base_offs + hidden_offs_r, ko_r, mask=mask_r[None, None, :] & mask_bh_seq
    )


def triton_rotary_embd_forward(
    q, k, cos_ptr, sin_ptr, offset=0, max_seq_len=None, seq_dim=0
):
    if max_seq_len is None:
        max_seq_len = k.shape[seq_dim]
        max_seq_len += offset
    query_rot = torch.empty_like(q)
    key_rot = torch.empty_like(k)
    _, B, H, D = q.shape
    _, _, nKV, _ = k.shape

    method_name = (
        "rotary_embedding_"
        + str(max_seq_len)
        + "_"
        + str(B)
        + "_"
        + str(H)
        + "_"
        + str(nKV)
        + "_"
        + str(D)
    )
    kwargs = [
        query_rot,
        key_rot,
        q,
        k,
        cos_ptr,
        sin_ptr,
        max_seq_len,
        B,
        H,
        nKV,
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        offset,
    ]

    def grid(META):
        return (
            max(
                1,
                (
                    triton.cdiv(
                        max_seq_len * B * H,
                        META["BLOCK_SIZE_SEQ"] * META["BLOCK_SIZE_BH"],
                    )
                    * triton.cdiv(D // 2, META["BLOCK_SIZE_D"])
                ),
            ),
            1,
            1,
        )

    rotary_embedding = triton.autotune(
        configs=_get_autotune_configs(),
        key=["seq_len", "batch_size", "num_heads", "num_kv", "hidden_size"],
    )(_rotary_embedding_kernel)

    rotary_embedding[grid](*kwargs)
    return query_rot, key_rot
