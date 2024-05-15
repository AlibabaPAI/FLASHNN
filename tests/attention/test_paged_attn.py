import os
import random
import unittest

import torch

import flashnn

torch.manual_seed(0)
from parameterized import parameterized


class PagedAttnTest(unittest.TestCase):
    def setUp(self):
        os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"

    @parameterized.expand(
        [
            (1, 32, 128, 16, 1024, torch.float16, 1024),
            (10, 32, 128, 32, 1024, torch.float16, 2048),
        ]
    )
    def test_paged_attn(
        self,
        batch_size: int,
        num_heads: int,
        head_size: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype,
        MAX_SEQ_LEN: int = 2048,
        num_kv_heads: int = None,
    ) -> None:
        query = torch.randn(
            (batch_size, num_heads, head_size), dtype=dtype, device="cuda"
        )
        key_cache = torch.randn(
            (num_blocks, num_heads, block_size, head_size), dtype=dtype, device="cuda"
        )
        value_cache = torch.randn(
            (num_blocks, num_heads, block_size, head_size), dtype=dtype, device="cuda"
        )

        context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(batch_size)]
        max_context_len = max(context_lens)
        context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables = []  # [batch_size, max_num_blocks_per_seq]
        for _ in range(batch_size):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")
        head_mapping = torch.arange(num_heads, dtype=torch.int32, device="cuda")

        scale = float(1.0 / (head_size**0.5))

        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        assert num_heads % num_kv_heads == 0
        num_queries_per_kv = num_heads // num_kv_heads
        head_mapping = torch.repeat_interleave(
            torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"),
            num_queries_per_kv,
        )

        tri_blade.set_use_triton(False)
        torch_attn = tri_blade.PagedAttention()
        torch_out = torch_attn(
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            max_context_len=max_context_len,
        )

        tri_blade.set_use_triton(True)
        tri_attn = tri_blade.PagedAttention(version=1)
        tri_out_v1 = torch_attn(
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            max_context_len=max_context_len,
        )
        assert torch.allclose(
            torch_out, tri_out_v1, atol=1e-2, rtol=1e-2
        ), f"{torch_out} vs {tri_out_v1}"

        tri_attn = tri_blade.PagedAttention(version=1)
        tri_out_v2 = torch_attn(
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            max_context_len=max_context_len,
        )
        assert torch.allclose(
            torch_out, tri_out_v1, atol=1e-2, rtol=1e-2
        ), f"{torch_out} vs {tri_out_v2}"


if __name__ == "__main__":
    unittest.main()
