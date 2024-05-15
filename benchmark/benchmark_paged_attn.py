import os
import random
import unittest
from typing import Optional

import numpy as np
import torch
import triton
from parameterized import parameterized


TEST_SEED = 0


try:
    from vllm._custom_C import paged_attention_custom
    from vllm._C import ops
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


def ref_masked_attention(
    query: torch.Tensor,  # [1, num_heads, head_size]
    key: torch.Tensor,  # [context_len, num_heads, head_size]
    value: torch.Tensor,  # [context_len, num_heads, head_size]
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query = query * scale
    attn = torch.einsum('qhd,khd->hqk', query, key)
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum('hqk,khd->qhd', attn, value)
    return out


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,  # [num_tokens, q_heads, head_size]
    query: torch.Tensor,  # [num_tokens, q_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, kv_heads, head_size/x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, kv_heads, head_size, block_size]
    block_tables: torch.Tensor,  # [num_tokens, max_num_blocks_per_seq]
    context_lens: torch.Tensor,  # [num_tokens]
) -> None:
    q_heads = query.shape[1]
    kv_heads = key_cache.shape[1]
    h_ratio = q_heads // kv_heads
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]

    num_input_tokens = query.shape[0]
    for i in range(num_input_tokens):
        q = query[i].unsqueeze(0)  # [1, q_heads, head_size]
        block_table = block_tables[i]  # [max_num_blocks_per_seq]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]  # [kv_heads, head_size/x, x]
            k = k.reshape(kv_heads, head_size)  # [kv_heads, head_size]
            k = k.repeat_interleave(h_ratio, 0)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]  # [kv_heads, head_size]
            v = v.repeat_interleave(h_ratio, 0)
            values.append(v)

        keys = torch.stack(keys, dim=0)  # [context_len, kv_heads, head_size]
        values = torch.stack(values, dim=0)

        scale = 1.0 / (head_size**0.5)
        out = ref_masked_attention(q, keys, values, scale)
        out = out.view(q_heads, head_size)
        output[i].copy_(out, non_blocking=True)


class PagedAttentionTest(unittest.TestCase):
    def setUp(self):
        os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"

    def run_single_query_cached_kv_attention(
        self,
        num_tokens: int,
        num_heads: int,
        head_size: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.float16,
        max_seq_len: int,
        num_kv_heads: int,
    ):
        query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype, device='cuda')
        query.uniform_(-1, 1)

        x = 16 // torch.tensor([], dtype=dtype).element_size()
        key_block_shape = (num_kv_heads, head_size // x, block_size, x)
        key_cache = torch.randn(size=(num_blocks, *key_block_shape), dtype=dtype, device='cuda')
        value_block_shape = (num_kv_heads, head_size, block_size)
        value_cache = torch.randn(size=(num_blocks, *value_block_shape), dtype=dtype, device='cuda')

        # context_lens = [random.randint(1, max_seq_len) for _ in range(num_tokens)]
        context_lens = [max_seq_len for _ in range(num_tokens)]
        max_context_len = max(context_lens)
        context_lens = torch.tensor(context_lens, dtype=torch.int, device='cuda')

        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables = []
        for _ in range(num_tokens):
            block_table = [random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)]
            block_tables.append(block_table)
        block_tables = torch.tensor(block_tables, dtype=torch.int, device='cuda')
        head_mapping = torch.arange(num_heads, dtype=torch.int32, device="cuda")

        scale = float(1.0 / (head_size**0.5))

        assert num_heads % num_kv_heads == 0
        num_queries_per_kv = num_heads // num_kv_heads
        head_mapping = torch.repeat_interleave(
            torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"), num_queries_per_kv
        )

        ef_out = torch.empty_like(query)

        ref_single_query_cached_kv_attention(
            ref_out,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
        )

        if HAS_VLLM:
            vllm_out_v1 = torch.empty_like(query)
            ops.paged_attention_v1(
                vllm_out_v1,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                context_lens,
                block_size,
                max_context_len,
                None,
                "auto",
                1.0,
            )
            vllm_out_v2 = torch.empty_like(query)
            PARTITION_SIZE = 256
            num_partitions = ((max_context_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
            tmp_output = torch.empty(
                size=(num_tokens, num_heads, num_partitions, head_size),
                dtype=vllm_out_v2.dtype,
                device=vllm_out_v2.device,
            )
            exp_sums = torch.empty(
                size=(num_tokens, num_heads, num_partitions),
                dtype=torch.float32,
                device=vllm_out_v2.device,
            )
            max_logits = torch.empty_like(exp_sums)
            ops.paged_attention_v2(
                vllm_out_v2,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                context_lens,
                block_size,
                max_context_len,
                None,
                "auto",
                1.0,
            )
            diff = ~np.isclose(vllm_custome_out.cpu().numpy(), vllm_out.cpu().numpy(), atol=1e-2)
            self.assertTrue(diff.sum() < 10, f"diff.sum={diff.sum()}, output={vllm_out}, ref_output={vllm_out}")

    @parameterized.expand(
        [
            (512,),
            (1024,),
            (8192,),
        ]
    )
    def test_paged_attention(self, max_seq_len):
        torch.random.manual_seed(TEST_SEED)
        torch.cuda.manual_seed(TEST_SEED)
        for num_kv_heads in [8, 16, 32, 64]:
        # for num_kv_heads in [64]:
            print(f"Test paged attn with bs=1, num_q_heads=64, num_kv_heads={num_kv_heads}, seq_len={max_seq_len}")
            self.run_single_query_cached_kv_attention(
                num_tokens=1,
                num_heads=64,
                head_size=128,
                block_size=16,
                num_blocks=10240,
                dtype=torch.half,
                max_seq_len=max_seq_len,
                num_kv_heads=num_kv_heads,
            )

configs = []
HEAD_DIM = 128
tmp = [
    (1, 16, 16),
    (1, 32, 32),
    (1, 32, 4),
    (64, 32, 4),
    (1, 52, 4),
    (64, 52, 4),
    (1, 16, 2),
    (64, 16, 2),
    (1, 26, 2),
    (64, 26, 2),
    (1, 8, 1),
    (64, 8, 1),
    (1, 13, 1),
    (64, 13, 1),
]
for bs, q_head, kv_head in tmp:
    configs.append(
        triton.testing.Benchmark(
            x_names=['seq_len'],
            x_vals=[2**i for i in range(8, 14)],
            # x_vals=[8192],
            line_arg='provider',
            line_vals=(
                ['triton_mix', 'triton_fma', 'triton_mma']
                + (['bladnn_v1', 'bladnn_v2'] if HAS_BLADNN else [])
                + (['vllm'] if HAS_VLLM else [])
            ),
            line_names=(
                ['Triton mix', "Triton FMA", "Triton MMA"]
                + (['BlaDNN v1', 'BlaDNN v2'] if HAS_BLADNN else [])
                + (['vLLM'] if HAS_VLLM else [])
            ),
            styles=[('red', '-'), ('yellow', '-'), ('blue', '-'), ('green', '-'), ('orange', '-'), ('purple', '-')],
            ylabel='ms',
            plot_name=f'BS={bs},num_head_q={q_head},num_heads_kv={kv_head},head_size=128, block_size=16,num_blocks=10240',
            args={
                "num_seqs": bs,
                "q_head": q_head,
                "kv_head": kv_head,
                "head_size": 128,
                "block_size": 16,
                "num_blocks": 10240,
                "dtype": torch.float16,
            },
        )
    )


@triton.testing.perf_report(configs)
def benchmark(
    num_seqs, seq_len, q_head, kv_head, head_size, block_size, num_blocks, dtype, provider, eps=1e-5, device='cuda'
):
    query = torch.empty(num_seqs, q_head, head_size, dtype=dtype, device='cuda')

    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_block_shape = (kv_head, head_size // x, block_size, x)
    key_cache = torch.randn((num_blocks, *key_block_shape), dtype=dtype, device='cuda')
    value_block_shape = (kv_head, head_size, block_size)
    value_cache = torch.randn((num_blocks, *value_block_shape), dtype=dtype, device='cuda')

    context_lens = [seq_len for _ in range(num_seqs)]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device='cuda')

    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for i in range(num_seqs):
        block_table = [random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device='cuda')
    head_mapping = torch.arange(q_head, dtype=torch.int32, device="cuda")

    scale = float(1.0 / (head_size**0.5))

    assert q_head % kv_head == 0
    num_queries_per_kv = q_head // kv_head
    head_mapping = torch.repeat_interleave(torch.arange(kv_head, dtype=torch.int32, device="cuda"), num_queries_per_kv)

    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_seqs)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int, device='cuda')
    out = torch.empty_like(query)
    key_cache_tri = key_cache.permute(0, 1, 3, 2, 4).flatten(3, 4).contiguous().cuda()
    value_cache_tri = value_cache.permute(0, 1, 3, 2).contiguous().cuda()
    quantiles = [0.5, 0.2, 0.8]
    
    if provider.startswith("triton"):
        device = torch.cuda.device_of(query)
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        if num_seqs * kv_head > 2 * num_sms:
            num_splits = 1
            partition_size = 0
            if max_context_len >= 8192:
                partition_size = max(256, block_size)
                num_splits = triton.cdiv(max_context_len, partition_size)
        else:
            partition_size = max(256, block_size)
            num_splits = triton.cdiv(max_context_len, partition_size)
            if max_context_len <= 1024 or block_size >= 256:
                num_splits = 1
                partition_size = 0
  
    if provider == 'triton_mix':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: paged_attn_new(
                out,
                query,
                key_cache_tri,
                value_cache_tri,
                context_lens,
                block_tables,
                scale,
                max_context_len,
            ),
            warmup=20,
            rep=100,
            quantiles=quantiles,
        )
  
    if provider == 'triton_fma':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: tt_paged_attn_fma(
                out,
                query,
                key_cache_tri,
                value_cache_tri,
                context_lens,
                block_tables,
                scale,
                max_context_len,
                num_splits,
                partition_size,
                device,
            ),
            warmup=20,
            rep=100,
            quantiles=quantiles,
        )
  
    if provider == 'triton_mma':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: tt_paged_attn_mma(
                out,
                query,
                key_cache_tri,
                value_cache_tri,
                context_lens,
                block_tables,
                scale,
                max_context_len,
                num_splits,
                partition_size, 
                device,
            ),
            warmup=20,
            rep=100,
            quantiles=quantiles,
        )

    if provider == 'bladnn_v1':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: bladnn.single_query_cached_kv_attention(
                out,
                query,
                key_cache,
                value_cache,
                head_mapping,
                scale,
                block_tables,
                context_lens,
                block_size,
                max_context_len,
                None,  # ALiBi slopes.
            ),
            warmup=20,
            rep=100,
            quantiles=quantiles,
        )

    if provider == 'bladnn_v2':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: bladnn.single_query_cached_kv_attention_v2(
                out,
                query,
                key_cache,
                value_cache,
                head_mapping,
                scale,
                block_tables,
                context_lens,
                block_size,
                max_context_len,
                None,
            ),
            warmup=20,
            rep=100,
            quantiles=quantiles,
        )

    if provider == 'vllm':
        PARTITION_SIZE = 256
        num_partitions = ((max_context_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
        tmp_output = torch.empty(
            size=(num_seqs, q_head, num_partitions, head_size),
            dtype=out.dtype,
            device=out.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, q_head, num_partitions),
            dtype=torch.float32,
            device=out.device,
        )
        max_logits = torch.empty_like(exp_sums)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: paged_attention_custom(
                out,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                kv_head,
                scale,
                block_tables,
                context_lens,
                block_size,
                max_context_len,
                None,
                "auto",
            ),
            warmup=20,
            rep=100,
            quantiles=quantiles,
        )

    def perf_us(ms):
        return ms * 1000
    
    # return ms, min_ms, max_ms
    return perf_us(ms), perf_us(min_ms), perf_us(max_ms)


if __name__ == "__main__":
    unittest.main()
    benchmark.run(show_plots=True, print_data=True)
