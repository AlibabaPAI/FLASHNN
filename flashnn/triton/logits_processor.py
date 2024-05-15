import triton
import triton.language as tl

from flashnn.triton.triton_utils import compile_and_cache_kernels


@triton.jit
def _triton_repetition_penalty_kernel(
    scores,  # [num_tokens, vocab_size]
    penalty,  # [num_tokens]
    input_ids_ptr,  # [num_tokens]
    input_ids_length,  # [num_tokens]
    vocab_size: tl.constexpr,
    power_2_of_max_ids_length: tl.constexpr,
):
    token_id = tl.program_id(0)
    penalty_val = tl.load(penalty + token_id)
    if tl.abs(penalty_val - 1.0) > 1e-9:
        input_ids_address = tl.load(input_ids_ptr + token_id).to(tl.pointer_type(tl.int64))
        current_input_ids_length = tl.load(input_ids_length + token_id)
        ids_offs = tl.arange(0, power_2_of_max_ids_length)
        ids = tl.load(
            input_ids_address + ids_offs,
            mask=ids_offs < current_input_ids_length,
            other=vocab_size,
        )
        ori_scores = tl.load(scores + token_id * vocab_size + ids[None, :], mask=ids[None, :] < vocab_size, other=0.0)
        tl.debug_barrier()
        new_scores = tl.where(ori_scores <= 0, ori_scores * penalty_val, ori_scores / penalty_val)
        tl.store(scores + token_id * vocab_size + ids[None, :], new_scores, mask=ids[None, :] < vocab_size)


def triton_repetition_penalty(scores, penalty, input_ids_ptr, input_ids_length, max_ids_length):
    num_tokens, vocab_size = scores.shape
    power_2_of_vocab_size = triton.next_power_of_2(vocab_size)
    power_2_of_max_ids_length = triton.next_power_of_2(max_ids_length)
    kwargs = [
        scores,
        penalty,
        input_ids_ptr,
        input_ids_length,
    ]
    const_kwargs = {
        "vocab_size": vocab_size,
        "power_2_of_max_ids_length": power_2_of_max_ids_length,
        "num_warps": 8,
    }
    method_name = "repetition_penalty" + str(vocab_size) + "_" + str(power_2_of_max_ids_length)
    grid = (num_tokens, 1, 1)
    compile_and_cache_kernels(_triton_repetition_penalty_kernel, method_name, grid, kwargs, const_kwargs)


@triton.jit
def _triton_presence_penalty_kernel(
    scores,  # [num_tokens, vocab_size]
    penalty,  # [num_tokens]
    input_ids_ptr,  # [num_tokens]
    input_ids_length,  # [num_tokens]
    vocab_size: tl.constexpr,
    power_2_of_max_ids_length: tl.constexpr,
):
    token_id = tl.program_id(0)
    penalty_val = tl.load(penalty + token_id)
    if tl.abs(penalty_val - 1.0) > 1e-9:
        input_ids_address = tl.load(input_ids_ptr + token_id).to(tl.pointer_type(tl.int64))
        current_input_ids_length = tl.load(input_ids_length + token_id)
        ids_offs = tl.arange(0, power_2_of_max_ids_length)
        ids = tl.load(
            input_ids_address + ids_offs,
            mask=ids_offs < current_input_ids_length,
            other=vocab_size,
        )
        ori_scores = tl.load(scores + token_id * vocab_size + ids[None, :], mask=ids[None, :] < vocab_size, other=0.0)
        tl.debug_barrier()
        new_scores = ori_scores - penalty_val
        tl.store(scores + token_id * vocab_size + ids[None, :], new_scores, mask=ids[None, :] < vocab_size)


def triton_presence_penalty(scores, penalty, input_ids_ptr, input_ids_length, max_ids_length):
    num_tokens, vocab_size = scores.shape
    power_2_of_vocab_size = triton.next_power_of_2(vocab_size)
    power_2_of_max_ids_length = triton.next_power_of_2(max_ids_length)
    kwargs = [
        scores,
        penalty,
        input_ids_ptr,
        input_ids_length,
    ]
    const_kwargs = {
        "vocab_size": vocab_size,
        "power_2_of_max_ids_length": power_2_of_max_ids_length,
        "num_warps": 8,
    }
    method_name = "presence_penalty_" + str(vocab_size) + "_" + str(power_2_of_max_ids_length)
    grid = (num_tokens, 1, 1)
    compile_and_cache_kernels(_triton_presence_penalty_kernel, method_name, grid, kwargs, const_kwargs)
