import json
import os
import types
from typing import Dict, List

import torch.distributed as dist
import triton
from loguru import logger

TRITON_KERNELS_CACHE = {}

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
configs_dir = os.path.join(current_dir, "configs")
json_filename = "triton_autotune_best_configs.json"
json_file_path = os.path.join(configs_dir, json_filename)

FORCE_RE_TUNE_TRITON_KERNELS = os.getenv("FORECE_RE_TUNE_TRITON_KERNELS", "None").split(
    ";"
)


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def dist_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def serialize_best_config(best_config: triton.runtime.autotuner.Config) -> List[Dict]:
    return [
        best_config.kwargs,
        {
            k: v
            for (k, v) in (
                ("num_warps", best_config.num_warps),
                ("num_ctas", best_config.num_ctas),
                ("num_stages", best_config.num_stages),
                ("maxnreg", best_config.maxnreg),
            )
            if v is not None
        },
    ]


def deserialize_best_config(
    best_config_list: List[Dict],
) -> triton.runtime.autotuner.Config:
    return triton.runtime.autotuner.Config(best_config_list[0], **best_config_list[1])


def init_file():
    os.makedirs(configs_dir, exist_ok=True)
    with open(json_file_path, "w") as f:
        json.dump({}, f, indent=4)


def read_json_file():
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def update_json_file(data):
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def force_re_tune(method_name):
    for kernel in FORCE_RE_TUNE_TRITON_KERNELS:
        if method_name.startswith(kernel):
            return True
    return False


def compile_and_cache_kernels(
    fn, method_name, grid, kwargs, const_kwargs, grid_kwargs=None
):
    """
    Upon first encountering a kernel with method_name, we compile it and cache the compiled kernel.
    Subsequent encounters with the same method_name will fetch the corresponding compiled kernel
    directly from the cache for execution.

    :param fn: the triton kernel to be compiled and executed,
        must be `triton.runtime.jit.JITFunction` or `triton.runtime.autotuner.Autotuner`
    :param method_name: str, the unique name corresponding to the triton kernel
    :param grid: a tuple or a function returning a tuple, the final tuple must contain three elements that
        correspond to the x/y/z dimensions of the CUDA blocks.
    :param kwargs: a list of args corresponding non `tl.constexpr` in triton kernel's parameter list
    :param const_kwargs: a map of args corresponding `tl.constexpr` in triton kernel's parameter list
    :param grid_kwargs: a map of args used for `grid` if the grid is a function turning a tuple
    """
    global TRITON_KERNELS_CACHE
    rank = get_rank()
    if method_name in TRITON_KERNELS_CACHE:
        kernel_and_tile_size = TRITON_KERNELS_CACHE[method_name]
        kernel = kernel_and_tile_size[0]
        tile_size = kernel_and_tile_size[-1]
        grid_tuple = grid
        if isinstance(grid, types.FunctionType):
            grid_tuple = grid(tile_size)
        kernel[grid_tuple](*kwargs)
    else:
        logger.info("rank-{} triton compile kernel: {}", rank, method_name)
        if isinstance(fn, triton.runtime.jit.JITFunction) or isinstance(
            fn, triton.runtime.autotuner.Heuristics
        ):
            kernel = (
                fn[grid](*kwargs)
                if const_kwargs is None
                else fn[grid](*kwargs, **const_kwargs)
            )
            TRITON_KERNELS_CACHE[method_name] = [kernel, {}]
        else:
            assert isinstance(fn, triton.runtime.autotuner.Autotuner)
            best_config = []
            if not os.path.exists(json_file_path):
                if rank == 0:
                    init_file()
                dist_barrier()
            data = read_json_file()
            if method_name in data and not force_re_tune(method_name):
                best_config = data[method_name]
                fn.configs = [deserialize_best_config(best_config)]
                kernel = (
                    fn[grid](*kwargs)
                    if const_kwargs is None
                    else fn[grid](*kwargs, **const_kwargs)
                )
            else:
                logger.info("rank-{} triton autotune kernel: {}", rank, method_name)
                kernel = (
                    fn[grid](*kwargs)
                    if const_kwargs is None
                    else fn[grid](*kwargs, **const_kwargs)
                )
                best_config = serialize_best_config(fn.best_config)
                data[method_name] = best_config
                if rank == 0:
                    update_json_file(data)
                dist_barrier()
            tile_size_config = best_config[0]
            TRITON_KERNELS_CACHE[method_name] = [kernel, tile_size_config]
