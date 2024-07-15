import types

import torch
import triton

THREADS_PER_WARP = 64 if torch.version.hip is not None else 32

TRITON_KERNELS_CACHE = {}


def compile_and_cache_kernels(fn, method_name, grid, kwargs, const_kwargs, grid_kwargs=None):
    """
    Upon first encountering a kernel with method_name, we compile it and cache the compiled results.
    Subsequent encounters with the same method_name will fetch the corresponding compiled results
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
    assert isinstance(fn, triton.runtime.autotuner.Autotuner) or isinstance(fn, triton.runtime.jit.JITFunction)
    if method_name in TRITON_KERNELS_CACHE:
        kernel = TRITON_KERNELS_CACHE[method_name][0]
        if isinstance(grid, types.FunctionType) and isinstance(fn, triton.runtime.autotuner.Autotuner):
            grid = TRITON_KERNELS_CACHE[method_name][-1]
        kernel[grid](*kwargs)
    else:
        results = fn[grid](*kwargs) if const_kwargs is None else fn[grid](*kwargs, **const_kwargs)
        TRITON_KERNELS_CACHE[method_name] = list(results) if isinstance(results, tuple) else [results]
        if isinstance(fn, triton.runtime.autotuner.Autotuner):
            best_config = fn.best_config
            if isinstance(grid, types.FunctionType):
                META = {key: best_config.kwargs[key] for key in grid_kwargs}
                grid = grid(META)
                TRITON_KERNELS_CACHE[method_name].append(grid)

