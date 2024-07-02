# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import os

import torch
import torch.nn

autotune_triton_kernels = os.getenv("autotune_triton_kernels", False)
use_triton = os.getenv("USE_TRITON_KERNELS", True)


def get_autotune_triton_kernels():
    return autotune_triton_kernels


def set_autotune_triton_kernels(v: bool):
    global autotune_triton_kernels
    autotune_triton_kernels = v


def get_use_triton():
    return use_triton


def set_use_triton(v: bool):
    global use_triton
    use_triton = v


def is_hip():
    import torch

    return torch.version.hip is not None


THREADS_PER_WARP = 32

if is_hip():
    THREADS_PER_WARP = 64


class BackendKernel(torch.nn.Module):
    """
    This class is used to implement a kernel for multiple backends.
    The kernel implementation is defined in the method named _{backend}_impl.
    The method _{backend}_impl should be implemented in the subclass of BackendKernel.
    Right now the backend can be "triton" and "torch".
    """

    def __init__(self):
        super().__init__()
        self.backends = ["triton", "torch"]

    def _triton_impl(self, *args, **kwargs):
        raise NotImplementedError

    def _torch_impl(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        for backend in self.backends:
            if backend == "triton" and not get_use_triton():
                continue
            method = getattr(self, f"_{backend}_impl")
            if method.__func__ != getattr(BackendKernel, f"_{backend}_impl"):
                return method(*args, **kwargs)
        raise ValueError("No valid backend found.")
