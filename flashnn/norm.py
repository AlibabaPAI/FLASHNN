# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import numbers
from typing import List, Union

import torch
import torch.nn as nn
from torch import Size

from .kernel_backend import BackendKernel
from .triton_kernels.layer_norm import triton_layer_norm_forward
from .triton_kernels.layer_norm_dquant import triton_layer_norm_dquant_forward
from .triton_kernels.rms_norm import triton_rmsnorm_forward
from .triton_kernels.rms_norm_dquant import triton_rmsnorm_dquant_forward


class RMSNorm(BackendKernel):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def _triton_impl(self, x):
        return triton_rmsnorm_forward(x, self.weight, self.eps).type_as(x)

    def _torch_impl(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def forward(self, x):
        return BackendKernel.forward(self, x)


class RMSNormDquant(BackendKernel):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm_dquant(self, x):
        norm_out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        norm_out *= self.weight
        max_norm_out = torch.max(torch.abs(norm_out), dim=-1, keepdim=True)[0]
        scale = max_norm_out / 127.0
        quantized_norm_out = torch.round(norm_out / scale)
        quantized_norm_out = torch.clamp(quantized_norm_out, -128.0, 127.0)
        scale = scale.reshape(x.shape[:-1])
        return quantized_norm_out.to(torch.int8), scale

    def _triton_impl(self, x):
        return triton_rmsnorm_dquant_forward(x, self.weight, self.eps)

    def _torch_impl(self, x):
        torch_out, torch_scale = self._norm_dquant(x.float())
        return torch_out, torch_scale.type_as(x)

    def forward(self, x):
        return BackendKernel.forward(self, x)


_shape_t = Union[int, List[int], Size]


class LayerNorm(BackendKernel):
    def __init__(self, dim: _shape_t, eps: float = 1e-5):
        super().__init__()
        if isinstance(dim, numbers.Integral):
            dim = (dim,)
        if isinstance(dim, tuple):
            self.dim = dim
        else:
            self.dim = tuple(dim)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))

    def _torch_impl(self, x):
        return torch.nn.functional.layer_norm(
            x, self.dim, self.weight, self.bias, self.eps
        )

    def _triton_impl(self, x):
        return triton_layer_norm_forward(x, self.weight, self.bias, self.eps)

    def forward(self, x):
        return BackendKernel.forward(self, x)


class LayernormDquant(BackendKernel):
    def __init__(self, dim: _shape_t, eps: float = 1e-5):
        super().__init__()
        if isinstance(dim, numbers.Integral):
            dim = (dim,)
        if isinstance(dim, tuple):
            self.dim = dim
        else:
            self.dim = tuple(dim)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(self.dim))
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))

    def _triton_impl(self, x):
        _, quantized_norm_out, scale = triton_layer_norm_dquant_forward(
            x, self.weight, self.bias, self.eps
        )
        return quantized_norm_out, scale.reshape(-1, 1)

    def _torch_impl(self, x):
        norm_out = torch.nn.functional.layer_norm(
            x, self.dim, self.weight, self.bias, self.eps
        )
        max_norm_out = torch.max(torch.abs(norm_out), dim=-1, keepdim=True)[0]
        scale = max_norm_out / 127.0
        quantized_norm_out = torch.round(norm_out / scale)
        quantized_norm_out = torch.clamp(quantized_norm_out, -128.0, 127.0)
        return quantized_norm_out.to(torch.int8), scale.reshape(-1, 1)

    def forward(self, x):
        return BackendKernel.forward(self, x)
