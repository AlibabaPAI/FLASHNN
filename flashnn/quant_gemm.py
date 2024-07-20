# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import torch

from flashnn.kernel_backend import BackendKernel
from flashnn.triton_kernels.dynamic_quant import triton_dynamic_quantize
from flashnn.triton_kernels.gemm_a8w8 import triton_gemm_a8w8_forward
from flashnn.triton_kernels.gemm_a16w4 import triton_gemm_a16w4_forward
from flashnn.triton_kernels.gemm_a16w8 import triton_gemm_a16w8_forward


def _channel_dequantize(x, scales, zero_points=None):
    if zero_points is None:
        zero_points = torch.zeros_like(scales)
    return (x - zero_points) * scales


def _block_dequantize(x, scales, block_size, zero_points=None):
    res = torch.zeros_like(x, dtype=torch.half)
    assert x.dim() == 2, "block_fake_quant only support tensor with dim=2."
    if zero_points is None:
        zero_points = torch.zeros_like(scales)
    k = x.shape[0]
    inters_in_k = scales.shape[0]
    assert k // block_size == inters_in_k
    for i in range(inters_in_k):
        k_start = i * block_size
        k_end = (i + 1) * block_size
        x_in_iter = x[k_start:k_end, :]
        scale_in_iter = scales[i]
        zero_points_in_iter = zero_points[i]
        x_in_iter_dequant = (x_in_iter - zero_points_in_iter) * scale_in_iter
        res[k_start:k_end, :] = x_in_iter_dequant
    return res


class DynamicQuantize(BackendKernel):
    """Dynamic quantization.
        input type: float16/bfloat16.
        output type: int8.
        scale type: float16.

    Returns:
        out <- Quantize(input)
        scale <- updated scale
    """

    def __init__(self):
        super().__init__()

    def _torch_impl(self, input):
        max_input = input.abs().max(-1, keepdim=True)[0]
        scale = max_input / 127.0
        out = torch.round(input / scale)
        return out.to(torch.int8), scale.half().squeeze(-1)

    def _triton_impl(self, input):
        out = torch.empty(input.shape, dtype=torch.int8, device=input.device)
        scale = torch.empty(input.shape[0], dtype=torch.half, device=input.device)
        triton_dynamic_quantize(out, input, scale)
        return out, scale

    def forward(self, input):
        return BackendKernel.forward(self, input)


class GemmA8W8(BackendKernel):
    """GEMM with triton A8W8 quantization.

    Args:
        m: number of rows of A.
        n: number of columns of B.
        alpha_row(optional): scale of activation, default is 1.
        alpha_col(optional): scale of weight, default is 1,
        dtype: type of output

    Returns:
        (fp16/bf16)out <- ((int8)A[m, k] * (int8)trans(B[n, k])) *
               ((fp16/bf16)scale_row[m, 1] * (fp16/bf16)scale_col[1, n])
    """

    def __init__(
        self,
        out_ty: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.out_ty = out_ty

    def _triton_impl(self, a, b, alpha_row, alpha_col):
        out = torch.empty([a.shape[0], b.shape[0]], dtype=self.out_ty, device=a.device)
        triton_gemm_a8w8_forward(out, a, b, alpha_row, alpha_col)
        return out

    def _torch_impl(self, a, b, alpha_row, alpha_col):
        b = b.transpose(0, 1)
        x = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        scale = torch.matmul(
            torch.squeeze(alpha_row).flatten().unsqueeze(-1),
            torch.squeeze(alpha_col).flatten().unsqueeze(0),
        )
        out = torch.mul(x, scale)
        return out.to(self.out_ty)

    def forward(self, a, b, alpha_row, alpha_col):
        return BackendKernel.forward(self, a, b, alpha_row, alpha_col)


class GemmWeightOnly(BackendKernel):
    """GEMM with triton weight only quantization."""

    def __init__(self, out_ty: torch.dtype = torch.float16):
        super().__init__()
        self.out_ty = out_ty

    def _triton_impl(self, act, quant_w, scale_w, bias=None, zero_points=None):
        assert quant_w.dtype == torch.int8, "Weight must be int8 type"
        assert act.is_contiguous(), "Activation must be contiguous"
        assert quant_w.is_contiguous(), "Weight must be contiguous"
        assert act.shape[1] == quant_w.shape[1], "Matrix B must be transposed"
        if quant_w.size(0) == scale_w.size(-1) / 2:
            weight_type = torch.quint4x2
        elif quant_w.size(0) == scale_w.size(-1):
            weight_type = torch.int8
        else:
            weight_type = None
        if weight_type == torch.int8:
            triton_out = torch.zeros(
                [act.shape[0], quant_w.shape[0]], dtype=self.out_ty, device=act.device
            )
            triton_gemm_a16w8_forward(
                triton_out, act, quant_w, scale_w, bias=bias, zero_points=zero_points
            )
        elif weight_type == torch.quint4x2:
            triton_out = torch.zeros(
                [act.shape[0], quant_w.shape[0] * 2],
                dtype=self.out_ty,
                device=act.device,
            )
            triton_gemm_a16w4_forward(
                triton_out, act, quant_w, scale_w, bias=bias, zero_points=zero_points
            )
        else:
            raise AssertionError("Gemm WeightOnly only support int8 or int4x2")
        return triton_out

    def _torch_impl(self, act, quant_w, scale_w, bias=None, zero_points=None):
        assert act.shape[1] == quant_w.shape[1], "Matrix B must be transposed"
        if quant_w.size(0) == scale_w.size(-1) / 2:
            weight_type = torch.quint4x2
        elif quant_w.size(0) == scale_w.size(-1):
            weight_type = torch.int8
        else:
            weight_type = None
        if weight_type == torch.int8:
            quant_w = quant_w.transpose(0, 1)
            scale_w = scale_w.squeeze()
            if scale_w.dim() == 1:  # channel dequantize
                dq_weights = _channel_dequantize(
                    quant_w, scale_w.unsqueeze(0), zero_points=zero_points
                )
            else:  # block dequantize
                k_per_scale = int(act.shape[1] / scale_w.shape[0])
                dq_weights = _block_dequantize(
                    quant_w, scale_w, k_per_scale, zero_points=zero_points
                )
            out = torch.matmul(act.to(torch.float32), dq_weights.to(torch.float32)).to(
                self.out_ty
            )
            if bias is not None:
                out += bias
            return out.to(self.out_ty)
        elif weight_type == torch.quint4x2:
            b_int4_low = (quant_w & 0x0F).to(torch.int8)  # (N/2) x K
            b_int4_low = (
                torch.where(b_int4_low > 7, (b_int4_low | 0xF0), b_int4_low)
            ).to(torch.int8)
            b_int4_low = b_int4_low.transpose(0, 1)  # K x (N/2)

            b_int4_high = (((quant_w & 0xF0) >> 4) & 0x0F).to(torch.int8)
            b_int4_high = (
                torch.where(
                    b_int4_high > 7, (b_int4_high | 0xF0).to(torch.int8), b_int4_high
                )
            ).to(torch.int8)
            b_int4_high = b_int4_high.transpose(0, 1)

            b = torch.stack((b_int4_low, b_int4_high), dim=2)
            b = b.view(b_int4_low.size(0), -1)  # K x N
            scale_w = scale_w.squeeze()
            if scale_w.dim() == 1:  # channel dequantize
                dq_weights = _channel_dequantize(
                    b, scale_w.unsqueeze(0), zero_points=zero_points
                )
            else:  # block dequantize
                k_per_scale = int(act.shape[1] / scale_w.shape[0])
                dq_weights = _block_dequantize(
                    b, scale_w, k_per_scale, zero_points=zero_points
                )
            out = torch.matmul(act.to(torch.float32), dq_weights.to(torch.float32))
            if bias is not None:
                out += bias
            return out.to(self.out_ty)
        else:
            raise AssertionError("Gemm WeightOnly only support int8 or int4x2")

    def forward(self, act, quant_w, scale_w, bias=None, zero_points=None):
        return BackendKernel.forward(self, act, quant_w, scale_w, bias, zero_points)
