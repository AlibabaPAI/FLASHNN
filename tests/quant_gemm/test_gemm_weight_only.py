import unittest

import torch

import flashnn

torch.manual_seed(0)
from torch.ao.quantization.observer import PerChannelMinMaxObserver


def block_quantize(x, scales, block_size, quant_min, quant_max, zero_points=None):
    res = torch.zeros_like(x, dtype=torch.int8)
    assert x.dim() == 2, "block_fake_quant only support tensor with dim=2."
    k = x.shape[0]
    inters_in_k = scales.shape[0]
    if zero_points is None:
        zero_points = torch.zeros_like(scales)
    assert k // block_size == inters_in_k
    for i in range(inters_in_k):
        k_start = i * block_size
        k_end = (i + 1) * block_size
        x_in_iter = x[k_start:k_end, :]
        scale_in_iter = scales[i]
        zero_points_in_iter = zero_points[i]
        x_in_iter_dequant = (
            (x_in_iter / scale_in_iter + zero_points_in_iter)
            .round()
            .clamp(quant_min, quant_max)
        )
        res[k_start:k_end, :] = x_in_iter_dequant
    return res


def channel_quantize(x, scales, quant_min, quant_max, zero_points=None):
    res = torch.zeros_like(x, dtype=torch.int8)
    assert x.dim() == 2, "block_fake_quant only support tensor with dim=2."
    if zero_points is None:
        zero_points = torch.zeros_like(scales)
    q_x = (x / scales + zero_points).round().clamp(quant_min, quant_max)
    res.copy_(q_x)
    return res


def block_dequantize(x, scales, block_size, zero_points=None):
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


def channel_dequantize(x, scales, zero_points=None):
    if zero_points is None:
        zero_points = torch.zeros_like(scales)
    return (x - zero_points) * scales


def random_tensor(shape, dtype, device, mean=0, std=1):
    return torch.empty(shape, dtype=dtype, device=device).normal_(mean, std)


class TestWeightOnlyQGemm(unittest.TestCase):
    def _run_test_weight_layout_transform(
        self,
        gemm_m,
        gemm_n,
        gemm_k,
        compute_type,
        weight_dtype,
        use_bias=False,
        is_sub_channel=False,
        quant_method="symmetric",
    ):
        weights = random_tensor(
            (gemm_k, gemm_n), dtype=compute_type, device="cuda", mean=0, std=0.002
        )
        # weights= torch.randn(gemm_k, gemm_n, dtype=compute_type, device="cuda")

        inputs = torch.randn(gemm_m, gemm_k, dtype=compute_type, device="cuda")
        bias = torch.randn(gemm_n, dtype=compute_type, device="cuda")

        if weight_dtype == torch.int8:
            quant_min, quant_max = -128, 127
        elif weight_dtype == torch.quint4x2:
            quant_min, quant_max = -8, 7
        else:
            raise RuntimeError("Unsupported weight dtype")
        zero_points = None
        k_per_scale = 0
        if not is_sub_channel:
            qscheme = torch.per_channel_symmetric
            if quant_method == "asymmetric":
                qscheme = torch.per_channel_affine
            min_max_observer = PerChannelMinMaxObserver.with_args(
                ch_axis=1,
                quant_min=quant_min,
                quant_max=quant_max,
                qscheme=qscheme,
            )

            obs = min_max_observer().to(compute_type).cuda()
            obs(weights)
            if quant_method == "asymmetric":
                scales, zero_points = obs.calculate_qparams()
                zero_points = zero_points.to(scales.dtype)
            else:
                scales, _ = obs.calculate_qparams()
                zero_points = None
            q_weights = channel_quantize(
                weights, scales.unsqueeze(0), quant_min, quant_max, zero_points
            )
            if quant_method == "asymmetric":
                unsqueeze_zero_points = zero_points.unsqueeze(0)
            else:
                unsqueeze_zero_points = None
            dq_weights = channel_dequantize(
                q_weights, scales.unsqueeze(0), zero_points=unsqueeze_zero_points
            )
        else:
            k_per_scale = 64
            scale_k = gemm_k // k_per_scale
            scales = torch.randn(scale_k, gemm_n, dtype=compute_type, device="cuda")
            zero_points = None
            if quant_method == "asymmetric":
                zero_points = torch.randint_like(scales, low=quant_min, high=quant_max)
            q_weights = block_quantize(
                weights, scales, k_per_scale, quant_min, quant_max, zero_points
            )
            dq_weights = block_dequantize(q_weights, scales, k_per_scale, zero_points)
        reference_result = torch.matmul(inputs, dq_weights)
        q_weights = q_weights.cpu()
        # if weight_dtype == torch.quint4x2:
        #     q_weights = pack_int8_tensor_to_packed_int4(q_weights)
        q_weights = q_weights.permute(1, 0).contiguous().cuda()
        if use_bias:
            reference_result += bias.unsqueeze(0)
        else:
            bias = None
        tri_result = tri_blade.GemmWeightOnly()(
            inputs, q_weights, scales, bias, zero_points
        )
        torch.testing.assert_close(
            tri_result, reference_result, rtol=0.001, atol=0.002, check_dtype=False
        )

    def test_weight_layout_transform(self):
        gemm_m, gemm_n, gemm_k = 3, 1024, 4096
        compute_type = torch.float16
        weight_dtype = [torch.int8]
        use_bias = [True, False]
        is_sub_channel = [True, False]
        quant_methods = ["symmetric", "asymmetric"]
        for quant_method in quant_methods:
            for w in weight_dtype:
                for b in use_bias:
                    for s in is_sub_channel:
                        print(
                            f"Testing weight_layout_transform with weight_dtype={w} use_bias={b} is_sub_channel={s} quant_method={quant_method}.....",
                            flush=True,
                        )
                        self._run_test_weight_layout_transform(
                            gemm_m, gemm_n, gemm_k, compute_type, w, b, s, quant_method
                        )
                        print(
                            f"Pass test of weight_layout_transform with weight_dtype={w} use_bias={b} is_sub_channel={s} quant_method={quant_method}!",
                            flush=True,
                        )


if __name__ == "__main__":
    unittest.main()
