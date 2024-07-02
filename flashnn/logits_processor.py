# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import torch

from .kernel_backend import BackendKernel
from .triton_kernels.logits_processor import triton_logits_processor_forward


class LogitsProcessor(BackendKernel):
    def __init__(self, penalty_ty: str):
        super().__init__()
        assert penalty_ty in ["REPETITION", "PRESENCE"]
        self.penalty_ty = penalty_ty

    def _triton_impl(
        self,
        scores,
        penalty,
        all_input_ids_ptr,
        input_ids_length,
        max_ids_length,
        input_ids=None,
    ):
        triton_logits_processor_forward(
            scores,
            penalty,
            all_input_ids_ptr,
            input_ids_length,
            max_ids_length,
            self.penalty_ty,
        )

    def _torch_impl(
        self,
        scores,
        penalty,
        all_input_ids_ptr,
        input_ids_length,
        max_ids_length,
        ids_list,
    ):
        assert ids_list is not None
        for batch_id in range(scores.shape[0]):
            input_ids = ids_list[batch_id]
            scores_batch = scores[batch_id, None, :]
            score = torch.gather(scores_batch, 1, input_ids)
            if self.penalty_ty == "REPETITION":
                score = torch.where(
                    score < 0, score * penalty[batch_id], score / penalty[batch_id]
                )
            elif self.penalty_ty == "PRESENCE":
                score -= penalty[batch_id]

            scores_batch.scatter_(1, input_ids, score)
            scores[batch_id, ...] = scores_batch

    def forward(
        self,
        scores: torch.FloatTensor,  # shape: [request_nums, out_dim]
        penalty: torch.FloatTensor,  # shape: [request_nums]
        all_input_ids_ptr: torch.LongTensor,  # shape: [request_nums]
        input_ids_length: torch.IntTensor,  # shape: [request_nums]
        max_ids_length: int,
        ids_list=None,
    ):
        BackendKernel.forward(
            self,
            scores,
            penalty,
            all_input_ids_ptr,
            input_ids_length,
            max_ids_length,
            ids_list,
        )
