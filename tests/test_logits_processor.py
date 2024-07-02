# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import flashnn
import torch

torch.manual_seed(0)


class TestLogitsProcessor(unittest.TestCase):
    def test_bladnn_repetition_and_presence_penalty(self):
        M, N = 2048, 1048
        dtype = torch.half
        scores = torch.randn((M, N), dtype=dtype).cuda()
        repetition_penalty_tensor = (torch.rand(M) + 1.0).float().cuda()
        repetition_penalty_tensor[0] = 1.0
        repetition_penalty_tensor[1] = 1.0
        presence_penalty_tensor = (torch.rand(M)).float().cuda()
        presence_penalty_tensor[2] = 0.0
        presence_penalty_tensor[4] = 0.0
        ids_list = []
        input_ids_ptr = []
        input_ids_length = []
        max_ids_length = 0
        for index in range(M):
            input_ids = torch.randperm(N)[: (index + 1)]
            input_ids = (input_ids.unsqueeze(0)).cuda()
            ids_list.append(input_ids)
            input_ids_length.append(input_ids.shape[-1])
            input_ids_ptr.append(input_ids.contiguous().data_ptr())
            if max_ids_length < input_ids.shape[-1]:
                max_ids_length = input_ids.shape[-1]

        input_ids_ptr_tensor = torch.tensor(input_ids_ptr, dtype=torch.long).cuda()
        input_ids_length_tensor = torch.tensor(
            input_ids_length, dtype=torch.int32
        ).cuda()

        copy_scores_repetition = scores.clone()
        # run repetition reference
        flashnn.set_use_triton(False)
        p = flashnn.LogitsProcessor(penalty_ty="REPETITION")
        p(
            copy_scores_repetition,
            repetition_penalty_tensor,
            input_ids_ptr_tensor,
            input_ids_length_tensor,
            max_ids_length,
            ids_list,
        )

        # run triton repetition forward
        flashnn.set_use_triton(True)
        p = flashnn.LogitsProcessor(penalty_ty="REPETITION")
        p(
            scores,
            repetition_penalty_tensor,
            input_ids_ptr_tensor,
            input_ids_length_tensor,
            max_ids_length,
        )
        torch.testing.assert_close(
            scores, copy_scores_repetition, rtol=0.001, atol=0.001
        )

        copy_scores_presence = scores.clone()
        # run presence reference
        flashnn.set_use_triton(False)
        p = flashnn.LogitsProcessor(penalty_ty="PRESENCE")
        p(
            copy_scores_presence,
            presence_penalty_tensor,
            input_ids_ptr_tensor,
            input_ids_length_tensor,
            max_ids_length,
            ids_list,
        )

        # run triton presence forward
        flashnn.set_use_triton(True)
        presence_p = flashnn.LogitsProcessor(penalty_ty="PRESENCE")
        presence_p(
            scores,
            presence_penalty_tensor,
            input_ids_ptr_tensor,
            input_ids_length_tensor,
            max_ids_length,
        )
        torch.testing.assert_close(scores, copy_scores_presence, rtol=0.001, atol=0.001)

    def test_bladnn_repetition_and_presence_penalty_with_repeat_input_ids(self):
        M, N = 150, 5000
        dtype = torch.half
        scores = torch.randn((M, N), dtype=dtype).cuda()
        repetition_penalty_tensor = (torch.rand(M) + 1.0).float().cuda()
        repetition_penalty_tensor[0] = 1.0
        repetition_penalty_tensor[1] = 1.0
        presence_penalty_tensor = (torch.rand(M)).float().cuda()
        presence_penalty_tensor[2] = 0.0
        presence_penalty_tensor[4] = 0.0
        ids_list = []
        input_ids_ptr = []
        input_ids_length = []
        max_ids_length = 0
        for _ in range(M):
            input_ids = torch.randint(0, N // 10, (N,))
            input_ids = (input_ids.unsqueeze(0)).cuda()
            ids_list.append(input_ids)
            input_ids_length.append(input_ids.shape[-1])
            input_ids_ptr.append(input_ids.contiguous().data_ptr())
            if max_ids_length < input_ids.shape[-1]:
                max_ids_length = input_ids.shape[-1]

        input_ids_ptr_tensor = torch.tensor(input_ids_ptr, dtype=torch.long).cuda()
        input_ids_length_tensor = torch.tensor(
            input_ids_length, dtype=torch.int32
        ).cuda()

        copy_scores_repetition = scores.clone()

        # run repetition reference
        flashnn.set_use_triton(False)
        p = flashnn.LogitsProcessor(penalty_ty="REPETITION")
        p(
            copy_scores_repetition,
            repetition_penalty_tensor,
            input_ids_ptr_tensor,
            input_ids_length_tensor,
            max_ids_length,
            ids_list,
        )

        # run triton repetition forward
        flashnn.set_use_triton(True)
        p = flashnn.LogitsProcessor(penalty_ty="REPETITION")
        p(
            scores,
            repetition_penalty_tensor,
            input_ids_ptr_tensor,
            input_ids_length_tensor,
            max_ids_length,
        )
        torch.testing.assert_close(
            scores, copy_scores_repetition, rtol=0.001, atol=0.001
        )

        copy_scores_presence = scores.clone()
        # run presence reference
        flashnn.set_use_triton(False)
        p = flashnn.LogitsProcessor(penalty_ty="PRESENCE")
        p(
            copy_scores_presence,
            presence_penalty_tensor,
            input_ids_ptr_tensor,
            input_ids_length_tensor,
            max_ids_length,
            ids_list,
        )

        # run triton presence forward
        flashnn.set_use_triton(True)
        presence_p = flashnn.LogitsProcessor(penalty_ty="PRESENCE")
        presence_p(
            scores,
            presence_penalty_tensor,
            input_ids_ptr_tensor,
            input_ids_length_tensor,
            max_ids_length,
        )
        torch.testing.assert_close(scores, copy_scores_presence, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
