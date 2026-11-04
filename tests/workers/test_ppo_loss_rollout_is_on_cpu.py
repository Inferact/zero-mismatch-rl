# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The engine workers' actor loss applies the trainer's rollout IS weights.

The trainer adds ``rollout_is_weights`` to the batch (rollout correction, decoupled mode); the update
converts the batch to the no-padding layout and the engine hands each micro-batch to ``ppo_loss``.

This guards existing behaviour (it passes on code before the K31 --bi arm and needs no mLite); the composed
--bi 0 test in tests/utils/test_k31_recipe_on_cpu.py (TestBatchVariantCorrection) is the one that fails on a
recipe without --bi.
"""

from itertools import accumulate

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import ActorConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding

PROMPT, RESPONSE = 3, 4
PROMPT_LENS, RESPONSE_LENS = (3, 2), (4, 2)


def _batch(weights):
    g = torch.Generator().manual_seed(0)
    bs = len(PROMPT_LENS)
    attention_mask = torch.zeros(bs, PROMPT + RESPONSE, dtype=torch.long)
    response_mask = torch.zeros(bs, RESPONSE, dtype=torch.long)
    for i, (p, r) in enumerate(zip(PROMPT_LENS, RESPONSE_LENS, strict=True)):
        attention_mask[i, PROMPT - p : PROMPT + r] = 1
        response_mask[i, :r] = 1
    input_ids = torch.randint(1, 100, (bs, PROMPT + RESPONSE), generator=g) * attention_mask
    old_log_probs = -torch.rand(bs, RESPONSE, generator=g) * response_mask
    advantages = torch.randn(bs, RESPONSE, generator=g) * response_mask
    data = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": (attention_mask.cumsum(-1) - 1).clamp(min=0),
            "response_mask": response_mask,
            "prompts": input_ids[:, :PROMPT],
            "responses": input_ids[:, PROMPT:],
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "rollout_is_weights": weights * response_mask,
        },
        batch_size=[bs],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(response_mask.sum()), global_batch_size=bs)
    data = left_right_2_no_padding(data)
    # On-policy model output: each response token's log-prob sits one position before the token.
    values = []
    for i, (p, r) in enumerate(zip(PROMPT_LENS, RESPONSE_LENS, strict=True)):
        seq = torch.zeros(p + r)
        seq[p - 1 : p + r - 1] = old_log_probs[i, :r]
        values.append(seq)
    offsets = torch.tensor([0, *accumulate(p + r for p, r in zip(PROMPT_LENS, RESPONSE_LENS, strict=True))])
    log_probs = torch.nested.nested_tensor_from_jagged(torch.cat(values), offsets=offsets)
    return data, {"log_probs": log_probs}, advantages, response_mask


class TestPpoLossRolloutIS:
    @pytest.mark.parametrize(
        "weights",
        [
            torch.ones(2, RESPONSE),
            torch.tensor([[0.5, 2.0, 1.0, 0.0], [1.5, 0.25, 0.0, 0.0]]),
        ],
        ids=["unit-weights", "truncated-is-weights"],
    )
    def test_weights_scale_the_policy_gradient_term(self, weights):
        data, model_output, advantages, mask = _batch(weights)
        config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size=2, clip_ratio=0.2)
        loss, _ = ppo_loss(config, model_output, data)
        # ratio 1 (on policy): the token-mean loss is -mean(A * w) over response tokens
        expected = -(advantages * weights * mask).sum() / mask.sum()
        torch.testing.assert_close(loss, expected)
