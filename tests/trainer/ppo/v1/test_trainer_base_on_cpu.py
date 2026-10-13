# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _CustomSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _trainer_with_filter_groups(filter_groups: dict, trainer_mode: str = "sync") -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = trainer_mode
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": filter_groups},
            "data": {"train_batch_size": 64, "gen_batch_size": 8},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "trainer": {
                "v1": {
                    trainer_mode: {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            },
        }
    )
    return trainer


def test_builtin_sampler_class_follows_trainer_mode():
    sync_sampler = _trainer_with_filter_groups({"enable": False}, trainer_mode="sync")._build_replay_buffer()
    async_samplers = [
        _trainer_with_filter_groups({"enable": True, "metric": "acc"}, trainer_mode=mode)._build_replay_buffer()
        for mode in ("colocate_async", "separate_async")
    ]

    assert type(sync_sampler) is ReplayBuffer
    assert all(type(sampler) is ReplayBufferAsync for sampler in async_samplers)
    assert all(sampler.filter_groups_metric == "acc" for sampler in async_samplers)
    assert all(sampler.train_batch_size is None for sampler in async_samplers)
    assert all(sampler.gen_batch_size is None for sampler in async_samplers)


def test_custom_sampler_skips_builtin_filter_groups_validation():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})
    trainer.config.trainer.v1.sampler.custom_sampler = {"path": "custom.py", "name": "CustomSampler"}

    with (
        patch("verl.trainer.ppo.v1.trainer_base.load_extern_type", return_value=_CustomSampler),
        patch.object(trainer, "_resolve_filter_groups_metric") as resolve_filter_groups_metric,
    ):
        sampler = trainer._build_replay_buffer()

    resolve_filter_groups_metric.assert_not_called()
    assert isinstance(sampler, _CustomSampler)
    assert "filter_groups_metric" not in sampler.kwargs
    assert "train_batch_size" not in sampler.kwargs
    assert "gen_batch_size" not in sampler.kwargs
    assert "max_inflight_gen_batches" not in sampler.kwargs
    assert "sync_refill_failed_groups" not in sampler.kwargs


def test_builtin_filter_groups_uses_default_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})

    sampler = trainer._build_replay_buffer()

    assert sampler.filter_groups_metric == "acc"
    assert sampler.train_batch_size == 64
    assert sampler.gen_batch_size == 1
    assert sampler.max_inflight_gen_batches == 1


def test_builtin_filter_groups_forwards_configured_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_inflight_gen_batches": 3})

    sampler = trainer._build_replay_buffer()

    assert sampler.max_inflight_gen_batches == 3


def test_builtin_sync_failure_refill_forces_single_prompt_generation():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True

    sampler = trainer._build_replay_buffer()

    assert sampler.sync_refill_failed_groups is True
    assert sampler.gen_batch_size == 1


@pytest.mark.parametrize(
    ("sampler_key", "value"), [("sync_refill_failed_groups", True), ("sync_over_provision_prompts", 2)]
)
def test_exact_prompt_counts_override_dataloader_generation_batch_size(sampler_key, value):
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler[sampler_key] = value
    trainer.config.data.update(
        {
            "train_files": [],
            "val_files": [],
            "train_max_samples": -1,
            "val_max_samples": -1,
            "dataloader_num_workers": 0,
            "val_batch_size": 1,
            "validation_shuffle": False,
        }
    )
    trainer.config.trainer.total_epochs = 1
    trainer.config.trainer.total_training_steps = None
    trainer.parameter_sync_step = 1
    trainer.tokenizer = None
    trainer.processor = None

    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", side_effect=[[{}, {}], [{}]]),
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        trainer._init_dataloader()

    assert trainer.config.data.gen_batch_size == 1
    assert dataloader.call_args_list[0].kwargs["batch_size"] == 1
    warning.assert_any_call("data.gen_batch_size=8 is overridden to 1.")


def test_builtin_filter_groups_warns_when_total_generation_limit_is_configured():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_num_gen_batches": 10})

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        trainer._build_replay_buffer()

    warning.assert_called_once_with(
        "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
        "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
        10,
    )


@pytest.mark.parametrize("score_centering", [True, False])
def test_update_actor_passes_score_centering_flag(score_centering):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 2,
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "ppo_epochs": 1,
                    "data_loader_seed": 1,
                    "shuffle": False,
                    "policy_loss": {"rollout_correction": {"score_centering": score_centering}},
                },
                "rollout": {"n": 4, "temperature": 1.0},
            }
        }
    )
    trainer.actor_rollout_wg = MagicMock()
    trainer.actor_rollout_wg.update_actor.return_value = {"metrics": {"mfu": 0.0}}
    batch = SimpleNamespace(extra_info={})

    trainer._update_actor(batch, {})

    assert batch.extra_info["score_centering"] is score_centering


def _over_provisioned_trainer(over_provision: int) -> _StubTrainer:
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.data.update({"train_batch_size": 4, "gen_batch_size": 1})
    trainer.config.trainer.v1.sampler.sync_over_provision_prompts = over_provision
    trainer.global_steps = 5
    trainer._fetch_one_gen_batch = lambda: tu.get_tensordict(
        tensor_dict={"uid": np.array([str(uuid.uuid4())], dtype=object), "raw_prompt": [[{"content": "p"}]]},
        non_tensor_dict={},
    )
    trainer.submitted = []
    trainer._submit_batch_to_rollout = trainer.submitted.append
    return trainer


@pytest.mark.parametrize(("over_provision", "expected_target"), [(0, None), (2, 4)])
def test_sync_over_provision_submits_spare_prompts_and_stamps_target(over_provision, expected_target):
    """The step launches train_batch_size + K prompts; the stamped target tells the
    rollout how many groups to keep, so the replay buffer's batch stays train_batch_size."""
    trainer = _over_provisioned_trainer(over_provision)

    trainer._add_batch_to_generate()

    (batch,) = trainer.submitted
    assert len(batch) == 4 + over_provision
    assert tu.get(batch, "__target_groups__") == expected_target
    assert tu.get(batch, "global_steps") == 5


@pytest.mark.parametrize(
    "incompatible",
    [
        {"trainer_mode": "colocate_async"},
        {"filter_groups": {"enable": True, "metric": "acc"}},
        {"sync_refill_failed_groups": True},
    ],
)
def test_sync_over_provision_rejects_paths_that_refill_or_evict_groups(incompatible):
    trainer = _trainer_with_filter_groups(
        incompatible.get("filter_groups", {"enable": False}), trainer_mode=incompatible.get("trainer_mode", "sync")
    )
    trainer.config.trainer.v1.sampler.sync_over_provision_prompts = 2
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = incompatible.get("sync_refill_failed_groups", False)

    with pytest.raises(ValueError, match="sync_over_provision_prompts requires trainer_mode=sync"):
        trainer._sync_over_provision_prompts()
