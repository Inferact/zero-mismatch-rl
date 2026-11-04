"""The rollout sampling settings whose logprobs the K31 trainer reproduces."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from megatron.lite.model.k31.vllm.protocol import validate_rollout_sampling


class TestRolloutSamplingContract:
    @pytest.mark.parametrize(
        "rollout",
        [
            {},
            {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
            {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
            {"temperature": 1.0, "logprobs_mode": "raw_logprobs"},
            {"temperature": 1.0, "top_p": None, "top_k": None},
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "val_kwargs": {"do_sample": False, "temperature": 0.0, "top_k": -1},
            },
            SimpleNamespace(
                temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0
            ),
        ],
        ids=[
            "empty",
            "verl-defaults",
            "temperature-only",
            "raw-logprobs-at-t1",
            "unset-values",
            "greedy-validation",
            "attribute-config",
        ],
    )
    def test_reproducible_settings_pass(self, rollout):
        validate_rollout_sampling(rollout)

    @pytest.mark.parametrize(
        "rollout,knob",
        [
            ({"top_p": 0.95}, "top_p"),
            ({"top_k": 50}, "top_k"),
            ({"min_p": 0.05}, "min_p"),
            ({"repetition_penalty": 1.1}, "repetition_penalty"),
            ({"presence_penalty": 0.5}, "presence_penalty"),
            ({"frequency_penalty": 0.5}, "frequency_penalty"),
            ({"temperature": 0.0}, "temperature"),
            ({"temperature": 0.005}, "temperature"),
            ({"temperature": 0.7, "logprobs_mode": "raw_logprobs"}, "logprobs_mode"),
            (SimpleNamespace(top_p=0.9, top_k=-1), "top_p"),
            ({"do_sample": False}, "do_sample"),
            ({"logit_bias": {"5": -100.0}}, "logit_bias"),
        ],
        ids=[
            "top-p",
            "top-k",
            "min-p",
            "repetition",
            "presence",
            "frequency",
            "greedy",
            "clamped-temperature",
            "raw-logprobs",
            "attribute-config",
            "no-sampling",
            "logit-bias",
        ],
    )
    def test_sampling_processing_is_refused(self, rollout, knob):
        with pytest.raises(ValueError, match=knob):
            validate_rollout_sampling(rollout)


class TestRolloutSamplingOnDictConfig:
    """The worker's rollout config is a struct DictConfig built from VERL's rollout.yaml keys."""

    @pytest.mark.parametrize(
        "override,knob",
        [
            ({}, None),
            ({"do_sample": False}, "do_sample"),
            ({"logit_bias": {"5": -100.0}}, "logit_bias"),
            ({"top_p": 0.9}, "top_p"),
        ],
        ids=["deploy", "no-sampling", "logit-bias", "top-p"],
    )
    def test_struct_config(self, override, knob):
        omegaconf = pytest.importorskip("omegaconf")
        rollout = omegaconf.OmegaConf.create(
            {
                "temperature": 1.0,
                "top_k": -1,
                "top_p": 1,
                "do_sample": True,
                "val_kwargs": {
                    "top_k": -1,
                    "top_p": 1.0,
                    "temperature": 0,
                    "n": 1,
                    "do_sample": False,
                },
                **override,
            }
        )
        omegaconf.OmegaConf.set_struct(rollout, True)
        if knob is None:
            validate_rollout_sampling(rollout)
        else:
            with pytest.raises(ValueError, match=knob):
                validate_rollout_sampling(rollout)
