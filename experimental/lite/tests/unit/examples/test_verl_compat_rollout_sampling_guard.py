# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

VERL_EXAMPLE_ROOT = Path(__file__).resolve().parents[3] / "examples" / "verl"
if str(VERL_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))

pytestmark = pytest.mark.optional


def _compat():
    from verl_mlite import compat

    return compat


def _protocol():
    from megatron.lite.model.k31.vllm import protocol

    return protocol


def _worker_module(protocol):
    module = ModuleType("verl.workers.engine_workers")

    class ActorRolloutRefWorker:
        def __init__(self, rollout):
            self.config = {"rollout": rollout}

        def init_model(self):
            handle = SimpleNamespace(_extras={"protocol": protocol})
            self.actor = SimpleNamespace(engine=SimpleNamespace(handle=handle))
            return "built"

    module.ActorRolloutRefWorker = ActorRolloutRefWorker
    return module


class TestRolloutSamplingGuard:
    @pytest.mark.parametrize(
        "rollout",
        [
            {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
            {"temperature": 0.6},
            {
                "temperature": 1.0,
                "val_kwargs": {"do_sample": False, "temperature": 0.0},
            },
        ],
        ids=["verl-defaults", "temperature-only", "greedy-validation"],
    )
    def test_reproducible_rollout_builds(self, rollout):
        module = _worker_module(_protocol())
        assert _compat()._install_rollout_sampling_guard(module)
        assert module.ActorRolloutRefWorker(rollout).init_model() == "built"

    @pytest.mark.parametrize(
        "rollout", [{"top_p": 0.9}, {"top_k": 20}], ids=["top-p", "top-k"]
    )
    def test_sampling_processing_fails_at_init(self, rollout):
        module = _worker_module(_protocol())
        _compat()._install_rollout_sampling_guard(module)
        with pytest.raises(ValueError, match="rollout sampling"):
            module.ActorRolloutRefWorker(rollout).init_model()

    def test_models_without_the_contract_are_untouched(self):
        module = _worker_module(SimpleNamespace())
        _compat()._install_rollout_sampling_guard(module)
        assert module.ActorRolloutRefWorker({"top_p": 0.9}).init_model() == "built"

    def test_dispatch_metadata_survives_and_install_is_idempotent(self):
        module = _worker_module(_protocol())
        module.ActorRolloutRefWorker.init_model.dispatch_mode = "one_to_all"
        assert _compat()._install_rollout_sampling_guard(module)
        assert not _compat()._install_rollout_sampling_guard(module)
        assert module.ActorRolloutRefWorker.init_model.dispatch_mode == "one_to_all"


# A process with VERL's real worker class and rollout.yaml, patched the way every VERL process is
# (apply_runtime_patches); only the engine build inside init_model is replaced.
_REAL_WORKER = r"""
import json
import os
import sys
from types import SimpleNamespace

import verl
import verl.workers.engine_workers as engine_workers
from omegaconf import OmegaConf

from megatron.lite.model.k31.vllm import protocol


def init_model(self):
    handle = SimpleNamespace(_extras={"protocol": protocol})
    self.actor = SimpleNamespace(engine=SimpleNamespace(handle=handle))
    return "built"


engine_workers.ActorRolloutRefWorker.init_model = init_model
from verl_mlite import compat

compat.apply_runtime_patches()
base = OmegaConf.load(os.path.join(os.path.dirname(verl.__file__), "trainer/config/rollout/rollout.yaml"))
results = {}
for case, override in json.loads(sys.argv[1]).items():
    rollout = OmegaConf.merge(base, OmegaConf.create(override))
    config = OmegaConf.create({"rollout": rollout})
    OmegaConf.set_struct(config, True)
    worker = object.__new__(engine_workers.ActorRolloutRefWorker)
    worker.config = config
    try:
        results[case] = worker.init_model()
    except ValueError as exc:
        results[case] = "refused: " + str(exc)
print("RESULTS " + json.dumps(results))
"""

_CASES = {
    "deploy": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
    "top-p": {"top_p": 0.9},
    "no-sampling": {"do_sample": False},
    "logit-bias": {"logit_bias": {"5": -100.0}},
}


@pytest.fixture(scope="module")
def real_worker_results():
    pytest.importorskip(
        "verl", reason="VERL is required for this optional example test."
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(VERL_EXAMPLE_ROOT), *sys.path])
    proc = subprocess.run(
        [sys.executable, "-c", _REAL_WORKER, json.dumps(_CASES)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULTS ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULTS ") :])


class TestGuardOnVerlWorker:
    @pytest.mark.parametrize(
        "case,knob",
        [
            ("deploy", None),
            ("top-p", "top_p"),
            ("no-sampling", "do_sample"),
            ("logit-bias", "logit_bias"),
        ],
        ids=["deploy", "top-p", "no-sampling", "logit-bias"],
    )
    def test_struct_rollout_config(self, real_worker_results, case, knob):
        result = real_worker_results[case]
        if knob is None:
            assert result == "built"
        else:
            assert result.startswith("refused") and knob in result
