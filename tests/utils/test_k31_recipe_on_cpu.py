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
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "examples" / "grpo_trainer" / "run_k31_true_on_policy_preview_megatron.sh"
DOCKERFILE = REPO / "docker" / "Dockerfile.k31_true_on_policy_preview"
FORWARD = "+ray_kwargs.ray_init.runtime_env.env_vars."
# Megatron-LM checkout with experimental/lite (the recipe's Hydra search path); composition tests skip without it.
MEGATRON_ROOT = os.environ.get("K31_RECIPE_TEST_MEGATRON_ROOT", "")
_FORWARDED_PREFIXES = ("K31_", "K3_", "K31L2_", "MLITE_", "VERL_", "NCCL_", "GLOO_", "CUDA_CACHE_", "WANDB_")


def _run(tmp_path, env=None, args=(), compose=False):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}")
    for name in ("train.parquet", "val.parquet"):
        (tmp_path / name).write_text("")
    full = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(_FORWARDED_PREFIXES) and k not in ("K31_RAY_FORWARD_ENV", "RAY_JOB_CONFIG_JSON_ENV_VAR")
    }
    full.update(
        VERL_ROOT=str(REPO) if compose else str(tmp_path),
        MEGATRON_ROOT=MEGATRON_ROOT if compose else str(tmp_path),
        MODEL_PATH=str(model),
        LORA_ADAPTER_PATH=str(adapter),
        TRAIN_FILES=str(tmp_path / "train.parquet"),
        VAL_FILES=str(tmp_path / "val.parquet"),
        OUTPUT_ROOT=str(tmp_path / "out"),
        DRY_RUN="1",
    )
    full.update(env or {})
    return subprocess.run(
        ["bash", str(RECIPE), "--mode", "quick_alignment_test", *args],
        env=full,
        capture_output=True,
        text=True,
    )


def _composed(tmp_path, env=None, args=()):
    """COMPOSE_ONLY=1: the configuration Hydra composes from the recipe's arguments."""
    yaml = pytest.importorskip("yaml")
    result = _run(tmp_path, {"DRY_RUN": "0", "COMPOSE_ONLY": "1", **(env or {})}, args, compose=True)
    assert result.returncode == 0, result.stderr[-4000:]
    text = result.stdout[result.stdout.index("model_engine:") :]
    return yaml.safe_load(text)


def _checked(tmp_path, env=None, args=()):
    """COMPOSE_ONLY=1: the launch up to its checks of the composed configuration."""
    return _run(tmp_path, {"DRY_RUN": "0", "COMPOSE_ONLY": "1", **(env or {})}, args, compose=True)


needs_mlite = pytest.mark.skipif(
    not (MEGATRON_ROOT and (Path(MEGATRON_ROOT) / "experimental" / "lite" / "examples" / "verl").is_dir()),
    reason="set K31_RECIPE_TEST_MEGATRON_ROOT to a Megatron-LM checkout with experimental/lite",
)


def _forwarded(stdout):
    return [arg.split("=", 1)[0][len(FORWARD) :] for arg in stdout.split() if arg.startswith(FORWARD)]


class TestRayForwarding:
    @pytest.mark.parametrize(
        "name",
        [
            "VERL_LORA_EXPORT_BEFORE_WAKE",
            "VERL_FREE_GRADS_BEFORE_WAKE",
            "K31_SLEEP_DISK_DIR",
            "MLITE_ACTOR_TIMING",
        ],
        ids=["export-before-wake", "free-grads", "sleep-disk-dir", "mlite-actor-timing"],
    )
    def test_switch_reaches_ray_workers(self, tmp_path, name):
        result = _run(tmp_path, {name: "1"})
        assert result.returncode == 0, result.stderr
        assert name in _forwarded(result.stdout)

    @pytest.mark.parametrize(
        "name",
        [
            "NCCL_IB_HCA",
            "GLOO_SOCKET_IFNAME",
            "LD_LIBRARY_PATH",
            "TORCHINDUCTOR_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "CUTE_DSL_CACHE_DIR",
            "TILELANG_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "CUDA_CACHE_MAXSIZE",
            "HF_HUB_OFFLINE",
            "MIMALLOC_PURGE_DELAY",
            "K31L2_HOOKS",
            "WANDB_PROJECT",
        ],
    )
    def test_runtime_settings_reach_ray_workers(self, tmp_path, name):
        result = _run(tmp_path, {name: "1"})
        assert result.returncode == 0, result.stderr
        assert name in _forwarded(result.stdout)

    def test_extra_names_reach_ray_workers(self, tmp_path):
        env = {"SITE_SETTING": "1", "OTHER_SETTING": "2", "K31_RAY_FORWARD_ENV": "SITE_SETTING, OTHER_SETTING"}
        result = _run(tmp_path, env)
        assert result.returncode == 0, result.stderr
        assert {"SITE_SETTING", "OTHER_SETTING"} <= set(_forwarded(result.stdout))

    @pytest.mark.parametrize("tp", ["1", "8"], ids=["tp1", "tp8"])
    def test_batch_invariance_is_the_only_bi_switch(self, tmp_path, tp):
        result = _run(tmp_path, {"ROLLOUT_TP": tp})
        assert result.returncode == 0, result.stderr
        names = set(_forwarded(result.stdout))
        assert {"VLLM_BATCH_INVARIANT", "VLLM_DISABLE_SHARED_EXPERTS_STREAM"} <= names
        retired = {"K31_MOE_FI_W4A16", "K31_TP_TREE", "K31_MOE_TREE", "K31_TP_ORDERED", "K31_BI_NCCL_UNPIN"}
        assert not names & (retired | {"K31_MOE_TREE_MAX_T"})

    def test_node_local_disk_cap_stays_on_its_node(self, tmp_path):
        result = _run(tmp_path, {"K31_SLEEP_DISK_MAX_GIB": "170", "K31_SLEEP_DISK_DIR": "/raid/snap"})
        assert result.returncode == 0, result.stderr
        names = _forwarded(result.stdout)
        assert "K31_SLEEP_DISK_DIR" in names and "K31_SLEEP_DISK_MAX_GIB" not in names

    def test_credentials_are_forwarded_but_not_printed(self, tmp_path):
        secret = "k31-test-credential-value"
        result = _run(tmp_path, {"WANDB_API_KEY": secret})
        assert result.returncode == 0, result.stderr
        assert secret not in result.stdout + result.stderr
        assert "WANDB_API_KEY" not in _forwarded(result.stdout)
        line = next(line for line in result.stdout.splitlines() if line.startswith("RAY_JOB_CONFIG env_vars"))
        assert "WANDB_API_KEY" in line.split()

    def test_ray_jobs_runtime_env_merges(self, tmp_path):
        runtime_env = pytest.importorskip("ray.runtime_env.runtime_env")
        job = {"runtime_env": {"env_vars": {"NCCL_DEBUG": "INFO", "K31_SITE": "job"}}}
        env = {
            "NCCL_DEBUG": "INFO",
            "K31_SITE": "job",
            "K31_OTHER": "1",
            "WANDB_API_KEY": "k31-test-credential-value",
            "RAY_JOB_CONFIG_JSON_ENV_VAR": json.dumps(job),
        }
        result = _run(tmp_path, env)
        assert result.returncode == 0, result.stderr
        driver = {"env_vars": {name: "x" for name in _forwarded(result.stdout)}}
        merged = runtime_env._merge_runtime_env(job["runtime_env"], driver)
        assert merged is not None
        assert {"NCCL_DEBUG", "K31_SITE", "K31_OTHER", "VLLM_BATCH_INVARIANT"} <= set(merged["env_vars"])
        assert merged["env_vars"]["K31_SITE"] == "job"

    @pytest.mark.parametrize(
        "job_env,args,refused",
        [
            ({"VLLM_BATCH_INVARIANT": "1"}, ("--bi", "0"), "VLLM_BATCH_INVARIANT"),
            ({"VLLM_BATCH_INVARIANT": "0"}, (), "VLLM_BATCH_INVARIANT"),
            ({"VLLM_DISABLE_SHARED_EXPERTS_STREAM": "0"}, (), "VLLM_DISABLE_SHARED_EXPERTS_STREAM"),
            ({"VLLM_BATCH_INVARIANT": "1"}, (), None),
        ],
        ids=["bi1-job-bi0-launch", "bi0-job-bi1-launch", "shared-experts-stream", "same-arm"],
    )
    def test_ray_job_config_cannot_swap_the_arm(self, tmp_path, job_env, args, refused):
        env = {"RAY_JOB_CONFIG_JSON_ENV_VAR": json.dumps({"runtime_env": {"env_vars": job_env}})}
        result = _run(tmp_path, env, args)
        if refused is None:
            assert result.returncode == 0, result.stderr
            assert not set(job_env) & set(_forwarded(result.stdout))
        else:
            assert result.returncode != 0
            assert f"the Ray job config sets {refused}=" in result.stderr

    @pytest.mark.parametrize("key", ["py_modules", "working_dir", "pip", "worker_process_setup_hook"])
    def test_ray_job_config_cannot_bring_code(self, tmp_path, key):
        job = {"runtime_env": {key: ["/elsewhere"] if key in ("py_modules", "pip") else "/elsewhere"}}
        result = _run(tmp_path, {"RAY_JOB_CONFIG_JSON_ENV_VAR": json.dumps(job)})
        assert result.returncode != 0
        assert f"the Ray job config's runtime_env sets {key}" in result.stderr

    def test_ray_job_config_cannot_set_the_pythonpath(self, tmp_path):
        job = {"runtime_env": {"env_vars": {"PYTHONPATH": "/elsewhere/vllm-src"}}}
        result = _run(tmp_path, {"RAY_JOB_CONFIG_JSON_ENV_VAR": json.dumps(job)})
        assert result.returncode != 0
        assert "the Ray job config sets PYTHONPATH='/elsewhere/vllm-src'" in result.stderr

    def test_inherited_pythonpath_stays_off_the_workers(self, tmp_path):
        result = _run(tmp_path, {"PYTHONPATH": "/elsewhere/vllm-src"})
        assert result.returncode == 0, result.stderr
        assert "/elsewhere" not in _forwarded_values(result.stdout)["PYTHONPATH"]

    def test_each_variable_is_forwarded_once(self, tmp_path):
        result = _run(tmp_path, {"VERL_LORA_EXPORT_BEFORE_WAKE": "1"})
        assert result.returncode == 0, result.stderr
        names = _forwarded(result.stdout)
        assert len(names) == len(set(names))
        assert {"VLLM_BATCH_INVARIANT", "VLLM_DISABLE_SHARED_EXPERTS_STREAM", "VERL_FILE_LOGGER_PATH"} <= set(names)

    def test_value_with_interpolation_is_refused(self, tmp_path):
        result = _run(tmp_path, {"K31_PATTERN": "${HOME}/x"})
        assert result.returncode != 0
        assert "cannot forward K31_PATTERN" in result.stderr

    @needs_mlite
    def test_forwarded_values_survive_composition(self, tmp_path):
        values = {
            "K31_JSON": '{"mode":"test"}',
            "K31_QUOTES": "it's \"quoted\" \\ and \\'",
            "K31_LIST": "a,b=c [x] {y}",
            "K31_LINES": "first\nsecond",
            "K31_EMPTY": "",
            "K31_WORD": "null",
            "K31_TRAILING": "dir\\",
        }
        env_vars = _composed(tmp_path, values)["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]
        assert {name: env_vars.get(name) for name in values} == values


class TestDataFiles:
    @needs_mlite
    def test_file_lists_compose_as_lists(self, tmp_path):
        files = []
        for name in ("a.parquet", "b.parquet", "val.parquet"):
            path = tmp_path / name
            path.write_text("")
            files.append(str(path))
        cfg = _composed(tmp_path, {"TRAIN_FILES": f"{files[0]},{files[1]}", "VAL_FILES": files[2]})
        assert cfg["data"]["train_files"] == files[:2]
        assert cfg["data"]["val_files"] == files[2:]

    def test_every_listed_file_is_checked(self, tmp_path):
        present = tmp_path / "a.parquet"
        present.write_text("")
        env = {"DRY_RUN": "0", "COMPOSE_ONLY": "1", "TRAIN_FILES": f"{present},{tmp_path}/missing.parquet"}
        result = _run(tmp_path, env)
        assert result.returncode != 0
        assert "missing data file" in result.stderr


@needs_mlite
class TestBypassGuard:
    @pytest.mark.parametrize("bi", ["1", "0"], ids=["bi1", "bi0"])
    def test_bypass_is_the_default_of_both_arms(self, tmp_path, bi):
        rc = _composed(tmp_path, args=("--bi", bi))
        assert rc["algorithm"]["rollout_correction"]["bypass_mode"] is True
        assert rc["algorithm"]["rollout_correction"]["loss_type"] == "ppo_clip"
        assert rc["algorithm"]["rollout_correction"]["rollout_is"] is None
        assert rc["algorithm"]["rollout_correction"]["rollout_rs"] is None
        assert rc["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] == "vanilla"
        assert rc["actor_rollout_ref"]["actor"]["ppo_epochs"] == 1

    def test_bypass_accepts_overrides_with_the_same_values(self, tmp_path):
        args = (
            "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
            "algorithm.rollout_correction.bypass_mode=True",
            "algorithm.rollout_correction.rollout_is=null",
        )
        result = _checked(tmp_path, {"K31_BYPASS": "1"}, args)
        assert result.returncode == 0, result.stderr[-4000:]

    @pytest.mark.parametrize(
        "env,args",
        [
            ({"PPO_MINI_BATCH_SIZE": "2"}, ()),
            ({}, ("actor_rollout_ref.actor.ppo_mini_batch_size=2",)),
            ({}, ("++actor_rollout_ref.actor.ppo_epochs=2",)),
            ({}, ("++algorithm.rollout_correction.rollout_rs=token",)),
            ({}, ("~algorithm.rollout_correction.bypass_mode",)),
            ({}, ("algorithm.rollout_correction={rollout_is:token}",)),
            ({}, ("algorithm.rollout_correction.loss_type=reinforce",)),
        ],
        ids=[
            "several-updates-per-step",
            "mini-batch-override",
            "plus-plus-epochs",
            "plus-plus-rejection-sampling",
            "deleted-bypass-mode",
            "parent-dict",
            "reinforce-loss",
        ],
    )
    def test_bypass_refuses_non_equivalent_settings(self, tmp_path, env, args):
        result = _checked(tmp_path, {"K31_BYPASS": "1", **env}, args)
        assert result.returncode != 0
        assert "K31_BYPASS=1" in result.stderr


@needs_mlite
class TestMliteEngineSettings:
    def test_offload_and_checkpoint_settings_are_set(self, tmp_path):
        engine = _composed(tmp_path)["actor_rollout_ref"]["actor"]["engine"]
        assert engine["offload_persist"] is True
        assert engine["ckpt_trainable_only"] is True


def _forwarded_values(stdout):
    values = {}
    for arg in stdout.split():
        if arg.startswith(FORWARD):
            name, value = arg[len(FORWARD) :].split("=", 1)
            values[name] = value.replace("\\'", "'").strip("'")
    return values


VLLM_ARG = "+actor_rollout_ref.rollout.engine_kwargs.vllm."


class TestBatchInvariantArm:
    def test_default_is_the_batch_invariant_arm(self, tmp_path):
        result = _run(tmp_path)
        assert result.returncode == 0, result.stderr
        env = _forwarded_values(result.stdout)
        assert env["VLLM_BATCH_INVARIANT"] == "1" and env["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == "1"
        args = result.stdout.split()
        assert f"{VLLM_ARG}compilation_config.cudagraph_mode=FULL_AND_PIECEWISE" in args
        assert "algorithm.rollout_correction.rollout_is=token" not in args
        assert not [arg for arg in args if "enable_flashinfer_autotune" in arg]

    def test_bi0_flips_the_switch_and_its_rollout_knobs(self, tmp_path):
        shell = {"CUBLAS_WORKSPACE_CONFIG": ":16:8", "CUBLASLT_WORKSPACE_SIZE": "1"}
        result = _run(tmp_path, shell, ("--bi", "0"))
        assert result.returncode == 0, result.stderr
        env = _forwarded_values(result.stdout)
        assert env["VLLM_BATCH_INVARIANT"] == "0" and "VLLM_DISABLE_SHARED_EXPERTS_STREAM" not in env
        assert not set(shell) & set(env)
        args = result.stdout.split()
        assert [arg for arg in args if arg.startswith(VLLM_ARG)] == [
            f"{VLLM_ARG}enable_moe_shared_loras=True",
            next(arg for arg in args if arg.startswith(f"{VLLM_ARG}lora_target_modules=")),
        ]
        assert "algorithm.rollout_correction.rollout_is=token" not in args
        assert "--bi" not in args and "BI=0" in result.stdout

    def test_bi0_without_bypass_adds_token_tis(self, tmp_path):
        result = _run(tmp_path, {"K31_BYPASS": "0"}, ("--bi", "0"))
        assert result.returncode == 0, result.stderr
        assert "algorithm.rollout_correction.rollout_is=token" in result.stdout.split()

    @pytest.mark.parametrize(
        "bi,env,r3,replay",
        [
            ("1", {}, "0", ("disabled", "False")),
            ("0", {}, "1", ("R3", "True")),
            ("1", {"K31_R3": "0"}, "0", ("disabled", "False")),
            ("1", {"K31_R3": "1"}, "1", ("R3", "True")),
        ],
        ids=["bi1-default-r3-off", "bi0-default-r3-on", "bi1-r3-off-explicit", "bi1-r3-on-for-debugging"],
    )
    def test_r3_follows_the_arm_and_prefix_caching_stays_on(self, tmp_path, bi, env, r3, replay):
        result = _run(tmp_path, env, ("--bi", bi))
        assert result.returncode == 0, result.stderr
        assert f"BI={bi} R3={r3} " in result.stdout
        args = result.stdout.split()
        assert f"actor_rollout_ref.actor.engine.router_replay_mode={replay[0]}" in args
        assert f"actor_rollout_ref.rollout.enable_rollout_routing_replay={replay[1]}" in args
        assert "actor_rollout_ref.rollout.enable_prefix_caching=True" in args
        note = "K31_R3=1 under --bi 1 is for debugging" in result.stderr
        assert note == (bi == "1" and r3 == "1")

    @pytest.mark.parametrize(
        "bi,env,prefix",
        [("1", {}, "True"), ("0", {}, "True"), ("1", {"ROLLOUT_PREFIX_CACHING": "False"}, "False")],
        ids=["bi1-default", "bi0-default", "bi1-off"],
    )
    def test_prefix_caching_is_printed(self, tmp_path, bi, env, prefix):
        result = _run(tmp_path, env, ("--bi", bi))
        assert result.returncode == 0, result.stderr
        assert f"PREFIX_CACHING={prefix} " in result.stdout
        assert f"actor_rollout_ref.rollout.enable_prefix_caching={prefix}" in result.stdout.split()

    @pytest.mark.parametrize(
        "env,args,message",
        [
            ({"K31_BYPASS": "yes"}, ("--bi", "0"), "K31_BYPASS takes 1 or 0, got 'yes'"),
            ({"ROLLOUT_PREFIX_CACHING": "no"}, (), "ROLLOUT_PREFIX_CACHING takes True or False, got 'no'"),
            ({"K31_R3": "0"}, ("--bi", "0"), "K31_R3=1"),
            ({}, ("--bi", "2"), "--bi takes 1 or 0"),
            ({"K31_R3": "true"}, ("--bi", "1"), "K31_R3 takes 1 or 0, got 'true'"),
            ({"K31_R3": "2"}, ("--bi", "0"), "K31_R3 takes 1 or 0, got '2'"),
            ({}, ("--bi",), "--bi requires a value"),
        ],
        ids=[
            "bypass-not-a-number",
            "prefix-caching-not-a-bool",
            "bi0-without-r3",
            "bad-value",
            "r3-not-a-number",
            "r3-out-of-range",
            "missing-value",
        ],
    )
    def test_bi_refuses(self, tmp_path, env, args, message):
        result = _run(tmp_path, env, args)
        assert result.returncode != 0
        assert message in result.stderr


@needs_mlite
class TestComposedGuard:
    @pytest.mark.parametrize(
        "bi,env,override",
        [
            ("0", {}, "++actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=False"),
            ("0", {}, "++actor_rollout_ref.rollout.engine_kwargs.vllm.enable-prefix-caching=False"),
            ("0", {}, "++actor_rollout_ref.rollout.engine_kwargs.vllm.no_enable_prefix_caching=True"),
            ("1", {"K31_R3": "1"}, "++actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=False"),
            ("0", {}, "++actor_rollout_ref.rollout.engine_kwargs.vllm.enable_return_routed_experts=False"),
            ("1", {}, "++actor_rollout_ref.rollout.engine_kwargs.vllm.enable_return_routed_experts=True"),
            ("1", {}, "++actor_rollout_ref.rollout.engine_kwargs.vllm.enable-return-routed-experts=True"),
            ("1", {}, "++actor_rollout_ref.rollout.engine_kwargs.vllm.enforce-eager=True"),
        ],
        ids=[
            "bi0-engine-prefix-off",
            "bi0-engine-prefix-hyphenated",
            "bi0-engine-no-prefix",
            "bi1-r3-engine-prefix-off",
            "bi0-engine-routes-off",
            "bi1-engine-routes-on",
            "bi1-engine-routes-hyphenated",
            "bi1-engine-eager-hyphenated",
        ],
    )
    def test_engine_kwargs_cannot_override_the_checked_settings(self, tmp_path, bi, env, override):
        result = _checked(tmp_path, env, ("--bi", bi, override))
        assert result.returncode != 0
        assert (
            f"error: --bi {bi}: actor_rollout_ref.rollout.engine_kwargs.vllm." in result.stderr
            and "the recipe sets" in result.stderr
        ), result.stderr[-2000:]

    @pytest.mark.parametrize(
        "bi,override,subtree",
        [
            (
                "1",
                "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[1]",
                "actor_rollout_ref.rollout.engine_kwargs.vllm",
            ),
            (
                "0",
                "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.custom_ops=['-rms_norm']",
                "actor_rollout_ref.rollout.engine_kwargs.vllm",
            ),
            (
                "1",
                "++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8",
                "actor_rollout_ref.rollout.engine_kwargs.vllm",
            ),
            (
                "1",
                "++actor_rollout_ref.rollout.engine_kwargs.vllm.kv-cache-dtype=fp8",
                "actor_rollout_ref.rollout.engine_kwargs.vllm",
            ),
            (
                "0",
                "++actor_rollout_ref.rollout.engine_kwargs.vllm.fully_sharded_loras=True",
                "actor_rollout_ref.rollout.engine_kwargs.vllm",
            ),
            (
                "1",
                "actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.sparse_mla_force_mqa=False",
                "actor_rollout_ref.rollout.engine_kwargs.vllm",
            ),
            (
                "1",
                "++actor_rollout_ref.actor.engine.impl_cfg.use_deepep=False",
                "actor_rollout_ref.actor.engine.impl_cfg",
            ),
            (
                "0",
                "++actor_rollout_ref.actor.engine.impl_cfg.runtime_plugins={x: y}",
                "actor_rollout_ref.actor.engine.impl_cfg",
            ),
            ("1", "+ray_kwargs.ray_init.runtime_env.py_modules=[/elsewhere]", "ray_kwargs.ray_init.runtime_env"),
            ("0", "++ray_kwargs.ray_init.runtime_env.working_dir=/elsewhere", "ray_kwargs.ray_init.runtime_env"),
            (
                "1",
                "++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=hook",
                "ray_kwargs.ray_init.runtime_env",
            ),
            ("1", "++ray_kwargs.ray_init.address=elsewhere:6379", "ray_kwargs.ray_init"),
            (
                "1",
                "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ATTENTION_BACKEND=FLASH_ATTN",
                "ray_kwargs.ray_init.runtime_env",
            ),
        ],
        ids=[
            "capture-sizes",
            "custom-ops",
            "kv-cache-dtype",
            "kv-cache-dtype-hyphenated",
            "fully-sharded-loras",
            "attention-config",
            "impl-cfg-deepep",
            "impl-cfg-plugins",
            "runtime-env-py-modules",
            "runtime-env-working-dir",
            "runtime-env-setup-hook",
            "ray-init-address",
            "runtime-env-vllm-env",
        ],
    )
    def test_pinned_subtrees_are_the_recipes(self, tmp_path, bi, override, subtree):
        result = _checked(tmp_path, args=("--bi", bi, override))
        assert result.returncode != 0
        assert f"error: --bi {bi}: {subtree}." in result.stderr and "the recipe sets" in result.stderr, result.stderr[
            -2000:
        ]

    def test_rollout_and_actor_keep_one_max_model_len(self, tmp_path):
        result = _checked(tmp_path, args=("actor_rollout_ref.rollout.max_model_len=8192",))
        assert result.returncode != 0
        assert "actor_rollout_ref.rollout.max_model_len=8192 but actor_rollout_ref.actor.engine.impl_cfg" in (
            result.stderr
        ), result.stderr[-2000:]

    def test_bi1_composes_r3_off_with_prefix_caching(self, tmp_path):
        cfg = _composed(tmp_path)
        assert cfg["actor_rollout_ref"]["actor"]["engine"]["router_replay_mode"] == "disabled"
        rollout = cfg["actor_rollout_ref"]["rollout"]
        assert rollout["enable_rollout_routing_replay"] is False and rollout["enable_prefix_caching"] is True

    @pytest.mark.parametrize(
        "env,override",
        [
            ({}, "actor_rollout_ref.actor.engine.router_replay_mode=R3"),
            ({}, "actor_rollout_ref.rollout.enable_rollout_routing_replay=True"),
            ({"K31_R3": "1"}, "actor_rollout_ref.actor.engine.router_replay_mode=disabled"),
            ({"K31_R3": "1"}, "actor_rollout_ref.rollout.enable_prefix_caching=False"),
            ({}, "actor_rollout_ref.rollout.enable_prefix_caching=False"),
        ],
        ids=[
            "bi1-r3-on-behind-the-launcher",
            "bi1-routes-on-behind-the-launcher",
            "bi1-r3-debug-off",
            "bi1-r3-debug-no-prefix",
            "bi1-prefix-off-behind-the-launcher",
        ],
    )
    def test_bi1_router_replay_follows_k31_r3(self, tmp_path, env, override):
        result = _checked(tmp_path, env, ("--bi", "1", override))
        assert result.returncode != 0
        assert f"error: --bi 1, K31_R3={env.get('K31_R3', '0')}: actor_rollout_ref." in result.stderr, result.stderr[
            -2000:
        ]

    def test_bi0_composes_the_corrected_configuration(self, tmp_path):
        cfg = _composed(tmp_path, {"K31_BYPASS": "0"}, ("--bi", "0"))
        rc = cfg["algorithm"]["rollout_correction"]
        assert rc["bypass_mode"] is False and rc["rollout_is"] == "token"
        assert cfg["actor_rollout_ref"]["actor"]["engine"]["router_replay_mode"] == "R3"
        rollout = cfg["actor_rollout_ref"]["rollout"]
        assert rollout["enable_rollout_routing_replay"] is True and rollout["enable_prefix_caching"] is True

    @pytest.mark.parametrize(
        "override",
        [
            "algorithm.rollout_correction.bypass_mode=True",
            "algorithm.rollout_correction.rollout_is=null",
            "~algorithm.rollout_correction.rollout_is",
            "algorithm.rollout_correction.rollout_is=sequence",
            "actor_rollout_ref.actor.engine.router_replay_mode=disabled",
            "actor_rollout_ref.rollout.enable_rollout_routing_replay=False",
            "actor_rollout_ref.rollout.enable_prefix_caching=False",
        ],
        ids=[
            "bypass-mode",
            "no-is",
            "deleted-is",
            "sequence-is",
            "router-replay-off",
            "rollout-routes-off",
            "prefix-caching-off",
        ],
    )
    def test_bi0_refuses_overrides_that_undo_the_correction(self, tmp_path, override):
        result = _checked(tmp_path, {"K31_BYPASS": "0"}, ("--bi", "0", override))
        assert result.returncode != 0
        assert "error: --bi 0" in result.stderr

    @pytest.mark.parametrize(
        "bi,override",
        [
            (
                "1",
                "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL",
            ),
            ("1", "++actor_rollout_ref.rollout.engine_kwargs.vllm.enable_flashinfer_autotune=True"),
            ("0", "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL"),
            ("0", "++actor_rollout_ref.rollout.engine_kwargs.vllm.enable_flashinfer_autotune=False"),
            ("0", "++actor_rollout_ref.rollout.engine_kwargs.vllm.enable_flashinfer_autotune=True"),
            ("1", "actor_rollout_ref.rollout.enforce_eager=True"),
            ("0", "actor_rollout_ref.rollout.enforce_eager=True"),
            ("1", "++actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=True"),
            ("1", "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.mode=0"),
            ("0", "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.mode=0"),
            ("0", "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.level=0"),
            ("0", "actor_rollout_ref.rollout.full_determinism=True"),
            ("1", "actor_rollout_ref.rollout.full_determinism=True"),
            ("0", "actor_rollout_ref.rollout.calculate_log_probs=False"),
            ("0", "~actor_rollout_ref.rollout.calculate_log_probs"),
            ("1", "actor_rollout_ref.rollout.calculate_log_probs=False"),
            ("1", "++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=auto"),
            ("0", "++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8_e4m3"),
            ("0", "++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8_ds_mla"),
            ("1", "++actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=FLASHINFER_MLA_SPARSE"),
            ("1", "++actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.sparse_indexer_topk_backend=persistent"),
            ("0", "++actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=FLASHMLA_SPARSE"),
            ("0", "++actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.sparse_mla_force_mqa=True"),
            ("0", "++actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.sparse_indexer_topk_backend=deep_select"),
            ("0", "++actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_tp_mode=data"),
            ("0", "++actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.moe_backend=flashinfer_trtllm"),
        ],
        ids=[
            "bi1-piecewise",
            "bi1-autotune",
            "bi0-full-only",
            "bi0-autotune-off",
            "bi0-autotune-pinned",
            "bi1-eager",
            "bi0-eager",
            "bi1-vllm-eager",
            "bi1-compile-mode",
            "bi0-compile-mode",
            "bi0-compile-level",
            "bi0-full-determinism",
            "bi1-full-determinism",
            "bi0-no-rollout-log-probs",
            "bi0-deleted-rollout-log-probs",
            "bi1-no-rollout-log-probs",
            "bi1-kv-auto",
            "bi0-kv-per-tensor-fp8",
            "bi0-kv-ds-mla",
            "bi1-flashinfer-sparse",
            "bi1-other-topk",
            "bi0-flashmla-pinned",
            "bi0-force-mqa",
            "bi0-deepselect-pinned",
            "bi0-mm-encoder-data",
            "bi0-moe-pinned",
        ],
    )
    def test_each_arm_keeps_its_rollout_knobs(self, tmp_path, bi, override):
        result = _checked(tmp_path, args=("--bi", bi, override))
        assert result.returncode != 0
        assert f"error: --bi {bi}: actor_rollout_ref.rollout." in result.stderr, result.stderr[-2000:]

    @pytest.mark.parametrize(
        "bi,override,job_env",
        [
            ("0", "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BATCH_INVARIANT='1'", {}),
            ("1", "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BATCH_INVARIANT='0'", {}),
            ("1", "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_DISABLE_SHARED_EXPERTS_STREAM='0'", {}),
            ("1", "++ray_kwargs.ray_init.runtime_env.env_vars.CUBLAS_WORKSPACE_CONFIG=':4096:8'", {}),
            ("1", "++ray_kwargs.ray_init.runtime_env.env_vars.CUBLASLT_WORKSPACE_SIZE='0'", {}),
            ("1", "~ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BATCH_INVARIANT", {}),
            ("0", "~ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BATCH_INVARIANT", {}),
            ("1", "ray_kwargs.ray_init.runtime_env={env_vars:{VLLM_BATCH_INVARIANT:'0'}}", {}),
            ("1", "ray_kwargs.ray_init.runtime_env.env_vars=null", {}),
            ("0", "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_DISABLE_SHARED_EXPERTS_STREAM='1'", {}),
        ],
        ids=[
            "bi0-override-to-1",
            "bi1-override-to-0",
            "shared-experts-stream",
            "cublas-workspace",
            "cublaslt-workspace",
            "bi1-deleted",
            "bi0-deleted",
            "parent-dict",
            "parent-dict-null",
            "bi0-shared-experts-stream-pinned",
        ],
    )
    def test_worker_environment_keeps_the_arm(self, tmp_path, bi, override, job_env):
        env = {"RAY_JOB_CONFIG_JSON_ENV_VAR": json.dumps({"runtime_env": {"env_vars": job_env}})} if job_env else {}
        result = _checked(tmp_path, env, ("--bi", bi, override))
        assert result.returncode != 0
        assert f"error: --bi {bi}: the Ray workers get " in result.stderr, result.stderr[-2000:]

    @pytest.mark.parametrize(
        "bi,args,job_env",
        [
            ("1", ("++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BATCH_INVARIANT='1'",), {}),
            ("0", ("++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BATCH_INVARIANT='0'",), {}),
            ("1", (), {"VLLM_BATCH_INVARIANT": "1"}),
        ],
        ids=["bi1-same-override", "bi0-same-override", "from-the-job-config"],
    )
    def test_worker_environment_accepts_the_same_values(self, tmp_path, bi, args, job_env):
        env = {"RAY_JOB_CONFIG_JSON_ENV_VAR": json.dumps({"runtime_env": {"env_vars": job_env}})} if job_env else {}
        result = _checked(tmp_path, env, ("--bi", bi, *args))
        assert result.returncode == 0, result.stderr[-4000:]


class TestRayAddress:
    """A launch (DRY_RUN=0, not COMPOSE_ONLY) runs on the Ray cluster at RAY_ADDRESS, a single node included: the
    package probe checks that cluster's GPU nodes, so a launch without an address stops before anything else."""

    @pytest.mark.parametrize(
        "nnodes,address",
        [("1", ""), ("2", ""), ("1", "local"), ("2", "local")],
        ids=["one-node", "two-nodes", "one-node-local", "two-nodes-local"],
    )
    def test_launch_without_ray_address_is_refused(self, tmp_path, nnodes, address):
        result = _run(tmp_path, {"DRY_RUN": "0", "NNODES": nnodes, "RAY_ADDRESS": address})
        assert result.returncode == 64, result.stderr[-2000:]
        assert "error: set RAY_ADDRESS" in result.stderr, result.stderr[-2000:]


class TestWorkerProbeDigest:
    """The package probe compares, launcher against every worker node, a digest of megatron.lite, megatron.core and
    every PYTHONPATH tree (VERL_ROOT, MEGATRON_ROOT): any file edited, added, removed or linked in changes it,
    bytecode and .git do not; a worker whose digest, versions or verifier result differ, or a cluster whose GPU node
    count is not the launch's, stops the launch."""

    @classmethod
    def _tree_digest(cls):
        return cls._probe()["tree_digest"]

    @pytest.mark.parametrize(
        "change,differs",
        [
            (lambda top: (top / "pkg" / "a.py").write_text("x = 2\n"), True),
            (lambda top: (top / "pkg" / "b.py").write_text(""), True),
            (lambda top: (top / "pkg" / "a.py").unlink(), True),
            (lambda top: (top / "pkg" / "a.py").rename(top / "a.py"), True),
            (lambda top: (top / "lib").symlink_to(top / "pkg", target_is_directory=True), True),
            (lambda top: (top / "pkg" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0"), False),
            (lambda top: (top / ".git" / "index").write_bytes(b"\1"), False),
        ],
        ids=["edited", "added", "removed", "moved", "linked-directory", "bytecode", "git"],
    )
    def test_digest_follows_the_files(self, tmp_path, change, differs):
        tree_digest = self._tree_digest()
        top = tmp_path / "top"
        (top / "pkg" / "__pycache__").mkdir(parents=True)
        (top / ".git").mkdir()
        (top / "pkg" / "a.py").write_text("x = 1\n")
        (tmp_path / "outside").mkdir()
        (tmp_path / "outside" / "__init__.py").write_text("x = 3\n")
        before = tree_digest(str(top))
        change(top)
        assert (tree_digest(str(top)) != before) is differs

    @staticmethod
    def _probe():
        text = RECIPE.read_text()
        source = text[text.index('WHERE = r"""') + len('WHERE = r"""') :]
        ns = {}
        exec(source[: source.index('"""')], ns)
        return ns

    @staticmethod
    def _result(**changes):
        result = {
            "vllm": ["/site/vllm/__init__.py", "0.22.0+k31"],
            "megatron.lite": ["/site/megatron/lite/__init__.py", "0.1.0", "a" * 64],
            "megatron.core": ["/site/megatron/core", "b" * 64],
            "PYTHONPATH": [["/verl", "c" * 64], ["/megatron", "d" * 64]],
            "image": {"SOURCES": "f" * 64, "verify_k31.sh": "9" * 64},
            "packages": {"python": ["/usr/bin/python3.12", "3.12.3"], "torch": ["2.11.0", "8" * 64]},
            "verify_k31.sh": [0, ["VERIFY_K31_OK ... vllm=/site/vllm"], True],
        }
        result.update(changes)
        return result

    @pytest.mark.parametrize(
        "worker,nodes,error",
        [
            ({}, 2, None),
            ({"megatron.lite": ["/site/megatron/lite/__init__.py", "0.1.0", "e" * 64]}, 2, "megatron.lite is"),
            ({"PYTHONPATH": [["/verl", "e" * 64], ["/megatron", "d" * 64]]}, 2, "PYTHONPATH is"),
            ({"megatron.core": ["/site/megatron/core", "e" * 64]}, 2, "megatron.core is"),
            ({"verify_k31.sh": [0, ["VERIFY_K31_OK ... vllm=/venv/vllm"], False]}, 2, "verify_k31.sh"),
            ({"verify_k31.sh": [1, ["differs"], False]}, 2, "verify_k31.sh"),
            ({"megatron.core": None}, 2, "megatron.core does not import"),
            ({"image": {"SOURCES": "e" * 64, "verify_k31.sh": "9" * 64}}, 2, "image is"),
            ({"packages": {"python": ["/usr/bin/python3.12", "3.12.3"], "torch": ["2.11.0", "e" * 64]}}, 2,
             "packages is"),
            ({}, 3, "the Ray cluster has 2 GPU nodes, this launch runs on 3"),
            ({}, 1, "the Ray cluster has 2 GPU nodes, this launch runs on 1"),
        ],
        ids=[
            "same",
            "lite-content",
            "pythonpath-content",
            "core-content",
            "verify-other-vllm",
            "verify-failed",
            "core-not-importable",
            "image-file",
            "package-record",
            "node-missing",
            "extra-node",
        ],
    )
    def test_a_differing_worker_stops_the_launch(self, worker, nodes, error):
        mismatches = self._probe()["mismatches"]
        bad = mismatches(self._result(), [("n0", self._result()), ("n1", self._result(**worker))], nodes)
        if error is None:
            assert bad == []
        else:
            assert any(error in item for item in bad), bad

    @pytest.mark.skipif(not Path("/opt/k31/vllm.manifest").is_file(), reason="needs the K31 image (/opt/k31)")
    def test_where_digests_lite_and_every_pythonpath_tree(self, tmp_path, monkeypatch):
        import importlib.util

        ns = self._probe()
        (tmp_path / "a.py").write_text("x = 1\n")
        # every interpreter verify_k31.sh starts warns on stderr after its result, as torch does in the workers
        (tmp_path / "sitecustomize.py").write_text(
            "import atexit, sys\natexit.register(lambda: print('W some library warning', file=sys.stderr))\n"
        )
        monkeypatch.setenv("PYTHONPATH", f"{tmp_path}{os.pathsep}{tmp_path / 'missing'}")
        result = ns["where"]()
        assert result["verify_k31.sh"][0] == 0 and result["verify_k31.sh"][2] is True, result["verify_k31.sh"]
        lite = os.path.dirname(os.path.realpath(importlib.util.find_spec("megatron.lite").origin))
        core = os.path.dirname(os.path.realpath(importlib.util.find_spec("megatron.core").origin))
        assert result["megatron.lite"][2] == ns["tree_digest"](lite)
        assert result["megatron.core"] == [core, ns["tree_digest"](core)]
        assert result["PYTHONPATH"] == [[str(tmp_path), ns["tree_digest"](str(tmp_path))], [str(tmp_path / "missing"), None]]
        assert "verify_k31.sh" in result["image"] and {"python", "torch"} <= set(result["packages"])

    FAKE_RAY = (
        "import os\n"
        "N = int(os.environ['FAKE_RAY_GPU_NODES'])\n"
        "def init(**kwargs): pass\n"
        "def shutdown(): pass\n"
        "def nodes():\n"
        "    return [{'Alive': True, 'Resources': {'GPU': 4.0}, 'NodeManagerAddress': f'n{i}', 'NodeID': str(i)}\n"
        "            for i in range(N)]\n"
        "class _Remote:\n"
        "    def __init__(self, f): self.f = f\n"
        "    def options(self, **kwargs): return self\n"
        "    def remote(self, *args): return self.f(*args)\n"
        "def remote(**kwargs): return _Remote\n"
        "def get(results, timeout=None): return results\n"
    )

    @pytest.mark.skipif(not Path("/opt/k31/vllm.manifest").is_file(), reason="needs the K31 image (/opt/k31)")
    @pytest.mark.parametrize("gpu_nodes,passes", [(2, True), (3, False), (1, False)], ids=["two", "extra", "missing"])
    def test_probe_script_stops_on_node_count(self, tmp_path, gpu_nodes, passes):
        """The recipe's own probe script, run against a stand-in Ray whose cluster lists gpu_nodes GPU nodes for a
        two-node launch (every node is this machine, so only the count differs)."""
        text = RECIPE.read_text()
        head = text.index("\n", text.index('python3 - "${COMPOSED}" "${NNODES}" <<\'PY\''))
        body = text[head + 1 :]
        (tmp_path / "probe.py").write_text(body[: body.index("\nPY\n")] + "\n")
        (tmp_path / "fake" / "ray" / "util").mkdir(parents=True)
        (tmp_path / "fake" / "ray" / "__init__.py").write_text(self.FAKE_RAY)
        (tmp_path / "fake" / "ray" / "util" / "__init__.py").write_text("")
        (tmp_path / "fake" / "ray" / "util" / "scheduling_strategies.py").write_text(
            "def NodeAffinitySchedulingStrategy(node_id, soft): return node_id\n"
        )
        (tmp_path / "composed.yaml").write_text("ray_kwargs:\n  ray_init:\n    runtime_env:\n      env_vars: {}\n")
        env = {**os.environ, "RAY_ADDRESS": "fake:1", "FAKE_RAY_GPU_NODES": str(gpu_nodes)}
        env["PYTHONPATH"] = os.pathsep.join(p for p in [str(tmp_path / "fake"), env.get("PYTHONPATH", "")] if p)
        result = subprocess.run(
            [sys.executable, str(tmp_path / "probe.py"), str(tmp_path / "composed.yaml"), "2"],
            env=env,
            capture_output=True,
            text=True,
        )
        if passes:
            assert result.returncode == 0 and "WORKER_PACKAGES nodes=2" in result.stdout, result.stderr[-2000:]
        else:
            assert result.returncode == 1, result.stdout[-2000:]
            assert f"the Ray cluster has {gpu_nodes} GPU nodes, this launch runs on 2" in result.stderr

    @pytest.mark.parametrize(
        "target",
        ["outside", "outside/__init__.py", "top/.git/vendor", "top/pkg/__pycache__"],
        ids=["directory", "file", "into-git", "into-pycache"],
    )
    def test_link_out_of_the_tree_fails_the_digest(self, tmp_path, target):
        tree_digest = self._tree_digest()
        top = tmp_path / "top"
        (top / "pkg" / "__pycache__").mkdir(parents=True)
        (top / ".git" / "vendor").mkdir(parents=True)
        (top / ".git" / "vendor" / "__init__.py").write_text("x = 4\n")
        (tmp_path / "outside").mkdir()
        (tmp_path / "outside" / "__init__.py").write_text("x = 3\n")
        (top / "pkg" / "sub").symlink_to(tmp_path / target)
        with pytest.raises(ValueError, match="links outside"):
            tree_digest(str(top))

    @pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs directory permissions to apply")
    def test_unlistable_directory_fails_the_digest(self, tmp_path):
        tree_digest = self._tree_digest()
        hidden = tmp_path / "pkg" / "hidden"
        hidden.mkdir(parents=True)
        (hidden / "__init__.py").write_text("x = 1\n")
        hidden.chmod(0o111)  # searchable (importable) but not listable
        try:
            with pytest.raises(PermissionError):
                tree_digest(str(tmp_path))
        finally:
            hidden.chmod(0o755)


@pytest.mark.skipif(not Path("/opt/k31/vllm.manifest").is_file(), reason="needs the K31 image (/opt/k31)")
class TestImageVerifier:
    """docker/k31/verify_k31.sh checks the packages of the interpreter K31_PYTHON names (the package probe passes
    the worker's own), megatron.lite and megatron.core file by file included, and its success line names the vllm
    directory it checked. A venv whose own vllm shadows the image's, whose megatron.lite differs from the installed
    one in one file, or that shadows megatron.core alone, fails it, whatever python3 is first on PATH."""

    @pytest.mark.parametrize(
        "shadow",
        [None, "vllm", "megatron.lite", "megatron.core", "megatron.lite+link"],
        ids=["image-interpreter", "venv-own-vllm", "venv-edited-lite", "venv-edited-core-only", "venv-lite-linked-dir"],
    )
    def test_checks_the_interpreter_it_is_given(self, tmp_path, shadow):
        import importlib.util
        import shutil

        vllm = pytest.importorskip("vllm")
        python, checked = sys.executable, os.path.dirname(os.path.realpath(vllm.__file__))
        if shadow:
            venv = tmp_path / "venv"
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv)], check=True)
            python = str(venv / "bin" / "python3")
            purelib = Path(
                subprocess.run(
                    [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
            if shadow == "vllm":
                (purelib / "vllm").mkdir(parents=True)
                (purelib / "vllm" / "__init__.py").write_text("")
                (purelib / "vllm" / "_version.py").write_text(f"__version__ = {vllm.__version__!r}\n")
            else:
                # copies of the installed packages, one file of the shadowing package changed by a line: lite with
                # core beside it, or core alone (lite then still resolves in the image)
                site = Path(importlib.util.find_spec("megatron.lite").origin).parents[2]
                edited = shadow.split(".")[1].split("+")[0]
                for name in ("core", "lite") if edited == "lite" else ("core",):
                    shutil.copytree(
                        site / "megatron" / name,
                        purelib / "megatron" / name,
                        ignore=shutil.ignore_patterns("__pycache__"),
                    )
                if shadow.endswith("+link"):
                    # every file unchanged, plus a linked package directory outside the tree
                    (tmp_path / "outside").mkdir()
                    (tmp_path / "outside" / "__init__.py").write_text("x = 1\n")
                    (purelib / "megatron" / "lite" / "extra").symlink_to(tmp_path / "outside", target_is_directory=True)
                else:
                    with open(purelib / "megatron" / edited / "__init__.py", "a") as f:
                        f.write("\n# edited\n")
        verify = REPO / "docker" / "k31" / "verify_k31.sh"
        result = subprocess.run(
            ["bash", str(verify)], env={**os.environ, "K31_PYTHON": python}, capture_output=True, text=True
        )
        if shadow:
            assert result.returncode != 0, result.stdout[-2000:]
            if shadow == "megatron.lite":
                assert "megatron.lite in" in result.stderr, result.stderr[-2000:]
            if shadow == "megatron.core":
                assert "megatron.core resolves in" in result.stdout, result.stdout[-2000:]
            if shadow == "megatron.lite+link":
                assert "symbolic links differ" in result.stderr, result.stderr[-2000:]
        else:
            assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
            assert result.stdout.strip().splitlines()[-1].endswith(f" vllm={checked}"), result.stdout[-2000:]


class TestPreflight:
    """Outside DRY_RUN: vllm (the +k31 build) and megatron.lite with its kernel are files their installed
    distributions recorded, in the site-packages of the interpreter the recipe runs; megatron.core is not
    required.  A source tree reaches the launcher only through the recipe's own PYTHONPATH (MEGATRON_ROOT)."""

    PACKAGES = {
        "lite": (
            "nemotron-megatron-lite",
            (
                "megatron/lite/__init__.py",
                "megatron/lite/primitive/__init__.py",
                "megatron/lite/primitive/kernels/__init__.py",
                "megatron/lite/primitive/kernels/_k31_mxfp4_dequant.py",
            ),
        ),
        "vllm": ("vllm", ("vllm/__init__.py",)),
    }

    @classmethod
    def _install(cls, root, package, record, version):
        name, files = cls.PACKAGES[package]
        for rel in files:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(f"__version__ = {version!r}\n" if package == "vllm" else "")
        if record:
            info = root / f"{name.replace('-', '_')}-{version}.dist-info"
            info.mkdir()
            (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
            (info / "RECORD").write_text("".join(f"{rel},,\n" for rel in files))

    @pytest.mark.parametrize(
        "lite_at,vllm_at,record,version,error",
        [
            ("venv", "venv", True, "0.22.0+k31", None),
            ("src", "venv", True, "0.22.0+k31", "not the installed experimental/lite wheel"),
            ("src/site-packages", "venv", True, "0.22.0+k31", "not the installed experimental/lite wheel"),
            ("venv", "venv", False, "0.22.0+k31", "not the installed experimental/lite wheel"),
            ("venv", "src", True, "0.22.0+k31", "not the image's vllm"),
            ("venv", "venv", True, "0.22.0", "not the image's +k31 build"),
        ],
        ids=[
            "installed-wheels-without-core",
            "source-tree-lite",
            "source-tree-named-site-packages",
            "no-record",
            "source-tree-vllm",
            "vllm-without-k31",
        ],
    )
    def test_preflight_checks_where_vllm_and_megatron_lite_come_from(
        self, tmp_path, lite_at, vllm_at, record, version, error
    ):
        venv = tmp_path / "venv"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
        purelib = subprocess.run(
            [str(venv / "bin" / "python3"), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        source = next((tmp_path / at for at in (lite_at, vllm_at) if at != "venv"), tmp_path / "megatron-root")
        source.mkdir(parents=True, exist_ok=True)
        for package, at in (("lite", lite_at), ("vllm", vllm_at)):
            root = Path(purelib) if at == "venv" else tmp_path / at
            self._install(root, package, record, version if package == "vllm" else "0.0.0")
        env = {
            "DRY_RUN": "0",
            "COMPOSE_ONLY": "1",
            "MEGATRON_ROOT": str(source),
            "PATH": f"{venv / 'bin'}:{os.environ['PATH']}",
            "PYTHONPATH": "",
        }
        result = _run(tmp_path, env)
        refused = "are not the image's packages" in result.stderr
        if error is None:
            assert not refused, result.stderr[-2000:]
        else:
            assert refused and error in result.stderr, result.stderr[-2000:]


@needs_mlite
class TestBatchVariantCorrection:
    """--bi 0 with K31_BYPASS=0: the composed rollout correction turns the rollout/actor log-prob gap into IS
    weights, and the actor's ppo_loss (configured from the composed actor section's loss settings) applies them.

    It fails on a recipe without --bi, or whose --bi 0 composes no rollout IS (rollout_is null) or a loss that
    does not apply the weights; tests/workers/test_ppo_loss_rollout_is_on_cpu.py guards ppo_loss itself without
    mLite."""

    def test_composed_tis_weights_change_the_actor_loss(self, tmp_path):
        torch = pytest.importorskip("torch")
        from tensordict import TensorDict

        from verl.protocol import DataProto
        from verl.trainer.ppo.core_algos import agg_loss
        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
        from verl.utils import tensordict_utils as tu
        from verl.workers.config.actor import ActorConfig, PolicyLossConfig
        from verl.workers.utils.losses import ppo_loss
        from verl.workers.utils.padding import left_right_2_no_padding

        cfg = _composed(tmp_path, {"K31_BYPASS": "0"}, ("--bi", "0"))
        actor = cfg["actor_rollout_ref"]["actor"]
        prompt, response, lens = 3, 4, ((3, 4), (2, 2))
        g = torch.Generator().manual_seed(0)
        attention_mask = torch.zeros(len(lens), prompt + response, dtype=torch.long)
        mask = torch.zeros(len(lens), response, dtype=torch.long)
        for i, (p, r) in enumerate(lens):
            attention_mask[i, prompt - p : prompt + r] = 1
            mask[i, :r] = 1
        old = -torch.rand(len(lens), response, generator=g) * mask
        gap = torch.tensor([[0.5, -1.0, 0.9, 0.2], [-0.3, 0.6, 0.0, 0.0]])
        corrected, _ = compute_rollout_correction_and_add_to_batch(
            DataProto.from_dict(
                tensors={"old_log_probs": old, "rollout_log_probs": (old - gap) * mask, "response_mask": mask.float()}
            ),
            cfg["algorithm"]["rollout_correction"],
        )
        weights = corrected.batch["rollout_is_weights"]
        assert not torch.allclose(weights[mask.bool()], torch.ones(int(mask.sum())))

        input_ids = torch.randint(1, 100, attention_mask.shape, generator=g) * attention_mask
        advantages = torch.randn(len(lens), response, generator=g) * mask
        data = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": (attention_mask.cumsum(-1) - 1).clamp(min=0),
                "response_mask": mask,
                "prompts": input_ids[:, :prompt],
                "responses": input_ids[:, prompt:],
                "old_log_probs": old,
                "advantages": advantages,
                "rollout_is_weights": weights,
            },
            batch_size=[len(lens)],
        )
        tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(mask.sum()), global_batch_size=len(lens))
        data = left_right_2_no_padding(data)
        values = []
        for i, (p, r) in enumerate(lens):
            seq = torch.zeros(p + r)
            seq[p - 1 : p + r - 1] = old[i, :r]
            values.append(seq)
        offsets = torch.tensor([0, *torch.tensor([p + r for p, r in lens]).cumsum(0).tolist()])
        log_probs = torch.nested.nested_tensor_from_jagged(torch.cat(values), offsets=offsets)
        config = ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            ppo_micro_batch_size=2,
            clip_ratio=actor["clip_ratio"],
            clip_ratio_low=actor["clip_ratio_low"],
            clip_ratio_high=actor["clip_ratio_high"],
            clip_ratio_c=actor["clip_ratio_c"],
            loss_agg_mode=actor["loss_agg_mode"],
            policy_loss=PolicyLossConfig(loss_mode=actor["policy_loss"]["loss_mode"]),
        )
        loss, _ = ppo_loss(config, {"log_probs": log_probs}, data)
        # on policy (ratio 1) the policy-gradient term is -A * w, aggregated as the composed actor aggregates it
        expected = agg_loss(-advantages * weights, mask.bool(), config.loss_agg_mode, **config.global_batch_info)
        unweighted = agg_loss(-advantages, mask.bool(), config.loss_agg_mode, **config.global_batch_info)
        torch.testing.assert_close(loss, expected)
        assert not torch.allclose(loss, unweighted)


class TestBypassDefault:
    @pytest.mark.parametrize("bi", ["1", "0"], ids=["bi1", "bi0"])
    def test_bypass_is_on_by_default(self, tmp_path, bi):
        result = _run(tmp_path, args=("--bi", bi))
        assert result.returncode == 0, result.stderr
        assert "algorithm.rollout_correction.bypass_mode=True" in result.stdout.split()

    def test_bypass_off_recomputes_old_log_prob(self, tmp_path):
        result = _run(tmp_path, {"K31_BYPASS": "0"})
        assert result.returncode == 0, result.stderr
        assert "algorithm.rollout_correction.bypass_mode=True" not in result.stdout.split()


class TestDockerfile:
    def test_heredocs_run_in_bash(self):
        instructions = [
            line.split(None, 1) for line in DOCKERFILE.read_text().splitlines() if re.match(r"^[A-Z]+\s", line)
        ]
        names = [name for name, _ in instructions]
        assert "SHELL" in names and names.index("SHELL") < names.index("RUN")
        assert json.loads(instructions[names.index("SHELL")][1]) == ["/bin/bash", "-c"]
