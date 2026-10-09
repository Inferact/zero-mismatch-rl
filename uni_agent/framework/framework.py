from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import json
import logging
import os
import random
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from uuid import uuid4

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from ray.exceptions import RayTaskError
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from uni_agent.gateway.session import SessionHandle, Trajectory
from uni_agent.logging import LogContext, sample_logging
from uni_agent.rl_insight.adapter import agent_loop_session
from uni_agent.tasks import TaskResult
from verl.tools.tool_registry import initialize_tools_from_config
from verl.utils import tensordict_utils as tu
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.transferqueue_utils import tq

from .base import AgentFramework
from .multi_modal_postprocess import compute_multi_modal_inputs, compute_position_ids

logger = logging.getLogger(__name__)


TrajectoryPostprocessor = Callable[..., list[Trajectory] | Awaitable[list[Trajectory]]]


@dataclass
class _RunnerConfig:
    runner_fqn: str
    runner_kwargs: dict[str, object]
    dispatch_mode: str
    max_concurrent_sessions: int
    trajectory_selection: str = "all"
    session_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.runner_fqn:
            raise ValueError("runner_fqn is required")
        if self.dispatch_mode not in {"inline_async", "ray_task"}:
            raise ValueError(f"Unknown dispatch mode: {self.dispatch_mode}")
        if self.max_concurrent_sessions < 0:
            raise ValueError(f"max_concurrent_sessions must be non-negative, got {self.max_concurrent_sessions}")
        if self.trajectory_selection not in {"all", "longest"}:
            raise ValueError(f"Unknown trajectory selection: {self.trajectory_selection}. Expected 'all' or 'longest'")
        if self.session_timeout_seconds is not None and self.session_timeout_seconds <= 0:
            raise ValueError(f"session_timeout_seconds must be positive, got {self.session_timeout_seconds}")

    @classmethod
    def from_config(cls, runner_name: object, runner_cfg) -> _RunnerConfig:
        runner_fqn = runner_cfg.get("runner_fqn")
        runner_kwargs = dict(
            OmegaConf.to_container(OmegaConf.create(runner_cfg.get("runner_kwargs", {})), resolve=True) or {}
        )
        tool_config_path = runner_cfg.get("tool_config_path")
        if tool_config_path:
            tool_config = initialize_tools_from_config(str(tool_config_path))
            if not tool_config:
                raise ValueError(
                    f"agent_runners.{runner_name}.tool_config_path did not initialize any tools: {tool_config_path}"
                )
            runner_kwargs["tool_config"] = tool_config
        dispatch_mode = str(runner_cfg.get("dispatch_mode", "inline_async"))
        max_concurrent_sessions = int(runner_cfg.get("max_concurrent_sessions", 0) or 0)
        trajectory_selection = str(runner_cfg.get("trajectory_selection", "all"))
        raw_timeout = runner_cfg.get("session_timeout_seconds")
        session_timeout_seconds = None if raw_timeout is None else float(raw_timeout)
        try:
            return cls(
                runner_fqn="" if runner_fqn is None else str(runner_fqn),
                runner_kwargs=runner_kwargs,
                dispatch_mode=dispatch_mode,
                max_concurrent_sessions=max_concurrent_sessions,
                trajectory_selection=trajectory_selection,
                session_timeout_seconds=session_timeout_seconds,
            )
        except ValueError as exc:
            raise ValueError(f"agent_runners.{runner_name}: {exc}") from exc


def _materialize_runner(runner_fqn: str, runner_kwargs: dict[str, object]):
    runner = load_class_from_fqn(runner_fqn, description="agent runner")
    if isinstance(runner, type):
        return runner(**runner_kwargs)
    if runner_kwargs:
        return partial(runner, **runner_kwargs)
    return runner


def _log_scope(log_context: LogContext | None):
    if log_context is None:
        return contextlib.nullcontext()
    return sample_logging.from_context(log_context)


# No Ray-level retries: a retry would replay the runner against the same, already
# partially-used Gateway session. A crashed runner is a failed session.
@ray.remote(max_retries=0)
def _run_agent_runner_ray_task(
    *,
    runner_fqn: str,
    runner_kwargs: dict[str, object],
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: object | None,
    log_context: LogContext | None,
) -> TaskResult | None:
    """Run only the user runner in Ray; parent owns session lifecycle outputs."""
    runner = _materialize_runner(runner_fqn, runner_kwargs)
    with _log_scope(log_context):
        try:
            return asyncio.run(
                runner(
                    raw_prompt=raw_prompt,
                    session=session,
                    sample_index=sample_index,
                    **({"tools_kwargs": tools_kwargs} if tools_kwargs is not None else {}),
                )
            )
        except asyncio.CancelledError as exc:
            # ray.cancel arrives as KeyboardInterrupt, so a CancelledError leaving
            # asyncio.run was raised inside the runner (e.g. a leaked timeout).
            logger.exception("runner for sample_index=%s was cancelled from inside its event loop", sample_index)
            raise RuntimeError(
                f"runner for sample_index={sample_index} was cancelled from inside its event loop"
            ) from exc


def _short_failure_reason(error: BaseException) -> str:
    message = str(error)
    if not message:
        message = error.__class__.__name__
    return f"{error.__class__.__name__}:{message}"[:512]


def _select_session_trajectories(
    session_id: str,
    trajectories: list[Trajectory],
    selection: str,
) -> list[Trajectory]:
    """Apply the runner's trajectory-retention policy before scoring and TQ writes."""
    if selection == "all" or len(trajectories) <= 1:
        return trajectories

    index, trajectory = max(
        enumerate(trajectories),
        key=lambda item: (
            sum(item[1].response_mask),
            len(item[1].response_ids),
            item[1].num_turns,
            item[0],
        ),
    )
    logger.info(
        "session %s: selected longest trajectory index=%s model_tokens=%s response_tokens=%s turns=%s candidates=%s",
        session_id,
        index,
        sum(trajectory.response_mask),
        len(trajectory.response_ids),
        trajectory.num_turns,
        len(trajectories),
    )
    return [trajectory]


@dataclass
class _PromptGroup:
    """One prompt's finished sessions, held in memory until written to TQ."""

    uid: str
    num_sessions: int
    completed: list[tuple[int, list[Trajectory], dict[str, object]]] = field(default_factory=list)
    failed_sessions: int = 0
    failure_reasons: list[str] = field(default_factory=list)
    dropped_sessions: int = 0
    drop_reason: str | None = None


def _new_batch_stats(num_input_prompts: int) -> dict:
    return {
        "num_input_prompts": num_input_prompts,
        "num_success_sessions": 0,
        "num_failed_sessions": 0,
        "num_success_outputs": 0,
        "num_unfinished_episodes": 0,
        "num_failed_uids": 0,
        "num_dropped_uids": 0,
        "num_dropped_sessions": 0,
        "num_cut_groups": 0,
        "failure_reasons": [],
    }


def _add_failed_prompt(stats: dict, num_sessions: int, reason: str) -> None:
    stats["num_failed_sessions"] += num_sessions
    stats["num_failed_uids"] += 1
    stats["failure_reasons"].append(reason)


def _add_prompt_stats(stats: dict, outcome: dict) -> None:
    for key in (
        "num_success_sessions",
        "num_failed_sessions",
        "num_success_outputs",
        "num_unfinished_episodes",
        "num_failed_uids",
        "num_dropped_uids",
        "num_dropped_sessions",
    ):
        stats[key] += outcome[key]
    stats["failure_reasons"].extend(outcome["failure_reasons"])


_TQ_NESTED_SEQUENCE_FIELDS = {
    "prompts",
    "responses",
    "response_mask",
    "loss_mask",
    "input_ids",
    "attention_mask",
    "position_ids",
    "rollout_log_probs",
    "routed_experts",
    "rm_scores",
    "teacher_logprobs",
    "teacher_ids",
}


def _json_default(obj: object) -> object:
    """Best-effort JSON coercion for reward/extra fields (numpy scalars, tensors, sets)."""
    if isinstance(obj, set | frozenset):
        return list(obj)
    for attr in ("item", "tolist"):  # numpy scalar / 0-d tensor, then ndarray / tensor
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return str(obj)


def _align_routed_experts(source: object, seq_len: int) -> torch.Tensor | None:
    """Return R3 routing as ``[seq_len, layers, topk]`` aligned to input_ids."""
    experts = torch.as_tensor(source, device="cpu")
    if experts.dim() != 3:
        return None
    out = torch.zeros((seq_len, experts.shape[1], experts.shape[2]), dtype=experts.dtype)
    covered = min(experts.shape[0], seq_len)
    if covered > 0:
        out[:covered] = experts[:covered]
    return out


def _list_of_tq_fields_to_tensordict(fields: list[dict[str, object]]) -> TensorDict:
    # Optional per-sample fields (e.g. routed_experts) can be missing on degenerate
    # trajectories; drop any column not present on every sample so the stacker never
    # KeyErrors on a partially-present key (list_of_dict_to_tensordict keys off row 0).
    if fields:
        shared_keys = set(fields[0]).intersection(*(set(f) for f in fields[1:]))
        for f in fields:
            for key in list(f):
                if key not in shared_keys:
                    f.pop(key, None)
    td = tu.list_of_dict_to_tensordict(fields)
    for key in _TQ_NESTED_SEQUENCE_FIELDS:
        if key not in fields[0]:
            continue
        values = [field[key] for field in fields]
        if not all(isinstance(value, torch.Tensor) for value in values):
            continue
        if key == "routed_experts":
            ragged_idx = 1  # [seq, layers, topk]: ragged on the sequence dim
        elif key == "position_ids" and values[0].dim() == 2:
            ragged_idx = 2
        else:
            ragged_idx = None
        td[key] = tu.nested_tensor_from_tensor_list(values, ragged_idx=ragged_idx)
    return td


def _attach_runner_reward_info(trajectories: list[Trajectory], task_result: TaskResult) -> list[Trajectory]:
    """Preserve runner scoring context separately from final reward metrics."""
    task_metrics = {} if task_result.accuracy is None else {"acc": task_result.accuracy}
    return [
        replace(
            traj,
            extra_fields={
                **traj.extra_fields,
                "runner_reward_info": {
                    "reward": task_result.reward,
                    "metrics": dict(task_metrics),
                    "reward_context": dict(task_result.extra_info),
                },
            },
        )
        for traj in trajectories
    ]


def _trajectory_to_reward_dataproto(trajectory, sample_fields, task_result: TaskResult):
    """Build a single-sample DataProto for RewardLoopWorker.compute_score.

    Field shape matches AgentLoopWorker._compute_score
    (verl/experimental/agent_loop/agent_loop.py:753-772). Only fields actually
    consumed by NaiveRewardManager.run_single / RewardLoopWorker dispatch are
    populated; ``__num_turns__`` rides in non_tensor_batch for parity.
    """
    from verl.protocol import DataProto

    prompt_ids = torch.tensor(trajectory.prompt_ids, dtype=torch.long).unsqueeze(0)
    response_ids = torch.tensor(trajectory.response_ids, dtype=torch.long).unsqueeze(0)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    batch = TensorDict(
        {
            "prompts": prompt_ids,
            "responses": response_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        batch_size=1,
    )

    non_tensor_batch: dict[str, object] = {}
    for key in ("raw_prompt", "data_source", "tools_kwargs", "agent_name"):
        if key in sample_fields:
            non_tensor_batch[key] = np.array([sample_fields[key]], dtype=object)
    non_tensor_batch["reward_model"] = np.array(
        [sample_fields.get("reward_model", {"ground_truth": None})],
        dtype=object,
    )
    non_tensor_batch["extra_info"] = np.array(
        [
            {
                **dict(sample_fields.get("extra_info") or {}),
                "runner_reward_info": {
                    "reward": task_result.reward,
                    "metrics": {} if task_result.accuracy is None else {"acc": task_result.accuracy},
                    "reward_context": dict(task_result.extra_info),
                },
            }
        ],
        dtype=object,
    )
    non_tensor_batch["__num_turns__"] = np.array([trajectory.num_turns])

    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class GatewayAgentFramework(AgentFramework):
    """Reference AgentFramework implementation for Gateway-backed agent loops.

    Each sample in the batch is run as an independent Gateway session: the agent
    communicates through the Gateway's provider adapter (OpenAI Chat Completions
    or Anthropic Messages), and the Gateway collects token-level trajectories. After
    finalization, a configured custom scorer receives the managed runner's reward
    information and owns the final score. Without a custom scorer, a non-None
    runner reward is retained; a RewardLoopWorker is used when the runner omits it.
    The framework then writes them to the TransferQueue schema consumed by sync training.
    """

    #: Grace period for a timed-out runner Ray task to observe a graceful
    #: ray.cancel (letting its sandbox context manager tear down) before the
    #: framework escalates to a force-kill.
    _RUNNER_CANCEL_GRACE_SECONDS = 30.0
    #: Bound on a best-effort TQ tag write/listing made while failing prompts.
    _TQ_TAG_TIMEOUT_SECONDS = 60.0
    #: Bound on a Gateway ``abort_session`` call made while failing a session.
    _SESSION_ABORT_TIMEOUT_SECONDS = 60.0
    #: How long a timed-out session may spend unwinding (runner cancel, Gateway
    #: abort) before the framework stops waiting for it.
    _SESSION_CLEANUP_GRACE_SECONDS = 120.0
    #: Bound on waiting for groups cut by over-provisioning to tear down their
    #: sessions; covers the per-session cleanup grace above.
    _CUT_UNWIND_TIMEOUT_SECONDS = 300.0
    #: How long ``pop_rollout_metrics`` waits for a step's metrics to be published.
    _METRICS_PUBLISH_TIMEOUT_SECONDS = 120.0
    #: Over-provisioned steps whose metrics nobody claimed are dropped past this count.
    _MAX_UNCLAIMED_STEP_METRICS = 8

    def __init__(
        self,
        gateway_manager,  # GatewayManager: framework calls create_session/finalize_session/abort_session
        *,
        runner_registry: dict[str, _RunnerConfig],
        reward_loop_worker_handles=None,
        custom_reward_function_configured: bool = False,
        processor=None,
        rollout_config=None,
        log_dir: str | None = None,
        mask_unfinished_episode: bool = False,
        max_failed_prompt_fraction: float = 1.0,
        max_failed_session_fraction: float = 1.0,
        min_group_size: int = 1,
        over_provision_groups: int = 0,
        trajectory_postprocessor: TrajectoryPostprocessor | None = None,
        trajectory_postprocessor_kwargs: dict[str, object] | None = None,
    ):
        self.gateway_manager = gateway_manager
        self.runner_registry = runner_registry
        # Materialize inline runners at construction since they run in-process and may maintain state;
        # Ray-dispatched runners are materialized per-run since they run remotely.
        self._inline_runners = {
            runner_name: _materialize_runner(runner_config.runner_fqn, runner_config.runner_kwargs)
            for runner_name, runner_config in runner_registry.items()
            if runner_config.dispatch_mode == "inline_async"
        }
        self.reward_loop_worker_handles = list(reward_loop_worker_handles) if reward_loop_worker_handles else None
        self._custom_reward_function_configured = custom_reward_function_configured
        if self.reward_loop_worker_handles is None:
            logger.info("No streaming reward worker handles; using the non-streaming reward path")
        self._processor = processor
        self._rollout_config = rollout_config
        self._runner_semaphores: dict[str, asyncio.Semaphore] = {}
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self._log_dir = log_dir
        self._mask_unfinished_episode = mask_unfinished_episode
        self._max_failed_prompt_fraction = max_failed_prompt_fraction
        self._max_failed_session_fraction = max_failed_session_fraction
        self._min_group_size = min_group_size
        self._over_provision_groups = over_provision_groups
        self._step_metrics: dict[int | None, tuple[asyncio.Event, dict[str, float]]] = {}
        self._trajectory_postprocessor = trajectory_postprocessor
        self._trajectory_postprocessor_kwargs = trajectory_postprocessor_kwargs or {}
        for runner_name, runner_config in runner_registry.items():
            if runner_config.session_timeout_seconds is None:
                logger.warning(
                    "agent_runners.%s.session_timeout_seconds is unset: one hung session can stall "
                    "the whole rollout step",
                    runner_name,
                )

    @classmethod
    def from_config(
        cls,
        *,
        config,
        gateway_manager,
        processor=None,
        reward_loop_worker_handles=None,
    ) -> GatewayAgentFramework:
        # TODO(phase-b): switch this to actor_rollout_ref.rollout.agent_framework.*
        af_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}
        reward_model_enabled = OmegaConf.select(config, "reward.reward_model.enable", default=False)
        reward_model_uses_separate_resources = OmegaConf.select(
            config, "reward.reward_model.enable_resource_pool", default=True
        )
        custom_reward_function_path = OmegaConf.select(config, "reward.custom_reward_function.path", default=None)
        if (
            reward_model_enabled is True
            and reward_model_uses_separate_resources is False
            and custom_reward_function_path is not None
        ):
            logger.warning(
                "Detected a custom reward function with a colocated reward model. The colocated reward path "
                "does not receive AgentRunner TaskResult/runner_reward_info and replaces Runner-provided "
                "rm_scores, so hybrid scoring such as model score plus task reward is not supported. Use "
                "separate reward-model resources or a future VERL TransferQueue contract extension."
            )
        runner_registry: dict[str, _RunnerConfig] = {}
        agent_runners_cfg = af_cfg.get("agent_runners")
        if not agent_runners_cfg:
            raise ValueError("actor_rollout_ref.rollout.custom.agent_framework.agent_runners is required")

        for runner_name, runner_cfg in agent_runners_cfg.items():
            runner_registry[str(runner_name)] = _RunnerConfig.from_config(runner_name, runner_cfg)

        if "log_dir" in af_cfg:
            configured_log_dir = af_cfg.get("log_dir")
            log_dir = str(configured_log_dir) if configured_log_dir else None
        else:
            log_dir = os.environ.get("UNI_AGENT_LOG_DIR") or "/tmp/uni_agent_logs"

        mask_unfinished_episode = af_cfg.get("mask_unfinished_episode", False)
        if type(mask_unfinished_episode) is not bool:
            raise ValueError("actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode must be a bool")

        max_failed_prompt_fraction = float(af_cfg.get("max_failed_prompt_fraction", 1.0))
        if not 0.0 <= max_failed_prompt_fraction <= 1.0:
            raise ValueError(
                "actor_rollout_ref.rollout.custom.agent_framework.max_failed_prompt_fraction must be in [0, 1]"
            )
        max_failed_session_fraction = float(af_cfg.get("max_failed_session_fraction", 1.0))
        if not 0.0 <= max_failed_session_fraction <= 1.0:
            raise ValueError(
                "actor_rollout_ref.rollout.custom.agent_framework.max_failed_session_fraction must be in [0, 1]"
            )
        min_group_size = af_cfg.get("min_group_size", 1)
        if type(min_group_size) is not int or min_group_size < 1:
            raise ValueError(
                "actor_rollout_ref.rollout.custom.agent_framework.min_group_size must be a positive integer"
            )

        over_provision_groups = af_cfg.get("over_provision_groups", 0)
        if type(over_provision_groups) is not int or over_provision_groups < 0:
            raise ValueError(
                "actor_rollout_ref.rollout.custom.agent_framework.over_provision_groups must be a non-negative integer"
            )

        postprocessor_fqn = af_cfg.get("trajectory_postprocessor_fqn")
        postprocessor_kwargs = af_cfg.get("trajectory_postprocessor_kwargs")
        if postprocessor_kwargs is None:
            postprocessor_kwargs = {}
        elif OmegaConf.is_config(postprocessor_kwargs):
            postprocessor_kwargs = OmegaConf.to_container(postprocessor_kwargs, resolve=True)

        trajectory_postprocessor = None
        trajectory_postprocessor_kwargs = {}
        if postprocessor_fqn is None:
            if postprocessor_kwargs:
                raise ValueError("trajectory_postprocessor_kwargs requires trajectory_postprocessor_fqn")
        else:
            if not isinstance(postprocessor_fqn, str) or not postprocessor_fqn.strip():
                raise ValueError("trajectory_postprocessor_fqn must be a non-empty string")
            if not isinstance(postprocessor_kwargs, dict):
                raise TypeError("trajectory_postprocessor_kwargs must be a mapping")
            postprocessor_fqn = postprocessor_fqn.strip()
            trajectory_postprocessor = load_class_from_fqn(
                postprocessor_fqn,
                description="trajectory postprocessor",
            )
            if not callable(trajectory_postprocessor):
                raise TypeError(f"Trajectory postprocessor {postprocessor_fqn!r} must resolve to a callable")
            trajectory_postprocessor_kwargs = dict(postprocessor_kwargs)

        return cls(
            gateway_manager=gateway_manager,
            runner_registry=runner_registry,
            reward_loop_worker_handles=reward_loop_worker_handles,
            custom_reward_function_configured=custom_reward_function_path is not None,
            processor=processor,
            rollout_config=config.actor_rollout_ref.rollout,
            log_dir=log_dir,
            mask_unfinished_episode=mask_unfinished_episode,
            max_failed_prompt_fraction=max_failed_prompt_fraction,
            max_failed_session_fraction=max_failed_session_fraction,
            min_group_size=min_group_size,
            over_provision_groups=over_provision_groups,
            trajectory_postprocessor=trajectory_postprocessor,
            trajectory_postprocessor_kwargs=trajectory_postprocessor_kwargs,
        )

    async def _apply_trajectory_postprocessor(
        self,
        trajectories: list[Trajectory],
        task_result: TaskResult,
    ) -> list[Trajectory]:
        """Apply the optional sync/async postprocessor and validate its result."""
        expected_reward_fields = (
            (
                trajectories[-1].finished,
                trajectories[-1].reward_score,
                deepcopy(trajectories[-1].reward_metrics),
            )
            if trajectories
            else (None, None, {})
        )
        result = self._trajectory_postprocessor(
            tuple(trajectories), task_result=deepcopy(task_result), **self._trajectory_postprocessor_kwargs
        )
        if inspect.isawaitable(result):
            result = await result

        if not isinstance(result, list):
            raise TypeError(f"trajectory postprocessor must return list[Trajectory], got {type(result).__name__}")
        if any(not isinstance(trajectory, Trajectory) for trajectory in result):
            raise TypeError("trajectory postprocessor returned a non-Trajectory item")
        if any(
            (trajectory.finished, trajectory.reward_score, trajectory.reward_metrics) != expected_reward_fields
            for trajectory in result
        ):
            raise ValueError("trajectory postprocessor must preserve finalized reward fields")
        return result

    def _build_session_sampling_params(
        self,
        *,
        partition_id: str,
        sample_fields: dict[str, object],
    ) -> dict[str, object]:
        """Build trusted per-session sampling defaults using VERL rollout semantics."""
        config = self._rollout_config
        sampling_params: dict[str, object] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": 1.0,
            "logprobs": config.calculate_log_probs,
        }
        if partition_id == "val":
            sampling_params.update(
                temperature=config.val_kwargs.temperature,
                top_p=config.val_kwargs.top_p,
                top_k=config.val_kwargs.top_k,
            )
        # An explicit greedy flag overrides the partition sampling defaults.
        if "__do_sample__" in sample_fields and not bool(sample_fields["__do_sample__"]):
            sampling_params.update(temperature=0, top_p=1.0, top_k=-1)
        return sampling_params

    def _resolve_target_groups(self, prompts: TensorDict, partition_id: str) -> int | None:
        """Return how many groups to keep from an over-provisioned training batch, or None.

        A trainer that over-provisions stamps ``__target_groups__`` on the batch; that
        wins over ``over_provision_groups`` so the two sides cannot disagree on the
        number of terminal groups the trainer waits for.
        """
        if partition_id != "train":
            return None
        num_prompts = len(prompts)
        stamped = tu.get(prompts, "__target_groups__")
        if stamped is not None:
            target_groups = int(stamped)
            if self._over_provision_groups and num_prompts - self._over_provision_groups != target_groups:
                logger.warning(
                    "batch stamps __target_groups__=%s for %s prompts; ignoring over_provision_groups=%s",
                    target_groups,
                    num_prompts,
                    self._over_provision_groups,
                )
        elif self._over_provision_groups:
            target_groups = num_prompts - self._over_provision_groups
        else:
            return None
        if target_groups < 1:
            raise ValueError(
                f"over-provisioned batch of {num_prompts} prompts leaves target_groups={target_groups}; "
                "need at least one group to keep"
            )
        return target_groups if target_groups < num_prompts else None

    async def generate_sequences(self, prompts: TensorDict) -> None:
        """Run rollout-manager generation and write outputs into TransferQueue."""
        if self._rollout_config is None:
            raise RuntimeError("GatewayAgentFramework requires rollout_config for generate_sequences")

        is_validate = bool(tu.get(prompts, "validate", False))
        partition_id = "val" if is_validate else "train"
        global_steps = tu.get(prompts, "global_steps")
        if global_steps is None and not is_validate:
            raise ValueError("GatewayAgentFramework requires prompts['global_steps'] for training")

        if partition_id == "val":
            val_kwargs = self._rollout_config.get("val_kwargs", {})
            num_sessions = int(val_kwargs.get("n"))
        else:
            num_sessions = int(self._rollout_config.get("n"))

        uids = tu.get(prompts, "uid")
        if uids is None:
            raise ValueError("GatewayAgentFramework requires prompts['uid'] for TransferQueue output")

        target_groups = self._resolve_target_groups(prompts, partition_id)
        if target_groups is None:
            try:
                stats = await self._run_batch_rollouts(
                    prompts,
                    global_steps=global_steps,
                    partition_id=partition_id,
                    num_sessions=num_sessions,
                )
            except BaseException:
                # The trainer polls TQ tags and never sees this actor call fail;
                # an untagged prompt would stall it forever.
                logger.exception(
                    "generate_sequences died at global_steps=%s; tagging unresolved prompts as failure",
                    global_steps,
                )
                await self._fail_unresolved_prompts([str(uid) for uid in uids], partition_id)
                raise
        else:
            stats = await self._run_batch_rollouts_with_cut(
                prompts,
                global_steps=global_steps,
                partition_id=partition_id,
                num_sessions=num_sessions,
                target_groups=target_groups,
            )
        logger.info(
            "generate_sequences summary: num_input_prompts=%s num_success_sessions=%s "
            "num_failed_sessions=%s num_success_outputs=%s num_unfinished_episodes=%s "
            "num_failed_uids=%s num_dropped_uids=%s num_dropped_sessions=%s num_cut_groups=%s failure_reasons=%s",
            stats["num_input_prompts"],
            stats["num_success_sessions"],
            stats["num_failed_sessions"],
            stats["num_success_outputs"],
            stats["num_unfinished_episodes"],
            stats["num_failed_uids"],
            stats["num_dropped_uids"],
            stats["num_dropped_sessions"],
            stats["num_cut_groups"],
            stats["failure_reasons"][:3],
        )
        if stats["num_success_outputs"] == 0:
            raise RuntimeError(
                f"All rollouts failed at global_steps={global_steps}. "
                f"failures={stats['num_failed_uids']}/{stats['num_input_prompts']}"
            )
        # Groups cut by over-provisioning never finished, so they are neither
        # failures nor part of the denominator of the failure guards below.
        num_resolved_prompts = stats["num_input_prompts"] - stats["num_cut_groups"]
        # Failed prompts are dropped from the batch; past this fraction the step
        # would silently train on a different batch than configured.
        if stats["num_failed_uids"] > self._max_failed_prompt_fraction * num_resolved_prompts:
            raise RuntimeError(
                f"Too many failed prompts at global_steps={global_steps}: "
                f"{stats['num_failed_uids']}/{num_resolved_prompts} > "
                f"max_failed_prompt_fraction={self._max_failed_prompt_fraction}; "
                f"reasons={stats['failure_reasons'][:3]}"
            )
        # Partial session loss keeps every prompt alive but shrinks groups, which
        # the prompt guard above cannot see.
        num_planned_sessions = num_resolved_prompts * num_sessions
        if stats["num_failed_sessions"] > self._max_failed_session_fraction * num_planned_sessions:
            raise RuntimeError(
                f"Too many failed sessions at global_steps={global_steps}: "
                f"{stats['num_failed_sessions']}/{num_planned_sessions} > "
                f"max_failed_session_fraction={self._max_failed_session_fraction}; "
                f"reasons={stats['failure_reasons'][:3]}"
            )
        return None

    async def pop_rollout_metrics(self, global_steps: int | None) -> dict[str, float]:
        """Return (once) the rollout metrics of an over-provisioned training step.

        The trainer calls this right after sampling the step's batch. It can sample as
        soon as the last kept group is tagged, a moment before the metrics are published;
        the bounded wait covers that gap. Steps that were not over-provisioned have no
        metrics and return ``{}``.
        """
        entry = self._step_metrics.get(global_steps)
        if entry is None:
            return {}
        published, metrics = entry
        try:
            await asyncio.wait_for(published.wait(), timeout=self._METRICS_PUBLISH_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("rollout metrics for global_steps=%s were not published in time", global_steps)
        self._step_metrics.pop(global_steps, None)
        return dict(metrics)

    async def _run_batch_rollouts(
        self,
        prompts: TensorDict,
        *,
        global_steps: int | None,
        partition_id: str,
        num_sessions: int = 1,
    ) -> dict:
        """Run all prompts in a batch and aggregate prompt/session stats."""
        assert len(prompts) > 0, "generate_sequences requires a non-empty batch"
        if num_sessions <= 0:
            raise ValueError(f"num_sessions must be positive, got {num_sessions}")

        # Batch layer: each sample/prompt owns its own group of rollout.n sessions.
        # Prompt tasks are isolated so one prompt failure does not drop the whole batch.
        tasks = []
        uids = []
        for sample_index in range(len(prompts)):
            sample_fields = self._extract_sample_fields(prompts=prompts, sample_index=sample_index)
            uids.append(str(sample_fields.get("uid")))
            tasks.append(
                self._run_prompt_rollouts(
                    sample_fields=sample_fields,
                    sample_index=sample_index,
                    global_steps=global_steps,
                    partition_id=partition_id,
                    num_sessions=num_sessions,
                )
            )
        # An outer cancellation makes gather itself raise; a CancelledError among
        # the outcomes was raised from inside one prompt and fails only that prompt.
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        stats = _new_batch_stats(len(prompts))
        for uid, outcome in zip(uids, outcomes, strict=True):
            if isinstance(outcome, Exception | asyncio.CancelledError):
                _add_failed_prompt(stats, num_sessions, _short_failure_reason(outcome))
                await self._tag_prompt_failure(uid, partition_id)
                continue
            if isinstance(outcome, BaseException):
                raise outcome
            _add_prompt_stats(stats, outcome)
        return stats

    async def _tag_prompt_failure(self, uid: str, partition_id: str) -> None:
        try:
            await asyncio.wait_for(
                tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"}),
                timeout=self._TQ_TAG_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.exception("failed to tag prompt uid=%s as failure in TQ", uid)

    async def _fail_unresolved_prompts(self, uids: list[str], partition_id: str) -> None:
        """Tag every prompt of ``uids`` still pending/running in TQ as ``failure``."""
        try:
            listing = await asyncio.wait_for(tq.async_kv_list(partition_id), timeout=self._TQ_TAG_TIMEOUT_SECONDS)
            tags = listing.get(partition_id, {})
        except Exception:
            logger.exception(
                "failed to list TQ partition %s; tagging all %s prompts as failure", partition_id, len(uids)
            )
            tags = {}
        unresolved = [uid for uid in uids if (tags.get(uid) or {}).get("status", "running") in ("pending", "running")]
        if unresolved:
            logger.error("tagging %s unresolved prompts as failure: %s", len(unresolved), unresolved[:8])
        for uid in unresolved:
            await self._tag_prompt_failure(uid, partition_id)

    async def _run_batch_rollouts_with_cut(
        self,
        prompts: TensorDict,
        *,
        global_steps: int | None,
        partition_id: str,
        num_sessions: int,
        target_groups: int,
    ) -> dict:
        """Run an over-provisioned batch and keep the first ``target_groups`` groups to finish.

        Once ``target_groups`` groups have finished (at least ``min_group_size``
        sessions survived and reached TQ), every group still in flight is cut: its
        sessions are cancelled, which cancels their runner Ray tasks, aborts their
        Gateway sessions and tears down their sandboxes. Kept groups are written to
        TQ in prompt order, not completion order.

        The trainer waits for exactly ``target_groups`` terminal groups, so this
        writes exactly that many terminal tags: kept groups are ``finished``; if
        fewer than ``target_groups`` groups could finish, failed groups fill the
        rest as ``failure`` (the ordinary failed-prompt path). Every other group's
        prompt key is cleared from TQ, so it never reaches this or a later step.
        """
        num_prompts = len(prompts)
        assert 0 < target_groups < num_prompts
        if num_sessions <= 0:
            raise ValueError(f"num_sessions must be positive, got {num_sessions}")
        published = asyncio.Event()
        metrics: dict[str, float] = {}
        self._step_metrics[global_steps] = (published, metrics)
        while len(self._step_metrics) > self._MAX_UNCLAIMED_STEP_METRICS:
            self._step_metrics.pop(next(iter(self._step_metrics)))

        uids: list[str] = []
        pending: dict[asyncio.Future, int] = {}
        for sample_index in range(num_prompts):
            sample_fields = self._extract_sample_fields(prompts=prompts, sample_index=sample_index)
            uids.append(str(sample_fields.get("uid")))
            task = asyncio.ensure_future(
                self._run_prompt_sessions(
                    sample_fields=sample_fields,
                    sample_index=sample_index,
                    global_steps=global_steps,
                    partition_id=partition_id,
                    num_sessions=num_sessions,
                )
            )
            pending[task] = sample_index

        stats = _new_batch_stats(num_prompts)
        held: dict[int, _PromptGroup] = {}  # finished, waiting for every lower index to resolve
        kept: list[int] = []
        failed: dict[int, str] = {}
        cut: dict[int, str] = {}
        next_to_write = 0

        async def write_resolved_prefix() -> None:
            # Writing a group only once every lower prompt index has resolved keeps
            # TQ insertion in prompt order while still streaming most writes.
            nonlocal next_to_write
            while next_to_write < num_prompts:
                index = next_to_write
                if index in held:
                    outcome = await self._write_prompt_group(
                        held.pop(index), global_steps=global_steps, partition_id=partition_id, tag_failure=False
                    )
                    _add_prompt_stats(stats, outcome)
                    if outcome["num_success_sessions"] > 0:
                        kept.append(index)
                    else:
                        failed[index] = "every TQ write failed"
                elif index not in failed and index not in cut:
                    return
                next_to_write += 1

        cut_tasks: list[asyncio.Future] = []
        try:
            while pending and len(kept) + len(held) < target_groups:
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in sorted(done, key=pending.__getitem__):
                    index = pending.pop(task)
                    if len(kept) + len(held) >= target_groups:
                        cut[index] = "finished after target_groups was reached"
                        continue
                    try:
                        group = task.result()
                    except (Exception, asyncio.CancelledError) as exc:
                        # Groups are only cancelled from here after this loop; a
                        # cancelled group in ``done`` cancelled itself from inside.
                        _add_failed_prompt(stats, num_sessions, _short_failure_reason(exc))
                        failed[index] = _short_failure_reason(exc)
                        continue
                    if group.completed:
                        held[index] = group
                    else:
                        _add_prompt_stats(
                            stats,
                            await self._write_prompt_group(
                                group, global_steps=global_steps, partition_id=partition_id, tag_failure=False
                            ),
                        )
                        failed[index] = group.drop_reason or "every session failed"
                await write_resolved_prefix()

            for task, index in pending.items():
                cut[index] = "in flight when target_groups was reached"
                task.cancel()
            cut_tasks.extend(pending)
            pending.clear()
            await write_resolved_prefix()
            assert not held and next_to_write == num_prompts

            # Fill up to target_groups terminal groups with failures, lowest prompt
            # index first, exactly as a non-over-provisioned batch would report them.
            failure_tagged = sorted(failed)[: max(0, target_groups - len(kept))]
            unused_failed = sorted(set(failed) - set(failure_tagged))
            await self._clear_prompt_keys([uids[index] for index in unused_failed], partition_id)
            stats["num_cut_groups"] = len(cut)
            metrics.update(
                {
                    "rollout/groups_launched": num_prompts,
                    "rollout/groups_target": target_groups,
                    "rollout/groups_kept": len(kept),
                    "rollout/groups_cut": len(cut),
                    "rollout/groups_failed": len(failed),
                    "rollout/sessions_failed": stats["num_failed_sessions"],
                }
            )
            published.set()
            for index in failure_tagged:
                await tq.async_kv_put(key=uids[index], partition_id=partition_id, tag={"status": "failure"})
            logger.info(
                "over-provision cut at global_steps=%s: kept %s/%s groups (target %s) in prompt order %s; "
                "cut %s; failed %s; failed groups reported as failure %s",
                global_steps,
                len(kept),
                num_prompts,
                target_groups,
                kept,
                [(index, uids[index], reason) for index, reason in sorted(cut.items())],
                [(index, uids[index], reason) for index, reason in sorted(failed.items())],
                failure_tagged,
            )
        except BaseException:
            for task in pending:
                task.cancel()
            cut_tasks.extend(pending)
            raise
        finally:
            published.set()
            await self._await_cut_groups(cut_tasks)
            await self._clear_prompt_keys([uids[index] for index in sorted(cut)], partition_id)
        return stats

    async def _await_cut_groups(self, tasks: list[asyncio.Future]) -> None:
        """Wait (bounded) for cancelled groups to tear down their sessions and sandboxes."""
        if not tasks:
            return
        _, still_running = await asyncio.wait(tasks, timeout=self._CUT_UNWIND_TIMEOUT_SECONDS)
        for task in tasks:
            if task.done() and not task.cancelled():
                task.exception()  # retrieved so asyncio does not log it as never retrieved
        if still_running:
            logger.error(
                "%s cut prompt groups did not unwind within %ss; abandoning them",
                len(still_running),
                self._CUT_UNWIND_TIMEOUT_SECONDS,
            )

    async def _clear_prompt_keys(self, uids: list[str], partition_id: str) -> None:
        """Drop prompt keys that will never reach a terminal status, so no later step sees them."""
        if not uids:
            return
        try:
            await tq.async_kv_clear(keys=uids, partition_id=partition_id)
        except Exception:
            logger.exception("failed to clear %s cut/unused prompt keys from TQ: %s", len(uids), uids[:8])

    async def _run_prompt_rollouts(
        self,
        *,
        sample_fields: dict[str, object],
        sample_index: int,
        global_steps: int | None,
        partition_id: str,
        num_sessions: int,
    ) -> dict:
        """Run ``rollout.n`` independent sessions for one prompt and persist their outputs."""
        group = await self._run_prompt_sessions(
            sample_fields=sample_fields,
            sample_index=sample_index,
            global_steps=global_steps,
            partition_id=partition_id,
            num_sessions=num_sessions,
        )
        return await self._write_prompt_group(group, global_steps=global_steps, partition_id=partition_id)

    async def _run_prompt_sessions(
        self,
        *,
        sample_fields: dict[str, object],
        sample_index: int,
        global_steps: int | None,
        partition_id: str,
        num_sessions: int,
    ) -> _PromptGroup:
        """Run ``rollout.n`` sessions for one prompt and apply the group-size guard; no TQ writes."""
        uid = sample_fields.get("uid")
        if uid is None:
            raise ValueError("GatewayAgentFramework requires prompts['uid'] for TransferQueue output")
        uid = str(uid)
        sampling_params = self._build_session_sampling_params(
            partition_id=partition_id,
            sample_fields=sample_fields,
        )

        # Prompt layer: rollout.n sessions race independently for the same uid.
        # Successful sessions are written to TQ; failed sessions only affect this uid's stats.
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        tasks = [
            self._run_agent_episode_with_concurrency_limit(
                sample_fields=sample_fields,
                sample_index=sample_index,
                session_index=session_index,
                global_steps=global_steps,
                sampling_params=sampling_params,
            )
            for session_index in range(num_sessions)
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        group = _PromptGroup(uid=uid, num_sessions=num_sessions)
        for session_index, outcome in enumerate(outcomes):
            # An outer cancellation makes gather itself raise; a CancelledError
            # among the outcomes came from inside that session and fails only it.
            if isinstance(outcome, Exception | asyncio.CancelledError):
                group.failed_sessions += 1
                group.failure_reasons.append(_short_failure_reason(outcome))
                continue
            # Propagate other control-flow exceptions such as SystemExit.
            if isinstance(outcome, BaseException):
                raise outcome

            trajectories, session_sample_fields = outcome
            if not trajectories:
                group.failed_sessions += 1
                group.failure_reasons.append(f"empty trajectories for uid={uid} session_index={session_index}")
                continue
            group.completed.append((session_index, trajectories, session_sample_fields))

        # A group shrunk by failures gives GRPO a high-variance baseline that is
        # biased toward whatever outcome survives failures; drop it instead.
        min_group_size = min(self._min_group_size, num_sessions) if partition_id == "train" else 1
        if 0 < len(group.completed) < min_group_size:
            group.dropped_sessions = len(group.completed)
            group.drop_reason = (
                f"group uid={uid} dropped: {len(group.completed)}/{num_sessions} sessions survived "
                f"< min_group_size={min_group_size}"
            )
            logger.warning(group.drop_reason)
            group.failure_reasons.append(group.drop_reason)
            group.completed = []
        return group

    async def _write_prompt_group(
        self,
        group: _PromptGroup,
        *,
        global_steps: int | None,
        partition_id: str,
        tag_failure: bool = True,
    ) -> dict:
        """Write a group's surviving sessions to TQ and tag the prompt finished (or failure)."""
        uid = group.uid
        failed_sessions = group.failed_sessions
        failure_reasons = list(group.failure_reasons)
        success_sessions = 0
        success_outputs = 0
        unfinished_episodes = 0
        for session_index, trajectories, session_sample_fields in group.completed:
            try:
                await self._write_session_trajectories_to_tq(
                    uid=uid,
                    session_index=session_index,
                    trajectories=trajectories,
                    sample_fields=session_sample_fields,
                    global_steps=global_steps,
                    partition_id=partition_id,
                )
            except Exception as e:
                logger.exception(f"TQ write failed for uid={uid} session={session_index}: {e}")
                failed_sessions += 1
                failure_reasons.append(f"TQ write error: {e}")
            else:
                success_sessions += 1
                success_outputs += len(trajectories)
                # One session is one episode; its trajectories all carry the same
                # session-level completion flag, so this counts episodes, not tokens.
                if any(traj.finished is False for traj in trajectories):
                    unfinished_episodes += 1

        if success_sessions > 0:
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
            failed_uids = 0
        else:
            if tag_failure:
                await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})
            failed_uids = 1

        return {
            "num_success_sessions": success_sessions,
            "num_failed_sessions": failed_sessions,
            "num_success_outputs": success_outputs,
            "num_unfinished_episodes": unfinished_episodes,
            "num_failed_uids": failed_uids,
            "num_dropped_uids": int(group.dropped_sessions > 0),
            "num_dropped_sessions": group.dropped_sessions,
            "failure_reasons": failure_reasons,
        }

    async def _run_agent_episode_with_concurrency_limit(
        self,
        *,
        sample_fields: dict[str, object],
        sample_index: int,
        session_index: int,
        global_steps: int | None,
        sampling_params: dict[str, object],
    ) -> tuple[list[Trajectory], dict[str, object]]:
        # Lazy-init semaphores on first use and rebind if the running loop
        # changed: asyncio.Semaphore binds to the loop at construction, but
        # Ray actors may run sessions on a different loop than __init__.
        loop = asyncio.get_running_loop()
        if self._semaphore_loop is not loop:
            self._runner_semaphores = {}
            self._semaphore_loop = loop

        if len(self.runner_registry) == 1:
            runner_name, runner_config = next(iter(self.runner_registry.items()))
        else:
            agent_name = sample_fields.get("agent_name")
            if agent_name is None:
                raise ValueError("agent_name is required when multiple agent_runners are configured")
            if not isinstance(agent_name, str):
                raise ValueError(f"agent_name must be a string, got {type(agent_name).__name__}")
            try:
                runner_name = agent_name
                runner_config = self.runner_registry[runner_name]
            except KeyError as exc:
                raise ValueError(f"Unknown agent runner: {agent_name}") from exc

        runner_cap = runner_config.max_concurrent_sessions
        if runner_cap <= 0:
            return await self._run_agent_episode_with_deadline(
                sample_fields=sample_fields,
                sample_index=sample_index,
                session_index=session_index,
                global_steps=global_steps,
                runner_name=runner_name,
                runner_config=runner_config,
                sampling_params=sampling_params,
            )

        runner_semaphore = self._runner_semaphores.get(runner_name)
        if runner_semaphore is None:
            runner_semaphore = asyncio.Semaphore(runner_cap)
            self._runner_semaphores[runner_name] = runner_semaphore

        async with runner_semaphore:
            return await self._run_agent_episode_with_deadline(
                sample_fields=sample_fields,
                sample_index=sample_index,
                session_index=session_index,
                global_steps=global_steps,
                runner_name=runner_name,
                runner_config=runner_config,
                sampling_params=sampling_params,
            )

    async def _run_agent_episode_with_deadline(
        self,
        *,
        runner_config: _RunnerConfig,
        **episode_kwargs,
    ) -> tuple[list[Trajectory], dict[str, object]]:
        """Bound the whole session (runner, finalize, scoring) by ``session_timeout_seconds``.

        The deadline covers every await of the session, not just the runner, so a
        hang anywhere fails only this session. A timed-out session raises
        ``TimeoutError`` and is counted as a failed session by the group/batch
        guards. Its cleanup is itself bounded: if it does not unwind within
        ``_SESSION_CLEANUP_GRACE_SECONDS`` it is abandoned rather than awaited.
        """
        timeout = runner_config.session_timeout_seconds
        episode = self._run_agent_episode(runner_config=runner_config, **episode_kwargs)
        task = asyncio.ensure_future(episode)
        # An abandoned task may still fail later; retrieve it so asyncio does
        # not log "exception was never retrieved".
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.wait({task}, timeout=self._SESSION_CLEANUP_GRACE_SECONDS)
            raise
        if done:
            if task.cancelled():
                # Nobody cancelled this session from outside: a CancelledError
                # raised inside it is a session failure, not batch cancellation.
                raise RuntimeError(
                    f"session (sample_index={episode_kwargs.get('sample_index')} "
                    f"session_index={episode_kwargs.get('session_index')}) was cancelled from inside"
                )
            return task.result()

        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=self._SESSION_CLEANUP_GRACE_SECONDS)
        if not done:
            logger.error(
                "session (sample_index=%s session_index=%s) did not unwind within %ss after timing out; abandoning it",
                episode_kwargs.get("sample_index"),
                episode_kwargs.get("session_index"),
                self._SESSION_CLEANUP_GRACE_SECONDS,
            )
        raise TimeoutError(f"session exceeded session_timeout_seconds={timeout:g}")

    async def _abort_session(self, session_id: str) -> None:
        """Abort a Gateway session without letting a hung or failing abort mask the session error."""
        try:
            await asyncio.wait_for(
                asyncio.shield(self.gateway_manager.abort_session(session_id)),
                timeout=self._SESSION_ABORT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "session %s: Gateway abort timed out after %ss", session_id, self._SESSION_ABORT_TIMEOUT_SECONDS
            )
        except Exception:
            logger.exception("session %s: Gateway abort failed", session_id)

    async def _run_agent_episode(
        self,
        *,
        sample_fields: dict[str, object],
        sample_index: int,
        session_index: int,
        global_steps: int | None,
        runner_name: str,
        runner_config: _RunnerConfig,
        sampling_params: dict[str, object],
    ) -> tuple[list[Trajectory], dict[str, object]]:
        """Run one AgentRunner episode inside a Framework-created Gateway session."""
        session_id = f"session-sample-{sample_index}-rollout-{session_index}-{uuid4().hex}"
        uid = str(sample_fields.get("uid", ""))
        session_trace = agent_loop_session(
            sample=sample_index,
            session=session_index,
            uid=uid,
            global_steps=global_steps,
            session_id=session_id,
        )
        trace_identity = session_trace.identity
        if self._log_dir:
            log_root = Path(self._log_dir)
            run_dir = (log_root if global_steps is None else log_root / f"step_{int(global_steps)}") / session_id
            framework_log = LogContext(session_id, str(run_dir / "framework.log"))
            task_log = LogContext(session_id, str(run_dir / "task.log"))
            parent_log = framework_log if runner_config.dispatch_mode == "ray_task" else task_log
        else:
            run_dir = None
            framework_log = None
            task_log = None
            parent_log = None

        raw_prompt = sample_fields["raw_prompt"]
        tools_kwargs = sample_fields.get("tools_kwargs")
        tools_kwargs = dict(tools_kwargs or {})
        tools_kwargs["_trace_identity"] = trace_identity
        async with _log_scope(parent_log):
            session = await self.gateway_manager.create_session(
                session_id,
                metadata={"_trace_identity": trace_identity},
                sampling_params=dict(sampling_params),
            )
            logger.info(
                "session %s start: runner=%s sample_index=%s session_index=%s global_steps=%s",
                session_id,
                runner_name,
                sample_index,
                session_index,
                global_steps,
            )
            try:
                if runner_config.dispatch_mode == "ray_task":
                    # Ray workers run only the runner. Gateway token truth,
                    # finalization, reward scoring, and TQ writes stay in parent.
                    object_ref = _run_agent_runner_ray_task.remote(
                        runner_fqn=runner_config.runner_fqn,
                        runner_kwargs=runner_config.runner_kwargs,
                        raw_prompt=raw_prompt,
                        session=session,
                        sample_index=sample_index,
                        tools_kwargs=tools_kwargs,
                        log_context=task_log,
                    )
                    # The session deadline (_run_agent_episode_with_deadline)
                    # arrives here as cancellation. It only stops the parent's
                    # await; the Ray task must be cancelled explicitly or it (and
                    # its sandbox) would keep running. Graceful cancel first so the
                    # runner's asyncio.run unwinds and its sandbox context manager
                    # tears down cleanly; force-kill only if it ignores the cancel.
                    try:
                        task_result = await object_ref
                    except RayTaskError as exc:
                        # Ray re-raises a runner's CancelledError as a class that
                        # also subclasses CancelledError; it is a runner failure,
                        # not cancellation of this session.
                        if isinstance(exc, asyncio.CancelledError):
                            raise RuntimeError(f"runner task raised CancelledError: {exc}") from None
                        raise
                    except asyncio.CancelledError:
                        await self._cancel_runner_task(object_ref, session_id)
                        raise
                else:
                    runner = self._inline_runners[runner_name]
                    task_result = await runner(
                        raw_prompt=raw_prompt,
                        session=session,
                        sample_index=sample_index,
                        **({"tools_kwargs": tools_kwargs} if tools_kwargs is not None else {}),
                    )
                if task_result is None:
                    task_result = TaskResult()
                elif not isinstance(task_result, TaskResult):
                    raise TypeError(
                        f"Agent runner {runner_name!r} must return TaskResult or None, got {type(task_result).__name__}"
                    )
                session_trajectories = await self.gateway_manager.finalize_session(session_id)
                session_trajectories = _select_session_trajectories(
                    session_id,
                    session_trajectories,
                    runner_config.trajectory_selection,
                )
            except asyncio.CancelledError:
                # Parent shutdown/cancellation must not leave the Gateway route
                # and actor-owned session live after the runner task is gone.
                await self._abort_session(session_id)
                raise
            except Exception:
                logger.exception("session %s failed (runner=%s); aborting session", session_id, runner_name)
                await self._abort_session(session_id)
                session_trace.finish(
                    runner_name=runner_name,
                    status="failure",
                    trajectories=[],
                )
                raise

            # Make the Runner's canonical result visible to postprocessors before
            # they filter or reorder trajectories.  These fields are the
            # Framework-owned trajectory representation of the session result;
            # scorer metadata remains a separate Worker output channel.
            task_metrics = {} if task_result.accuracy is None else {"acc": task_result.accuracy}
            if session_trajectories:
                session_trajectories = [
                    replace(
                        trajectory,
                        finished=task_result.finished,
                        reward_score=task_result.reward,
                        reward_metrics=dict(task_metrics),
                    )
                    for trajectory in session_trajectories
                ]

            if self._trajectory_postprocessor is not None:
                session_trajectories = await self._apply_trajectory_postprocessor(session_trajectories, task_result)

            if not session_trajectories:
                session_trace.finish(
                    runner_name=runner_name,
                    status="empty",
                    trajectories=[],
                )
                return session_trajectories, sample_fields

            if self.reward_loop_worker_handles and self._custom_reward_function_configured:
                annotations = await self._score_trajectories(
                    session_trajectories,
                    sample_fields,
                    task_result,
                )
                reward_source = "reward_loop_worker"
            elif task_result.reward is not None:
                annotations = [(float(task_result.reward), {}) for _ in session_trajectories]
                reward_source = "agent_runner"
            elif self.reward_loop_worker_handles:
                annotations = await self._score_trajectories(session_trajectories, sample_fields, task_result)
                reward_source = "reward_loop_worker"
            else:
                annotations = None
                reward_source = None

            if annotations is None:
                logger.info("session %s: Framework produced no reward; rm_scores remain zero", session_id)
                result_trajectories = session_trajectories
            else:
                logger.info("session %s: scored via %s", session_id, reward_source)
                result_trajectories = [
                    replace(
                        traj,
                        finished=task_result.finished,
                        reward_score=score,
                        reward_metrics=extra if reward_source == "reward_loop_worker" else dict(task_metrics),
                    )
                    for traj, (score, extra) in zip(session_trajectories, annotations, strict=True)
                ]

            result_trajectories = _attach_runner_reward_info(result_trajectories, task_result)

            self._log_trajectory_summary(session_id, result_trajectories)
            if run_dir is not None:
                await asyncio.to_thread(self._dump_trajectories, run_dir, session_id, result_trajectories)
            session_trace.finish(
                runner_name=runner_name,
                status="success",
                trajectories=result_trajectories,
                reward_source=reward_source,
                finished=result_trajectories[0].finished if result_trajectories else None,
            )
            return result_trajectories, sample_fields

    async def _cancel_runner_task(self, object_ref, session_id: str) -> None:
        """Cancel a dispatched runner Ray task after its session timed out.

        Graceful cancel (``force=False``) lets the worker's ``asyncio.run`` raise
        ``CancelledError`` and unwind, so runner-side cleanup (e.g. the task's
        sandbox context manager) runs. If the task is still pending after a short
        grace period, force-kill it: correctness of the batch (freeing the worker
        and its resources) beats the risk of skipping graceful teardown.
        """
        try:
            ray.cancel(object_ref)
        except Exception:
            logger.exception("session %s: ray.cancel failed for runner task", session_id)
            return
        try:
            await asyncio.wait_for(object_ref, timeout=self._RUNNER_CANCEL_GRACE_SECONDS)
        except asyncio.TimeoutError:
            logger.warning(
                "session %s: runner task ignored graceful cancel after %ss; force-killing",
                session_id,
                self._RUNNER_CANCEL_GRACE_SECONDS,
            )
            try:
                ray.cancel(object_ref, force=True)
            except Exception:
                logger.exception("session %s: force ray.cancel failed", session_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Task terminated while being cancelled (e.g. TaskCancelledError);
            # that is the expected outcome.
            pass

    def _log_trajectory_summary(self, session_id: str, trajectories: list[Trajectory]) -> None:
        """Log a per-session trajectory summary -- the info the task layer can't emit,
        since trajectories exist only after the session finalizes."""
        lines = [f"session {session_id}: {len(trajectories)} trajectory(ies)"]
        for i, traj in enumerate(trajectories):
            model_tokens = sum(traj.response_mask) if traj.response_mask else 0
            finished = traj.finished
            extra = traj.extra_fields or {}
            reason = extra.get("materialization_reason")
            lines.append(
                f"  [{i}] turns={traj.num_turns} prompt_tokens={len(traj.prompt_ids)} "
                f"response_tokens={len(traj.response_ids)} model_tokens={model_tokens} "
                f"finished={finished} "
                f"logprobs={'yes' if traj.response_logprobs else 'no'} "
                f"experts={'yes' if traj.routed_experts is not None else 'no'} "
                f"reward_score={traj.reward_score} reward_metrics={traj.reward_metrics}"
                + (f" materialization_reason={reason}" if reason else "")
            )
            if extra.get("num_coalesced_requests") or extra.get("rollback_count"):
                lines[-1] += (
                    f" coalesced_waiters={extra.get('num_coalesced_requests', 0)}"
                    f" rollback_count={extra.get('rollback_count', 0)}"
                    f" rollback_dropped_trainable_tokens={extra.get('rollback_dropped_trainable_tokens_total', 0)}"
                )
        logger.info("\n".join(lines))

    def _dump_trajectories(self, run_dir: Path, session_id: str, trajectories: list[Trajectory]) -> None:
        """Persist finalized trajectories next to ``task.log``.

        Split by cost: a small human-readable summary (reward, turns, lengths) is written
        to ``trajectory.json``; the bulky per-token arrays (ids / mask / logprobs) go to a
        compressed ``trajectory.npz``. The arrays serialize at C speed and compress well
        (token ids repeat, the mask is runs of 0/1), so this is far smaller and faster than
        the old indented-JSON dump -- which matters most on a network / HDFS log_dir.

        Runs off the event loop (caller wraps this in ``asyncio.to_thread``) and is
        best-effort: an IO / serialization error is logged but never aborts the rollout.
        """
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "session_id": session_id,
                "num_trajectories": len(trajectories),
                "trajectories": [self._trajectory_meta(traj) for traj in trajectories],
            }
            (run_dir / "trajectory.json").write_text(
                json.dumps(meta, ensure_ascii=False, separators=(",", ":"), default=_json_default),
                encoding="utf-8",
            )
            arrays: dict[str, np.ndarray] = {}
            for i, traj in enumerate(trajectories):
                arrays[f"traj{i}_prompt_ids"] = np.asarray(traj.prompt_ids, dtype=np.int32)
                arrays[f"traj{i}_response_ids"] = np.asarray(traj.response_ids, dtype=np.int32)
                arrays[f"traj{i}_response_mask"] = np.asarray(traj.response_mask, dtype=np.int8)
                if traj.response_logprobs is not None:
                    arrays[f"traj{i}_response_logprobs"] = np.asarray(traj.response_logprobs, dtype=np.float32)

            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)
            (run_dir / "trajectory.npz").write_bytes(buf.getvalue())
        except Exception:
            logger.exception("session %s: failed to write trajectory dump under %s", session_id, run_dir)

    def _trajectory_meta(self, traj: Trajectory) -> dict[str, object]:
        """Small, human-readable per-trajectory summary; the token arrays live in the npz."""
        extra = traj.extra_fields or {}
        return {
            "num_turns": traj.num_turns,
            "finished": traj.finished,
            "reward_score": traj.reward_score,
            "reward_metrics": dict(traj.reward_metrics),
            "num_coalesced_requests": extra.get("num_coalesced_requests"),
            "rollback_count": extra.get("rollback_count"),
            "rollback_dropped_trainable_tokens_total": extra.get("rollback_dropped_trainable_tokens_total"),
            "materialization_reason": extra.get("materialization_reason"),
            "prompt_len": len(traj.prompt_ids),
            "response_len": len(traj.response_ids),
            "model_token_count": sum(traj.response_mask) if traj.response_mask else 0,
            "has_routed_experts": traj.routed_experts is not None,
            "has_logprobs": traj.response_logprobs is not None,
        }

    async def _score_trajectories(
        self,
        session_trajectories: list[Trajectory],
        sample_fields: dict[str, object],
        task_result: TaskResult,
    ) -> list[tuple[float, dict[str, object]]]:
        """Score the session's final trajectory and broadcast (score, extra_info) to all.

        Mirrors AgentLoopWorkerTQ._agent_loop_postprocess
        (verl/trainer/main_ppo_sync.py:353-396): only the final trajectory (the
        session's last interaction segment) is dispatched to RewardLoopWorker.
        The runner payload is available under ``extra_info.runner_reward_info``;
        the Worker's score + reward_extra_info are then broadcast to every
        trajectory in the session. Subclasses can override this method to
        implement custom session-to-trajectory scoring policies.
        """
        assert self.reward_loop_worker_handles is not None
        assert session_trajectories, "expected non-empty session_trajectories"

        final_trajectory = session_trajectories[-1]
        data = _trajectory_to_reward_dataproto(final_trajectory, sample_fields, task_result)
        worker = random.choice(self.reward_loop_worker_handles)
        result = await worker.compute_score.remote(data)

        if not isinstance(result, dict) or "reward_score" not in result:
            raise ValueError(
                f"RewardLoopWorker result missing 'reward_score' key or invalid for uid={sample_fields.get('uid')}"
            )
        score = float(result["reward_score"])
        if not np.isfinite(score):
            raise ValueError("RewardLoopWorker reward_score must be finite")
        extra = result.get("reward_extra_info")
        if extra is None:
            extra = {}
        if not isinstance(extra, dict):
            raise ValueError("RewardLoopWorker reward_extra_info must be a dict")
        for key in extra:
            if not isinstance(key, str):
                raise ValueError("RewardLoopWorker reward_extra_info keys must be strings")
            if key == "reward":
                raise ValueError("RewardLoopWorker reward_extra_info key 'reward' is reserved")
        # Each trajectory needs its own dict: downstream code merges into it.
        return [(score, dict(extra)) for _ in session_trajectories]

    def _extract_sample_fields(self, *, prompts: TensorDict, sample_index: int) -> dict[str, object]:
        sample_fields = {}
        for key, value in prompts.items():
            if isinstance(value, torch.Tensor):
                sample_fields[key] = value if value.ndim == 0 else value[sample_index]
            elif isinstance(value, NonTensorStack):
                sample_fields[key] = tu.get(prompts, key)[sample_index]
            else:
                assert isinstance(value, NonTensorData)
                sample_fields[key] = value.data
        return sample_fields

    async def _write_session_trajectories_to_tq(
        self,
        *,
        uid: str,
        session_index: int,
        trajectories: list[Trajectory],
        sample_fields: dict[str, object],
        global_steps: int | None,
        partition_id: str,
    ) -> None:
        keys = []
        fields = []
        tags = []
        for index, trajectory in enumerate(trajectories):
            field, tag = self._trajectory_to_tq_field_and_tag(
                trajectory=trajectory,
                sample_fields=sample_fields,
                session_index=session_index,
                global_steps=global_steps,
                uid=uid,
            )
            keys.append(f"{uid}_{session_index}_{index}")
            fields.append(field)
            tags.append(tag)

        await tq.async_kv_batch_put(
            keys=keys,
            fields=_list_of_tq_fields_to_tensordict(fields),
            tags=tags,
            partition_id=partition_id,
        )

    def _trajectory_to_tq_field_and_tag(
        self,
        *,
        trajectory: Trajectory,
        sample_fields: dict[str, object],
        session_index: int,
        global_steps: int | None,
        uid: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        prompts = torch.tensor(trajectory.prompt_ids, dtype=torch.long)
        responses = torch.tensor(trajectory.response_ids, dtype=torch.long)
        source_response_mask = torch.tensor(trajectory.response_mask, dtype=torch.long)
        finished = trajectory.finished
        if finished is not None and type(finished) is not bool:
            raise ValueError("Trajectory.finished must be a bool or None")
        response_mask = (
            torch.zeros_like(source_response_mask)
            if self._mask_unfinished_episode and finished is False
            else source_response_mask
        )
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        mm_processor_kwargs = trajectory.extra_fields.get("mm_processor_kwargs")
        if mm_processor_kwargs is not None and not isinstance(mm_processor_kwargs, dict):
            raise ValueError("trajectory extra_fields.mm_processor_kwargs must be a dict or null")
        multi_modal_inputs = compute_multi_modal_inputs(
            self._processor,
            input_ids.unsqueeze(0),
            trajectory.multi_modal_data,
            mm_processor_kwargs,
        )
        if self._processor is None:
            position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0)).squeeze(0)
        else:
            position_ids = compute_position_ids(
                self._processor,
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                multi_modal_inputs,
            ).squeeze(0)

        field: dict[str, object] = {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "multi_modal_inputs": multi_modal_inputs,
        }
        if trajectory.response_logprobs is not None:
            field["rollout_log_probs"] = torch.tensor(trajectory.response_logprobs, dtype=torch.float32)
        if trajectory.routed_experts is not None:
            aligned_experts = _align_routed_experts(trajectory.routed_experts, input_ids.size(0))
            if aligned_experts is not None:
                field["routed_experts"] = aligned_experts
        rm_scores = torch.zeros_like(responses, dtype=torch.float32)
        if trajectory.reward_score is not None and responses.numel() > 0:
            rm_scores[-1] = float(trajectory.reward_score)
        field["rm_scores"] = rm_scores

        field["extra_fields"] = {
            **trajectory.extra_fields,
            "reward_extra_info": dict(trajectory.reward_metrics),
        }
        # Framework-owned masks must win over same-named Gateway extra fields.
        field["response_mask"] = response_mask
        field["loss_mask"] = response_mask
        field.pop("multi_modal_data", None)
        for key in (
            "uid",
            "raw_prompt",
            "data_source",
            "reward_model",
            "extra_info",
            "tools_kwargs",
            "agent_name",
        ):
            if key in sample_fields:
                field[key] = sample_fields[key]
        field["session_id"] = session_index
        field["global_steps"] = global_steps
        field["num_turns"] = torch.tensor(int(trajectory.num_turns), dtype=torch.long)

        prompt_len = prompts.size(0)
        response_len = responses.size(0)

        # The gateway reports the weight versions a trajectory spanned. Fall back to
        # the dataloader step only when absent (backends that report no version):
        # the trainer casts these tags with dtype=int, so None must not reach it.
        min_global_steps = trajectory.extra_fields.get("min_global_steps", global_steps)
        max_global_steps = trajectory.extra_fields.get("max_global_steps", global_steps)
        tag = {
            "global_steps": global_steps,
            "min_global_steps": global_steps if min_global_steps is None else min_global_steps,
            "max_global_steps": global_steps if max_global_steps is None else max_global_steps,
            "status": "success",
            "prompt_len": prompt_len,
            "response_len": response_len,
            "seq_len": prompt_len + response_len,
            "uid": uid,
        }
        materialization_reason = trajectory.extra_fields.get("materialization_reason")
        if materialization_reason is not None:
            tag["materialization_reason"] = materialization_reason
        return field, tag
