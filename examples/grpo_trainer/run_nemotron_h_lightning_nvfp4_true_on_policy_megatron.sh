#!/usr/bin/env bash
# Nemotron-H Lightning NVFP4 true-on-policy for VERL.
# VLLM_BATCH_INVARIANT=1 (default): batch-invariant vLLM rollout and a Megatron
# Lite actor on the same vLLM kernels, for exact probabilities.
# VLLM_BATCH_INVARIANT=0: the conventional baseline, not true on-policy: vLLM's
# default kernels, the verl Megatron engine (Megatron-Bridge/mcore) on the BF16
# release and rollout routing replay (R3); every sync requantizes the weights
# into the checkpoint's ModelOpt format. Both arms keep vLLM prefix caching on
# for the multi-turn agent workload. The actor runs
# weight QAT by default (ACTOR_PRECISION=qat: forward on the deployed NVFP4/FP8
# weights); ACTOR_PRECISION=bf16 trains on the BF16 masters as they are. Like the
# baseline-r3 recipe, it logs to W&B project verl-nemotron-h-baseline-r3 and
# forwards NCCL_CUMEM_ENABLE to an existing Ray cluster when set.
# ACTIVATION_PRECISION (BI=1) chooses how actor and rollout serve the release
# checkpoint's activations: release (default) as stored, FP8 W8A8 Mamba in/out_proj
# and FP8 query and KV cache; bf16 keeps them BF16: W8A16 Mamba in/out_proj (the
# FP8 weights dequantized) and a BF16 KV cache. The NVFP4 routed experts are
# W4A16 in both; the checkpoint is used as released. ACTOR_ROUTER_REPLAY (BI=1)
# is R2 by default: the actor's old-log-prob forward records each MoE router's
# experts and every PPO update of the step replays them; disabled recomputes them.
# TASK chooses the workload: math (default, DAPO-format parquet, single turn) or
# swe_bench_pro (uni-agent ReAct agent in E2B sandboxes, uni-agent-format parquet).
# swe_bench_pro is strictly on-policy (one update per step: mini batch = train
# batch, no routing replay at BI=1) with a 32K agent context; E2B credentials
# come from the environment of every Ray node, never through Hydra.
# ROLLOUT_NNODES=N (swe_bench_pro only, default 0) adds N nodes of standalone
# rollout replicas (same vLLM settings) next to the 1-node actor and its
# colocated rollout: the v1 separate_async trainer at zero staleness (weights
# synced over NCCL after every step, no warmup batch, stale groups dropped), so
# training stays strictly on-policy. The Ray cluster then has 1+N nodes.
# PIN_DP_RANK_BY_SESSION (swe_bench_pro, default True) sends every turn of an
# agent session to one DP rank of its rollout replica (stable hash of the
# session id), so multi-turn sessions keep their prefix cache.
# SWE_OVERPROVISION_GROUPS=K (swe_bench_pro sync only, default 0) launches
# TRAIN_BATCH_SIZE+K prompt groups per step and trains on the first
# TRAIN_BATCH_SIZE to finish; uni-agent cuts the K slowest (cancels their
# runners, aborts their sessions, kills their sandboxes).
# Modes: quick_alignment_test (1x4, short workload, three steps), aligned.
# Hardware: gb200 (1x4, PP1/EP4 actor, rollout DP4/EP4).
# Image builds on `Dockerfile.nemotron_h_true_on_policy`, which installs vLLM,
# Megatron-LM, Megatron-Bridge and VERL; nothing is mounted.
set -euo pipefail

# Required env: MODEL_PATH, BF16_MASTER_PATH, TRAIN_FILES, VAL_FILES.
# Optional env: VLLM_BATCH_INVARIANT, training steps, batch size, lengths, output
# paths, WANDB_API_KEY, WANDB_ENTITY, WANDB_MODE, WANDB_BASE_URL, and Hydra
# overrides; with VLLM_BATCH_INVARIANT=1 also ACTIVATION_PRECISION and
# ACTOR_ROUTER_REPLAY and ACTOR_PP_MAX_IN_FLIGHT, with VLLM_BATCH_INVARIANT=0 also
# ACTOR_PRECISION; at either setting MTP (+MTP_SPEC_TOKENS, MTP_LR, MTP_LOSS_COEF,
# MTP_SPEC_MOE_BACKEND); with
# TASK=swe_bench_pro also UNI_AGENT_TASK_CONFIG (required), UNI_AGENT_CONCURRENCY,
# ROLLOUT_NNODES, STANDALONE_GPU_MEMORY_UTILIZATION, PIN_DP_RANK_BY_SESSION and
# SWE_OVERPROVISION_GROUPS.
SEED="${SEED:-42}"
ACTOR_LR="${ACTOR_LR:-1e-6}"

usage() {
  echo "usage: $0 --hardware gb200 --mode {quick_alignment_test|aligned} [Hydra overrides...]"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 64
}

MODE="${MODE:-}"
HARDWARE="${HARDWARE:-}"
HYDRA_OVERRIDES=()
while (( $# > 0 )); do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires a value"
      MODE="$2"
      shift 2
      ;;
    --hardware)
      [[ $# -ge 2 ]] || die "--hardware requires a value"
      HARDWARE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      HYDRA_OVERRIDES+=("$@")
      break
      ;;
    *)
      HYDRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

[[ -n "${MODE}" ]] || die "set --mode quick_alignment_test or aligned"
[[ -n "${HARDWARE}" ]] || die "set --hardware gb200"
[[ "${HARDWARE}" == gb200 ]] || die "unknown hardware: ${HARDWARE}"

# --- Required inputs ---
: "${MODEL_PATH:?set MODEL_PATH to the Nemotron-H Lightning NVFP4 checkpoint}"
# The BF16 release the NVFP4 checkpoint was quantized from: the actor's masters.
: "${BF16_MASTER_PATH:?set BF16_MASTER_PATH to the Nemotron-H Lightning BF16 release}"
: "${TRAIN_FILES:?set TRAIN_FILES to DAPO-format training parquet}"
: "${VAL_FILES:?set VAL_FILES to DAPO-format validation parquet}"

# --- Task presets ---
TASK="${TASK:-math}"
: "${ROLLOUT_NNODES:=0}"
[[ "${ROLLOUT_NNODES}" =~ ^[0-9]+$ ]] || die "ROLLOUT_NNODES must be a non-negative integer, got '${ROLLOUT_NNODES}'"
case "${TASK}" in
  math)
    [[ "${ROLLOUT_NNODES}" == 0 ]] || die "ROLLOUT_NNODES applies to TASK=swe_bench_pro"
    : "${MAX_PROMPT_LENGTH:=2048}"
    TASK_ARGS=(trainer.use_v1=False)
    ;;
  swe_bench_pro)
    : "${UNI_AGENT_TASK_CONFIG:?set UNI_AGENT_TASK_CONFIG to the uni-agent Task Config YAML}"
    : "${TRAIN_BATCH_SIZE:=32}"
    : "${PPO_MINI_BATCH_SIZE:=${TRAIN_BATCH_SIZE}}"
    [[ "${PPO_MINI_BATCH_SIZE}" == "${TRAIN_BATCH_SIZE}" ]] \
      || die "TASK=swe_bench_pro updates once per step: PPO_MINI_BATCH_SIZE must equal TRAIN_BATCH_SIZE"
    : "${ROLLOUT_N:=8}"
    : "${MAX_PROMPT_LENGTH:=8192}"
    : "${MAX_RESPONSE_LENGTH:=24576}"
    # One micro-batch holds a whole trajectory; verl packs up to this times
    # the actor's CP size, which splits the pack over the CP ranks.
    : "${PPO_MAX_TOKEN_LEN_PER_GPU:=$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + ${ACTOR_CP:-1} - 1) / ${ACTOR_CP:-1} ))}"
    # Fixed split-KV FA4 scratch grows with the prefill chunk; 8192 leaves the
    # BI=1 rollout room next to the resident actor state.
    : "${ROLLOUT_MAX_NUM_BATCHED_TOKENS:=8192}"
    : "${ACTOR_ROUTER_REPLAY:=disabled}"
    # The E2B provider waits for sandbox capacity (E2B_CAPACITY_WAIT, 1800 s)
    # inside start(); the generic startup cap must not cut that wait short.
    export SANDBOX_STARTUP_TIMEOUT="${SANDBOX_STARTUP_TIMEOUT:-2400}"
    : "${SAVE_FREQ:=5}"
    : "${TEST_FREQ:=-1}"
    AF=actor_rollout_ref.rollout.custom.agent_framework
    TRAINER_MODE=sync
    (( ROLLOUT_NNODES == 0 )) || TRAINER_MODE=separate_async
    : "${SWE_OVERPROVISION_GROUPS:=0}"
    [[ "${SWE_OVERPROVISION_GROUPS}" =~ ^[0-9]+$ ]] \
      || die "SWE_OVERPROVISION_GROUPS must be a non-negative integer, got '${SWE_OVERPROVISION_GROUPS}'"
    [[ "${SWE_OVERPROVISION_GROUPS}" == 0 || "${TRAINER_MODE}" == sync ]] \
      || die "SWE_OVERPROVISION_GROUPS needs the sync trainer (ROLLOUT_NNODES=0)"
    # OLD_LOG_PROB=rollout skips the actor's old-log-prob pass and uses the
    # rollout's log-probs (bypass mode). With one update per step actor/ppo_kl
    # then measures the rollout/actor mismatch, and entropy comes from the
    # update forward.
    case "${PIN_DP_RANK_BY_SESSION:=True}" in
      True|False) ;;
      *) die "PIN_DP_RANK_BY_SESSION must be True or False, got '${PIN_DP_RANK_BY_SESSION}'" ;;
    esac
    case "${OLD_LOG_PROB:=recompute}" in
      recompute) BYPASS_OLD_LOG_PROB=False ;;
      rollout) BYPASS_OLD_LOG_PROB=True ;;
      *) die "OLD_LOG_PROB must be recompute or rollout, got '${OLD_LOG_PROB}'" ;;
    esac
    TASK_ARGS=(
      trainer.use_v1=True
      trainer.v1.trainer_mode="${TRAINER_MODE}"
      trainer.v1.sampler.sync_over_provision_prompts="${SWE_OVERPROVISION_GROUPS}"
      actor_rollout_ref.actor.ppo_epochs=1
      algorithm.rollout_correction.bypass_mode="${BYPASS_OLD_LOG_PROB}"
      actor_rollout_ref.actor.calculate_entropy="${BYPASS_OLD_LOG_PROB}"
      transfer_queue.enable=True
      data.prompt_key=prompt
      data.return_raw_chat=True
      data.filter_overlong_prompts=True
      data.truncation=error
      actor_rollout_ref.rollout.multi_turn.enable=True
      actor_rollout_ref.rollout.pin_dp_rank_by_session="${PIN_DP_RANK_BY_SESSION}"
      actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
      ++actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
      ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter
      "++${AF}.gateway_count=8"
      "++${AF}.allowed_request_sampling_param_keys=[temperature,top_p,top_k]"
      "++${AF}.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task"
      "++${AF}.agent_runners.task.dispatch_mode=ray_task"
      "++${AF}.agent_runners.task.max_concurrent_sessions=${UNI_AGENT_CONCURRENCY:-512}"
      # A session (all turns and tool calls) that outlives this deadline is aborted.
      "++${AF}.agent_runners.task.session_timeout_seconds=${UNI_AGENT_SESSION_TIMEOUT:-3600}"
      "++${AF}.agent_runners.task.trajectory_selection=longest"
      "++${AF}.agent_runners.task.runner_kwargs.task_config_path=${UNI_AGENT_TASK_CONFIG}"
      "++${AF}.mask_unfinished_episode=False"
      # The gateway rolls a rewritten last assistant back on its own token buffer.
      "++${AF}.enable_last_assistant_rollback=True"
      # Fail the step instead of training on a batch that lost prompts to sandbox errors.
      "++${AF}.max_failed_prompt_fraction=${UNI_AGENT_MAX_FAILED_PROMPT_FRACTION:-0.25}"
      # ...or too many sessions, and drop groups too small for a GRPO baseline.
      "++${AF}.max_failed_session_fraction=${UNI_AGENT_MAX_FAILED_SESSION_FRACTION:-0.25}"
      "++${AF}.min_group_size=${UNI_AGENT_MIN_GROUP_SIZE:-$(( (ROLLOUT_N + 1) / 2 ))}"
      reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner
      reward.custom_reward_function.name=score_from_runner_result
      reward.reward_kwargs.overlong_buffer_cfg.enable=False
    )
    ;;
  *) die "TASK must be math or swe_bench_pro, got '${TASK}'" ;;
esac

# --- Mode presets ---
# quick_alignment_test keeps the full model (the actor requires the complete
# Lightning checkpoint); only the workload shrinks.
case "${MODE}" in
  quick_alignment_test)
    : "${TOTAL_TRAINING_STEPS:=3}"
    : "${TRAIN_BATCH_SIZE:=16}"
    : "${PPO_MINI_BATCH_SIZE:=8}"
    : "${OVERLONG_BUFFER_LEN:=512}"
    : "${ROLLOUT_N:=4}"
    : "${MAX_RESPONSE_LENGTH:=2048}"
    : "${ROLLOUT_MAX_NUM_SEQS:=64}"
    : "${PPO_MAX_TOKEN_LEN_PER_GPU:=8192}"
    : "${SAVE_FREQ:=-1}"
    : "${TEST_FREQ:=-1}"
    ;;
  aligned)
    : "${TOTAL_TRAINING_STEPS:=100}"
    : "${TRAIN_BATCH_SIZE:=128}"
    : "${PPO_MINI_BATCH_SIZE:=32}"
    : "${OVERLONG_BUFFER_LEN:=4096}"
    : "${ROLLOUT_N:=8}"
    : "${MAX_RESPONSE_LENGTH:=14000}"
    : "${ROLLOUT_MAX_NUM_SEQS:=128}"
    # Without context parallelism one micro-batch holds the longest sequence.
    : "${PPO_MAX_TOKEN_LEN_PER_GPU:=16384}"
    : "${SAVE_FREQ:=5}"
    : "${TEST_FREQ:=10}"
    ;;
  *)
    die "unknown mode '${MODE}'"
    ;;
esac
if [[ -n "${TRAINER_LOGGERS:-}" ]]; then
  :
elif [[ "${WANDB_MODE:-}" == disabled ]]; then
  TRAINER_LOGGERS='[console,file]'
elif [[ -v WANDB_API_KEY || -v WANDB_ENTITY || -v WANDB_MODE ]]; then
  TRAINER_LOGGERS='[console,file,wandb]'
else
  TRAINER_LOGGERS='[console,file]'
fi

# --- Hardware profile ---
# NNODES is the Ray cluster size; the actor (and its colocated rollout) always
# takes one node, the standalone rollout pool the other ROLLOUT_NNODES.
: "${NNODES:=$((1 + ROLLOUT_NNODES))}"
TRAINER_NNODES="${NNODES}"
if (( ROLLOUT_NNODES > 0 )); then
  [[ "${NNODES}" == $((1 + ROLLOUT_NNODES)) ]] \
    || die "ROLLOUT_NNODES=${ROLLOUT_NNODES} needs a 1+${ROLLOUT_NNODES}-node Ray cluster, got NNODES=${NNODES}"
  TRAINER_NNODES=1
fi
: "${NGPUS_PER_NODE:=4}"
: "${ACTOR_PP:=1}"
: "${ACTOR_CP:=1}"
: "${ACTOR_EP:=4}"
: "${ROLLOUT_DP:=4}"
: "${ROLLOUT_EP:=4}"
: "${ROLLOUT_AGENT_WORKERS:=8}"

ROLLOUT_TP="${ROLLOUT_TP:-1}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# --- Alignment behavior belongs to VLLM_BATCH_INVARIANT ---
: "${VLLM_BATCH_INVARIANT:=1}"
: "${ROLLOUT_MAX_NUM_BATCHED_TOKENS:=16384}"
# Routed experts: FlashInfer CuTe-DSL W4A16, the kernel the aligned actor calls
# and the fastest BI=0 backend ("auto" picks Marlin, which fails on GB200).
: "${ROLLOUT_MOE_BACKEND:=flashinfer_cutedsl}"
# Routed-experts capture (R3) needs Model Runner V2 as well.
export VLLM_USE_V2_MODEL_RUNNER=1
if [[ "${VLLM_BATCH_INVARIANT}" == 1 ]]; then
  # ACTOR_CP splits each packed micro-batch contiguously over the EP ranks:
  # attention gathers K/V and Mamba runs head-parallel, so log-probs are the
  # CP=1 ones bitwise. CP must divide the 8 Mamba groups and ACTOR_EP.
  case "${ACTOR_CP}" in
    1) ;;
    2|4) [[ "${ACTOR_PP}" == 1 && "${ACTOR_EP}" == 4 ]] \
      || die "ACTOR_CP=${ACTOR_CP} requires ACTOR_PP=1 ACTOR_EP=4" ;;
    *) die "ACTOR_CP must be 1, 2 or 4 with VLLM_BATCH_INVARIANT=1, got '${ACTOR_CP}'" ;;
  esac
  : "${ACTOR_OPTIMIZER:=fsdp2}"
  case "${ACTOR_OPTIMIZER}" in
    # The actor's training state stays on the GPU during rollout; larger values
    # run out of memory in vLLM's startup or FlashInfer autotune warmup.
    fsdp2) : "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.65}" ;;
    dist_opt) : "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.55}" ;;
    *) die "ACTOR_OPTIMIZER must be dist_opt or fsdp2, got '${ACTOR_OPTIMIZER}'" ;;
  esac
  : "${ACTIVATION_PRECISION:=release}"
  case "${ACTIVATION_PRECISION}" in
    release)
      ROLLOUT_KV_CACHE_DTYPE=fp8_e4m3
      ACTIVATION_ARGS=()
      ;;
    bf16)
      # vLLM serves the FP8 Mamba in/out_proj as W8A16 (weights dequantized
      # at load) and the KV cache in BF16; the actor mirrors both.
      ROLLOUT_KV_CACHE_DTYPE=bfloat16
      ACTIVATION_ARGS=(
        +actor_rollout_ref.actor.engine.impl_cfg.activation_precision=bf16
        +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.bf16_activations=true
      )
      ;;
    *) die "ACTIVATION_PRECISION must be release or bf16, got '${ACTIVATION_PRECISION}'" ;;
  esac
  : "${ACTOR_ROUTER_REPLAY:=R2}"
  case "${ACTOR_ROUTER_REPLAY}" in
    # The actor records routes only without pipeline parallelism.
    R2) [[ "${ACTOR_PP}" == 1 ]] || die "ACTOR_ROUTER_REPLAY=R2 requires ACTOR_PP=1" ;;
    disabled) ;;
    *) die "ACTOR_ROUTER_REPLAY must be R2 or disabled, got '${ACTOR_ROUTER_REPLAY}'" ;;
  esac
  ARM=
  PROJECT_NAME="${PROJECT_NAME:-verl-nemotron-h-true-on-policy}"
  ARM_ENV_NAMES=()
  ACTOR_SUMMARY="OPTIMIZER=${ACTOR_OPTIMIZER} ACTIVATIONS=${ACTIVATION_PRECISION}"
  ACTOR_SUMMARY+=" ROUTING_REPLAY=${ACTOR_ROUTER_REPLAY} CP=${ACTOR_CP}"
  # ACTOR_PP_MAX_IN_FLIGHT=c caps the micro-batches a PP stage holds activations
  # for at c + 1 (empty: plain 1F1B, stage r holds pp - r); gradients are
  # bitwise identical. SWE trajectories reach 128K tokens, and at PP4 1F1B's
  # four in-flight micro-batches add ~45 GiB on stage 0, so SWE runs them one
  # at a time.
  if [[ -z "${ACTOR_PP_MAX_IN_FLIGHT+x}" && "${TASK}" == swe_bench_pro && "${ACTOR_PP}" -gt 1 ]]; then
    ACTOR_PP_MAX_IN_FLIGHT=0
  fi
  PP_IN_FLIGHT_ARGS=()
  if [[ -n "${ACTOR_PP_MAX_IN_FLIGHT:-}" ]]; then
    [[ "${ACTOR_PP_MAX_IN_FLIGHT}" =~ ^[0-9]+$ ]] \
      || die "ACTOR_PP_MAX_IN_FLIGHT must be a non-negative integer, got '${ACTOR_PP_MAX_IN_FLIGHT}'"
    PP_IN_FLIGHT_ARGS=(+actor_rollout_ref.actor.engine.impl_cfg.pp_max_in_flight="${ACTOR_PP_MAX_IN_FLIGHT}")
    ACTOR_SUMMARY+=" PP_MAX_IN_FLIGHT=${ACTOR_PP_MAX_IN_FLIGHT}"
  fi
  MODE_ARGS=(
    actor_rollout_ref.actor.engine.impl=vllm
    +actor_rollout_ref.actor.engine.seed="${SEED}"
    +actor_rollout_ref.actor.engine.full_determinism=True
    actor_rollout_ref.actor.engine.attention_backend_override=null
    actor_rollout_ref.actor.engine.router_replay_mode="${ACTOR_ROUTER_REPLAY}"
    # True gives the n samples of a prompt one vLLM seed; batch invariance, not
    # full determinism, makes rollout and actor agree.
    actor_rollout_ref.rollout.full_determinism=False
    actor_rollout_ref.rollout.seed="${SEED}"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.all2all_backend=flashinfer_nvlink_one_sided
  )
  OPTIMIZER_ARGS=(
    +actor_rollout_ref.actor.engine.impl_cfg.optimizer="${ACTOR_OPTIMIZER}"
    actor_rollout_ref.actor.engine.param_offload=False
    actor_rollout_ref.actor.engine.optimizer_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.offload_fraction=1.0
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.decoupled_weight_decay=True
  )
  runtime_config_root="$(mktemp -d /tmp/nemotron-h-true-on-policy-config.XXXXXX)"
  trap 'rm -rf "${runtime_config_root}"' EXIT
  mkdir -p "${runtime_config_root}/critic" "${runtime_config_root}/model_engine"
  printf '%s\n' \
    '# @package _global_' \
    'model_engine: mlite' \
    >"${runtime_config_root}/model_engine/mlite.yaml"
  printf '%s\n' \
    '_target_: verl.workers.config.CriticConfig' \
    'enable: false' \
    'strategy: mlite' \
    >"${runtime_config_root}/critic/mlite_critic.yaml"
  ACTOR_ARGS=(
    actor_rollout_ref.actor.engine.pp="${ACTOR_PP}"
    actor_rollout_ref.actor.engine.cp="${ACTOR_CP}"
    actor_rollout_ref.actor.engine.ep="${ACTOR_EP}"
    # The actor exports its NVFP4/FP8 deployment bytes as stored.
    actor_rollout_ref.actor.engine.export_dtype=null
    +actor_rollout_ref.actor.engine.impl_cfg.bf16_master_path="${BF16_MASTER_PATH}"
    +actor_rollout_ref.actor.engine.impl_cfg.recompute=full
    "${PP_IN_FLIGHT_ARGS[@]}"
  )
  ENGINE_ARGS=("hydra.searchpath=[file://${runtime_config_root},pkg://verl_mlite.config]" model_engine=mlite)
elif [[ "${VLLM_BATCH_INVARIANT}" == 0 ]]; then
  # Weight QAT by default: the actor's forward sees the deployed NVFP4/FP8
  # weights (STE to the BF16 masters); bf16 trains on the masters as they are.
  : "${ACTOR_PRECISION:=qat}"
  : "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.45}"
  [[ "${ACTIVATION_PRECISION:-release}" == release ]] \
    || die "ACTIVATION_PRECISION applies to VLLM_BATCH_INVARIANT=1; BI=0 uses ACTOR_PRECISION"
  [[ -z "${ACTOR_PP_MAX_IN_FLIGHT:-}" ]] \
    || die "ACTOR_PP_MAX_IN_FLIGHT applies to the VLLM_BATCH_INVARIANT=1 Megatron Lite actor"
  ROLLOUT_KV_CACHE_DTYPE=fp8_e4m3
  ACTIVATION_ARGS=()
  # Packed Mamba sequences need their per-sequence seq_idx, which needs CP=1; with
  # CP>1 every micro-batch holds exactly one sequence instead (no packing).
  if [[ "${ACTOR_CP}" != 1 ]]; then
    ACTOR_DYNAMIC_BSZ=False
  fi
  ARM=_bi0
  PROJECT_NAME="${PROJECT_NAME:-verl-nemotron-h-baseline-r3}"
  ARM_ENV_NAMES=(NCCL_CUMEM_ENABLE)
  ACTOR_SUMMARY="BI=0 PRECISION=${ACTOR_PRECISION} ROUTING_REPLAY=R3"
  MODE_ARGS=(
    actor_rollout_ref.rollout.seed="${SEED}"
    # A dummy-initialized engine serves garbage after the first refit at BI=0
    # (state derived from the dummy weights survives layerwise reload); start
    # from the checkpoint instead.
    actor_rollout_ref.rollout.load_format=auto
    +actor_rollout_ref.rollout.engine_kwargs.vllm.all2all_backend=flashinfer_nvlink_one_sided
    actor_rollout_ref.actor.megatron.router_replay.mode=R3
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
  )
  case "${ACTOR_PRECISION}" in
    bf16) ;;
    qat) MODE_ARGS+=(
          actor_rollout_ref.actor.megatron.qat.enable=True
          actor_rollout_ref.actor.megatron.qat.mode=modelopt_mixed
        ) ;;
    *) die "ACTOR_PRECISION must be qat or bf16, got '${ACTOR_PRECISION}'" ;;
  esac
  # Optimizer state and FP32 masters stay on the host between updates (as DS4's
  # baseline-r3); full offload of params/grads as well exhausts host memory.
  OPTIMIZER_ARGS=(
    actor_rollout_ref.actor.megatron.param_offload=False
    actor_rollout_ref.actor.megatron.optimizer_offload=False
    +actor_rollout_ref.actor.optim.override_optimizer_config.chunked_optimizer_state_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_state_offload_chunk_size_mb=1024
    # Checkpoints hold Megatron dist-checkpoint shards; the HF safetensors export
    # of the model fails at this host-memory level.
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
  )
  ACTOR_ARGS=(
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP}"
    actor_rollout_ref.actor.megatron.context_parallel_size="${ACTOR_CP}"
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="${ACTOR_EP}"
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.seed="${SEED}"
    # Masters come from the BF16 release; MODEL_PATH (NVFP4) is the rollout's,
    # and every sync is requantized into its ModelOpt format.
    actor_rollout_ref.actor.megatron.hf_weights_path="${BF16_MASTER_PATH}"
    actor_rollout_ref.actor.megatron.mixed_precision_export=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    # No MTP block unless MTP=1 (which trains it detached); frozen expert-bias updates.
    ++actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=null
    ++actor_rollout_ref.actor.megatron.override_transformer_config.mtp_hybrid_override_pattern=null
    ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_bias_update_rate=0.0
    ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    actor_rollout_ref.model.use_fused_kernels=False
    # The entropy of the old-log-prob forward would otherwise hold a second
    # full-vocabulary copy of the logits (OOM at 64K tokens per rank).
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
  )
  ENGINE_ARGS=(model_engine=megatron)
else
  die "VLLM_BATCH_INVARIANT must be 0 or 1, got '${VLLM_BATCH_INVARIANT}'"
fi
export VLLM_BATCH_INVARIANT

# --- Optional MTP: speculative decoding in the rollout, trained head (default off) ---
# MTP=1 drafts MTP_SPEC_TOKENS tokens per rollout step with the checkpoint's MTP
# block (rejection sampling keeps the target distribution; the returned
# logprobs are the target's) and trains that block during RL, so its proposals
# follow the policy. The block sees the backbone's hidden states, the token
# embedding and the LM head detached: its loss (next-next-token cross-entropy
# on the sampled response tokens, scaled by MTP_LOSS_COEF) reaches only the
# MTP parameters, which train at MTP_LR and are clipped on their own gradient
# norm, so the policy update is unchanged. Each refit sends mtp.* to the
# drafter, whose embedding and LM head are the target's.
: "${MTP:=0}"
MTP_ARGS=()
case "${MTP}" in
  0) ;;
  1)
    : "${MTP_SPEC_TOKENS:=1}"
    : "${MTP_LR:=1e-5}"
    : "${MTP_LOSS_COEF:=0.1}"
    # The MTP block is BF16, which the NVFP4-only CuTe-DSL MoE backend rejects;
    # Triton's unquantized MoE rejects the one-sided NVLink all2all under EP.
    : "${MTP_SPEC_MOE_BACKEND:=flashinfer_cutlass}"
    SPEC_CONFIG="moe_backend:${MTP_SPEC_MOE_BACKEND}"
    # The BI=1 engine dummy-loads the target and refits it before the first
    # rollout; the drafter's block loads from the checkpoint until its own refit.
    [[ "${VLLM_BATCH_INVARIANT}" == 1 ]] && SPEC_CONFIG+=",draft_load_config:{load_format:auto}"
    MTP_ARGS=(
      actor_rollout_ref.model.mtp.enable=True
      actor_rollout_ref.model.mtp.enable_train=True
      actor_rollout_ref.model.mtp.enable_rollout=True
      actor_rollout_ref.model.mtp.method=mtp
      actor_rollout_ref.model.mtp.num_speculative_tokens="${MTP_SPEC_TOKENS}"
      actor_rollout_ref.model.mtp.mtp_loss_scaling_factor="${MTP_LOSS_COEF}"
      actor_rollout_ref.model.mtp.lr="${MTP_LR}"
      "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config={${SPEC_CONFIG}}"
    )
    # Megatron (BI=0): detach the block's inputs and tag its parameters for
    # their own norm. Megatron Lite (BI=1) always trains the block that way.
    [[ "${VLLM_BATCH_INVARIANT}" == 0 ]] \
      && MTP_ARGS+=(++actor_rollout_ref.actor.megatron.override_transformer_config.mtp_detach_heads=True)
    # Keep the checkpoint's MTP block in the actor.
    kept=()
    for arg in "${ACTOR_ARGS[@]}"; do
      [[ "${arg}" == *override_transformer_config.mtp_num_layers=null \
        || "${arg}" == *override_transformer_config.mtp_hybrid_override_pattern=null ]] || kept+=("${arg}")
    done
    ACTOR_ARGS=("${kept[@]}")
    ACTOR_SUMMARY+=" MTP=k${MTP_SPEC_TOKENS},lr${MTP_LR}"
    ;;
  *) die "MTP must be 0 or 1, got '${MTP}'" ;;
esac

# The ModelOpt checkpoint refits through vLLM's layerwise reload.
VLLM_WORKER_EXTENSION="verl_mlite.rollout.vllm_worker.MLiteVLLMColocateWorkerExtension"

# The BI=0 arm keeps its own checkpoints and logs (resume_mode=auto).
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/nemotron_h_true_on_policy/${HARDWARE}/${MODE}${ARM}}"
RUN_NAME="${RUN_NAME:-nemotron_h_${HARDWARE}_${MODE}${ARM}}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.log}"
JSONL_FILE="${JSONL_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.jsonl}"
if [[ "${TASK}" == swe_bench_pro ]]; then
  TASK_ARGS+=("++${AF}.log_dir=${OUTPUT_ROOT}/agent_logs")
fi
if (( ROLLOUT_NNODES > 0 )); then
  # Zero staleness: each step submits exactly one train batch with the weights
  # synced at the end of the previous step, waits for all of it, and drops any
  # group submitted under older weights. The colocated rollout serves until the
  # whole batch is ready (ratio 1.0, never adapted).
  SEP=trainer.v1.separate_async
  TASK_ARGS+=(
    "${SEP}.num_warmup_batches=0"
    "${SEP}.parameter_sync_step=1"
    "${SEP}.hybrid_rollout.enable_switch=True"
    "${SEP}.hybrid_rollout.switch_threshold_ratio=1.0"
    "${SEP}.hybrid_rollout.adaptive_switch_threshold=False"
    trainer.v1.sampler.max_off_policy_threshold=1
    trainer.v1.sampler.max_off_policy_strategy=drop
    actor_rollout_ref.hybrid_engine=True
    actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES}"
    actor_rollout_ref.rollout.n_gpus_per_node="${NGPUS_PER_NODE}"
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
  )
  if [[ -n "${STANDALONE_GPU_MEMORY_UTILIZATION:-}" ]]; then
    TASK_ARGS+=(
      "actor_rollout_ref.rollout.standalone_gpu_memory_utilization=${STANDALONE_GPU_MEMORY_UTILIZATION}"
    )
  fi
  # The cross-node weight sync runs NCCL inside the container, where both must be off; GB200 images
  # set them to 1, so a fallback default never applies. ROLLOUT_NCCL_MNNVL/CUMEM override.
  export NCCL_MNNVL_ENABLE="${ROLLOUT_NCCL_MNNVL:-0}" NCCL_CUMEM_ENABLE="${ROLLOUT_NCCL_CUMEM:-0}"
  [[ " ${ARM_ENV_NAMES[*]} " == *" NCCL_CUMEM_ENABLE "* ]] || ARM_ENV_NAMES+=(NCCL_CUMEM_ENABLE)
fi
if [[ "${DRY_RUN:-0}" != 1 && "${NNODES}" -gt 1 ]]; then
  : "${RAY_ADDRESS:?multi-node modes require an existing Ray cluster}"
fi

mkdir -p \
  "${OUTPUT_ROOT}" \
  "${CKPT_DIR}" \
  "$(dirname "${LOG_FILE}")" \
  "$(dirname "${JSONL_FILE}")"
export VERL_FILE_LOGGER_PATH="${JSONL_FILE}"

# --- Internal container/Ray environment; normally do not edit ---
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${SEED}"
# Cold starts (no compile caches) take longer than vLLM's 600 s default.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"

# Exporting in this launcher is not enough for an existing Ray cluster.
RAY_ENV_NAMES=(
  PYTHONNOUSERSITE PYTHONHASHSEED
  VLLM_BATCH_INVARIANT VLLM_USE_V2_MODEL_RUNNER VLLM_ENGINE_READY_TIMEOUT_S
  VERL_FILE_LOGGER_PATH
)
RAY_RUNTIME_ENV=()
for name in "${RAY_ENV_NAMES[@]}"; do
  RAY_RUNTIME_ENV+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.${name}=\"${!name}\""
  )
done

# Cluster launchers should point MLITE_DCP_LOCAL_STAGE_DIR at node-local disk.
# Sandbox timeouts reach agent runners on every node, not only the driver's.
for name in WANDB_ENTITY WANDB_MODE WANDB_BASE_URL HF_HUB_OFFLINE NCCL_MNNVL_ENABLE MLITE_DCP_LOCAL_STAGE_DIR \
  SANDBOX_STARTUP_TIMEOUT E2B_CAPACITY_WAIT E2B_EXEC_HARD_TIMEOUT "${ARM_ENV_NAMES[@]}"; do
  if [[ -v "${name}" ]]; then
    RAY_RUNTIME_ENV+=(
      "+ray_kwargs.ray_init.runtime_env.env_vars.${name}=\"${!name}\""
    )
  fi
done

if [[ "${DRY_RUN:-0}" != 1 && "${COMPOSE_ONLY:-0}" != 1 ]]; then
  [[ -s "${MODEL_PATH}/config.json" ]] || die "missing ${MODEL_PATH}/config.json"
  [[ -s "${BF16_MASTER_PATH}/config.json" ]] || die "missing ${BF16_MASTER_PATH}/config.json"
  IFS=, read -r -a train_files <<<"${TRAIN_FILES}"
  IFS=, read -r -a val_files <<<"${VAL_FILES}"
  for file in "${train_files[@]}"; do
    [[ -f "${file}" ]] || die "missing train file: ${file}"
  done
  for file in "${val_files[@]}"; do
    [[ -f "${file}" ]] || die "missing validation file: ${file}"
  done
fi

# --- Hydra overrides: only non-default behavior ---
HYDRA_ARGS=(
  # Algorithm and data.
  algorithm.adv_estimator=grpo
  algorithm.kl_ctrl.kl_coef=0.0
  algorithm.norm_adv_by_std_in_grpo=False
  data.train_files="${TRAIN_FILES}"
  data.val_files="${VAL_FILES}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.seed="${SEED}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  +data.apply_chat_template_kwargs.enable_thinking=True

  # Model and shared actor settings.
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.optim.betas="${ACTOR_BETAS:-[0.9,0.999]}"
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.optim.lr_warmup_steps=0
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz="${ACTOR_DYNAMIC_BSZ:-True}"
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.clip_ratio_c=10.0

  # vLLM rollout.
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}"
  actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}"
  actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_WORKERS}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.logprobs_mode=raw_logprobs
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
  +actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls="${VLLM_WORKER_EXTENSION}"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype="${ROLLOUT_KV_CACHE_DTYPE}"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend="${ROLLOUT_MOE_BACKEND}"
  # Reward and trainer.
  reward.reward_manager.name=dapo
  +reward.reward_kwargs.overlong_buffer_cfg.enable=True
  +reward.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN}"
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
  +reward.reward_kwargs.overlong_buffer_cfg.log=False
  +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}"
  trainer.logger="${TRAINER_LOGGERS}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${RUN_NAME}"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
  trainer.nnodes="${TRAINER_NNODES}"
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.max_actor_ckpt_to_keep=2
  trainer.default_local_dir="${CKPT_DIR}"
  trainer.val_before_train=False
)

# --- Launch ---
COMMAND=(
  python3 -m verl.trainer.main_ppo
  "${ENGINE_ARGS[@]}"
  "${HYDRA_ARGS[@]}"
  "${ACTOR_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${MODE_ARGS[@]}"
  "${MTP_ARGS[@]}"
  "${ACTIVATION_ARGS[@]}"
  "${TASK_ARGS[@]}"
  "${RAY_RUNTIME_ENV[@]}"
  "${HYDRA_OVERRIDES[@]}"
)

print_command() {
  local arg
  for arg in "${COMMAND[@]}"; do
    printf '%q ' "${arg}"
  done
  printf '\n'
}

printf 'MODE=%s HARDWARE=%s TOPOLOGY=%sx%s %s\n' \
  "${MODE}" "${HARDWARE}" "${NNODES}" "${NGPUS_PER_NODE}" "${ACTOR_SUMMARY}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  print_command
  exit 0
fi

if [[ "${COMPOSE_ONLY:-0}" == 1 ]]; then
  "${COMMAND[@]}" --cfg job --resolve
  exit 0
fi

set +e
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
run_rc="${PIPESTATUS[0]}"
set -e
exit "${run_rc}"
