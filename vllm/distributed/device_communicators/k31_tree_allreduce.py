# SPDX-License-Identifier: Apache-2.0
"""Fixed-tree all-reduce and EP-invariant MoE combine on symmetric memory."""

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_ELEMS = 8192 * 7168
MAX_ELEMS_F32 = 1 << 22
ONESHOT_MAX_ELEMS = 1 << 16
ONESHOT_MAX_BYTES = 32768
MOE_H = 3584
MOE_ONESHOT_T = 8


_EXT = None


def _ext():
    """csrc/k31/k31_tree_allreduce.cu, built with vLLM (sm_100a / sm_103a, --fmad=false)."""
    global _EXT
    if _EXT is None:
        from vllm import _k31_tree_C

        _EXT = _k31_tree_C
    return _EXT


def _align(x, a=4096):
    return (x + a - 1) // a * a


class _Inst:
    """One wire dtype; three Lamport buffers per region; sizes fixed at init (graph safe)."""

    def __init__(self, group, me, world, device, dtype, max_elems, os_max):
        import torch.distributed._symmetric_memory as symm_mem

        e = _ext()
        es = torch.tensor([], dtype=dtype).element_size()
        self.W, self.me, self.dtype, self.max = world, me, dtype, max_elems
        self.os_eb = _align(world * os_max * es)
        self.rs_eb = _align(max_elems * es)
        self.ag_eb = _align(max_elems * es)
        self.os_off, self.rs_off, self.ag_off = 0, 3 * self.os_eb, 3 * (self.os_eb + self.rs_eb)
        total = self.ag_off + 3 * self.ag_eb
        self.buf = symm_mem.empty(total, dtype=torch.uint8, device=device)
        self.hdl = symm_mem.rendezvous(self.buf, group.group_name)
        self.peers = torch.tensor([int(p) for p in self.hdl.buffer_ptrs], dtype=torch.int64)
        self.base = int(self.buf.data_ptr())
        mc = int(getattr(self.hdl, "multicast_ptr", 0) or 0)
        self.mc = mc
        e.fill_sentinel(self.base, total, dtype == torch.bfloat16)
        torch.cuda.synchronize(device)
        self.state = torch.zeros(16, dtype=torch.int64, device=device)
        self.oneshot_max_bytes = ONESHOT_MAX_BYTES
        logger.info("K31 tree all-reduce: world=%d dtype=%s max_elems=%d multicast=%s buffer=%.2f GiB",
                    world, dtype, max_elems, bool(mc), total / 2**30)

    def __call__(self, x):
        n = x.numel()
        es = x.element_size()
        out = torch.empty_like(x)
        if n == 0:
            return out
        two = not (n * es <= self.oneshot_max_bytes and self.W * n * es <= self.os_eb)
        if two and (n % (8 * self.W) != 0 or n > self.max):
            if self.W * n * es <= self.os_eb and n % 8 == 0:
                two = False
            else:
                raise RuntimeError(f"K31 tree all-reduce: unsupported size {tuple(x.shape)} (max {self.max})")
        _ext().tree_all_reduce(x.contiguous().view(-1), out.view(-1), self.peers, self.base, self.mc, self.state,
                               self.me, two, self.os_off, self.os_eb, self.rs_off, self.rs_eb,
                               self.ag_off, self.ag_eb)
        return out


class _MoeInst:
    """EP-invariant MoE combine buffers, one wire dtype per instance: bf16 = unweighted expert rows
    (leaf w*y formed by the owner), fp32 = expert-LoRA leaves w*y + d2 (formed by the pushing rank).
    bf16 latent out on every rank."""

    def __init__(self, group, me, world, device, max_t, h, os_t, f32=False):
        import torch.distributed._symmetric_memory as symm_mem

        e = _ext()
        sz = 4 if f32 else 2
        self.W, self.me, self.max_t, self.os_t, self.f32 = world, me, max_t, os_t, f32
        self.rs_eb = _align(max_t * 16 * h * sz // world)
        self.ag_eb = _align(max_t * h * 2)
        self.os_eb = _align(max(os_t, 1) * 16 * h * sz)
        self.rs_off, self.ag_off = 0, 3 * self.rs_eb
        self.os_off = self.ag_off + 3 * self.ag_eb
        total = self.os_off + 3 * self.os_eb
        self.buf = symm_mem.empty(total, dtype=torch.uint8, device=device)
        self.hdl = symm_mem.rendezvous(self.buf, group.group_name)
        self.peers = torch.tensor([int(p) for p in self.hdl.buffer_ptrs], dtype=torch.int64)
        self.base = int(self.buf.data_ptr())
        mc = int(getattr(self.hdl, "multicast_ptr", 0) or 0)
        self.mc = mc
        # rs / os regions carry the wire dtype's sentinel, the gather region is always bf16
        e.fill_sentinel(self.base, self.ag_off, not f32)
        e.fill_sentinel(self.base + self.ag_off, 3 * self.ag_eb, True)
        e.fill_sentinel(self.base + self.os_off, 3 * self.os_eb, not f32)
        torch.cuda.synchronize(device)
        self.state = torch.zeros(16, dtype=torch.int64, device=device)
        self.no_d2 = torch.empty(0, dtype=torch.bfloat16, device=device)
        logger.info("K31 tree MoE combine (%s leaves): world=%d max_tokens=%d latent=%d oneshot<=%d buffer=%.2f GiB",
                    "fp32" if f32 else "bf16", world, max_t, h, os_t, total / 2**30)

    def __call__(self, rows, e2p, ids, scales, out, d2=None):
        assert (d2 is not None) == self.f32, "the fp32-leaf instance takes the expert-LoRA delta d2"
        T = out.shape[0]
        if T > self.max_t:
            raise RuntimeError(f"K31 tree MoE combine: {T} tokens > max_t={self.max_t}")
        _ext().moe_combine(rows, self.no_d2 if d2 is None else d2, e2p, ids, scales, out, self.peers, self.base,
                           self.mc, self.state, self.me, T > self.os_t, self.rs_off, self.rs_eb, self.ag_off,
                           self.ag_eb, self.os_off, self.os_eb)
        return out


class K31TreeAllReduce:
    def __init__(self, group, me, world, device, moe_max_t):
        assert world in (2, 4, 8, 16), world
        self.inst = {
            torch.bfloat16: _Inst(group, me, world, device, torch.bfloat16, MAX_ELEMS, ONESHOT_MAX_ELEMS),
            torch.float32: _Inst(group, me, world, device, torch.float32, MAX_ELEMS_F32, ONESHOT_MAX_ELEMS),
        }
        self.ranks = dist.get_process_group_ranks(group)
        margs = (group, me, world, device, moe_max_t, MOE_H, MOE_ONESHOT_T)
        self.moe = _MoeInst(*margs)
        self.moe_f32 = _MoeInst(*margs, f32=True)
        dist.barrier(group=group)

    def all_reduce(self, x):
        inst = self.inst.get(x.dtype)
        if inst is None:
            if x.dtype.is_floating_point:
                raise RuntimeError(f"K31 tree all-reduce: dtype {x.dtype} has no fixed-tree path")
            return None  # integer: any order is exact
        return inst(x)


# ------------------------------------------------------------------ EP-invariant MoE combine hook
_MOE = {"reduced": None, "checked": False, "on": False, "owner": None}


def moe_tree_enabled():
    """True while this process holds a K31 tree (K31 BI model, a TP or EP group of size > 1)."""
    return _MOE["on"]


def activate(tree) -> None:
    """``tree`` (and the TP group's tp-ordered instance) is built: the MoE combine goes through it."""
    _MOE.update(reduced=None, checked=False, on=True, owner=tree)


def reset(tree) -> None:
    """``tree``'s communicator was destroyed: if it is the live tree, back to the no-tree state."""
    if _MOE["owner"] is tree:
        _MOE.update(reduced=None, checked=False, on=False, owner=None)


def _tp_tree():
    from vllm.distributed.parallel_state import get_ep_group, get_tp_group

    tree = getattr(get_tp_group().device_communicator, "k31_tree", None)
    if tree is None:  # K31 overlay moe-tree-dp: attention DP + EP, TP 1 -> the EP group's tree
        tree = getattr(get_ep_group().device_communicator, "k31_tree", None)
    if tree is None:
        return None
    if not _MOE["checked"]:
        ep = list(get_ep_group().ranks)
        assert ep == list(tree.ranks), f"K31 tree MoE combine needs EP group == TP group ({ep} vs {tree.ranks})"
        _MOE["checked"] = True
    return tree


def w4a16_moe_rows(
    x, token_selected_experts, token_final_scales, w1_weight, w1_weight_sf, w1_alpha, w2_weight, w2_weight_sf,
    w2_alpha, num_experts, num_local_experts, local_expert_offset, enable_pdl, activation_type, *,
    situ_beta=None, situ_linear_beta=None, tactic=None,
):
    """flashinfer's launch_w4a16_moe with use_fused_finalize=False and the default SwiGLU constants, up to its
    unpermute: workspace, sort, permute and both grouped GEMMs, launched with the same arguments. Returns the
    expert rows (GEMM2 output) and expanded_idx_to_permuted_idx; ``moe_unpermute`` of these is flashinfer's
    result bit for bit (tests/models/kimi_k3/k31_moe_lora_parity.py)."""
    from flashinfer.fused_moe.cute_dsl.blackwell import moe_w4a16 as fi
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        get_max_num_permuted_tokens,
        moe_sort,
        normalize_cute_dsl_moe_activation_type,
        validate_cute_dsl_moe_situ_config,
    )
    from flashinfer.tllm_enums import DEFAULT_SWIGLU_ALPHA, DEFAULT_SWIGLU_BETA, DEFAULT_SWIGLU_LIMIT

    top_k = int(token_selected_experts.size(1))
    intermediate_size = int(w2_weight.size(2)) * 2
    activation_type, gated = normalize_cute_dsl_moe_activation_type(activation_type)
    validate_cute_dsl_moe_situ_config(activation_type, situ_beta, situ_linear_beta)
    if int(w1_weight.size(1)) != intermediate_size * (2 if gated else 1):
        raise ValueError(f"w1_weight dim 1 does not match {activation_type.name}: {w1_weight.size(1)}")
    gemm1_tactic, gemm2_tactic = fi.DEFAULT_W4A16_MOE_TACTIC if tactic is None else tactic
    route_tile = gemm1_tactic[0][1]
    if gemm2_tactic[0][1] != route_tile:
        raise ValueError("W4A16 GEMM tactics must use the same routed-row tile")
    workspace = fi._get_workspace(x, top_k, num_experts, num_local_experts, intermediate_size, route_tile)
    (tile_idx_to_expert_idx, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx, _,
     num_non_exiting_tiles) = moe_sort(
        token_selected_experts=token_selected_experts, token_final_scales=token_final_scales,
        num_experts=num_experts, top_k=top_k, local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts, tile_tokens_dim=route_tile, enable_pdl=enable_pdl,
        **workspace.moe_sort_buffers,
    )
    num_tokens = int(x.size(0))
    route_slots = get_max_num_permuted_tokens(num_tokens, top_k, num_local_experts, route_tile)
    num_route_tiles = route_slots // route_tile
    tile_idx_to_expert_idx = tile_idx_to_expert_idx[:num_route_tiles]
    tile_idx_to_mn_limit = tile_idx_to_mn_limit[:num_route_tiles]
    expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx[:num_tokens]
    permuted_idx_to_expanded_idx = permuted_idx_to_expanded_idx[:route_slots]
    hidden_workspace = workspace.hidden_workspace[:route_slots]
    intermediate = workspace.intermediate[:route_slots]
    fi.moe_permute_w4a16(
        input=x, permuted_output=hidden_workspace, tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx, num_non_exiting_tiles=num_non_exiting_tiles,
        max_num_permuted_tokens=route_slots, top_k=top_k, tile_size=route_tile, enable_pdl=enable_pdl,
    )
    common = dict(tile_idx_to_expert_idx=tile_idx_to_expert_idx, tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                  num_non_exiting_tiles=num_non_exiting_tiles, num_local_experts=num_local_experts,
                  use_fused_finalize=False, permuted_idx_to_expanded_idx=None, token_final_scales=None,
                  enable_pdl=enable_pdl)
    fi._run_grouped_gemm(
        weight=w1_weight, weight_sf=w1_weight_sf, activations=hidden_workspace, alpha=w1_alpha,
        output=intermediate, activation_type=activation_type, swiglu_alpha=DEFAULT_SWIGLU_ALPHA,
        swiglu_beta=DEFAULT_SWIGLU_BETA, swiglu_limit=DEFAULT_SWIGLU_LIMIT, situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta, tactic=gemm1_tactic, **common,
    )
    fi._run_grouped_gemm(
        weight=w2_weight, weight_sf=w2_weight_sf, activations=intermediate, alpha=w2_alpha,
        output=hidden_workspace, activation_type=None, swiglu_alpha=DEFAULT_SWIGLU_ALPHA,
        swiglu_beta=DEFAULT_SWIGLU_BETA, swiglu_limit=DEFAULT_SWIGLU_LIMIT, situ_beta=None,
        situ_linear_beta=None, tactic=gemm2_tactic, **common,
    )
    return hidden_workspace, expanded_idx_to_permuted_idx


def launch_w4a16_moe_tree(
    x, token_selected_experts, token_final_scales, w1_weight, w1_weight_sf, w1_alpha, w2_weight, w2_weight_sf,
    w2_alpha, num_experts, num_local_experts, local_expert_offset, moe_output, use_fused_finalize, enable_pdl,
    activation_type, *, situ_beta=None, situ_linear_beta=None, tactic=None, tree_topk_ids,
):
    """flashinfer's launch_w4a16_moe whose last step, the unpermute (k-order sum of each token's rows), is the
    canonical-slot tree combine."""
    if _MOE["reduced"] is not None:
        raise RuntimeError(f"K31 tree MoE combine: previous call was not consumed (reduced={_MOE['reduced']})")
    tree = _tp_tree()
    if tree is None:
        raise RuntimeError("K31 tree MoE combine: no fixed tree on the TP or EP group")
    if use_fused_finalize:
        raise ValueError("K31 tree MoE combine replaces the unpermute; fused finalize has none")
    rows, e2p = w4a16_moe_rows(
        x, token_selected_experts, token_final_scales, w1_weight, w1_weight_sf, w1_alpha, w2_weight, w2_weight_sf,
        w2_alpha, num_experts, num_local_experts, local_expert_offset, enable_pdl, activation_type,
        situ_beta=situ_beta, situ_linear_beta=situ_linear_beta, tactic=tactic,
    )
    _moe_tree_unpermute(tree, tree_topk_ids, rows, moe_output, e2p, token_final_scales)
    return moe_output


def _moe_tree_unpermute(tree, topk_ids, rows, output, e2p, scales):
    # one kernel, no host-side ops: canonical slot positions and weights are computed in-kernel from the ids
    T, K = e2p.shape
    assert K == 16 and tuple(topk_ids.shape) == (T, K) and output.shape[0] == T, (e2p.shape, topk_ids.shape)
    out = output if output.is_contiguous() else torch.empty_like(output)
    tree.moe(rows.contiguous(), e2p.contiguous(), topk_ids.contiguous(), scales.float().contiguous(), out)
    if out is not output:
        output.copy_(out)
    _MOE["reduced"] = T


def moe_tree_lora(o2, e2p32, ww, d2, topk_ids, output):
    """Expert-LoRA path (K31FiW4A16Experts._k31_apply_lora): the leaf of slot (t, k) is
    fp32(w[t, k] * o2[e2p[t, k]]) + fp32(d2[t, k]) (k31_moe_lora_fuse slot_kernel), formed by the rank
    holding the expert while it pushes the leaf into the same canonical-slot tree as the no-LoRA path.
    Replaces the local k-order sum. Returns False (caller keeps its own sum) when the tree is off."""
    if not moe_tree_enabled():
        return False
    tree = _tp_tree()
    if tree is None:
        raise RuntimeError("K31 tree MoE combine: no fixed tree on the TP or EP group")
    if _MOE["reduced"] is not None:
        raise RuntimeError("K31 tree MoE combine: previous call was not consumed")
    T, K = topk_ids.shape
    assert K == 16 and e2p32.numel() == T * K and tuple(d2.shape[:2]) == (T, K), (e2p32.shape, d2.shape)
    out = output if output.is_contiguous() else torch.empty_like(output)
    tree.moe_f32(o2.contiguous(), e2p32.view(T, K), topk_ids.contiguous(), ww.float().contiguous(), out,
                 d2=d2.contiguous())
    if out is not output:
        output.copy_(out)
    _MOE["reduced"] = T
    return True


def moe_tree_chunk_done(check=True):
    """A row chunk of one MoE call reached the tree combine: let the next chunk's combine run (check=False: a
    chunk raised, drop whatever it left)."""
    if moe_tree_enabled():
        if check and _MOE["reduced"] is None:
            raise RuntimeError("K31 tree MoE combine: a row chunk did not reach the canonical-slot tree combine")
        _MOE["reduced"] = None


def moe_tree_chunks_reduced(rows):
    """Every row chunk of one MoE call (``rows`` rows in all) went through the tree combine."""
    if moe_tree_enabled():
        _MOE["reduced"] = rows


def k31_dp_local_rows(hidden_states, is_sequence_parallel):
    """K31 overlay moe-tree-dp, MoERunner._maybe_combine: when the canonical-slot tree already reduced the
    all-gathered batch (every rank holds every DP rank's rows), return this DP rank's rows (the same sizes and
    group the allgather_reducescatter dispatch used); None -> caller keeps get_ep_group().combine."""
    r = _MOE["reduced"]
    if r is None:
        if moe_tree_enabled():
            raise RuntimeError("K31 tree MoE combine: this MoE call did not reach the canonical-slot tree combine")
        return None
    from vllm.distributed.parallel_state import get_dp_group
    from vllm.forward_context import get_forward_context

    assert not is_sequence_parallel, "K31 tree DP combine: sequence-parallel MoE input not supported"
    assert r == hidden_states.shape[0], (r, tuple(hidden_states.shape))
    sizes = get_forward_context().dp_metadata.get_chunk_sizes_across_dp_rank()
    dp = get_dp_group()
    assert sizes is not None and len(sizes) == dp.world_size and sum(sizes) == r, (sizes, r)
    off = sum(sizes[: dp.rank_in_group])
    n = sizes[dp.rank_in_group]
    _MOE["reduced"] = n
    return hidden_states[off : off + n]


def take_moe_reduced(num_tokens):
    """latent tail: True when the routed output of this layer is already the EP-invariant tree sum."""
    r = _MOE["reduced"]
    _MOE["reduced"] = None
    if r is None:
        if moe_tree_enabled():
            raise RuntimeError("K31 tree MoE combine: this MoE call did not reach the canonical-slot tree combine")
        return False
    assert r == num_tokens, (r, num_tokens)
    return True
