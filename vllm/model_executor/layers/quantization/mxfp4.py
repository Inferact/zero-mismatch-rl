# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    TRITON_BACKENDS,
    Mxfp4MoeBackend,
    backend_to_kernel_cls,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    convert_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
    mxfp4_round_up_hidden_size_and_intermediate_size,
    select_deepseek_v4_mxfp4_moe_backend,
    select_mxfp4_moe_backend,
    triton_weight,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    is_layer_skipped,
)
from vllm.model_executor.utils import (
    is_weights_pre_processed,
    replace_parameter,
    set_weight_attrs,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

# ---------------------------------------------------------------- K31 overlay moe-fi-w4a16-bi
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts


def _k31_fi_moe(moe, vllm_config) -> bool:
    """The K31 BI MoE (SiTU MXFP4 experts of a Kimi-K3.1 checkpoint on sm_10x under VLLM_BATCH_INVARIANT) runs FI
    W4A16 with the CUDA expert LoRA and the canonical-slot combine (moe tree). Without BI the upstream MXFP4
    oracle picks the experts (moe_backend)."""
    from vllm.distributed.parallel_state import k31_bi_model

    cap = current_platform.get_device_capability() if current_platform.is_cuda() else None
    return moe.activation.value == "situ" and k31_bi_model(vllm_config) and cap is not None and cap.major == 10


def _k31_lora_cuda_ok(ctx, M, hdim, inter) -> bool:
    """Layout half of the CUDA expert-LoRA choice. Only K31FiW4A16Experts calls it, and those experts exist only
    where _k31_fi_moe holds (K31 checkpoint on sm_10x), which is the model/arch half; any other layout keeps
    punica's Triton kernel."""
    if ctx.max_loras != 1 or not ctx.enable_moe_shared_loras or ctx.fully_sharded:
        return False
    a, b = ctx.w13_lora_a_stacked, ctx.w13_lora_b_stacked
    return (len(a) == 2 and len(b) == 2 and len(ctx.w2_lora_a_stacked) == 1 and a[0].shape[1] == 1
            and a[0].shape[2] == 32 and ctx.w2_lora_b_stacked[0].shape[1] == 1 and hdim % 512 == 0
            and inter % 512 == 0 and hdim % 128 == 0 and inter % 128 == 0)


# Every tactic gives bitwise the same rows (K tile 256, full K per CTA); this only picks speed.
_K31_FI_TACTICS = {
    "default": (((256, 128, 256), (2, 1), True), ((256, 128, 256), (2, 1), True)),
    "r64": (((128, 64, 256), (2, 1), True), ((128, 64, 256), (2, 1), True)),
    "r16": (((128, 16, 256), (1, 1), True), ((128, 16, 256), (1, 1), True)),
}


def _k31_pick_tactic(m: int, topk: int, n_glob: int):
    """by mean routed rows per expert (CUDA-graph timings); host-side only."""
    rows = m * topk / n_glob
    return _K31_FI_TACTICS["r16" if rows <= 12 else "r64" if rows <= 48 else "default"]


def _k31_e8m0_to_e4m3_pow2(sc: torch.Tensor, name: str):
    """[E, N, K/32] E8M0 bytes -> ([E, N, K/16] E4M3 bytes, [E] fp32 alpha). Exact when each expert's
    scales span <= 17 binades (E4M3 holds 2^-9 .. 2^8 exactly); asserted."""
    p = sc.to(torch.int16) - 127
    pmax = p.flatten(1).amax(1)
    c = pmax - 8
    q = p - c[:, None, None]
    lossy = int((q < -9).sum())
    if lossy:
        raise RuntimeError(
            f"K31 moe-fi-w4a16-bi: {name} has {lossy} scale blocks below the E4M3 range of their expert"
        )
    normal = q >= -6
    byte = torch.where(
        normal, (q + 7) * 8, torch.bitwise_left_shift(torch.ones_like(q), (q + 9).clamp(0, 2))
    ).to(torch.uint8)
    dec = byte.view(torch.float8_e4m3fn).float()
    assert torch.equal(dec, torch.exp2(q.float())), f"K31 e4m3 encode mismatch {name}"
    return byte.repeat_interleave(2, -1), torch.exp2(c.float()).contiguous()


def _k31_swizzle_mma(lin: torch.Tensor) -> torch.Tensor:
    """[E, M, Kc] -> physical (E, M/128, Kc/4, 32, 4, 4), the layout of flashinfer's convert_sf_to_mma_layout."""
    E, M, Kc = lin.shape
    assert M % 128 == 0 and Kc % 4 == 0, (M, Kc)
    return lin.view(E, M // 128, 4, 32, Kc // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous()


def _k31_prepare_fi_w4a16(layer) -> None:
    from flashinfer.fused_moe.prepare import _interleave_linear_and_gate

    w13, w2 = layer.w13_weight.data, layer.w2_weight.data
    s13, s2 = layer.w13_weight_scale.data, layer.w2_weight_scale.data
    E, n2, kh = w13.shape
    inter = n2 // 2
    assert n2 % 128 == 0 and (kh * 2) % 256 == 0 and w2.shape[2] * 2 % 256 == 0, (w13.shape, w2.shape)
    b13, a13 = _k31_e8m0_to_e4m3_pow2(s13, "w13")
    b2, a2 = _k31_e8m0_to_e4m3_pow2(s2, "w2")
    # checkpoint w13 rows are [gate (w1); up (w3)]; the kernel wants [up; gate] interleaved per 64 rows
    w13o = torch.cat([w13[:, inter:], w13[:, :inter]], 1)
    b13o = torch.cat([b13[:, inter:], b13[:, :inter]], 1)
    del b13
    w13i = _interleave_linear_and_gate(w13o, group_size=64, dim=1).contiguous()
    del w13o
    sf13 = _k31_swizzle_mma(_interleave_linear_and_gate(b13o, group_size=64, dim=1).contiguous())
    del b13o
    sf2 = _k31_swizzle_mma(b2.contiguous())
    replace_parameter(layer, "w13_weight", w13i)
    replace_parameter(layer, "w13_weight_scale", sf13)
    replace_parameter(layer, "w2_weight_scale", sf2)
    layer.k31_w1_alpha = a13
    layer.k31_w2_alpha = a2


def _k31_fmlora_shape(m: int) -> tuple[int, int]:
    """fmlora-policy overlay: (BLOCK_SIZE_M, NPID) of punica's one-shot fused_moe_lora by gathered tokens M;
    NPID 0 = image launcher heuristic. Launch shape only (bits equal for every shape)."""
    if m <= 32:
        return 16, 6
    if m <= 128:
        return 16, 3
    if m <= 512:
        return 16, 4
    if m <= 2048:
        return 16, 3
    return 64, 0


@functools.cache
def _k31_moe_chunk() -> int:
    """Rows per FI W4A16 call (VLLM_K31_MOE_CHUNK, default 8192: the row range the K31 pipeline was measured
    bitwise batch invariant on), read once per process. Above the tree combine's one-shot size, so consecutive
    chunks never run one-shot combines (which push from the owning ranks only) back to back."""
    from vllm.distributed.device_communicators.k31_tree_allreduce import MOE_ONESHOT_T

    n = int(os.environ.get("VLLM_K31_MOE_CHUNK", "8192"))
    if n <= MOE_ONESHOT_T:
        raise ValueError(f"VLLM_K31_MOE_CHUNK must be > {MOE_ONESHOT_T}, got {n}")
    return n


class K31FiW4A16Experts(MarlinExperts):
    """FlashInfer CuTe-DSL sm100 W4A16 experts on the rewritten MXFP4 weights; output already combined."""

    @staticmethod
    def _supports_batch_invariance() -> bool:
        return True

    def process_weights_after_loading(self, layer) -> None:
        self._k31_w1a = layer.k31_w1_alpha
        self._k31_w2a = layer.k31_w2_alpha
        e_loc = layer.w13_weight.shape[0]
        em = getattr(layer, "expert_map", None)
        if em is None:
            self._k31_off, self._k31_nglob = 0, e_loc
        else:
            loc = (em >= 0).nonzero().flatten()
            off = int(loc[0])
            ar = torch.arange(off, off + e_loc, device=em.device)
            assert loc.numel() == e_loc and torch.equal(loc, ar), "K31 FI W4A16 needs contiguous local experts"
            assert torch.equal(em[loc].long(), torch.arange(e_loc, device=em.device)), "expert_map order"
            self._k31_off, self._k31_nglob = off, em.numel()
        beta = self.moe_config.activation_situ_beta
        lbeta = self.moe_config.activation_situ_linear_beta
        self._k31_situ = (None if beta is None else float(beta), None if lbeta is None else float(lbeta))
        print(f"K31 FI W4A16 experts: local={e_loc} offset={self._k31_off} global={self._k31_nglob} "
              f"situ={self._k31_situ}", flush=True)

    def workspace_shapes(self, M, N, K, topk, global_num_experts, local_num_experts, expert_tokens_meta,
                         activation):
        return ((M, K), (M, K), (M, K))

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
              expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        assert not apply_router_weight_on_input
        assert activation.value in ("situ", "silu"), activation
        situ_beta, situ_lbeta = self._k31_situ if activation.value == "situ" else (None, None)
        from vllm.distributed.device_communicators import k31_tree_allreduce as _k31_t

        # The DP all-gathered batch (every DP rank's rows) runs in row chunks of at most VLLM_K31_MOE_CHUNK:
        # every step is row-local, so the bits do not change, and the FI workspace and LoRA buffers scale with
        # the chunk instead of DP x max_num_batched_tokens. Every EP rank holds the same rows, so the chunks
        # (and the tree combine calls) line up across ranks.
        M = hidden_states.shape[0]
        C = _k31_moe_chunk()
        if M > C:
            tlm = self._k31_token_lora_mapping(M) if self._lora_context is not None else None
            try:
                for s in range(0, M, C):
                    e = min(M, s + C)
                    self._k31_apply_rows(output[s:e], hidden_states[s:e], w1, w2, topk_weights[s:e], topk_ids[s:e],
                                         expert_map, situ_beta, situ_lbeta, None if tlm is None else tlm[s:e])
                    _k31_t.moe_tree_chunk_done()
            except BaseException:
                _k31_t.moe_tree_chunk_done(check=False)
                raise
            _k31_t.moe_tree_chunks_reduced(M)
            return
        self._k31_apply_rows(output, hidden_states, w1, w2, topk_weights, topk_ids, expert_map, situ_beta,
                             situ_lbeta, None)

    def _k31_apply_rows(self, output, hidden_states, w1, w2, topk_weights, topk_ids, expert_map, situ_beta,
                        situ_lbeta, tlm):
        from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import launch_w4a16_moe
        from flashinfer.tllm_enums import ActivationType

        if self._lora_context is not None:
            self._k31_apply_lora(output, hidden_states, w1, w2, topk_weights, topk_ids, expert_map,
                                 situ_beta, situ_lbeta, tlm=tlm)
            return
        # K31 overlay tp-invariant: EP-invariant canonical-slot combine (no-op unless the K31 tree exists)
        from vllm.distributed.device_communicators import k31_tree_allreduce as _k31_t

        launch = (
            (lambda *a, **k: _k31_t.launch_w4a16_moe_tree(*a, tree_topk_ids=topk_ids, **k))
            if _k31_t.moe_tree_enabled()
            else launch_w4a16_moe
        )
        launch(
            hidden_states, topk_ids.to(torch.int32), topk_weights.to(torch.float32),
            w1, self.w1_scale, self._k31_w1a, w2, self.w2_scale, self._k31_w2a,
            num_experts=self._k31_nglob, num_local_experts=w1.size(0), local_expert_offset=self._k31_off,
            moe_output=output, use_fused_finalize=False, enable_pdl=False,
            activation_type=ActivationType.Swiglu, situ_beta=situ_beta, situ_linear_beta=situ_lbeta,
            tactic=_k31_pick_tactic(hidden_states.shape[0], topk_ids.shape[1], self._k31_nglob),
        )

    def _k31_token_lora_mapping(self, m: int) -> torch.Tensor:
        ctx = self._lora_context
        if ctx.local_token_lora_mapping is not None:
            return ctx.local_token_lora_mapping[:m]
        pw = ctx.punica_wrapper
        return pw.token_mapping_meta.meta_args(m, pw.lora_config.specialize_active_lora)[0][:m]

    @staticmethod
    def _k31_lora_cfgs(rank: int, m: int):
        """Pinned Triton configs for fused_moe_lora (the VLLM_BATCH_INVARIANT=1 defaults of the image after
        its LoRA post-processing), used for every M and both BI settings: SPLIT_K 1 (no atomics), fixed
        BLOCK_K, so a row's sums never depend on the batch."""
        r2 = 1 << max(rank - 1, 0).bit_length()
        base = {"BLOCK_SIZE_M": _k31_fmlora_shape(m)[0], "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8,
                "NUM_WARPS": 4, "NUM_STAGES": 3, "SPLIT_K": 1}
        shrink = dict(base, BLOCK_SIZE_N=min(64, r2))
        expand = dict(base, BLOCK_SIZE_K=max(16, min(32, r2)))
        return shrink, expand

    def _k31_lora_w13(self, ctx, y, x, topk_ids, topk_weights, expert_map, M, topk, tlm):
        """= LoRAExpertsMixin.apply_w13_lora / PunicaWrapperGPU.add_lora_w13 with pinned configs and no
        M-dependent naive block assignment."""
        pw = ctx.punica_wrapper
        a = ctx.w13_lora_a_stacked
        b = ctx.w13_lora_b_stacked
        if ctx.enable_moe_shared_loras:
            a = tuple(t.expand(-1, ctx.local_num_experts, -1, -1) for t in a)
        rank = a[0].shape[-2]
        shrink, expand = self._k31_lora_cfgs(rank, M)
        tlm, sorted_ids, eids, npad = pw.moe_lora_align_block_size(
            topk_ids, M, shrink["BLOCK_SIZE_M"], ctx.local_num_experts, ctx.max_loras, ctx.adapter_enabled,
            expert_map, naive_block_assignment=False, token_lora_mapping=tlm)
        pw.add_lora_fused_moe(
            y.view(-1, topk, y.shape[-1]), x, a, b, topk_weights, sorted_ids.view(ctx.max_loras, -1),
            eids.view(ctx.max_loras, -1), npad, rank, ctx.top_k, shrink, expand, ctx.adapter_enabled,
            fully_sharded=ctx.fully_sharded, token_lora_mapping=tlm, add_inputs=True)
        return sorted_ids, eids, npad, tlm

    def _k31_lora_w2(self, ctx, y, x, topk_weights, sorted_ids, eids, npad, tlm, topk):
        pw = ctx.punica_wrapper
        a = ctx.w2_lora_a_stacked
        b = ctx.w2_lora_b_stacked
        if ctx.enable_moe_shared_loras:
            b = tuple(t.expand(-1, ctx.local_num_experts, -1, -1) for t in b)
        rank = a[0].shape[-2]
        shrink, expand = self._k31_lora_cfgs(rank, topk_weights.shape[0])
        shard = b[0].shape[-2]
        offset = shard * ctx.tp_rank if ctx.fully_sharded else 0
        pw.add_lora_fused_moe(
            y, x, a, b, topk_weights, sorted_ids.view(ctx.max_loras, -1), eids.view(ctx.max_loras, -1), npad,
            rank, ctx.top_k, shrink, expand, ctx.adapter_enabled, mul_routed_weight=True,
            fully_sharded=ctx.fully_sharded, offset=offset, token_lora_mapping=tlm, add_inputs=True)

    def _k31_apply_lora(self, output, hidden_states, w1, w2, topk_weights, topk_ids, expert_map, situ_beta,
                        situ_lbeta, tlm=None):
        """Expert LoRA on the FI path, every step row-local and in a fixed order (same pipeline measured
        bitwise BI for M = 1..8192 with a non-zero adapter):
        FI GEMM1 without fused activation -> flat (token, k) rows [gate | up] -> + w13 LoRA delta (punica
        fused_moe_lora into BF16 zeros with pinned configs) added in FP32 -> SiTU
        in FP32 -> BF16 -> FI GEMM2 -> per slot FP32(w_k * base_k) + FP32(w_k * w2 delta_k) -> sum over k in
        k order in FP32 -> BF16 once. Graph-capturable (static shapes, no host sync besides punica's own)."""
        import flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 as F
        from flashinfer.tllm_enums import DEFAULT_SWIGLU_ALPHA, DEFAULT_SWIGLU_BETA, DEFAULT_SWIGLU_LIMIT

        assert situ_beta is not None, "K31 FI LoRA path implements SiTU only"
        ctx = self._lora_context
        M, topk = topk_ids.shape
        hdim = hidden_states.shape[1]
        e_loc = w1.size(0)
        inter = w1.size(1) // 2
        dev = hidden_states.device
        ids32 = topk_ids.to(torch.int32)
        ww = topk_weights.to(torch.float32)
        tactic = _k31_pick_tactic(M, topk, self._k31_nglob)
        rt = tactic[0][0][1]
        ws = F._get_workspace(hidden_states, topk, self._k31_nglob, e_loc, inter, rt)
        t2e, t2l, e2p, p2e, _, nnt = F.moe_sort(
            token_selected_experts=ids32, token_final_scales=ww, num_experts=self._k31_nglob, top_k=topk,
            local_expert_offset=self._k31_off, num_local_experts=e_loc, tile_tokens_dim=rt, enable_pdl=False,
            **ws.moe_sort_buffers)
        S = F.get_max_num_permuted_tokens(M, topk, e_loc, rt)
        t2e, t2l, e2p, p2e = t2e[: S // rt], t2l[: S // rt], e2p[:M], p2e[:S]
        hid = ws.hidden_workspace[:S]
        F.moe_permute_w4a16(input=hidden_states, permuted_output=hid, tile_idx_to_mn_limit=t2l,
                      permuted_idx_to_expanded_idx=p2e, num_non_exiting_tiles=nnt, max_num_permuted_tokens=S,
                      top_k=topk, tile_size=rt, enable_pdl=False)
        common = dict(tile_idx_to_expert_idx=t2e, tile_idx_to_mn_limit=t2l, num_non_exiting_tiles=nnt,
                      num_local_experts=e_loc, activation_type=None, swiglu_alpha=DEFAULT_SWIGLU_ALPHA,
                      swiglu_beta=DEFAULT_SWIGLU_BETA, swiglu_limit=DEFAULT_SWIGLU_LIMIT, situ_beta=None,
                      situ_linear_beta=None, use_fused_finalize=False, permuted_idx_to_expanded_idx=None,
                      token_final_scales=None, enable_pdl=False)
        g1 = torch.empty(S, 2 * inter, device=dev, dtype=torch.bfloat16)
        F._run_grouped_gemm(weight=w1, weight_sf=self.w1_scale, activations=hid, alpha=self._k31_w1a,
                            output=g1, tactic=tactic[0], **common)
        # moe-lora-fuse overlay: elementwise glue as row-local CUDA kernels, bitwise equal to the equivalent torch ops
        import vllm.distributed.device_communicators.k31_tree_allreduce as _k31_t
        from vllm.model_executor.layers.quantization import k31_moe_lora_fuse as _lf

        e2p32 = _lf.i32(e2p)
        ly = torch.zeros(M * topk, 2 * inter, device=dev, dtype=torch.bfloat16)
        if tlm is None:
            tlm = self._k31_token_lora_mapping(M)
        cu = _k31_lora_cuda_ok(ctx, M, hdim, inter)
        if cu:
            # fmlora-cuda overlay: same bits as punica's fused_moe_lora, rows from FI's moe_sort tiles
            from vllm.model_executor.layers.quantization import k31_moe_lora_mma as _lm

            p2e32 = _lf.i32(p2e)
            # tokens take slot 0 only while it is in active_lora_ids (same read as k31_lin_lora.apply)
            _pw = ctx.punica_wrapper
            _ma = _pw.token_mapping_meta.meta_args(M, _pw.lora_config.specialize_active_lora)
            lids, nlids = _ma[4], min(int(_ma[6].item()), _ma[4].numel())
            _lm.w13(ly, hidden_states, ctx.w13_lora_a_stacked[0][0, 0], ctx.w13_lora_a_stacked[1][0, 0],
                    ctx.w13_lora_b_stacked[0][0], ctx.w13_lora_b_stacked[1][0], t2e, t2l, nnt, p2e32, rt, topk,
                    tlm, ctx.adapter_enabled, lids, nlids)
        else:
            sorted_l, eids_l, npad_l, tlm_l = self._k31_lora_w13(
                ctx, ly, hidden_states, topk_ids, topk_weights, expert_map, M, topk, tlm)
        x32 = _lf.gather_add(g1, e2p32, ly, inter)
        del ly
        y32 = torch.empty(M * topk, inter, device=dev, dtype=torch.float32)
        torch.ops._C.situ_and_mul(y32, x32, situ_beta, -1.0 if situ_lbeta is None else situ_lbeta)
        del x32
        act, act_s = _lf.act_cast(y32, _lf.i32(p2e))
        del y32
        o2 = torch.empty(S, hdim, device=dev, dtype=torch.bfloat16)
        F._run_grouped_gemm(weight=w2, weight_sf=self.w2_scale, activations=act_s, alpha=self._k31_w2a,
                            output=o2, tactic=tactic[1], **common)
        d2 = torch.zeros(M, topk, hdim, device=dev, dtype=torch.bfloat16)
        if cu:
            _lm.w2(d2, act_s, ctx.w2_lora_a_stacked[0][0], ctx.w2_lora_b_stacked[0][0, 0], ww.reshape(-1),
                   t2e, t2l, nnt, p2e32, rt, topk, tlm, ctx.adapter_enabled, lids, nlids)
        else:
            self._k31_lora_w2(ctx, d2, act, topk_weights, sorted_l, eids_l, npad_l, tlm_l, topk)
        if _k31_t.moe_tree_lora(o2, e2p32, ww, d2, topk_ids, output):
            return
        if output.dtype != torch.bfloat16 or not output.is_contiguous():
            slot = _lf.slot(o2, e2p32, ww, d2)
            acc = slot[:, 0].clone()
            for k in range(1, topk):
                acc.add_(slot[:, k])
            output.copy_(acc.to(output.dtype))
            return
        _lf.slot_sum(o2, e2p32, ww, d2, output)


# ---------------------------------------------------------------- end K31 overlay


class Mxfp4Config(QuantizationConfig):
    """Canonical base config for MXFP4 quantization.

    Subclasses override get_name() and override_quantization_method() to
    register themselves as the handler for a specific checkpoint format.
    """

    def __init__(self, ignored_layers: list[str] | None = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def _make_moe_method(self, moe: FusedMoEConfig) -> FusedMoEMethodBase:
        """MoE method for RoutedExperts. Subclasses override to pick a
        checkpoint-specific kernel family."""
        return Mxfp4MoEMethod(moe)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            logger.debug_once(
                "MXFP4 linear layer is not implemented - falling back to "
                "UnquantizedLinearMethod.",
            )
            return UnquantizedLinearMethod()
        elif isinstance(layer, RoutedExperts):
            return self._make_moe_method(layer.moe_config)
        elif isinstance(layer, Attention):
            logger.debug_once(
                "MXFP4 attention layer is not implemented. "
                "Skipping quantization for this layer.",
            )
        return None


class GptOssMxfp4Config(Mxfp4Config):
    """MXFP4 config for GPT-OSS checkpoints.

    Checkpoints carry ``"quant_method": "mxfp4"`` in their JSON config.
    override_quantization_method() maps that to the canonical internal name
    so that the rest of the loading path uses "gpt_oss_mxfp4" consistently.
    """

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "gpt_oss_mxfp4"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        # Match both "mxfp4" (original checkpoint value) and "gpt_oss_mxfp4"
        # (already normalized by verify_and_update_model_config) so that
        # explicit --quantization mxfp4 from the user doesn't cause a mismatch.
        if not (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") in ("mxfp4", "gpt_oss_mxfp4")
        ):
            return None
        # Require explicit confirmation that this is a GPT-OSS model.
        # Do NOT fall back to returning the override when hf_config is None,
        # as that would silently claim all mxfp4 checkpoints.
        model_type = getattr(hf_config, "model_type", None)
        if model_type != "gpt_oss":
            return None
        return "gpt_oss_mxfp4"

    def _make_moe_method(self, moe: FusedMoEConfig) -> FusedMoEMethodBase:
        return GptOssMxfp4MoEMethod(moe)


class GptOssMxfp4MoEMethod(FusedMoEMethodBase):
    """MXFP4 MoE quantization method."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.weight_dtype = "gpt_oss_mxfp4"
        self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)

        self.max_capture_size = moe.max_capture_size

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs
        self.w13_precision_config = None
        self.w2_precision_config = None

    @property
    def skip_forward_padding(self) -> bool:
        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
        # so can skip the padding in the forward before applying the moe method
        return self.mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8

    # TODO(bnell): move to MK/expert_class?
    @property
    def has_unpadded_output(self) -> bool:
        return self.mxfp4_backend in [
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        ]

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        return mxfp4_round_up_hidden_size_and_intermediate_size(
            self.mxfp4_backend, hidden_size, intermediate_size_per_partition
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        mxfp4_block = 32

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w13_weight_scale.quant_method = "block"

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        w2_weight_scale.quant_method = "block"

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> None:
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size
        hidden_size = self.hidden_size
        sf_block_size = 32

        # Shape assertions
        assert (
            w13.dim() == 3
            and w13.shape[0] == num_experts
            and w13.shape[1] == intermediate_size * self.moe.w13_num_shards
            and w13.shape[2] == hidden_size // 2
        )
        assert (
            w13_scale.dim() == 3
            and w13_scale.shape[0] == num_experts
            and w13_scale.shape[1] == intermediate_size * self.moe.w13_num_shards
            and w13_scale.shape[2] == hidden_size // sf_block_size
        )
        assert (
            w2.dim() == 3
            and w2.shape[0] == num_experts
            and w2.shape[1] == hidden_size
            and w2.shape[2] == intermediate_size // 2
        )
        assert (
            w2_scale.dim() == 3
            and w2_scale.shape[1] == hidden_size
            and w2_scale.shape[2] == intermediate_size // sf_block_size
        )
        if w13_bias is not None:
            assert (
                w13_bias.dim() == 2
                and w13_bias.shape[0] == num_experts
                and w13_bias.shape[1] == intermediate_size * self.moe.w13_num_shards
            )
        if w2_bias is not None:
            assert (
                w2_bias.dim() == 2
                and w2_bias.shape[0] == num_experts
                and w2_bias.shape[1] == hidden_size
            )

        # Convert weights to kernel format
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=w13,
                w2_weight=w2,
                w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                _cache_permute_indices=self._cache_permute_indices,
            )
        )

        # For TRITON backends, weights are wrapped tensors from triton_kernels
        # that don't support .detach(). Manually assign parameters.
        if self.mxfp4_backend not in TRITON_BACKENDS:
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
        else:
            layer.w13_weight_triton = w13
            layer.w2_weight_triton = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        # Build quant config
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        # Build kernel (modular or monolithic)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )
            self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        w1_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend in TRITON_BACKENDS:
            # TRITON backends free w13/w2_weight_scale after swizzling; the
            # swizzled scales live inside the precision configs instead.
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config
        else:
            w1_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            swiglu_limit=7.0,
            layer=layer,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=triton_weight(layer, "w13_weight"),
            w2=triton_weight(layer, "w2_weight"),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=triton_weight(layer, "w13_weight"),
            w2=triton_weight(layer, "w2_weight"),
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
            routing_sink=layer.routing_sink,
            input_ids=input_ids,
            hash_indices_table=layer.hash_indices_table,
        )


def _use_k3_situ_aiter(moe: FusedMoEConfig) -> bool:
    """Route Kimi-K3 weight-only MXFP4 SiTU MoE to AITER A16W4 on gfx950."""
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.platforms.rocm import on_gfx950

    return (
        rocm_aiter_ops.is_fused_moe_enabled()
        and on_gfx950()
        and moe.activation == MoEActivation.SITU
        and moe.activation_situ_linear_beta is not None
        and rocm_aiter_ops.get_aiter_activation_type("situ") is not None
    )


def _use_k3_situ_int4_gfx942(moe: FusedMoEConfig) -> bool:
    """Opt-in gfx942 MXFP4-to-int4 requant for Kimi-K3 SiTU experts."""
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.platforms.rocm import on_gfx942

    return (
        rocm_aiter_ops.is_fused_moe_enabled()
        and on_gfx942()
        and moe.activation == MoEActivation.SITU
        and moe.activation_situ_linear_beta is not None
        and _moe_weight_override_is_int4()
    )


def _moe_weight_override_is_int4() -> bool:
    """True when --quantization-config.moe.weight int4_per_group_32 was set."""
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    if vllm_config is None:
        return False
    quant_args = getattr(vllm_config.model_config, "quantization_config", None)
    moe_spec = getattr(quant_args, "moe", None)
    weight = getattr(moe_spec, "weight", None)
    if weight is None:
        return False
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt4Static32,
    )

    return weight == kInt4Static32


class Mxfp4MoEMethod(FusedMoEMethodBase):
    """MXFP4 MoE quantization method."""

    supports_pre_processed_weights = True

    def __init__(self, moe: FusedMoEConfig, activation_key: QuantKey | None = None):
        super().__init__(moe)
        self.weight_dtype = "mxfp4"
        # Checkpoint-declared activation quantization. Only backends that pick
        # their activation format from the layer (Humming) consume it.
        self.activation_key = activation_key
        self.is_k3_situ_aiter = _use_k3_situ_aiter(moe)
        self.is_k3_situ_int4_gfx942 = _use_k3_situ_int4_gfx942(moe)
        self.experts_cls: type[mk.FusedMoEExperts] | None
        if self.is_k3_situ_aiter or self.is_k3_situ_int4_gfx942:
            self.mxfp4_backend = Mxfp4MoeBackend.AITER_MXFP4_BF16
            self.experts_cls = backend_to_kernel_cls(self.mxfp4_backend)[0]
            if self.is_k3_situ_int4_gfx942:
                logger.warning_once(
                    "Requantizing Kimi-K3 MXFP4 experts to groupwise int4 "
                    "for gfx942. This conversion is lossy and was explicitly "
                    "enabled by --quantization-config.moe.weight "
                    "int4_per_group_32."
                )
            else:
                logger.info_once("Using AITER_MXFP4_BF16 for Kimi-K3 SiTU MXFP4 MoE.")
            from vllm._aiter_ops import rocm_aiter_ops

            if rocm_aiter_ops.is_fused_moe_situv2_enabled():
                # a16w4 does not read this bound. Force 0 so leftover a8w4
                # config cannot keep bf16 activations.
                # TODO: Remove once AITER takes this as a kernel argument.
                os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
        else:
            from vllm.config import get_current_vllm_config

            k31 = _k31_fi_moe(moe, get_current_vllm_config())
            try:
                self.mxfp4_backend, self.experts_cls = (
                    select_deepseek_v4_mxfp4_moe_backend(moe)
                )
            except (ValueError, NotImplementedError) as _k31_e:
                # K31 overlay moe-fi-bi-start: under BI the image oracle rejects every
                # backend before the FI override below gets a chance
                from vllm import envs as _k31_envs

                if not k31 and moe.activation.value == "situ" and _k31_envs.VLLM_BATCH_INVARIANT:
                    raise ValueError(
                        "VLLM_BATCH_INVARIANT=1 serves this SiTU MXFP4 MoE only through the K31 FlashInfer "
                        "W4A16 experts: a Kimi-K3.1 checkpoint (linear_attn_config.use_full_rank_gate) on sm_10x"
                    ) from _k31_e
                if not k31:
                    raise
                logger.info_once("K31 moe-fi-bi-start: oracle refused (%s); using FI W4A16", _k31_e)
                self.mxfp4_backend = Mxfp4MoeBackend.MARLIN
            if k31:
                logger.info_once(
                    "K31: FlashInfer CuTe-DSL W4A16 for MXFP4 MoE instead of %s",
                    self.mxfp4_backend,
                )
                self.mxfp4_backend = Mxfp4MoeBackend.MARLIN
                self.experts_cls = K31FiW4A16Experts

        self.max_capture_size = moe.max_capture_size

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs
        self.w13_precision_config = None
        self.w2_precision_config = None

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def skip_forward_padding(self) -> bool:
        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
        # so can skip the padding in the forward before applying the moe method
        return self.mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8

    # TODO(bnell): move to MK/expert_class?
    @property
    def has_unpadded_output(self) -> bool:
        return self.mxfp4_backend in [
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        ]

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        return mxfp4_round_up_hidden_size_and_intermediate_size(
            self.mxfp4_backend,
            hidden_size,
            intermediate_size_per_partition,
            activation=self.moe.activation,
        )

    @staticmethod
    def _encode_mxfp4_weight_scale(loaded_weight: torch.Tensor) -> torch.Tensor:
        if loaded_weight.dtype == torch.uint8:
            return loaded_weight
        if loaded_weight.dtype == torch.float8_e8m0fnu:
            return loaded_weight.view(torch.uint8)
        if loaded_weight.is_floating_point():
            return loaded_weight.to(torch.float8_e8m0fnu).view(torch.uint8)
        return loaded_weight

    @staticmethod
    def get_scale_weight_loader(weight_loader):
        def mxfp4_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: bool = False,
        ) -> bool | None:
            loaded_weight = Mxfp4MoEMethod._encode_mxfp4_weight_scale(loaded_weight)
            return weight_loader(
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
                return_success=return_success,
            )

        return mxfp4_weight_loader

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        mxfp4_block = 32

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size
        weight_loader = extra_weight_attrs.pop("weight_loader")
        scale_weight_loader = Mxfp4MoEMethod.get_scale_weight_loader(weight_loader)

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        set_weight_attrs(w13_weight, {"weight_loader": weight_loader})

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_scale, {"weight_loader": scale_weight_loader})
        w13_weight_scale.quant_method = "block"

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        set_weight_attrs(w2_weight, {"weight_loader": weight_loader})

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, {"weight_loader": scale_weight_loader})
        w2_weight_scale.quant_method = "block"

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            set_weight_attrs(w13_bias, {"weight_loader": weight_loader})

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
            set_weight_attrs(w2_bias, {"weight_loader": weight_loader})

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> None:
        if self.experts_cls is K31FiW4A16Experts:
            _k31_prepare_fi_w4a16(layer)
            self._build_moe_kernel(layer)
            return
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size
        hidden_size = self.hidden_size
        sf_block_size = 32

        # Shape assertions — skipped for SITU since its kernel handles native
        # (non-256-aligned) intermediate sizes without prior round-up.
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        if self.moe.activation != MoEActivation.SITU:
            assert (
                w13.dim() == 3
                and w13.shape[0] == num_experts
                and w13.shape[1] == intermediate_size * self.moe.w13_num_shards
                and w13.shape[2] == hidden_size // 2
            )
            assert (
                w13_scale.dim() == 3
                and w13_scale.shape[0] == num_experts
                and w13_scale.shape[1] == intermediate_size * self.moe.w13_num_shards
                and w13_scale.shape[2] == hidden_size // sf_block_size
            )
            assert (
                w2.dim() == 3
                and w2.shape[0] == num_experts
                and w2.shape[1] == hidden_size
                and w2.shape[2] == intermediate_size // 2
            )
            assert (
                w2_scale.dim() == 3
                and w2_scale.shape[1] == hidden_size
                and w2_scale.shape[2] == intermediate_size // sf_block_size
            )
            if w13_bias is not None:
                assert (
                    w13_bias.dim() == 2
                    and w13_bias.shape[0] == num_experts
                    and w13_bias.shape[1] == intermediate_size * self.moe.w13_num_shards
                )
            if w2_bias is not None:
                assert (
                    w2_bias.dim() == 2
                    and w2_bias.shape[0] == num_experts
                    and w2_bias.shape[1] == hidden_size
                )

        # Convert weights to kernel format
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=w13,
                w2_weight=w2,
                w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                _cache_permute_indices=self._cache_permute_indices,
                activation=self.moe.activation,
                use_separated_a4w4=self.moe.use_mxfp4_w4a4_dsv4,
                activation_key=self.activation_key,
            )
        )

        # For TRITON backends, weights are wrapped tensors from triton_kernels
        # that don't support .detach(). Manually assign parameters.
        is_gfx1250 = False
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx1250

            is_gfx1250 = on_gfx1250()

        uses_triton_weight_format = self.mxfp4_backend in TRITON_BACKENDS or (
            self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16 and is_gfx1250
        )
        if not uses_triton_weight_format:
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
        else:
            layer.w13_weight_triton = w13
            layer.w2_weight_triton = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        # Build quant config
        self._build_moe_kernel(layer)

    def _build_moe_kernel(self, layer: RoutedExperts) -> None:
        """Build the modular MoE kernel from the (already in-format) weights."""
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )
            self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def _setup_kernel_k3_situ_gfx942(self, layer: RoutedExperts) -> None:
        import inspect

        from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
            flydsl_a16w4_gemm1,
        )

        # Fail closed if this AITER build would silently run SiLU.
        if "act" not in inspect.signature(flydsl_a16w4_gemm1).parameters:
            raise RuntimeError(
                "This AITER build ignores the SiTUv2 activation on the "
                "replacement packed-int4 MoE path and would silently compute "
                "SiLU. Rebuild with the replacement A16W4/int4 pipeline."
            )

        from aiter import dtypes as aiter_dtypes
        from aiter.ops.quant import per_1x32_i4_quant
        from aiter.ops.shuffle import (
            pack_int8_to_packed_int4,
            shuffle_scale_for_int4,
            shuffle_weight,
        )
        from aiter.utility import fp4_utils

        fp4_dtype = torch.float4_e2m1fn_x2
        e8m0_dtype = torch.float8_e8m0fnu

        # mxfp4_to_f32 grows the packed weight 8x and peaks above 20 GiB per
        # rank if all experts convert at once. Convert 8 experts at a time.
        _CONVERT_CHUNK = 8

        def convert(
            weight: torch.nn.Parameter,
            scale: torch.nn.Parameter,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Drop the source before returning. Packed int4 is the same size as
            # packed MXFP4, so holding both doubles the footprint. That fits
            # under expert parallel (1/ep_size experts per rank) and does not
            # fit under pure tensor parallel.
            w_all = weight.data.view(fp4_dtype)
            s_all = scale.data.view(e8m0_dtype)
            num_experts = w_all.shape[0]

            # Write chunks into a preallocated output. torch.cat of a list
            # would hold the full result twice at the join.
            out_packed: torch.Tensor | None = None
            out_scale: torch.Tensor | None = None
            scale_stride = 0
            for lo in range(0, num_experts, _CONVERT_CHUNK):
                hi = min(lo + _CONVERT_CHUNK, num_experts)
                weight_f32 = fp4_utils.mxfp4_to_f32(w_all[lo:hi])
                scale_f32 = fp4_utils.e8m0_to_f32(s_all[lo:hi])
                chunk_experts, output_size, input_size = weight_f32.shape
                weight_bf16 = (
                    (
                        weight_f32.view(
                            chunk_experts, output_size, input_size // 32, 32
                        )
                        * scale_f32.view(
                            chunk_experts, output_size, input_size // 32, 1
                        )
                    )
                    .view(chunk_experts, output_size, input_size)
                    .to(torch.bfloat16)
                )
                del weight_f32, scale_f32

                weight_int4, weight_scale = per_1x32_i4_quant(weight_bf16)
                del weight_bf16
                weight_int4 = weight_int4.view(aiter_dtypes.i4x2).view(
                    chunk_experts, output_size, input_size
                )
                weight_packed = pack_int8_to_packed_int4(
                    shuffle_weight(weight_int4.view(aiter_dtypes.i8), (16, 16))
                )
                del weight_int4
                chunk_packed = weight_packed.view(
                    chunk_experts, output_size, input_size // 2
                ).view(aiter_dtypes.i4x2)
                chunk_scale = (
                    shuffle_scale_for_int4(weight_scale, group_size=32)
                    .view(-1)
                    .contiguous()
                )
                if out_packed is None:
                    out_packed = torch.empty(
                        (num_experts,) + tuple(chunk_packed.shape[1:]),
                        dtype=chunk_packed.dtype,
                        device=chunk_packed.device,
                    )
                    scale_stride = chunk_scale.numel() // chunk_experts
                    out_scale = torch.empty(
                        num_experts * scale_stride,
                        dtype=chunk_scale.dtype,
                        device=chunk_scale.device,
                    )
                assert out_packed is not None
                assert out_scale is not None
                out_packed[lo:hi].copy_(chunk_packed)
                out_scale[lo * scale_stride : hi * scale_stride].copy_(chunk_scale)
                del weight_packed, weight_scale, chunk_packed, chunk_scale
                torch.accelerator.empty_cache()

            del w_all, s_all
            assert out_packed is not None
            assert out_scale is not None
            return out_packed, out_scale

        w13, w13_scale = convert(layer.w13_weight, layer.w13_weight_scale)
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        del w13, w13_scale
        torch.accelerator.empty_cache()

        w2, w2_scale = convert(layer.w2_weight, layer.w2_weight_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        del w2, w2_scale
        torch.accelerator.empty_cache()
        layer.w13_weight.is_shuffled = True
        layer.w2_weight.is_shuffled = True

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        if is_weights_pre_processed():
            if self.mxfp4_backend != Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8:
                raise RuntimeError(
                    "pre-processed weights require FLASHINFER_TRTLLM_MXFP4_MXFP8 "
                    f"moe backend, got {self.mxfp4_backend}"
                )
            self._build_moe_kernel(layer)
            return

        if self.is_k3_situ_int4_gfx942:
            self._setup_kernel_k3_situ_gfx942(layer)
            return

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)

    def get_fused_moe_quant_config(
        self,
        layer: RoutedExperts,
    ) -> FusedMoEQuantConfig | None:
        w1_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)
        swiglu_limit = getattr(layer, "swiglu_limit", None)

        is_gfx1250 = False
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx1250

            is_gfx1250 = on_gfx1250()

        if self.mxfp4_backend in TRITON_BACKENDS or (
            self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16 and is_gfx1250
        ):
            # TRITON backends free w13/w2_weight_scale after swizzling; the
            # swizzled scales live inside the precision configs instead.
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config
        else:
            w1_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale

        if self.mxfp4_backend == Mxfp4MoeBackend.EMULATION:
            # Canonical ``mxfp4`` checkpoints are weight-only W4A16. The
            # generic EMULATION config is W4A4, so preserve BF16 activations
            # while the fallback dequantizes only the weights.
            return mxfp4_w4a16_moe_quant_config(
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_bias=w1_bias,
                w2_bias=w2_bias,
                gemm1_clamp_limit=swiglu_limit,
            )

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            swiglu_limit=swiglu_limit,
            layer=layer,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=triton_weight(layer, "w13_weight"),
            w2=triton_weight(layer, "w2_weight"),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=triton_weight(layer, "w13_weight"),
            w2=triton_weight(layer, "w2_weight"),
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
            routing_sink=layer.routing_sink,
            input_ids=input_ids,
            hash_indices_table=layer.hash_indices_table,
        )
