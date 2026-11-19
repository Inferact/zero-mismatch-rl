"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Unmodified multi-token grouped kernel and compiler extracted from FlashInfer
c3af0c13328fd23b256c53b9b4f94d9e160d118c for independent golden testing.
Only unrelated imports/dispatchers were removed. Never import production helpers.
"""

from typing import Callable
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.utils import SmemAllocator
import tvm_ffi  # noqa: F401 -- initialize the CuTe FFI provider
GROUPED_LOG2E = 1.4426950408889634

@cute.kernel
def _grouped_kda_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    cu: cute.Tensor,
    ssm_idx: cute.Tensor,
    nat: cute.Tensor,
    state: cute.Tensor,
    src: cute.Tensor,
    src_idx: cute.Tensor,
    out: cute.Tensor,
    q_total: cutlass.Int32,
    scale: cutlass.Float32,
    lower_bound: cutlass.Float32,
    D: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    KS: cutlass.Constexpr[int],
    RATIO: cutlass.Constexpr[int],
    GATE_MODE: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    BETA_LOGIT: cutlass.Constexpr[int],
    USE_L2: cutlass.Constexpr[int],
    USE_SRC: cutlass.Constexpr[int],
    VSPLIT: cutlass.Constexpr[int],
):
    tid, _, _ = cute.arch.thread_idx()
    hv, n, vz = cute.arch.block_idx()
    _, n_seq, _ = cute.arch.grid_dim()
    h = hv // RATIO
    KC: cutlass.Constexpr = D // KS  # state elems per thread
    G: cutlass.Constexpr = KC // 8  # 16B granules per thread
    CPB: cutlass.Constexpr = D // VSPLIT  # columns per block
    NT: cutlass.Constexpr = (D * KS) // VSPLIT  # threads per block
    EPT: cutlass.Constexpr = max(D // NT, 1)  # preprocess elems per thread
    SW: cutlass.Constexpr = min(D, NT) // 32  # warp partials in L2 reduce

    v_idx = vz * CPB + tid // KS  # global column owned
    part = tid % KS  # slice of that column

    smem = SmemAllocator()
    s_eg = smem.allocate_tensor(cutlass.Float32, cute.make_layout(T * D), 16)
    s_kr = smem.allocate_tensor(cutlass.Float32, cute.make_layout(T * D), 16)
    s_qr = smem.allocate_tensor(cutlass.Float32, cute.make_layout(T * D), 16)
    s_red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(T * 16), 16)

    # per-thread strided views into the T-batched smem: (8, G, token) at `part`
    eg_v = cute.make_tensor(
        s_eg.iterator, cute.make_layout((8, G, KS, T), stride=(1, KS * 8, 8, D))
    )[None, None, part, None]
    kr_v = cute.make_tensor(
        s_kr.iterator, cute.make_layout((8, G, KS, T), stride=(1, KS * 8, 8, D))
    )[None, None, part, None]
    qr_v = cute.make_tensor(
        s_qr.iterator, cute.make_layout((8, G, KS, T), stride=(1, KS * 8, 8, D))
    )[None, None, part, None]

    token_base = cu[n]
    seq_len = cu[n + 1] - token_base

    # ---- initial-state checkpoint slot -------------------------------------
    s = cute.make_rmem_tensor((8, G), cutlass.Float32)
    if cutlass.const_expr(USE_SRC):
        slot0 = src_idx[n]
    else:
        ic = cutlass.Int32(0)
        if cutlass.const_expr(T > 1):
            ic = nat[n] - 1
            if ic < 0:
                ic = cutlass.Int32(0)
            if ic > T - 1:
                ic = cutlass.Int32(T - 1)
        slot0 = ssm_idx[n, ic]
    if slot0 < 0:
        slot0 = cutlass.Int32(0)

    if cutlass.const_expr(USE_SRC):
        row0 = src[slot0, hv, v_idx, None]
    else:
        row0 = state[slot0, hv, v_idx, None]
    grow0 = cute.make_tensor(
        row0.iterator, cute.make_layout((8, G, KS), stride=(1, KS * 8, 8))
    )
    gv0 = grow0[None, None, part]
    sb0 = cute.make_rmem_tensor((8, G), cutlass.BFloat16)
    cute.autovec_copy(
        gv0, sb0, l1c_evict_priority=cute.nvgpu.common.CacheEvictionPriority.NO_ALLOCATE
    )
    s.store(sb0.load().to(cutlass.Float32))

    # loop-invariant gate constants
    d = tid % D
    av = cutlass.Float32(1.0)
    if cutlass.const_expr(GATE_MODE != 0):
        av = cute.exp2(a_log[h] * GROUPED_LOG2E, fastmath=True)
    dtbs = []
    for e in cutlass.range_constexpr(EPT):
        if cutlass.const_expr(GATE_MODE != 0 and HAS_DT_BIAS != 0):
            dtbs.append(dt_bias[h * D + d + e * NT])
        else:
            dtbs.append(cutlass.Float32(0.0))

    # ---- phase A: preprocess ALL tokens, one barrier -----------------------
    slots = []
    ves = []
    bbs = []
    for t in cutlass.range_constexpr(T):
        slots.append(ssm_idx[n, t])
        pidx = token_base + t
        if t >= seq_len:
            pidx = cutlass.Int32(0)
        ves.append(v[pidx, hv, v_idx].to(cutlass.Float32))
        bbv = beta[pidx, hv].to(cutlass.Float32)
        if cutlass.const_expr(BETA_LOGIT):
            bbv = 1.0 / (1.0 + cute.exp2(-bbv * GROUPED_LOG2E, fastmath=True))
        bbs.append(bbv)

        sqp = cutlass.Float32(0.0)
        skp = cutlass.Float32(0.0)
        for e in cutlass.range_constexpr(EPT):
            de = d + e * NT
            qe = q[pidx, h, de].to(cutlass.Float32)
            ke = k[pidx, h, de].to(cutlass.Float32)
            ge = g[pidx, hv, de].to(cutlass.Float32)
            sqp += qe * qe
            skp += ke * ke

            gate = ge
            if cutlass.const_expr(GATE_MODE != 0):
                x = ge
                if cutlass.const_expr(HAS_DT_BIAS):
                    x = x + dtbs[e]
                if cutlass.const_expr(GATE_MODE == 1):
                    sp = cute.log1p(cute.exp2(x * GROUPED_LOG2E, fastmath=True))
                    if x > 20.0:
                        sp = x
                    gate = -av * sp
                else:
                    sig = 1.0 / (
                        1.0 + cute.exp2(-(av * x) * GROUPED_LOG2E, fastmath=True)
                    )
                    gate = lower_bound * sig

            s_eg[t * D + de] = cute.exp2(gate * GROUPED_LOG2E, fastmath=True)
            s_kr[t * D + de] = ke
            s_qr[t * D + de] = qe

        if cutlass.const_expr(USE_L2):
            sq = cute.arch.warp_reduction_sum(sqp)
            sk = cute.arch.warp_reduction_sum(skp)
            lane = tid % 32
            wid = tid // 32
            if (lane == 0) & (wid < SW):
                s_red[t * 16 + wid] = sq
                s_red[t * 16 + 8 + wid] = sk
    cute.arch.sync_threads()

    pf = cute.make_rmem_tensor((8,), cutlass.Float32)

    # ---- phase B: barrier-free sequential recurrence -----------------------
    for t in cutlass.range_constexpr(T):
        slot = slots[t]
        in_row = t < seq_len
        active = in_row & (slot >= 0)
        if active:
            pidx = token_base + t
            ve = ves[t]
            bb = bbs[t]

            rk = cutlass.Float32(1.0)
            rq = scale
            if cutlass.const_expr(USE_L2):
                sqt = cutlass.Float32(0.0)
                skt = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(SW):
                    sqt += s_red[t * 16 + w]
                    skt += s_red[t * 16 + 8 + w]
                rk = cute.rsqrt(skt + 1e-6)
                rq = cute.rsqrt(sqt + 1e-6) * scale

            # pass 1: decay state + raw prediction
            svec = s[None, 0].load() * eg_v[None, 0, t].load()
            s[None, 0].store(svec)
            pvec = kr_v[None, 0, t].load() * svec
            for gi in cutlass.range_constexpr(1, G, 1):
                svec = s[None, gi].load() * eg_v[None, gi, t].load()
                s[None, gi].store(svec)
                pvec = pvec + kr_v[None, gi, t].load() * svec
            pf.store(pvec)
            pred = ((pf[0] + pf[1]) + (pf[2] + pf[3])) + (
                (pf[4] + pf[5]) + (pf[6] + pf[7])
            )
            if cutlass.const_expr(KS > 1):
                for off_i in cutlass.range_constexpr((KS - 1).bit_length()):
                    pred += cute.arch.shuffle_sync_bfly(pred, 1 << off_i)
            deltak = rk * bb * (ve - rk * pred)

            # pass 2: rank-1 update + raw output projection
            svec = s[None, 0].load() + kr_v[None, 0, t].load() * deltak
            s[None, 0].store(svec)
            ovec = qr_v[None, 0, t].load() * svec
            for gi in cutlass.range_constexpr(1, G, 1):
                svec = s[None, gi].load() + kr_v[None, gi, t].load() * deltak
                s[None, gi].store(svec)
                ovec = ovec + qr_v[None, gi, t].load() * svec
            pf.store(ovec)
            o = ((pf[0] + pf[1]) + (pf[2] + pf[3])) + (
                (pf[4] + pf[5]) + (pf[6] + pf[7])
            )
            if cutlass.const_expr(KS > 1):
                for off_i in cutlass.range_constexpr((KS - 1).bit_length()):
                    o += cute.arch.shuffle_sync_bfly(o, 1 << off_i)
            if part == 0:
                out[pidx, hv, v_idx] = (rq * o).to(cutlass.BFloat16)

            # bf16 checkpoint write
            row_w = state[slot, hv, v_idx, None]
            grow_w = cute.make_tensor(
                row_w.iterator, cute.make_layout((8, G, KS), stride=(1, KS * 8, 8))
            )
            gw = grow_w[None, None, part]
            sb1 = cute.make_rmem_tensor((8, G), cutlass.BFloat16)
            sb1.store(s.load().to(cutlass.BFloat16))
            cute.autovec_copy(
                sb1,
                gw,
                l1c_evict_priority=cute.nvgpu.common.CacheEvictionPriority.NO_ALLOCATE,
            )
        else:
            if in_row:
                if part == 0:
                    out[token_base + t, hv, v_idx] = cutlass.BFloat16(0.0)

    # orphan packed-suffix zeroing (carrier tokens owned by no row)
    if (n == 0) & (vz == 0):
        if tid < D:
            covered = cu[n_seq]
            for pos in cutlass.range(covered, q_total, 1):
                for e in cutlass.range_constexpr(EPT):
                    out[pos, hv, tid + e * NT] = cutlass.BFloat16(0.0)


@cute.jit
def _grouped_kda_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    cu: cute.Tensor,
    ssm_idx: cute.Tensor,
    nat: cute.Tensor,
    state: cute.Tensor,
    src: cute.Tensor,
    src_idx: cute.Tensor,
    out: cute.Tensor,
    n_seq: cutlass.Int32,
    q_total: cutlass.Int32,
    g_stride_q: cutlass.Int32,
    state_stride0: cutlass.Int32,
    src_stride0: cutlass.Int32,
    src_idx_stride: cutlass.Int32,
    scale: cutlass.Float32,
    lower_bound: cutlass.Float32,
    stream: cuda.CUstream,
    D: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    KS: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    GATE_MODE: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    BETA_LOGIT: cutlass.Constexpr[int],
    USE_L2: cutlass.Constexpr[int],
    USE_SRC: cutlass.Constexpr[int],
    VSPLIT: cutlass.Constexpr[int],
):
    qt = q_total
    st0 = cute.assume(state_stride0, divby=8)
    ss0 = cute.assume(src_stride0, divby=8)

    q3 = cute.make_tensor(
        q.iterator, cute.make_layout((qt, H, D), stride=(H * D, D, 1))
    )
    k3 = cute.make_tensor(
        k.iterator, cute.make_layout((qt, H, D), stride=(H * D, D, 1))
    )
    v3 = cute.make_tensor(
        v.iterator, cute.make_layout((qt, HV, D), stride=(HV * D, D, 1))
    )
    g3 = cute.make_tensor(
        g.iterator, cute.make_layout((qt, HV, D), stride=(g_stride_q, D, 1))
    )
    b2 = cute.make_tensor(beta.iterator, cute.make_layout((qt, HV), stride=(HV, 1)))
    o3 = cute.make_tensor(
        out.iterator, cute.make_layout((qt, HV, D), stride=(HV * D, D, 1))
    )
    st4 = cute.make_tensor(
        state.iterator,
        cute.make_layout((n_seq * T + 8, HV, D, D), stride=(st0, D * D, D, 1)),
    )
    sr4 = cute.make_tensor(
        src.iterator, cute.make_layout((n_seq + 8, HV, D, D), stride=(ss0, D * D, D, 1))
    )
    si2 = cute.make_tensor(
        ssm_idx.iterator, cute.make_layout((n_seq, T), stride=(T, 1))
    )
    sx1 = cute.make_tensor(
        src_idx.iterator, cute.make_layout((n_seq,), stride=(src_idx_stride,))
    )

    kern = _grouped_kda_kernel(
        q3,
        k3,
        v3,
        g3,
        b2,
        a_log,
        dt_bias,
        cu,
        si2,
        nat,
        st4,
        sr4,
        sx1,
        o3,
        q_total,
        scale,
        lower_bound,
        D,
        T,
        KS,
        HV // H,
        GATE_MODE,
        HAS_DT_BIAS,
        BETA_LOGIT,
        USE_L2,
        USE_SRC,
        VSPLIT,
    )
    kern.launch(
        grid=(HV, n_seq, VSPLIT),
        block=(D * KS // VSPLIT, 1, 1),
        stream=stream,
        preferred_smem_carveout=25,
    )


_grouped_compiled_cache: dict[tuple[int, ...], Callable[..., None]] = {}


def _get_grouped_compiled(
    key,
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    initial_state,
    initial_state_source,
    initial_state_indices,
    output,
):
    fn = _grouped_compiled_cache.get(key)
    if fn is None:
        D, T, H, HV, gm, hdb, bl, ul2, us = key
        KS = 4 if T == 1 else 2
        VSPLIT = 4

        def dyn(t, ld):
            return from_dlpack(
                t, assumed_align=16, enable_tvm_ffi=True
            ).mark_layout_dynamic(leading_dim=ld)

        fn = cute.compile(
            _grouped_kda_host,
            dyn(q, 3),
            dyn(k, 3),
            dyn(v, 3),
            dyn(g, 3),
            dyn(beta, 2),
            dyn(A_log, 0),
            dyn(dt_bias, 0),
            dyn(cu_seqlens, 0),
            dyn(ssm_state_indices, 1),
            dyn(num_accepted_tokens, 0),
            dyn(initial_state, 3),
            dyn(initial_state_source, 3),
            from_dlpack(
                initial_state_indices, assumed_align=4, enable_tvm_ffi=True
            ).mark_layout_dynamic(),
            dyn(output, 3),
            1,
            1,
            1,
            1,
            1,
            1,
            1.0,
            1.0,
            make_fake_stream(),
            D,
            T,
            KS,
            H,
            HV,
            gm,
            hdb,
            bl,
            ul2,
            us,
            VSPLIT,
            options="--enable-tvm-ffi",
        )
        _grouped_compiled_cache[key] = fn
    return fn
