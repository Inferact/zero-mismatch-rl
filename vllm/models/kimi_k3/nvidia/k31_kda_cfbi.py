"""Runtime for the CF-BI KDA engines.

Contract: per (sequence, head) the state is E fp32 [V][K] + a row cache
fp32 [64][640] (kn, G, ut, w, u of the open chunk) + pos (int32, = tokens already in the open chunk). Chunks are
aligned to the sequence start. The engines (prefill K1/K2, decode rows_fast) are bitwise equal, so a token may go
through either of them.

Routing (VLLM_BATCH_INVARIANT=1):
  * pure decode batch: rows_fast (decode engine) over all rows, one token each.
  * any batch with prefill tokens (mixed / chunked prefill): per request with ctx tokens already done and L new
    tokens, the first min(L, (-ctx) mod 64) tokens (closing the open chunk) go through rows_fast, the rest
    (chunk-aligned, pos 0 at entry) through the prefill kernel. Decode rows of a mixed batch are L=1 requests and
    always take the rows_fast leg.
  * state lives in the vLLM mamba KV cache (slots: conv, E, rowcache, pos). The kernels index state by batch
    row, so each call gathers the rows' state by slot index, runs, and scatters back (cost recorded as a gap).
  * internal prefill checkpoint (align-mode prefix caching; MambaSpec prefill_checkpoint_alignment = BT): a row
    whose step crosses its last cache block boundary stores its state there into the checkpoint slot (mixed ckpt,
    export_conv_ckpt) instead of the scheduler ending the step at that boundary. The row's prefill runs as two
    prefill calls split at the boundary: the calls of a step that ends there and of the step that follows it.

Contract v6 (default; K31_KDA_CFBI_CONTRACT=v5 selects v5; csrc/k31/kda/kda_cfbi_v6.cu): 16-token chunks and a
row cache fp32 [17][256] (rows {kjr, u} of the open chunk, then {G0, Glast}); same routing, with the v6 row engine
as rows_fast and the v6 chunk engine as prefill.
K31_KDA_CFBI_V6_ENGINE=rows runs every token through the row engine (the contract's reference).

Kernel sources: csrc/k31/kda (built with vLLM for sm_100a / sm_103a, --fmad=false).
"""

import os

import torch

CONTRACT = os.environ.get("K31_KDA_CFBI_CONTRACT", "v6")
if CONTRACT not in ("v5", "v6"):
    raise ValueError(f"K31_KDA_CFBI_CONTRACT takes v5 or v6, got {CONTRACT!r}")
ROWS_ONLY = CONTRACT == "v6" and os.environ.get("K31_KDA_CFBI_V6_ENGINE", "") == "rows"
D, BT, ROWF = (128, 16, 256) if CONTRACT == "v6" else (128, 64, 640)
# v6: rows 0 .. BT-1 = {kjr, u} of the open chunk, row BT = {G0, Glast}
ROWS = BT + 1 if CONTRACT == "v6" else BT
NULL_SLOT = 0  # vllm.v1.attention.backends.utils.NULL_BLOCK_ID
_arange: dict[torch.device, torch.Tensor] = {}
_dec_conv: dict[tuple[int, torch.device], tuple] = {}


def enabled() -> bool:
    from vllm import envs

    return bool(envs.VLLM_BATCH_INVARIANT)


# ---------------------------------------------------------------------------------------------------------------
# state layout: (conv, E (H,D,D) fp32, rowcache (H,ROWS,ROWF) fp32, pos (H,) int32)
# ---------------------------------------------------------------------------------------------------------------
def check_conv_layout() -> None:
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    if is_conv_state_dim_first():
        raise RuntimeError("K31 KDA CF-BI needs the SD conv state layout, not VLLM_SSM_CONV_STATE_LAYOUT=DS")


def state_shape(base: tuple) -> tuple:
    """The KDA mamba state (conv, E) with the row cache and pos appended."""
    conv, rec = base
    h = rec[0]
    assert rec == (h, D, D), f"K31 kda-cfbi needs head_dim {D}, got {rec}"
    return (conv, rec, (h, ROWS, ROWF), (h,))


def state_dtype(base: tuple) -> tuple:
    conv, rec = base
    assert rec == torch.float32, f"K31 kda-cfbi needs an fp32 KDA state (mamba_ssm_cache_dtype), got {rec}"
    return (conv, rec, torch.float32, torch.int32)


def state_copy_func() -> tuple:
    """Align-mode prefix caching copies the KDA state into the slot of the next block through these copy funcs;
    the upstream pair (conv, temporal) skips the CF-BI rowcache and pos, so copy all four."""
    from vllm.model_executor.layers.mamba.mamba_utils import get_conv_copy_spec, get_temporal_copy_spec

    return (get_conv_copy_spec, get_temporal_copy_spec, get_temporal_copy_spec, get_temporal_copy_spec)


def state_bytes_per_seq(h: int, conv_dim: int, conv_w: int, conv_bytes: int = 2) -> dict:
    e = h * D * D * 4
    c = h * ROWS * ROWF * 4
    p = h * 4
    conv = conv_dim * conv_w * conv_bytes
    return {"E": e, "rowcache": c, "pos": p, "conv": conv, "total": e + c + p + conv, "baseline": e + conv}


_EXT: tuple = ()


def kernels():
    """(prefill, decode) engines: csrc/k31/kda/kda_cfbi6_pf4.cu and kda_cfbi6_dsync8.cu, built with vLLM
    (sm_100a / sm_103a, --fmad=false)."""
    global _EXT
    if not _EXT:
        if CONTRACT == "v6":
            from vllm import _k31_kda_v6_C

            _EXT = (_k31_kda_v6_C, _k31_kda_v6_C)
        else:
            from vllm import _k31_kda_dsync8_C, _k31_kda_pf4_C

            _EXT = (_k31_kda_pf4_C, _k31_kda_dsync8_C)
    return _EXT


def _cu_ones(n: int, dev) -> torch.Tensor:
    """0 .. n (int32): a view of a per-device arange. One allocated inside a CUDA-graph capture is not kept (graph-pool
    memory, written only when that graph replays)."""
    a = _arange.get(dev)
    if a is None or a.numel() < n + 1:
        a = torch.arange(max(n + 1, 1025), dtype=torch.int32, device=dev)
        if not torch.cuda.is_current_stream_capturing():
            _arange[dev] = a
    return a[: n + 1]


def _c(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _qkv(q, k, v):
    """q/k/v as the decode kernels read them: unit inner strides and one token stride (e.g. views into the
    packed conv output); anything else is copied."""
    if q.stride(-1) == 1 and q.stride(-2) == D and k.stride() == q.stride() and v.stride() == q.stride():
        return q, k, v
    return _c(q), _c(k), _c(v)


def decode_conv_meta(R: int, dev):
    """causal_conv1d metadata of a pure-decode batch of R one-token rows: a function of R alone, built once per
    (R, device) and kept. vLLM runs every CUDA-graph size eagerly before capturing it, so the capture finds the
    metadata built and the decode graph holds no host-to-device copy node (built inside the capture it put copies
    from pinned host memory into the graph). A size first met inside a capture is built there and not kept (its
    tensors are only initialised when that graph replays)."""
    key = (R, dev)
    meta = _dec_conv.get(key)
    if meta is None:
        import types

        from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

        nd, bp, tp = compute_causal_conv1d_metadata(torch.arange(R + 1, dtype=torch.int32), device=dev)
        meta = (types.SimpleNamespace(nums_dict=nd, batch_ptr=bp, token_chunk_offset_ptr=tp), _cu_ones(R, dev),
                torch.ones(R, dtype=torch.bool, device=dev))
        if not torch.cuda.is_current_stream_capturing():
            _dec_conv[key] = meta
    return meta


# ---------------------------------------------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------------------------------------------
def decode(q, k, v, g, beta, A_log, bias, idx, Epool, Cpool, Ppool, lower, out=None):
    """One token per row. q/k/v/g: [1,N,H,D] bf16 (q/k/v may be token-strided views), beta [1,N,H] bf16 (any
    token stride),
    idx [N] state slots (NULL_SLOT = padding). Writes `out` [1,N,H,D] when given. Graph safe (no host sync)."""
    _, dec = kernels()
    if not (Epool.stride()[1:] == (D * D, D, 1) and Cpool.stride()[1:] == (ROWS * ROWF, ROWF, 1)
            and Ppool.stride(1) == 1):
        raise RuntimeError("K31 KDA CF-BI decode: state pools are not in the slot layout")
    sl = idx if (idx.dtype == torch.int32 and idx.is_contiguous()) else idx.to(torch.int32).contiguous()
    if beta.stride(2) != 1:
        raise RuntimeError("K31 KDA CF-BI decode: beta needs a unit head stride")
    if CONTRACT == "v6":
        return dec.rows_slots(_c(q), _c(k), _c(v), _c(g), _c(beta), A_log, bias, _cu_ones(idx.numel(), q.device),
                              Epool, Cpool, Ppool, sl, lower, out=out)
    return dec.rows_fast_slots(*_qkv(q, k, v), _c(g), beta, A_log, bias, _cu_ones(idx.numel(), q.device),
                               Epool, Cpool, Ppool, sl, lower, out=out)


def store_slots(slots, *pairs):
    """pool[slots[i]] = val[i] for each (pool, val) pair; NULL_SLOT entries keep their content (no host sync)."""
    keep = slots != NULL_SLOT
    for pool, val in pairs:
        old = pool.index_select(0, slots)
        pool.index_copy_(0, slots, torch.where(keep.view(-1, *([1] * (val.dim() - 1))), val, old))


def export_conv_ckpt(x, conv_state, qsl, offsets, slots):
    """Conv state at the internal prefill checkpoints: conv_state[slots[r], :, i] = x[qsl[r] + offsets[r] - w + i]
    for rows with offsets[r] > 0 (w = conv_state.size(-1)): the raw q|k|v inputs a conv leaves as its state after
    a step that ends at the checkpoint. x [T, C] raw inputs, conv_state [slots, C, w]."""
    rows = [r for r, o in enumerate(offsets) if o > 0]
    if not rows:
        return
    w = conv_state.size(-1)
    assert all(offsets[r] >= w for r in rows) and x.size(-1) == conv_state.size(-2)
    dev = x.device
    tok = torch.tensor([qsl[r] + offsets[r] - w + i for r in rows for i in range(w)], dtype=torch.long)
    val = x.index_select(0, tok.to(dev, non_blocking=True)).view(len(rows), w, x.size(-1)).transpose(1, 2)
    sl = slots.index_select(0, torch.tensor(rows, dtype=torch.long).to(dev, non_blocking=True)).long()
    store_slots(sl, (conv_state, val.to(conv_state.dtype)))


def _prefill(pf, toks, A_log, bias, lower, segs, E, C, P, out):
    """One prefill-engine call: state row j of E/C/P (chunk aligned, pos 0) runs tokens [t0, t0 + L) of toks
    (q, k, v, g, beta) for segs[j] = (t0, L); their outputs go to out."""
    dev = toks[0].device
    tidx_l, tab, offs = [], [], [0]
    for j, (t0, L) in enumerate(segs):
        o = len(tidx_l)
        tidx_l.extend(range(t0, t0 + L))
        nc = (L + BT - 1) // BT
        for c in range(nc):
            tab.append([j, o + c * BT, min(BT, L - c * BT), int(c == nc - 1)])
        offs.append(offs[-1] + nc)
    tidx = torch.tensor(tidx_l, dtype=torch.long).to(dev, non_blocking=True)
    tabt = torch.tensor(tab, dtype=torch.int32).to(dev, non_blocking=True)
    offt = torch.tensor(offs, dtype=torch.int32).to(dev, non_blocking=True)
    bv = 316 if len(segs) == 1 else 332
    ot = pf.prefill(*(x.index_select(1, tidx) for x in toks), A_log, bias, tabt, offt, E, C, P, lower, bv)[0]
    out.index_copy_(1, tidx, ot)


def mixed(q, k, v, g, beta, A_log, bias, idx, starts, lens, ctx, Epool, Cpool, Ppool, lower, ckpt=None):
    """Prefill / chunked-prefill / mixed batch. Token tensors cover the real tokens [1,T,H,D]; request r owns
    tokens [starts[r], starts[r]+lens[r]) and has ctx[r] tokens of history (python ints, from CPU metadata).
    ckpt = (offsets, slots): offsets[r] > 0 (host ints) stores row r's state after its first offsets[r] new tokens
    (a chunk boundary) in cache slot slots[r] (device, per row; NULL_SLOT keeps the slot)."""
    if ROWS_ONLY:
        return _mixed_v6(q, k, v, g, beta, A_log, bias, idx, starts, lens, ctx, Epool, Cpool, Ppool, lower, ckpt)
    pf, dec = kernels()
    dev = q.device
    R = len(lens)
    q, k, v, g, beta = _c(q), _c(k), _c(v), _c(g), _c(beta)
    il = idx[:R].long()
    E = Epool.index_select(0, il)
    C = Cpool.index_select(0, il)
    P = Ppool.index_select(0, il)
    fresh = [c == 0 for c in ctx]
    if any(fresh):
        fm = torch.tensor(fresh, dtype=torch.bool).to(dev, non_blocking=True)
        E.masked_fill_(fm.view(R, 1, 1, 1), 0.0)
        P.masked_fill_(fm.view(R, 1), 0)
    out = torch.empty_like(v)
    head = [min(L, (-c) % BT) for L, c in zip(lens, ctx)]
    nh = sum(head)
    if nh:
        hidx_l, cu = [], [0]
        for s, h in zip(starts, head):
            hidx_l.extend(range(s, s + h))
            cu.append(cu[-1] + h)
        hidx = torch.tensor(hidx_l, dtype=torch.long).to(dev, non_blocking=True)
        cut = torch.tensor(cu, dtype=torch.int32).to(dev, non_blocking=True)
        oh = dec.rows_fast(*(x.index_select(1, hidx) for x in (q, k, v, g, beta)), A_log, bias, cut, E, C, P, lower)
        out.index_copy_(1, hidx, oh)
    tails = [(r, starts[r] + head[r], lens[r] - head[r]) for r in range(R) if lens[r] > head[r]]
    if tails:
        ckc = {}  # tail j -> chunks before its checkpoint
        if ckpt is not None:
            for j, (r, _, L) in enumerate(tails):
                o = ckpt[0][r]
                if o > 0:
                    assert head[r] == 0 and o % BT == 0 and o < L, f"K31 kda-cfbi: checkpoint {o} in a {L}-token tail"
                    ckc[j] = o // BT
        full = len(tails) == R
        if full:
            Et, Ct, Pt = E, C, P
        else:
            sel = torch.tensor([r for r, _, _ in tails], dtype=torch.long).to(dev, non_blocking=True)
            Et, Ct, Pt = E.index_select(0, sel), C.index_select(0, sel), P.index_select(0, sel)
        toks = (q, k, v, g, beta)
        _prefill(pf, toks, A_log, bias, lower, [(t0, ckc[j] * BT if j in ckc else L) for j, (_, t0, L) in
                                                enumerate(tails)], Et, Ct, Pt, out)
        if ckc:
            js = torch.tensor(list(ckc), dtype=torch.long).to(dev, non_blocking=True)
            rows = torch.tensor([tails[j][0] for j in ckc], dtype=torch.long).to(dev, non_blocking=True)
            Ek, Ck, Pk = Et.index_select(0, js), Ct.index_select(0, js), Pt.index_select(0, js)
            store_slots(ckpt[1].index_select(0, rows).long(), (Epool, Ek), (Cpool, Ck), (Ppool, Pk))
            _prefill(pf, toks, A_log, bias, lower, [(tails[j][1] + c * BT, tails[j][2] - c * BT) for j, c in
                                                    ckc.items()], Ek, Ck, Pk, out)
            Et.index_copy_(0, js, Ek)
            Ct.index_copy_(0, js, Ck)
            Pt.index_copy_(0, js, Pk)
        if not full:
            E.index_copy_(0, sel, Et)
            C.index_copy_(0, sel, Ct)
            P.index_copy_(0, sel, Pt)
    Epool.index_copy_(0, il, E)
    Cpool.index_copy_(0, il, C)
    Ppool.index_copy_(0, il, P)
    return out


def _mixed_v6(q, k, v, g, beta, A_log, bias, idx, starts, lens, ctx, Epool, Cpool, Ppool, lower, ckpt):
    """mixed() on the v6 row engine alone: every row runs its new tokens from its own state, split at the
    checkpoint when there is one."""
    eng, _ = kernels()
    dev = q.device
    R = len(lens)
    q, k, v, g, beta = _c(q), _c(k), _c(v), _c(g), _c(beta)
    il = idx[:R].long()
    E = Epool.index_select(0, il)
    C = Cpool.index_select(0, il)
    P = Ppool.index_select(0, il)
    fresh = [c == 0 for c in ctx]
    if any(fresh):
        fm = torch.tensor(fresh, dtype=torch.bool).to(dev, non_blocking=True)
        E.masked_fill_(fm.view(R, 1, 1, 1), 0.0)
        P.masked_fill_(fm.view(R, 1), 0)
    out = torch.empty_like(v)

    def run(segs):
        tidx_l, cu = [], [0]
        for t0, L in segs:
            tidx_l.extend(range(t0, t0 + L))
            cu.append(cu[-1] + L)
        if not tidx_l:
            return
        tidx = torch.tensor(tidx_l, dtype=torch.long).to(dev, non_blocking=True)
        cut = torch.tensor(cu, dtype=torch.int32).to(dev, non_blocking=True)
        ot = eng.rows(*(x.index_select(1, tidx) for x in (q, k, v, g, beta)), A_log, bias, cut, E, C, P, lower)
        out.index_copy_(1, tidx, ot)

    offs = [0] * R
    if ckpt is not None:
        for r in range(R):
            o = ckpt[0][r]
            if o > 0:
                assert o < lens[r] and (ctx[r] + o) % BT == 0, f"K31 kda-cfbi: checkpoint {o} of {lens[r]}"
                offs[r] = o
    run([(starts[r], offs[r] or lens[r]) for r in range(R)])
    rows = [r for r in range(R) if offs[r]]
    if rows:
        rt = torch.tensor(rows, dtype=torch.long).to(dev, non_blocking=True)
        store_slots(ckpt[1].index_select(0, rt).long(), (Epool, E.index_select(0, rt)),
                    (Cpool, C.index_select(0, rt)), (Ppool, P.index_select(0, rt)))
        run([(starts[r] + offs[r], lens[r] - offs[r]) if offs[r] else (starts[r], 0) for r in range(R)])
    Epool.index_copy_(0, il, E)
    Cpool.index_copy_(0, il, C)
    Ppool.index_copy_(0, il, P)
    return out
