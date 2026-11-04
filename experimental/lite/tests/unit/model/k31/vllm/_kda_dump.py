"""Loader for the KDA rollout dump (a vLLM run with a dump hook on the KDA layers).

A dump directory holds ``params_<layer>.pt`` (the vLLM-held conv / A_log / dt_bias / o_norm tensors and the
o_norm CustomOp method) and ``r*.pt`` records, one per KDA layer call while a request tag was on:
``gen<i>`` (greedy c=1 generation: prompt prefill, then one decode step per generated token, decode steps
harvested from CUDA graph replays) and ``score<i>`` (one prefill of prompt + generated tokens).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import torch

DUMP_ENV = "K31_KDA_G1_DUMP"
CASES_ENV = "K31_KDA_G1_CASES"
FIELDS = ("qkv", "g1", "beta", "g2", "o")


@dataclass
class LayerParams:
    prefix: str
    conv_weight: torch.Tensor  # [3HD, W] fp32
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    norm_weight: torch.Tensor
    norm_eps: float
    norm_method: str
    lower: float
    num_heads: int
    head_dim: int


@dataclass
class Sequence:
    tag: str
    rows: dict[str, torch.Tensor]  # FIELDS -> [L, ...]
    graph_rows: torch.Tensor  # bool [L], row came from a CUDA graph replay
    calls: int = 0
    meta: dict = field(default_factory=dict)


def available() -> bool:
    d = os.environ.get(DUMP_ENV, "")
    return bool(d) and os.path.isdir(d) and bool(glob.glob(os.path.join(d, "params_*.pt")))


def load_params(d: str) -> dict[str, LayerParams]:
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "params_*.pt"))):
        p = torch.load(f)
        w = p["conv1d_weight"]
        out[p["prefix"]] = LayerParams(
            prefix=p["prefix"],
            conv_weight=w.reshape(w.shape[0], w.shape[-1]).contiguous(),
            A_log=p["A_log"],
            dt_bias=p["dt_bias"],
            norm_weight=p["o_norm_weight"],
            norm_eps=p["o_norm_eps"],
            norm_method=p["o_norm_method"],
            lower=p["gate_lower_bound"],
            num_heads=p["local_num_heads"],
            head_dim=p["head_dim"],
        )
    return out


def _gather(records, want: int, tag: str) -> Sequence:
    rows = {k: [] for k in FIELDS}
    graph = []
    nxt = 0
    for r in records:
        pos = r["pos"].tolist()
        for i, p in enumerate(pos):
            if nxt >= want:
                break
            if p != nxt:  # c=1: real rows lead every record; the rest is graph / batch padding
                break
            for k in FIELDS:
                rows[k].append(r[k][i])
            graph.append(bool(r["graph"]))
            nxt += 1
    if nxt != want:
        raise AssertionError(f"{tag}: gathered {nxt} rows, want {want}")
    return Sequence(
        tag=tag,
        rows={k: torch.stack(v) for k, v in rows.items()},
        graph_rows=torch.tensor(graph),
        calls=len(records),
    )


def load_sequences(d: str, cases_path: str) -> tuple[dict[str, LayerParams], dict, list[dict]]:
    """Returns params, {(layer, tag): Sequence}, cases. gen<i> covers prompt + generated[:-1] (every token the
    rollout fed); score<i> covers prompt + generated."""
    params = load_params(d)
    cases = json.load(open(cases_path))
    by = {}
    for f in sorted(glob.glob(os.path.join(d, "r*.pt"))):
        r = torch.load(f)
        by.setdefault((r["prefix"], r["tag"]), []).append(r)
    seqs = {}
    for c in cases:
        P, N = len(c["prompt"]), len(c["gen"])
        for layer in params:
            for tag, want in ((f"gen{c['i']}", P + N - 1), (f"score{c['i']}", P + N)):
                recs = sorted(by.get((layer, tag), []), key=lambda r: r["seq"])
                if recs:
                    s = _gather(recs, want, tag)
                    s.meta = {"prompt": P, "gen": N}
                    seqs[(layer, tag)] = s
    return params, seqs, cases
