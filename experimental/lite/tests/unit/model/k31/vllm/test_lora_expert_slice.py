"""Expert-sliced LoRA export: the in-group exchange hands every rank its rollout expert block."""

from __future__ import annotations

import pytest
import torch

from megatron.lite.model.k31.vllm.checkpoint import (
    _expert_exchange_from_rows,
    _stage_plans_from_rows,
)


def _exchange(num_experts: int, actor_ep: int, rollout_ep: int, rollout_ranks: list[int]):
    """Run every group rank's plan on a full tensor; return (received, expected) per rank."""

    full = torch.arange(num_experts * 3, dtype=torch.int64).view(num_experts, 3)
    per_actor = num_experts // actor_ep
    rows = [(p, rollout_ranks[p], rollout_ep) for p in range(actor_ep)]
    plans = [
        _expert_exchange_from_rows(rows, p, num_experts, rollout_ranks[p], rollout_ep)
        for p in range(actor_ep)
    ]
    local = [full[p * per_actor : (p + 1) * per_actor] for p in range(actor_ep)]
    out = []
    for q in range(actor_ep):
        pieces = []
        for p in range(actor_ep):
            row, count = plans[p].send[q]
            assert count == plans[q].recv[p]
            pieces.append(local[p][row : row + count])
        start, count = plans[q].target
        out.append((torch.cat(pieces), full[start : start + count]))
    return out


class TestExpertExchange:
    @pytest.mark.parametrize(
        "num_experts,actor_ep,rollout_ep,rollout_ranks",
        [
            (896, 16, 16, list(range(16))),
            (896, 16, 4, [p % 4 for p in range(16)]),
            (896, 8, 8, list(range(8))),
            (896, 4, 16, [0, 1, 2, 3]),
            (896, 4, 16, [12, 13, 14, 15]),
            (896, 1, 4, [2]),
            (896, 64, 64, list(range(64))),
            (896, 16, 64, [16 + p for p in range(16)]),
        ],
        ids=[
            "ep16-dp16ep16",
            "ep16-4x-dp4ep4",
            "ep8-dp8ep8",
            "ep4-group0-of-ep16",
            "ep4-group3-of-ep16",
            "ep1-ep4",
            "ep64-dp64ep64",
            "ep16-group1-of-ep64",
        ],
    )
    def test_every_rank_gets_its_block(self, num_experts, actor_ep, rollout_ep, rollout_ranks):
        for got, want in _exchange(num_experts, actor_ep, rollout_ep, rollout_ranks):
            assert torch.equal(got, want)

    @pytest.mark.parametrize(
        "rows,ep_rank,rollout,error",
        [
            ([(0, 0, 4), (1, 1, 8)], 0, (0, 4), ValueError),
            ([(1, 0, 4), (0, 1, 4)], 0, (0, 4), RuntimeError),
            ([(0, 0, 4), (1, 1, 4)], 0, (3, 4), RuntimeError),
            ([(0, 0, 3)], 0, (0, 3), ValueError),
            ([(0, 4, 4)], 0, (4, 4), ValueError),
        ],
        ids=["mixed-rollout-ep", "group-order", "wrong-own-row", "indivisible", "rank-out-of-range"],
    )
    def test_rejects_inconsistent_requests(self, rows, ep_rank, rollout, error):
        with pytest.raises(error):
            _expert_exchange_from_rows(rows, ep_rank, 896, *rollout)


def _pipeline_exchange(num_experts: int, pp: int, ep: int, rollout_ep: int, rollout_rank):
    """Simulate _pipeline_entries for one per-expert factor of every stage.

    ``rollout_rank(t, e)`` is the rollout EP rank colocated with the trainer rank at PP
    stage ``t``, EP position ``e``.  Stage ``s`` runs plans[t] inside its EP group for every
    target stage ``t`` and scatters block ``t`` to its PP peer at ``t``.  Returns
    ``(received, expected)`` for every (source stage, target rank).
    """

    full = torch.arange(num_experts * 3, dtype=torch.int64).view(num_experts, 3)
    per_actor = num_experts // ep
    rows = [
        tuple([e] + [v for t in range(pp) for v in (rollout_rank(t, e), rollout_ep)])
        for e in range(ep)
    ]
    plans = [_stage_plans_from_rows(rows, e, num_experts, pp) for e in range(ep)]
    local = [full[e * per_actor : (e + 1) * per_actor] for e in range(ep)]
    out = []
    for _stage in range(pp):  # every stage holds its own layers' copy of the factor
        for t in range(pp):
            for q in range(ep):  # EP position q of the source stage, scattering to (t, q)
                pieces = []
                for p in range(ep):
                    row, count = plans[p][t].send[q]
                    assert count == plans[q][t].recv[p]
                    pieces.append(local[p][row : row + count])
                r = rollout_rank(t, q)
                per_rollout = num_experts // rollout_ep
                out.append((torch.cat(pieces), full[r * per_rollout : (r + 1) * per_rollout]))
    return out


class TestPipelineExpertExchange:
    @pytest.mark.parametrize(
        "num_experts,pp,ep,rollout_ep,order",
        [
            (896, 4, 4, 16, "pp-outer"),
            (896, 4, 4, 16, "pp-inner"),
            (896, 2, 8, 16, "pp-outer"),
            (896, 4, 16, 64, "pp-outer"),
            (896, 4, 4, 4, "per-node-replica"),
            (896, 2, 2, 4, "pp-inner"),
        ],
        ids=["pp4ep4-dp16ep16", "pp4ep4-dp16ep16-ppinner", "pp2ep8-dp16ep16",
             "pp4ep16-dp64ep64", "pp4ep4-4x-dp4ep4", "pp2ep2-dp4ep4-ppinner"],
    )
    def test_every_rank_gets_its_block(self, num_experts, pp, ep, rollout_ep, order):
        if order == "pp-outer":
            def rollout_rank(t, e):
                return (t * ep + e) % rollout_ep
        elif order == "pp-inner":
            def rollout_rank(t, e):
                return (e * pp + t) % rollout_ep
        else:
            def rollout_rank(t, e):
                return (t * ep + e) % rollout_ep
        for got, want in _pipeline_exchange(num_experts, pp, ep, rollout_ep, rollout_rank):
            assert torch.equal(got, want)

    def test_rejects_mixed_rollout_ep_across_stages(self):
        rows = [(0, 0, 4, 0, 8), (1, 1, 4, 1, 8)]
        with pytest.raises(ValueError):
            _stage_plans_from_rows(rows, 0, 896, 2)
