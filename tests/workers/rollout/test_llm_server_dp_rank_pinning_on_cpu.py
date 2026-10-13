# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""``rollout.pin_dp_rank_by_session``: every turn of a session reaches one DP
engine of its vLLM replica, so a multi-turn session keeps its prefix cache."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout.llm_server import LLMServerClient, session_dp_rank
from verl.workers.rollout.replica import TokenOutput


class _RecordingServer:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

        async def remote(**kwargs):
            self.calls.append(kwargs)
            return TokenOutput(token_ids=[1], log_probs=None, extra_fields={"global_steps": 0})

        self.generate = SimpleNamespace(remote=remote)


def _client(monkeypatch, *, name="vllm", pin=True, dp=4) -> tuple[LLMServerClient, _RecordingServer]:
    rollout = {"name": name, "data_parallel_size": dp, "pin_dp_rank_by_session": pin}
    client = LLMServerClient(config=OmegaConf.create({"actor_rollout_ref": {"rollout": rollout}}))
    server = _RecordingServer()

    async def acquire(request_id, **extra):
        return "server-0", server

    monkeypatch.setattr(client, "_acquire_server", acquire)
    monkeypatch.setattr(client, "_release_server", lambda *args, **kwargs: None)
    return client, server


async def _turns(client, session_ids):
    for session_id in session_ids:
        await client.generate(session_id, prompt_ids=[1, 2, 3], sampling_params={})


@pytest.mark.asyncio
async def test_turns_of_a_session_share_a_dp_rank(monkeypatch):
    client, server = _client(monkeypatch)
    sessions = [f"session-{i}" for i in range(64)]

    await _turns(client, sessions + sessions)

    ranks = [call["data_parallel_rank"] for call in server.calls]
    assert ranks[:64] == ranks[64:]
    assert ranks[:64] == [session_dp_rank(s, 4) for s in sessions]
    # Sessions still spread over every engine of the replica.
    assert set(ranks) == {0, 1, 2, 3}
    # Each turn stays its own vLLM request.
    assert len({call["request_id"] for call in server.calls}) == 128


def test_rank_is_a_stable_hash_not_python_hash():
    # zlib.crc32(b"abc") == 0x352441C2; hash() would differ per PYTHONHASHSEED.
    assert session_dp_rank("abc", 4) == 0x352441C2 % 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [{"pin": False}, {"dp": 1}, {"name": "sglang"}],
    ids=["disabled", "single-dp-engine", "not-vllm"],
)
async def test_no_rank_is_requested_otherwise(monkeypatch, overrides):
    client, server = _client(monkeypatch, **overrides)

    await _turns(client, ["session-0"])

    assert "data_parallel_rank" not in server.calls[0]
