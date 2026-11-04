# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Verified DCP shard writes: read-back check, publish of step_N.partial, partial dirs ignored."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed.checkpoint as dcp

from megatron.lite.primitive.ckpt import dcp as ckpt_dcp


def _state():
    g = torch.Generator().manual_seed(0)
    return {
        "step": 3,
        "model.w": torch.randn(64, 32, generator=g).to(torch.bfloat16),
        "model.b": torch.randn(7, generator=g),
    }


def _corrupt(path, how):
    shard = next(f for f in os.listdir(path) if f.endswith(".distcp"))
    f = os.path.join(path, shard)
    size = os.path.getsize(f)
    if how == "truncate":
        with open(f, "r+b") as fh:
            fh.truncate(size - 100)
    else:  # flip one byte inside the tensor payload region
        with open(f, "r+b") as fh:
            fh.seek(size // 2)
            b = fh.read(1)
            fh.seek(size // 2)
            fh.write(bytes([b[0] ^ 0xFF]))


class TestDcpShardVerification:
    def test_clean_save_passes(self, tmp_path):
        state = _state()
        meta = dcp.save(state, checkpoint_id=str(tmp_path))
        assert ckpt_dcp._dcp_own_shard_error(meta, state, str(tmp_path)) is None

    @pytest.mark.parametrize("how", ["truncate", "flip"], ids=["short-shard", "flipped-byte"])
    def test_damaged_shard_is_reported(self, tmp_path, how):
        state = _state()
        meta = dcp.save(state, checkpoint_id=str(tmp_path))
        _corrupt(str(tmp_path), how)
        assert ckpt_dcp._dcp_own_shard_error(meta, state, str(tmp_path)) is not None

    def test_verified_save_publishes_partial(self, tmp_path):
        final = str(tmp_path / "step_3")
        stage = ckpt_dcp._begin_partial(final)
        assert stage.endswith(".partial") and os.path.isdir(stage)
        ckpt_dcp._save_dcp_verified(_state(), stage)
        ckpt_dcp._publish_partial(stage, final)
        assert os.path.isdir(final) and not os.path.exists(stage)
        assert os.path.exists(os.path.join(final, ".metadata"))

    def test_partial_dirs_are_not_checkpoints(self, tmp_path):
        (tmp_path / "step_2").mkdir()
        (tmp_path / "step_3.partial").mkdir()
        assert ckpt_dcp._resolve_step_checkpoint_path(str(tmp_path)) == str(tmp_path / "step_2")

    @pytest.mark.parametrize(
        "name", ["step_3.partial", "step_3.partial/", "step_3.replaced"], ids=["partial", "trailing-slash", "replaced"]
    )
    def test_direct_unpublished_path_is_refused(self, tmp_path, name):
        (tmp_path / name.rstrip("/")).mkdir()
        with pytest.raises(ValueError, match="not a published checkpoint"):
            ckpt_dcp._resolve_step_checkpoint_path(os.path.join(str(tmp_path), name))

    def test_direct_published_path_resolves_to_itself(self, tmp_path):
        (tmp_path / "step_3").mkdir()
        assert ckpt_dcp._resolve_step_checkpoint_path(str(tmp_path / "step_3")) == str(tmp_path / "step_3")

    @pytest.mark.parametrize("fail", [False, True], ids=["replaced", "publish-fails"])
    def test_republish_keeps_the_published_copy(self, tmp_path, monkeypatch, fail):
        final = tmp_path / "step_3"
        final.mkdir()
        (final / "state").write_text("old")
        stage = ckpt_dcp._begin_partial(str(final))
        with open(os.path.join(stage, "state"), "w", encoding="utf-8") as f:
            f.write("new")
        if fail:
            real_replace = os.replace

            def replace(src, dst):
                if os.fspath(src) == stage:
                    raise OSError("simulated failure")
                real_replace(src, dst)

            monkeypatch.setattr(os, "replace", replace)
            with pytest.raises(OSError):
                ckpt_dcp._publish_partial(stage, str(final))
            monkeypatch.undo()
        else:
            ckpt_dcp._publish_partial(stage, str(final))
        assert (final / "state").read_text() == ("old" if fail else "new")
        assert sorted(os.listdir(tmp_path)) == (["step_3", "step_3.partial"] if fail else ["step_3"])


class TestRankLocalCheckpoint:
    """Node-local checkpoint disks: every rank saves / loads its own tensors (exact layout)."""

    def _model(self, seed):
        torch.manual_seed(seed)
        return torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Linear(4, 2)).to(torch.bfloat16)

    def test_roundtrip_is_bitwise(self, tmp_path):
        src, dst = self._model(0), self._model(1)
        stage = str(tmp_path / "step_5")
        os.makedirs(stage)
        local = {"step": 5, **{f"model.{n}": p.detach() for n, p in src.named_parameters()}}
        ckpt_dcp._save_rank_local_dcp(local, stage, None, trainable_only=False)
        step = ckpt_dcp._load_rank_local_dcp(
            dst, "model", stage, None, load_model=True, trainable_only=False
        )
        assert step == 5
        for (_, a), (_, b) in zip(src.named_parameters(), dst.named_parameters(), strict=True):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))

    def test_layout_mismatch_is_refused(self, tmp_path):
        import json

        src = self._model(0)
        stage = str(tmp_path / "step_5")
        os.makedirs(stage)
        ckpt_dcp._save_rank_local_dcp(
            {"step": 5, **{f"model.{n}": p.detach() for n, p in src.named_parameters()}},
            stage,
            None,
            trainable_only=False,
        )
        layout = os.path.join(ckpt_dcp._rank_local_dir(stage), ckpt_dcp._RANK_LOCAL_LAYOUT)
        rec = json.load(open(layout, encoding="utf-8"))
        rec["world_size"] = 16
        json.dump(rec, open(layout, "w", encoding="utf-8"))
        with pytest.raises(RuntimeError):
            ckpt_dcp._load_rank_local_dcp(
                self._model(1), "model", stage, None, load_model=True, trainable_only=False
            )

    def _lora_like(self, seed):
        m = self._model(seed)
        m[0].weight.requires_grad_(False)  # frozen base weight
        return m

    @pytest.mark.parametrize("trainable_only", [False, True], ids=["all-params", "trainable-only"])
    def test_trainable_only_roundtrip(self, tmp_path, trainable_only):
        src, dst = self._lora_like(0), self._lora_like(1)
        frozen_before = dst[0].weight.detach().clone()
        stage = str(tmp_path / "step_5")
        os.makedirs(stage)
        params = ckpt_dcp._rank_local_params(src, trainable_only)
        local = {"step": 5, **{f"model.{n}": p.detach() for n, p in params}}
        ckpt_dcp._save_rank_local_dcp(local, stage, None, trainable_only=trainable_only)
        ckpt_dcp._load_rank_local_dcp(
            dst, "model", stage, None, load_model=True, trainable_only=trainable_only
        )
        for (n, a), (_, b) in zip(src.named_parameters(), dst.named_parameters(), strict=True):
            if trainable_only and n == "0.weight":
                assert torch.equal(b.view(torch.uint8), frozen_before.view(torch.uint8))  # untouched
            else:
                assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        saved = set(dcp.FileSystemReader(ckpt_dcp._rank_local_dir(stage)).read_metadata().state_dict_metadata)
        assert ("model.0.weight" in saved) is (not trainable_only)

    @pytest.mark.parametrize("saved", [True, False], ids=["saved-trainable", "saved-all"])
    def test_trainable_only_flag_must_match(self, tmp_path, saved):
        src = self._lora_like(0)
        stage = str(tmp_path / "step_5")
        os.makedirs(stage)
        params = ckpt_dcp._rank_local_params(src, saved)
        ckpt_dcp._save_rank_local_dcp(
            {"step": 5, **{f"model.{n}": p.detach() for n, p in params}}, stage, None, trainable_only=saved
        )
        with pytest.raises(RuntimeError, match="layout"):
            ckpt_dcp._load_rank_local_dcp(
                self._lora_like(1), "model", stage, None, load_model=True, trainable_only=not saved
            )

    @pytest.mark.parametrize("trainable_only", [False, True], ids=["off-no-marker", "on-marker"])
    def test_layout_marks_trainable_only_when_on(self, tmp_path, trainable_only):
        import json

        src = self._lora_like(0)
        stage = str(tmp_path / "step_5")
        os.makedirs(stage)
        params = ckpt_dcp._rank_local_params(src, trainable_only)
        ckpt_dcp._save_rank_local_dcp(
            {"step": 5, **{f"model.{n}": p.detach() for n, p in params}},
            stage,
            None,
            trainable_only=trainable_only,
        )
        layout = os.path.join(ckpt_dcp._rank_local_dir(stage), ckpt_dcp._RANK_LOCAL_LAYOUT)
        rec = json.load(open(layout, encoding="utf-8"))
        assert rec.get("params") == ("trainable" if trainable_only else None)

    def test_checkpoint_written_before_the_flag_loads_with_it_off(self, tmp_path):
        import json

        src, dst = self._model(0), self._model(1)
        stage = str(tmp_path / "step_5")
        os.makedirs(stage)
        local = {"step": 5, **{f"model.{n}": p.detach() for n, p in src.named_parameters()}}
        ckpt_dcp._save_rank_local_dcp(local, stage, None, trainable_only=False)
        # the layout a checkpoint written before the flag existed carries: no "params" key
        layout = os.path.join(ckpt_dcp._rank_local_dir(stage), ckpt_dcp._RANK_LOCAL_LAYOUT)
        rec = json.load(open(layout, encoding="utf-8"))
        rec.pop("params", None)
        json.dump(rec, open(layout, "w", encoding="utf-8"))
        assert ckpt_dcp._load_rank_local_dcp(dst, "model", stage, None, load_model=True, trainable_only=False) == 5
        with pytest.raises(RuntimeError, match="layout"):
            ckpt_dcp._load_rank_local_dcp(dst, "model", stage, None, load_model=True, trainable_only=True)

    def _save_with_layout(self, tmp_path, pp_layout):
        from types import SimpleNamespace

        src = self._model(0)
        stage = str(tmp_path / "step_5")
        os.makedirs(stage)
        ps = SimpleNamespace(pp_size=2, pp_rank=1, pp_layout=pp_layout)
        local = {
            "step": 5,
            **{f"model.{n}": p.detach() for n, p in src.named_parameters()},
        }
        ckpt_dcp._save_rank_local_dcp(local, stage, ps, trainable_only=False)
        return stage, ps

    @pytest.mark.parametrize(
        "saved,current",
        [("Et|tL", "Ett|L"), ("Et|tL", None), (None, "Et|tL")],
        ids=["boundary-moved", "custom-to-auto", "auto-to-custom"],
    )
    def test_pp_layout_change_is_refused(self, tmp_path, saved, current):
        stage, ps = self._save_with_layout(tmp_path, saved)
        ps.pp_layout = current
        with pytest.raises(RuntimeError, match="layout"):
            ckpt_dcp._load_rank_local_dcp(
                self._model(1), "model", stage, ps, load_model=True, trainable_only=False
            )

    @pytest.mark.parametrize(
        "layout",
        ["Et|tL", [["embedding", "decoder"], ["decoder", "loss"]]],
        ids=["string", "list"],
    )
    def test_same_pp_layout_loads(self, tmp_path, layout):
        stage, ps = self._save_with_layout(tmp_path, layout)
        step = ckpt_dcp._load_rank_local_dcp(
            self._model(1), "model", stage, ps, load_model=True, trainable_only=False
        )
        assert step == 5
