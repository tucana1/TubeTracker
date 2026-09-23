"""rev13 W1 acceptance: strict metadata-driven loading everywhere.

- repeated loads under different global RNG states produce IDENTICAL
  operative weights and predictions for a fixed checkpoint;
- a fixture missing a required head (declared presence_head, absent
  weights) fails BEFORE inference;
- the app backend under the "strict" name never silently enables
  legacy semantics, resolves checkpoints against repo_root and
  reports unavailability explicitly.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")

CKPT = (REPO / "runs/prototypes/v30/rev12_propfit_smallcert.block2"
        ".weights.pt")


def _predict(model):
    from prototypes.v30_video_apex.model import build_owner_prompt
    torch.manual_seed(12345)  # different global state than load time
    clip = torch.rand(1, 9, 1, 64, 64)
    prompt = build_owner_prompt("o")
    route = [[8.0, 40.0], [56.0, 40.0]]
    with torch.no_grad():
        p = model.forward(clip, prompt, route_xy=route)
    return (float(p.front_logits.sum()), float(p.heat.sum()))


def test_repeated_loads_identical_under_different_rng():
    from prototypes.v30_video_apex.model_factory import (
        build_model_from_checkpoint)

    torch.manual_seed(0)
    m1, i1 = build_model_from_checkpoint(str(CKPT))
    p1 = _predict(m1)
    torch.manual_seed(9999)  # different global RNG state at load
    m2, i2 = build_model_from_checkpoint(str(CKPT))
    p2 = _predict(m2)
    assert i1["parameter_hash"] == i2["parameter_hash"]
    assert p1 == p2


def test_missing_required_head_fails_before_inference(tmp_path):
    from prototypes.v30_video_apex.model_factory import (
        build_model_from_checkpoint, CheckpointContractError)

    payload = torch.load(str(CKPT), map_location="cpu",
                         weights_only=False)
    bad = copy.deepcopy(payload)
    # declared presence_head=True stays in config; its weights vanish.
    bad["model_state"] = {k: v for k, v in payload["model_state"].items()
                          if "front_present" not in k}
    p = tmp_path / "missing_presence.pt"
    torch.save(bad, str(p))
    with pytest.raises(CheckpointContractError):
        build_model_from_checkpoint(str(p))


def test_backend_strict_no_silent_legacy(tmp_path):
    from tubetracker.pipeline_backends import load_backend

    payload = torch.load(str(CKPT), map_location="cpu",
                         weights_only=False)
    bad = copy.deepcopy(payload)
    # strip ALL declarations -> only a declared legacy replay could
    # accept it; the "strict" backend must refuse explicitly.
    for k in ("model_schema", "activation", "preprocessing"):
        bad.pop(k, None)
    bad["config"] = {k: v for k, v in payload["config"].items()
                     if k != "multiscale"}
    p = tmp_path / "undeclared.pt"
    torch.save(bad, str(p))
    fn, info = load_backend("v30-strict", checkpoint=str(p),
                            repo_root=str(REPO))
    assert fn is None and info["available"] is False
    assert "strict load refused" in info["reason"]


def test_backend_resolves_against_repo_root():
    from tubetracker.pipeline_backends import load_backend

    fn, info = load_backend("v30-strict", repo_root=str(REPO))
    assert fn is not None and info["available"] is True
    assert Path(info["checkpoint"]).is_absolute()
    assert Path(info["checkpoint"]).exists()
    assert info["parameter_hash"] and info["semantics"]
    # a bogus relative checkpoint resolves against repo_root, not CWD
    fn2, info2 = load_backend(
        "v30-strict", checkpoint="runs/does/not/exist.pt",
        repo_root=str(REPO))
    assert fn2 is None and info2["available"] is False
    assert str(REPO) in info2["checkpoint"]
