"""rev9 WP-A.4: a checkpoint must be able to rebuild its own model.

The audit reproduced `loads: false` for the movie runner (it always
constructed a non-multiscale model at --base, so a current checkpoint
could not load), an unchecked `strict=False` in the trainer reload, and
a config missing ten knobs. These tests pin the contract: declared
architecture, strict weights, named optimizer groups, and an explicit
declared exception for the one legitimate legacy gap.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from prototypes.v30_video_apex.model import build_model
from prototypes.v30_video_apex.model_factory import (
    CURRENT_ACTIVATION, CURRENT_MODEL_SCHEMA, CURRENT_PREPROCESSING,
    CheckpointContractError, build_model_from_checkpoint,
    checkpoint_config, model_kwargs_from_config, optimizer_groups_summary,
    parameter_hash, read_checkpoint_payload)
from prototypes.v30_video_apex.train import save_checkpoint

REPO = Path(__file__).resolve().parents[1]
LEARN3 = REPO / "runs/prototypes/v30/learn3/front.pt"
BEST_FRONT = (REPO / "runs/prototypes/v30/rev10_front_snap25"
              / "best_front_ep19.pt")

# The ten knobs the audit found omitted from saved configs.
AUDITED_KEYS = ("multiscale", "mask_jitter", "body_objective",
                "body_pixel_bce", "head_lr", "prompt_mix", "only_refs",
                "confusable_weight", "confusable_mode", "head_weights")


def _complete_config(**over):
    cfg = {"variant": "temporal", "base": 4, "lr": 0.003,
           "multiscale": True, "mask_jitter": 6.0,
           "body_objective": "dice_pixel", "body_pixel_bce": 0.1,
           "head_lr": 0.5, "prompt_mix": "auto:1.0",
           "only_refs": "mask-rev8m-000", "confusable_weight": 0.0,
           "confusable_mode": "balance", "head_weights": "",
           "optimizer_group_lrs": {"head": 0.5, "trunk": 0.003}}
    cfg.update(over)
    return cfg


def _save(tmp_path, name="ck.pt", cfg=None, model=None, drop=(),
          semantics=True):
    model = model or build_model("temporal", base=4, multiscale=True)
    state = {k: v for k, v in model.state_dict().items() if k not in drop}
    path = tmp_path / name
    payload = {"model_state": state, "config": cfg or _complete_config(),
               "config_hash": "test", "manifest": {}}
    if semantics:
        # rev10 WP-A: a checkpoint must declare its activation semantics;
        # ReLU and LeakyReLU keep identical shapes but differ in function.
        # rev11: preprocessing is validated too, so fixtures declare it.
        from prototypes.v30_video_apex.model_factory import (
            CURRENT_ACTIVATION, CURRENT_MODEL_SCHEMA,
            CURRENT_PREPROCESSING)
        payload["model_schema"] = int(CURRENT_MODEL_SCHEMA)
        payload["activation"] = str(CURRENT_ACTIVATION)
        payload["preprocessing"] = str(CURRENT_PREPROCESSING)
    torch.save(payload, str(path))
    return path, model


def test_factory_rebuilds_and_loads_strictly(tmp_path):
    path, _m = _save(tmp_path)
    model, info = build_model_from_checkpoint(path)
    assert info["missing_keys"] == [] and info["unexpected_keys"] == []
    assert info["model_kwargs"] == {"variant": "temporal", "base": 4,
                                    "multiscale": True}
    # the rebuilt model reproduces the saved weights exactly
    _m2 = build_model("temporal", base=4, multiscale=True)
    report = _m2.load_state_dict(torch.load(path,
                                            map_location="cpu",
                                            weights_only=False)["model_state"],
                                 strict=True)
    assert report.missing_keys == [] and report.unexpected_keys == []


def test_complete_config_records_every_audited_knob(tmp_path):
    cfg = _complete_config()
    for k in AUDITED_KEYS:
        assert k in cfg, f"{k} missing from the contract config"
    path, _m = _save(tmp_path, cfg=cfg)
    loaded = checkpoint_config(read_checkpoint_payload(path))
    for k in AUDITED_KEYS:
        assert k in loaded
    assert optimizer_groups_summary(loaded) == "head=0.5, trunk=0.003"
    assert isinstance(loaded["optimizer_group_lrs"], dict)


def test_checkpoint_without_multiscale_is_refused(tmp_path):
    cfg = {k: v for k, v in _complete_config().items()
           if k != "multiscale"}
    path, _m = _save(tmp_path, cfg=cfg)
    with pytest.raises(CheckpointContractError) as e:
        build_model_from_checkpoint(path)
    assert "multiscale" in str(e.value)


def test_weights_that_do_not_fit_are_refused_not_silently_loaded(tmp_path):
    """The old `strict=False` would have accepted this silently."""
    small = build_model("temporal", base=4, multiscale=True)
    path = tmp_path / "mismatch.pt"
    from prototypes.v30_video_apex.model_factory import (
        CURRENT_ACTIVATION, CURRENT_MODEL_SCHEMA)
    torch.save({"model_state": small.state_dict(),
                "config": _complete_config(base=8),
                "config_hash": "x", "manifest": {},
                "model_schema": int(CURRENT_MODEL_SCHEMA),
                "activation": str(CURRENT_ACTIVATION),
                "preprocessing": str(CURRENT_PREPROCESSING)}, str(path))
    with pytest.raises(CheckpointContractError) as e:
        build_model_from_checkpoint(path)
    assert "incompatible" in str(e.value)


def test_route_head_gap_is_declared_and_zeroed(tmp_path):
    model = build_model("temporal", base=4, multiscale=True)
    drop = tuple(k for k in model.state_dict() if "route_head" in k)
    assert drop, "expected route_head keys in a current model"
    path, _m = _save(tmp_path, name="noroute.pt", model=model, drop=drop)
    model2, info = build_model_from_checkpoint(
        path, allow_route_head_gap=True)
    assert info["missing_keys"], "the gap must be reported"
    assert all("route_head" in k for k in info["missing_keys"])
    assert info["route_head_fallback"] == "zeroed-absent-legacy-head"
    for p in model2.route_head.parameters():
        assert float(p.detach().abs().sum()) == 0.0
    # and without the declared exception it is refused
    with pytest.raises(CheckpointContractError):
        build_model_from_checkpoint(path)


@pytest.mark.skipif(not LEARN3.exists(), reason="learn3 checkpoint absent")
def test_historical_checkpoint_cannot_declare_its_architecture():
    """learn3 predates the contract: the factory refuses it rather than
    guessing, which is exactly why the runner's legacy path is explicit.
    """
    cfg = checkpoint_config(read_checkpoint_payload(LEARN3))
    assert "multiscale" not in cfg
    assert any(k not in cfg for k in AUDITED_KEYS)
    with pytest.raises(CheckpointContractError):
        build_model_from_checkpoint(LEARN3)
    # the CLI-declared legacy fallback still works, loudly
    kwargs = model_kwargs_from_config(dict(cfg, multiscale=False))
    assert kwargs["multiscale"] is False


def test_undeclared_semantics_are_refused_unless_declared_replay(tmp_path):
    """rev10: identical weight shapes do not make identical models."""
    path, _m = _save(tmp_path, name="old.pt", semantics=False)
    with pytest.raises(CheckpointContractError) as e:
        build_model_from_checkpoint(path)
    assert "activation" in str(e.value) or "model_schema" in str(e.value)
    # ... and an explicitly declared replay loads it
    model, info = build_model_from_checkpoint(
        path, allow_legacy_semantics=True)
    assert info["semantics"] == "legacy-declared-replay"
    assert info["missing_keys"] == []


def test_a_declared_activation_that_differs_is_refused(tmp_path):
    path, _m = _save(tmp_path, name="relu.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["activation"] = "relu"
    torch.save(payload, str(path))
    with pytest.raises(CheckpointContractError) as e:
        build_model_from_checkpoint(path)
    assert "relu" in str(e.value)
    # even the legacy-replay flag does not silently accept a WRONG
    # declared activation: that checkpoint must be replayed under its own
    # architecture, not this one.
    with pytest.raises(CheckpointContractError):
        build_model_from_checkpoint(path, allow_legacy_semantics=True)


def test_the_legacy_gap_covers_route_head_keys_only(tmp_path):
    """The audit's fixture case: a checkpoint missing `body.weight` must
    be refused even when the route-head exception is granted."""
    model = build_model("temporal", base=4, multiscale=True)
    path, _m = _save(tmp_path, name="missing_body.pt", model=model,
                     drop=("body.weight",))
    with pytest.raises(CheckpointContractError) as e:
        build_model_from_checkpoint(path, allow_route_head_gap=True)
    msg = str(e.value)
    assert "body.weight" in msg and "route_head" in msg
    # dropping ONLY the route head is still the one permitted gap
    drop = tuple(k for k in model.state_dict() if "route_head" in k)
    path2, _m2 = _save(tmp_path, name="gap.pt", model=model, drop=drop)
    model2, info2 = build_model_from_checkpoint(path2,
                                                allow_route_head_gap=True)
    # the gap is REPORTED (exactly the route-head keys), not swallowed
    assert sorted(info2["missing_keys"]) == sorted(
        k for k in drop)
    assert all("route_head" in k for k in info2["missing_keys"])
    for prm in model2.route_head.parameters():
        assert float(prm.detach().abs().sum()) == 0.0, \
            "the gap must be zeroed loud"


# ------------------------------------------------------------------ rev11
def test_partial_route_head_is_refused_even_with_the_declared_gap(tmp_path):
    """rev11: the declared legacy gap is the COMPLETE key set. A
    checkpoint carrying only half a route head has no declared meaning
    and must be refused, with or without the exception flag."""
    model = build_model("temporal", base=4, multiscale=True)
    path, _m = _save(tmp_path, name="partial.pt", model=model,
                     drop=("route_head.weight",))
    for kwargs in ({}, {"allow_route_head_gap": True}):
        with pytest.raises(CheckpointContractError) as e:
            build_model_from_checkpoint(path, **kwargs)  # type: ignore[arg-type]
        assert "partial" in str(e.value) or "route_head" in str(e.value)


def test_wrong_schema_is_refused_not_merely_recorded(tmp_path):
    """The audit: a synthetic schema-999 checkpoint was accepted by both
    factory and runner. It must now be refused; no flag rescues it."""
    path, _m = _save(tmp_path, name="schema999.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model_schema"] = 999
    torch.save(payload, str(path))
    with pytest.raises(CheckpointContractError) as e:
        build_model_from_checkpoint(path)
    assert "999" in str(e.value)
    with pytest.raises(CheckpointContractError):
        build_model_from_checkpoint(path, allow_legacy_semantics=True)


def test_wrong_preprocessing_is_refused(tmp_path):
    path, _m = _save(tmp_path, name="pre.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["preprocessing"] = "native/255 grayscale, 1-frame window"
    torch.save(payload, str(path))
    with pytest.raises(CheckpointContractError) as e:
        build_model_from_checkpoint(path)
    assert "preprocessing" in str(e.value)
    with pytest.raises(CheckpointContractError):
        build_model_from_checkpoint(path, allow_legacy_semantics=True)


@pytest.mark.skipif(not BEST_FRONT.exists(),
                    reason="best-front checkpoint absent")
def test_trained_route_head_survives_both_flag_states_exactly():
    """rev11 core case (loader audit): the current best checkpoint keeps
    its trained route head with the legacy-gap flag ON as well as OFF.
    Exact tensor comparison — the audited abs sum 1.2866277694702148 /
    bias 0.2249743789434433 come out of the tensors themselves."""
    payload = torch.load(BEST_FRONT, map_location="cpu", weights_only=False)
    want_w = payload["model_state"]["route_head.weight"]
    want_b = payload["model_state"]["route_head.bias"]
    for flag in (False, True):
        model, info = build_model_from_checkpoint(
            BEST_FRONT, allow_route_head_gap=flag)
        got = dict(model.named_parameters())
        assert torch.equal(got["route_head.weight"].detach(), want_w), \
            f"flag={flag} altered the trained route weights"
        assert torch.equal(got["route_head.bias"].detach(), want_b), \
            f"flag={flag} altered the trained route bias"
        assert info["route_head_fallback"] is None
        assert info["missing_keys"] == [] and info["unexpected_keys"] == []
    assert float(got["route_head.weight"].detach().abs().sum()) == \
        1.2866277694702148
    assert float(got["route_head.bias"].detach()) == 0.2249743789434433


@pytest.mark.skipif(not BEST_FRONT.exists(),
                    reason="best-front checkpoint absent")
def test_parameter_hash_reports_the_post_load_state():
    """The runner's provenance hash must equal the checkpoint payload's
    own tensor hash — a loader that rewrites trained weights cannot
    pass this comparison."""
    payload = torch.load(BEST_FRONT, map_location="cpu", weights_only=False)
    model, info = build_model_from_checkpoint(BEST_FRONT)
    assert info["parameter_hash"] == parameter_hash(model)
    assert info["parameter_hash"] == parameter_hash(payload["model_state"])
    assert info["manifest"].get("epoch") == 19
    assert info["config_hash"]
