"""rev11: exercise the RUNNER's checkpoint boundary, not only the helper.

The rev11 audit found `run_v30_movie.main` granting the route-head
exception unconditionally (zeroing a present, trained route head) and
catching every `CheckpointContractError` to silently rebuild a different
architecture. These tests run the real entry point with
`--stop-after-load`, which loads the model exactly as a movie run would,
writes `load_provenance.json`, and exits before any image inference.

Required fixtures (all in-repo): the current best-front checkpoint
(schema 2), the legacy `rev9_wpB8/ep29.pt` (pre-contract), snap25 and
the frozen v1 weights. A movie path is parsed but never opened.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "run_v30_movie_entry", REPO / "scripts" / "run_v30_movie.py")
run_v30_movie = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(run_v30_movie)  # type: ignore[union-attr]

from prototypes.v30_video_apex.model_factory import (  # noqa: E402
    parameter_hash)

BEST_FRONT = (REPO / "runs/prototypes/v30/rev10_front_snap25"
              / "best_front_ep19.pt")
LEGACY_EP29 = REPO / "runs/prototypes/v30/rev9_wpB8/ep29.pt"
SNAP = REPO / "runs/prototypes/v30/snapshots/snap25"
V1 = REPO / "runs/prototypes/timesfm/tip_cnn_v1/best-point-heatmap-model.pt"
MOVIE = "ld=/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"

pytestmark = pytest.mark.skipif(
    not (BEST_FRONT.exists() and SNAP.exists() and V1.exists()),
    reason="runner fixtures (checkpoint/snap25/v1 weights) absent")


def _args(tmp_path, checkpoint, extra=()):
    return ["--front-checkpoint", str(checkpoint),
            "--v1-weights", str(V1),
            "--snapshot", str(SNAP),
            "--movie", MOVIE,
            "--event-ref", "r4-p",
            "--out-dir", str(tmp_path / "run"),
            "--stop-after-load", *extra]


def _run(tmp_path, checkpoint, extra=()):
    return run_v30_movie.main(_args(tmp_path, checkpoint, extra))


def _mutated(tmp_path, name, mutate):
    payload = torch.load(BEST_FRONT, map_location="cpu", weights_only=False)
    mutate(payload)
    p = tmp_path / name
    torch.save(payload, str(p))
    return p


def _provenance(tmp_path):
    return json.loads((tmp_path / "run" / "load_provenance.json").read_text())


def test_runner_preserves_the_present_trained_head(tmp_path):
    """The audit: STRICT default must keep every trained parameter.
    The recorded post-load hash must equal the checkpoint's own tensor
    hash — exact, not approximate."""
    rc = _run(tmp_path, BEST_FRONT)
    assert rc == 0
    prov = _provenance(tmp_path)["front_load"]
    assert prov["mode"] == "strict-current"
    assert prov["route_head_fallback"] is None
    assert prov["missing_keys"] == [] and prov["unexpected_keys"] == []
    payload = torch.load(BEST_FRONT, map_location="cpu", weights_only=False)
    assert prov["parameter_hash_post_load"] == \
        parameter_hash(payload["model_state"])
    assert prov["manifest"].get("epoch") == 19


def test_runner_preserves_with_the_gap_flag_granted(tmp_path):
    """The exact regression: with --allow-route-head-gap the OLD runner
    zeroed the head. It must now be a no-op for a present head."""
    rc = _run(tmp_path, BEST_FRONT, ["--allow-route-head-gap"])
    assert rc == 0
    prov = _provenance(tmp_path)["front_load"]
    assert prov["route_head_fallback"] is None
    payload = torch.load(BEST_FRONT, map_location="cpu", weights_only=False)
    assert prov["parameter_hash_post_load"] == \
        parameter_hash(payload["model_state"])


def test_runner_legacy_checkpoint_refused_without_the_migration(tmp_path):
    if not LEGACY_EP29.exists():
        pytest.skip("legacy ep29 absent")
    rc = _run(tmp_path, LEGACY_EP29)
    assert rc == 1
    assert not (tmp_path / "run" / "load_provenance.json").exists()


def test_runner_legacy_migration_is_explicit_and_recorded(tmp_path):
    if not LEGACY_EP29.exists():
        pytest.skip("legacy ep29 absent")
    rc = _run(tmp_path, LEGACY_EP29, ["--allow-legacy-migration", "--base", "8"])
    assert rc == 0
    prov = _provenance(tmp_path)["front_load"]
    assert prov["mode"] == "legacy-migration"
    assert prov["parameter_hash_post_load"]
    assert prov["semantics"].startswith("legacy-declared-replay")


@pytest.mark.parametrize("case", ["missing-body", "partial-route-head",
                                  "wrong-schema", "wrong-preprocessing",
                                  "wrong-activation"])
def test_runner_refuses_every_other_contract_failure(tmp_path, case):
    """No fallback for anything that is not a legacy declaration."""
    def mutate_missing_body(p):
        del p["model_state"]["body.weight"]

    def mutate_partial(p):
        del p["model_state"]["route_head.weight"]

    def mutate_schema(p):
        p["model_schema"] = 999

    def mutate_pre(p):
        p["preprocessing"] = "native/255 grayscale, 1-frame window"

    def mutate_act(p):
        p["activation"] = "relu"

    mutators = {"missing-body": mutate_missing_body,
                "partial-route-head": mutate_partial,
                "wrong-schema": mutate_schema,
                "wrong-preprocessing": mutate_pre,
                "wrong-activation": mutate_act}
    ck = _mutated(tmp_path, f"{case}.pt", mutators[case])
    # even with every permissive flag on, a non-legacy failure is final
    rc = _run(tmp_path, ck, ["--allow-legacy-migration", "--base", "8",
                             "--allow-route-head-gap"])
    assert rc == 1
    assert not (tmp_path / "run" / "load_provenance.json").exists()
