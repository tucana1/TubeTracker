"""rev11 item 4: the typed prompt contract.

The audit: training drew an "auto-grain" disc at the annotation focus
while runtime drew one at the DETECTED grain center — centers 45–68 px
apart under one label. The fix is a typed constructor
(`inference.build_typed_prompt`) with one call site per consumer
(`train_v30_front.sample_prompt`, `run_v30_movie.deployment_prompt`,
`inference.build_query_prompt`), whose provenance records which declared
geometry was actually used.

These tests pin:
  * focus/gaze points are NEVER labelled auto-grain;
  * the type priority (human-grain > auto-grain > query-declared > focus);
  * matched movie/frame/owner/crop produces IDENTICAL prompt tensors in
    training and runtime (real r4-p00 pixels), and the evaluation entry
    declares its own geometry type;
  * consumers do not hand-roll grain discs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.inference import (  # noqa: E402
    build_query_prompt, build_typed_prompt, grain_disc_mask)

SNAP25 = REPO / "runs/prototypes/v30/snapshots/snap25"
FROZEN = REPO / "runs/prototypes/v30/rev10_front20/samples.json"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_focus_is_never_labelled_auto_grain():
    p, prov = build_typed_prompt("o", crop_wh=(64, 64),
                                 focus_xy=(10.0, 10.0))
    assert prov["prompt_kind"] == "focus"
    assert p.provenance == "focus"
    assert p.grain_mask is not None
    # auto-grain only ever comes from a detected center
    p2, prov2 = build_typed_prompt("o", crop_wh=(64, 64),
                                   detected_grain_xy=(20.0, 20.0),
                                   detected_radius_px=13.0)
    assert prov2["prompt_kind"] == "auto-grain"
    assert p2.provenance == "auto-grain"


def test_type_priority_is_declared_order():
    p, prov = build_typed_prompt(
        "o", crop_wh=(64, 64), human_grain_xy=(5.0, 5.0),
        detected_grain_xy=(20.0, 20.0), query_xy=(30.0, 30.0),
        focus_xy=(40.0, 40.0))
    assert prov["prompt_kind"] == "human-grain"
    p, prov = build_typed_prompt(
        "o", crop_wh=(64, 64), detected_grain_xy=(20.0, 20.0),
        query_xy=(30.0, 30.0), focus_xy=(40.0, 40.0))
    assert prov["prompt_kind"] == "auto-grain"
    p, prov = build_typed_prompt("o", crop_wh=(64, 64),
                                 query_xy=(30.0, 30.0),
                                 focus_xy=(40.0, 40.0))
    assert prov["prompt_kind"] == "query-declared"
    p, prov = build_typed_prompt("o", crop_wh=(64, 64), focus_xy=(40.0, 40.0))
    assert prov["prompt_kind"] == "focus"
    p, prov = build_typed_prompt("o", crop_wh=(64, 64))
    assert prov["prompt_kind"] == "none" and p.grain_mask is None


def test_disc_renderer_translation_and_radius():
    m = grain_disc_mask(64, 64, (32.0, 32.0), 10.0)
    assert m[32, 32] == 1.0
    assert m[32, 42] == 1.0        # distance exactly 10 -> inclusive
    assert m[32, 43] == 0.0        # distance 11 -> outside
    m2 = grain_disc_mask(64, 64, (52.0, 32.0), 10.0)
    assert np.array_equal(np.roll(m, 20, axis=1), m2)


def test_human_geometry_only_renders_the_human_arm():
    p, prov = build_typed_prompt(
        "o", crop_wh=(64, 64), attachment_xy=(10.0, 10.0),
        proximal_xy=[(10.0, 10.0), (12.0, 9.0)])
    assert prov["prompt_kind"] == "human"
    assert p.grain_mask is None
    assert p.attachment_xy == (10.0, 10.0)
    assert p.proximal_xy == [(10.0, 10.0), (12.0, 9.0)]


def test_evaluation_entry_declares_its_query_point():
    p = build_query_prompt("owner|g1", (16.0, 16.0), (64, 64))
    assert p.provenance == "query-declared"
    assert p.grain_mask is not None


def test_consumers_do_not_hand_roll_discs():
    """Drift guard: the grain disc is rendered in exactly one module."""
    for name in ("scripts/run_v30_movie.py", "scripts/train_v30_front.py"):
        src = (REPO / name).read_text()
        assert "mgrid" not in src, f"{name} hand-rolls a disc"
        assert ("build_typed_prompt(" in src or "deployment_prompt(" in src
                or "sample_prompt(" in src), name


@pytest.mark.skipif(not (SNAP25.exists() and FROZEN.exists()),
                    reason="snapshot/frozen samples absent")
def test_matched_inputs_identical_tensors_training_vs_runtime():
    """Real pixels, r4-p00: the trainer's construction and the runtime's
    construction must agree exactly for the same movie/frame/owner/crop.
    (Before rev11 they were a focus disc vs a detected disc, 45+ px
    apart, under one label.)"""
    from prototypes.v30_video_apex.targets import samples_from_snapshot
    from prototypes.v30_video_apex import routes as R
    from tubetracker.annotation_frames import FrameReader

    snap = SNAP25
    rows = samples_from_snapshot(str(snap))
    s = next((x for x in rows if str(x.tube_ref) == "obs-r4-p00"
              and x.kind == "path_tip"), None)
    assert s is not None
    fr = json.loads(FROZEN.read_text())
    e = next((x for x in fr
              if x.get("entry_id") == "ld|obs-r4-p00|42000"), None)
    assert e is not None
    ox, oy = int(e["crop_xywh"][0]), int(e["crop_xywh"][1])
    man = json.loads((snap / "snapshot_manifest.json").read_text())
    movie = man["movies"]["ld"]["path"]
    if not Path(movie).exists():
        pytest.skip("raw movie absent")
    r = FrameReader(movie)
    try:
        frame = r.read(int(s.source_frame)).frame
    finally:
        r.close()
    gray = np.asarray(frame[:, :, 0] if frame.ndim == 3 else frame)

    anchor = (float(s.path_xy[0][0]), float(s.path_xy[0][1]))
    rgc, rgr, rsrc = R.auto_grain(gray, anchor)

    tvf = _load_script("tvf_parity", REPO / "scripts" / "train_v30_front.py")
    rvm = _load_script("rvm_parity", REPO / "scripts" / "run_v30_movie.py")
    crop = gray[oy:oy + 288, ox:ox + 288]
    clip = np.stack([np.zeros_like(crop)] * 4 + [crop]
                    + [np.zeros_like(crop)] * 4)
    pt, prov_t, pk = tvf.sample_prompt(s, clip, ox, oy, 288, 288,
                                       "auto-grain", False)
    pr, prov_r = rvm.deployment_prompt(s.obs_uuid, ox, oy, 288, rgc, rgr)

    assert pk == "auto-grain" == prov_t["prompt_kind"]
    assert prov_r["prompt_kind"] == "auto-grain"
    assert prov_t["center_native"] == prov_r["center_native"]
    assert prov_t["radius_px"] == prov_r["radius_px"]
    assert np.array_equal(pt.grain_mask, pr.grain_mask)
    assert pt.provenance == pr.provenance == "auto-grain"
    assert pt.attachment_xy is None and pt.proximal_xy is None
    assert pt.owner_id == pr.owner_id
    # the detected center is NOT the annotation focus (that was the bug)
    focus = (float(s.focus_xy[0]), float(s.focus_xy[1]))
    d_focus = float(np.hypot(prov_t["center_native"][0] - focus[0],
                             prov_t["center_native"][1] - focus[1]))
    assert d_focus > 5.0, ("this event's focus and detected center "
                           "coincide; pick another for the anti-label "
                           "check")


@pytest.mark.skipif(not SNAP25.exists(), reason="snapshot absent")
def test_absent_sample_renders_none_not_a_guess():
    from prototypes.v30_video_apex.targets import samples_from_snapshot
    rows = samples_from_snapshot(str(SNAP25))
    s = next((x for x in rows if x.kind == "body_mask"
              and not x.focus_xy), None)
    if s is None:
        pytest.skip("every mask carries a focus point")
    tvf = _load_script("tvf_none", REPO / "scripts" / "train_v30_front.py")
    clip = np.zeros((9, 288, 288), np.float32)
    prompt, prov, pk = tvf.sample_prompt(s, clip, 100, 100, 288, 288,
                                         "auto-grain", False)
    assert prov["prompt_kind"] == "none"
    assert prompt.grain_mask is None
