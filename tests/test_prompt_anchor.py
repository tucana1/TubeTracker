"""rev8: a mask query must be anchored on its GRAIN, and must never
train query-blind.

Measured failure this pins: on the paired clump the "auto-grain" prompt
disc was drawn at the PAINT CENTROID, which sits 25-30px off the grain
(the disc landed on tube body), and 60% of body-mask queries trained
with no query geometry at all ("human" needs an attachment path that
mask samples do not have). With the clump trained, the swap matrix came
back with IDENTICAL rows (max row spread 0.000) — the head had learned
"where tubes are", not "which tube belongs to my grain".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.model import resolve_prompt_kind  # noqa: E402
from prototypes.v30_video_apex.targets import (  # noqa: E402
    encode_mask_raster, samples_from_snapshot)


def test_kinds_fall_back_to_what_a_sample_can_render():
    # a body mask has a grain but no attachment path
    assert resolve_prompt_kind("human", False, True) == "auto-grain"
    # a path sample has both
    assert resolve_prompt_kind("human", True, True) == "human"
    assert resolve_prompt_kind("auto-grain", True, False) == "human"
    # explicit zeros stay zeros, and nothing renders if the sample has
    # neither geometry (never a silently wrong prompt)
    assert resolve_prompt_kind("none", True, True) == "none"
    assert resolve_prompt_kind("human", False, False) == "none"
    assert resolve_prompt_kind("auto-grain", False, False) == "none"


def test_mask_sample_anchors_on_the_grain_not_the_paint(tmp_path):
    """The grain is the query AND the crop anchor: a 288 crop centred on
    the tube's middle frames the tube the way training never will at
    deployment (where the crop is centred on the grain)."""
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "observations.json").write_text("[]")
    (snap / "regions.json").write_text("[]")
    (snap / "snapshot_manifest.json").write_text(
        '{"movies": {"ld": {"path": "/m.mp4"}}}')
    paint = np.zeros((120, 120), bool)
    paint[70:76, 70:110] = True                 # a tube running right
    row = {
        "mask_uuid": "mask-000", "mask_revision": 1, "task_uuid": "rev8p-000",
        "project": "p", "movie": "ld", "movie_path": "/m.mp4",
        "source_frame": 100, "painted_xy": [], "brush_px": 9.0,
        "complete": True, "mask_raster": encode_mask_raster(paint),
        "source_obs_uuid": "", "review_region": [[0.0, 0.0], [120.0, 120.0]],
        "owner_uuid": "rev8p-100-g0", "owner_key": "ld|rev8p-100-g0",
        "tube_uuid": "rev8p-100-g0",
        "target_xy": [71.0, 73.0], "target_r": 14.0, "annotator": "t"}
    (snap / "body_masks.json").write_text(
        __import__("json").dumps([row]))
    got = [s for s in samples_from_snapshot(str(snap))
           if s.kind == "body_mask"]
    assert len(got) == 1
    s = got[0]
    # paint centroid would be (~89.5, 73); the grain is (71, 73)
    assert s.focus_xy == (71.0, 73.0), s.focus_xy
    assert s.target_xy == (71.0, 73.0)
    # and the auto-grain disc the trainer draws sits on the grain
    cx, cy = s.focus_xy
    assert abs(cx - 89.5) > 15.0, "must not anchor on the paint centroid"
