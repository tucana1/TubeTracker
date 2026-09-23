"""rev9 WP-A.3: one builder for trainer, dev probe and evaluator.

The audit found the dev probe rebuilding body targets that the training
path had already built, and an evaluator re-deriving the temporal
window. These tests pin the shared entry point and the anti-drift
guarantees: identical arrays from both entry points, stable label-source
ids, a single temporal window, and no nested re-implementation left in
the trainer/evaluator sources.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from prototypes.v30_video_apex.batch_builder import (
    QUERY_INDEX, QUERY_OFFSETS, BodyTargets, linked_mask_target,
    own_mask_target)
from prototypes.v30_video_apex.dataset import QUERY_OFFSETS as DS_OFFSETS
from prototypes.v30_video_apex.targets import (
    SampleDef, body_mask_from_points, body_mask_from_raster,
    encode_mask_raster, samples_from_snapshot)

REPO = Path(__file__).resolve().parents[1]
SNAP = "runs/prototypes/v30/snapshots/snap20"


def _raster_mask(cx=500.0, cy=430.0, r=8, canvas=900):
    m = np.zeros((canvas, canvas), dtype=bool)
    yy, xx = np.mgrid[0:canvas, 0:canvas]
    m[np.hypot(xx - cx, yy - cy) <= r] = True
    return m


def test_query_index_is_the_zero_offset():
    """One temporal window, defined once (rev9 evaluator mismatch)."""
    assert QUERY_OFFSETS == DS_OFFSETS
    assert QUERY_INDEX == QUERY_OFFSETS.index(0)
    assert QUERY_INDEX == 4 and len(QUERY_OFFSETS) == 9


def test_own_mask_target_matches_the_underlying_paint_paths():
    native = _raster_mask()
    raster = encode_mask_raster(native)
    origin = (380.0, 320.0)
    good = [[400.0, 380.0], [620.0, 520.0]]
    s = SampleDef(entry_id="t", movie="ld", movie_path="x.mp4",
                  source_frame=1, kind="body_mask",
                  mask_uuid="mask-x", mask_raster=raster, brush_px=9.0,
                  complete=True, review_region=good)
    b = own_mask_target(s, 288, 288, origin)
    t_ref, v_ref = body_mask_from_raster(288, 288, origin, raster,
                                         complete=True, review_region=good)
    assert np.array_equal(b.target, t_ref)
    assert np.array_equal(b.valid, v_ref)
    assert b.source == "own-mask-raster:mask-x"
    t, v = b.tensors()
    assert tuple(t.shape) == (1, 1, 288, 288)

    # the stamps path goes through the same entry point
    s2 = SampleDef(entry_id="t2", movie="ld", movie_path="x.mp4",
                   source_frame=1, kind="body_mask", mask_uuid="mask-y",
                   mask_points=[[500.0, 430.0], [505.0, 430.0]],
                   brush_px=9.0, complete=True, review_region=good)
    b2 = own_mask_target(s2, 288, 288, origin)
    t_ref2, v_ref2 = body_mask_from_points(
        288, 288, origin, s2.mask_points, brush_px=9.0, complete=True,
        review_region=good)
    assert np.array_equal(b2.target, t_ref2)
    assert np.array_equal(b2.valid, v_ref2)
    assert b2.source == "own-mask-stamps:mask-y"


def test_linked_mask_source_ids_keep_their_historical_spelling():
    native = _raster_mask()
    rec = {"mask_uuid": "mask-rev8p-000",
           "mask_raster": encode_mask_raster(native),
           "complete": False, "review_region": []}
    b = linked_mask_target(288, 288, (380.0, 320.0), rec, link="explicit")
    assert b.source == "raster:mask-rev8p-000|explicit"
    rec2 = {"mask_uuid": "mask-old", "painted_xy": [[500.0, 430.0]],
            "brush_px": 9.0, "complete": False, "review_region": []}
    b2 = linked_mask_target(288, 288, (380.0, 320.0), rec2, link="proximity")
    assert b2.source == "stamps-legacy:mask-old|proximity"


def test_quarantine_and_readout_are_visible_through_the_builder():
    native = _raster_mask()
    s = SampleDef(entry_id="t", movie="ld", movie_path="x.mp4",
                  source_frame=1, kind="body_mask", mask_uuid="mask-z",
                  mask_raster=encode_mask_raster(native), brush_px=9.0,
                  complete=True,
                  review_region=[[390.0, 330.0], [580.0, 400.0]])
    b = own_mask_target(s, 288, 288, (380.0, 320.0))
    r = b.readout()
    assert r["quarantine_reason"] == "misses-paint"
    assert r["extent_used"] is False and r["reviewed_bg_px"] == 0
    assert r["paint_px"] > 0


def test_real_snap20_mask_through_the_builder():
    samples = [s for s in samples_from_snapshot(SNAP)
               if s.kind == "body_mask" and s.mask_raster
               and s.review_region]
    assert samples, "no mask samples to check"
    for s in samples:
        r_ = s.mask_raster
        b = own_mask_target(
            s, 288, 288,
            (float(r_["x0"]) + float(r_["w"]) / 2.0 - 144.0,
             float(r_["y0"]) + float(r_["h"]) / 2.0 - 144.0))
        assert b.extent_used is False, s.obs_uuid
        assert b.bg_reviewed.sum() == 0
        assert b.quarantine_reason == "misses-paint"


@pytest.mark.parametrize("script", ["scripts/train_v30_front.py",
                                    "scripts/eval_whole_instance.py"])
def test_no_nested_target_reconstruction_left(script):
    """A direct paint-target call outside the shared builder is drift.

    WP-A.3 retires the nested dev-target reconstruction; if it (or an
    evaluator equivalent) comes back, this fails.
    """
    src = (REPO / script).read_text()
    body = re.sub(r"^\s*#.*$", "", src, flags=re.M)
    offenders = []
    for name in ("body_mask_from_raster(", "body_mask_from_points("):
        for m in re.finditer(re.escape(name), body):
            line = body[:m.start()].count("\n") + 1
            offenders.append((name, line))
    assert not offenders, (
        f"{script} builds paint targets directly at {offenders}; use "
        f"prototypes.v30_video_apex.batch_builder instead")
