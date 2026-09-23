"""rev9 WP-A.1: invalid reviewed extents must licence nothing.

The reviewer's contracts-audit reproduced four learn3 training views in
which a bad historical extent (H306 shape: y symmetric about 0) landed
inside the crop and turned unreviewed pixels into supervised
background (1300 / 644 / 1080 / 2475 px). These tests pin the fix: the
extent is quarantined before crop translation, so the mask falls back
to band-only validity, and the channels (fg / reviewed bg / unknown)
are explicit.

Coordinates: paint rasters are stored in NATIVE frame coords and the
loader crops (encode_mask_raster docstring); tests therefore build
native paint and decode it through a crop origin.
"""
from __future__ import annotations

import numpy as np
import pytest

from prototypes.v30_video_apex.targets import (
    _region_in_crop, _review_extent_background,
    body_mask_channels_from_raster, body_mask_from_points,
    body_mask_from_raster, decode_mask_raster, encode_mask_raster,
    review_region_usable, samples_from_snapshot)

SNAP = "runs/prototypes/v30/snapshots/snap20"
ORIGIN = (380.0, 320.0)          # crop covers native x 380-668, y 320-608
PAINT = (500.0, 430.0)           # native blob centre -> crop (120, 110)
PAINT_BBOX = (492.0, 422.0, 508.0, 438.0)
BAD = [[390.0, 330.0], [580.0, 400.0]]      # in-crop, misses the paint
GOOD = [[400.0, 380.0], [620.0, 520.0]]     # encloses the paint


def _paint_native(cx=PAINT[0], cy=PAINT[1], r=8, canvas=900):
    m = np.zeros((canvas, canvas), dtype=bool)
    yy, xx = np.mgrid[0:canvas, 0:canvas]
    m[np.hypot(xx - cx, yy - cy) <= r] = True
    return m


def _paint_raster(**kw):
    native = _paint_native(**kw)
    return native, encode_mask_raster(native)


def _paint_in_crop(h=288, w=288, origin=ORIGIN, **kw):
    _n, r = _paint_raster(**kw)
    return decode_mask_raster(h, w, origin, r)


def test_bad_extent_that_would_have_licensed_is_quarantined():
    """The H306 shape: a band at the top of the crop, paint below it.

    Before rev9 this rectangle rasterised into the crop and licensed
    every pixel inside it as reviewed background. Now it must be
    rejected, with the reason surfaced and nothing licensed.
    """
    _n, raster = _paint_raster()
    paint = _paint_in_crop()
    # the rectangle really does land inside this crop (the defect was
    # not hypothetical): prove it before asserting the quarantine
    would_have = _region_in_crop(288, 288, ORIGIN, BAD)
    assert would_have.sum() > 0
    assert not (would_have & paint).any()  # misses the paint pixels
    ok, reason = review_region_usable(BAD, PAINT_BBOX)
    assert not ok and reason == "misses-paint"
    ch = body_mask_channels_from_raster(
        288, 288, ORIGIN, raster, complete=True, review_region=BAD)
    assert ch["extent_used"] is False
    assert ch["quarantine_reason"] == "misses-paint"
    assert ch["bg_reviewed"].sum() == 0
    # nothing inside the bad rectangle is licensed at all
    assert not (ch["valid"] > 0)[would_have].any()
    yy, xx = np.mgrid[0:288, 0:288]
    d = np.hypot(xx + ORIGIN[0] - PAINT[0], yy + ORIGIN[1] - PAINT[1])
    # band validity is <=13.5 px from the PAINT BOUNDARY, i.e. <= r+13.5
    # = 21.5 px from the blob centre
    assert not ((ch["valid"] > 0) & (d > 21.5)).any()


def test_enclosing_extent_licenses_reviewed_background_only():
    paint = _paint_in_crop()
    _n, raster = _paint_raster()
    ch = body_mask_channels_from_raster(
        288, 288, ORIGIN, raster, complete=True, review_region=GOOD,
        review_region_provenance={'schema':'tubetracker.review_geometry.v1',
                                  'kind':'declared_field','region_xy':GOOD})
    assert ch["extent_used"] is True and ch["quarantine_reason"] == "ok"
    ext = _region_in_crop(288, 288, ORIGIN, GOOD)
    assert np.array_equal(ch["bg_reviewed"], ext & ~paint)
    assert np.array_equal(ch["fg"], paint)
    # REVISED (H344): the band is a NAMED CHANNEL ("the 13.5 px zone
    # the paint itself licenses" - this module's own docstring), not a
    # fallback a usable extent replaces. It used to be computed only
    # inside the `not used_extent` branch, so once every extent became
    # usable (snap24: 8/8) the channel was empty on every mask and any
    # consumer silently got nothing. It is now always the paint-adjacent
    # zone; this assertion is what was wrong.
    assert ch["band"].sum() > 300
    assert not (ch["band"] & ch["fg"]).any()
    from scipy.ndimage import distance_transform_edt
    _d = distance_transform_edt(~ch["fg"])
    assert (_d[ch["band"]] <= 13.5 + 1e-6).all()
    # explicit channels partition the crop
    assert not (ch["fg"] & ch["bg_reviewed"]).any()
    assert np.array_equal(
        ch["unknown"], ~(ch["fg"] | ch["band"] | ch["bg_reviewed"]))
    assert np.array_equal(
        ch["valid"] > 0, ch["fg"] | ch["band"] | ch["bg_reviewed"])
    # background here is genuinely supervised 0 outside the paint
    assert (ch["bg_reviewed"] & paint).sum() == 0


@pytest.mark.parametrize("region,reason", [
    (None, "no-review-region"),
    ([[390.0, 330.0], [390.0, 400.0]], "degenerate"),
    ([[np.nan, 330.0], [580.0, 400.0]], "nonfinite"),
    ([[390.0, 330.0], [400.0]], "shape"),
])
def test_rejected_extent_shapes(region, reason):
    _n, raster = _paint_raster()
    ch = body_mask_channels_from_raster(
        288, 288, ORIGIN, raster, complete=True, review_region=region)
    assert ch["extent_used"] is False
    assert ch["quarantine_reason"] == reason
    assert ch["bg_reviewed"].sum() == 0, "quarantined: nothing licensed"
    assert ch["band"].sum() > 0, "paint band still supervised"


def test_incomplete_task_never_licences_background():
    _n, raster = _paint_raster()
    ch = body_mask_channels_from_raster(
        288, 288, ORIGIN, raster, complete=False, review_region=GOOD)
    assert ch["extent_used"] is False
    assert ch["bg_reviewed"].sum() == 0
    assert ch["band"].sum() > 0


def test_points_path_quarantines_the_same_extent():
    stamps = [[PAINT[0] - 6, PAINT[1]], [PAINT[0] + 6, PAINT[1]]]
    t_bad, v_bad = body_mask_from_points(
        288, 288, ORIGIN, stamps, complete=True, review_region=BAD)
    t_good, v_good = body_mask_from_points(
        288, 288, ORIGIN, stamps, complete=True, review_region=GOOD)
    assert v_bad.sum() < v_good.sum(), "bad extent must licence less"
    # the bad path keeps band-only validity (stamp radius + 13.5 band)
    yy, xx = np.mgrid[0:288, 0:288]
    d = np.hypot(xx + ORIGIN[0] - PAINT[0], yy + ORIGIN[1] - PAINT[1])
    assert not ((v_bad > 0) & (d > 24.0)).any()
    assert np.array_equal(t_bad > 0, t_good > 0)
    # and the quarantine reason is preserved on the shared helper
    bg, used, reason = _review_extent_background(
        288, 288, ORIGIN, t_bad > 0, BAD)
    assert not used and reason == "misses-paint" and bg.sum() == 0


def test_snapshot_builder_uses_the_shared_predicate():
    from scripts.build_v30_snapshot import _extent_hits_paint
    bad = {"review_region": BAD,
           "mask_raster": {"x0": 492, "y0": 422, "w": 16, "h": 16}}
    good = {"review_region": GOOD,
            "mask_raster": {"x0": 492, "y0": 422, "w": 16, "h": 16}}
    assert _extent_hits_paint(bad) is False
    assert _extent_hits_paint(good) is True
    # and the builder agrees with the target-side predicate
    assert _extent_hits_paint(good) == review_region_usable(
        GOOD, PAINT_BBOX)[0]
    assert _extent_hits_paint(bad) == review_region_usable(
        BAD, PAINT_BBOX)[0]


def test_real_snap20_masks_are_all_quarantined():
    """Every surviving reviewed extent in snap20 is unusable (H306);
    after the fix none may licence background in a training target."""
    samples = samples_from_snapshot(SNAP)
    masks = [s for s in samples if s.kind == "body_mask"]
    assert masks, "no body masks in snap20"
    with_extent = [s for s in masks
                   if getattr(s, "review_region", None) and s.mask_raster]
    assert with_extent, "expected masks carrying review extents"
    for s in with_extent:
        r = s.mask_raster
        cx = float(r["x0"]) + float(r["w"]) / 2.0
        cy = float(r["y0"]) + float(r["h"]) / 2.0
        ch = body_mask_channels_from_raster(
            288, 288, (cx - 144.0, cy - 144.0), r,
            complete=bool(s.complete), review_region=s.review_region)
        assert ch["extent_used"] is False, (
            f"{s.obs_uuid}: unusable extent was accepted")
        assert ch["bg_reviewed"].sum() == 0
        assert ch["quarantine_reason"] == "misses-paint", (
            f"{s.obs_uuid}: expected the H306 signature, got "
            f"{ch['quarantine_reason']}")


def test_plain_target_unchanged_for_usable_extent():
    """Regression: the two-value API still returns paint-exact target."""
    paint = _paint_in_crop()
    _n, raster = _paint_raster()
    t, v = body_mask_from_raster(
        288, 288, ORIGIN, raster, complete=True, review_region=GOOD)
    assert np.array_equal(t > 0, paint)
    assert (v > 0).any()
