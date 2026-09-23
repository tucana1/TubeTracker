"""Tests for the protrusion-growth emergence onset detector."""
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototypes", "v30_video_apex"))
import importlib.util
_root = os.path.join(os.path.dirname(__file__), "..", "prototypes", "v30_video_apex", "emergence_onset.py")
spec = importlib.util.spec_from_file_location("emergence_onset", _root)
eo = importlib.util.module_from_spec(spec); spec.loader.exec_module(eo)


def test_growing_bump_detects_onset_at_knee():
    frames = list(range(0, 1000, 50))
    # flat pore baseline ~2.5, then growth from f500 to a 6px tube
    height = [2.5] * 10 + [2.6, 3.0, 3.6, 4.4, 5.2, 5.8, 6.0, 6.0, 6.0, 6.0]
    res = eo.detect_onset(frames, height)
    assert res["verdict"] == "emergence_detected"
    # onset at the knee, within the growth window (500..700), not in the flat part
    assert 450 <= res["onset_frame"] <= 750, res


def test_flat_no_growth_reports_none():
    frames = list(range(0, 1000, 50))
    height = [2.5 + 0.1 * np.sin(i) for i in range(len(frames))]  # pore, no growth
    res = eo.detect_onset(frames, height)
    assert res["verdict"] == "no_emergence_by_end", res


def test_single_spike_does_not_fire():
    frames = list(range(0, 1000, 50))
    height = [2.5] * len(frames)
    height[12] = 5.0  # one-frame rim glitch, no sustained growth
    res = eo.detect_onset(frames, height, smooth=5)
    assert res["verdict"] == "no_emergence_by_end", res


def test_boundary_reach_sees_protrusion():
    # a synthetic dark grain on bright bg with an outward bump reaches past radius
    tile = np.full((120, 120), 200.0)
    yy, xx = np.mgrid[:120, :120]
    g = (60.0, 60.0)
    dist = np.hypot(xx - g[0], yy - g[1])
    ang = np.arctan2(yy - g[1], xx - g[0])
    tile[dist <= 13] = 40.0
    wedge = (np.abs(ang) <= np.deg2rad(15)) & (dist <= 20)   # a bump to r=20 at 0deg
    tile[wedge] = 40.0
    prof, ok = eo.boundary_radius_profile(tile, g, 13.0)
    assert prof.max() >= 17.0, prof.max()      # reaches the bump
    assert np.median(prof[ok]) <= 14.0         # the grain body stays ~13
    assert (prof.max() - np.median(prof[ok])) >= 3.0   # a measurable protrusion
