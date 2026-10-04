"""The learned tip detector in the tracker (sparsetrack/tipdet.py): its onset rule, young lengths from its tips, and
the length series staying consistent (prototypes/tip_track)."""

import math
from dataclasses import replace

import numpy as np
import pytest

from sparsetrack import tipdet
from sparsetrack.analyze import Params

FPB, NB = 300, 60
R0 = 10.0


class FakeMaps:
    """Stands in for GrainTipMaps: ``tips[b]`` = [(x, y, value)] peaks at bin b (reference coordinates)."""

    def __init__(self, tips: dict, lo: int = 7, hi: int = NB - 2):
        self.tips, self.lo, self.hi = tips, lo, hi

    def all_peaks(self, b, min_value=0.03):
        return sorted(self.tips.get(b, []), key=lambda t: -t[2])


def reading(onset_i=None, lengths=None, status=None):
    frames = [b * FPB + FPB // 2 for b in range(NB)]
    px = [0.0] * NB if lengths is None else list(lengths)
    res = {"id": "g1", "x": 100.0, "y": 100.0, "r": R0, "flags": [], "length": {"frames": frames, "px": px},
           "tip": {"frames": frames, "xy": [[100.0, 100.0]] * NB}, "path": [[110.0, 100.0], [140.0, 100.0]],
           "exit_xy": [110.0, 100.0]}
    if onset_i is None:
        res.update(status=status or "no_emergence_by_end", onset_frame=None, onset_interval=None, path=[])
    else:
        res.update(status="emerged_within", onset_frame=frames[onset_i], onset_interval=[frames[onset_i - 1],
                                                                                         frames[onset_i]])
    return res


def growing_tip(start: int, value: float = 0.6, speed: float = 0.5, angle: float = 0.0) -> dict:
    """A tip leaving the grain at ``start``: r + 2 px out, then ``speed`` px per bin."""
    out = {}
    for b in range(start, NB):
        d = R0 + 2.0 + speed * (b - start)
        out[b] = [(100.0 + d * math.cos(angle), 100.0 + d * math.sin(angle), value)]
    return out


META = {"frames_per_bin": FPB, "n_bins": NB, "ref_start": 0}
POS = np.tile([100.0, 100.0], (NB, 1))
ON = replace(Params(), tipdet_model="detector.pt", tipdet_onset="later", tipdet_late_bins=10)


def test_off_by_default():
    p = Params()
    assert p.tipdet_model is None and p.tipdet_onset == "off" and p.tipdet_young is False


def test_sustained_needs_the_whole_hold():
    v = np.array([0.0, 0.4, 0.1, 0.4, 0.4, 0.2, 0.5, 0.5, 0.5, 0.1])
    assert tipdet.sustained(v, 0.3, 3) == 6
    assert tipdet.sustained(v, 0.3, 2) == 3
    assert tipdet.sustained(v, 0.3, 4) is None
    assert tipdet.sustained(np.array([0.0, 0.5, 0.5]), 0.3, 3) is None  # runs off the end


def test_isotonic_is_non_decreasing_and_keeps_monotone_series():
    y = np.array([0.0, 2.0, 1.0, 3.0, 2.5, 2.5, 6.0])
    f = tipdet.isotonic(y)
    assert np.all(np.diff(f) >= -1e-12)
    assert np.isclose(f.sum(), y.sum())  # pooled values keep the total
    m = np.array([0.0, 1.0, 1.0, 4.0])
    assert np.allclose(tipdet.isotonic(m), m)


def test_a_much_later_reading_onset_moves_to_the_detectors():
    res = reading(onset_i=35, lengths=[0.0] * 35 + [25.0 + 0.5 * k for k in range(NB - 35)])
    s = tipdet.apply(res, FakeMaps(growing_tip(20)), POS, None, META, ON)
    assert s["moved_onset_bins"] == 15
    assert res["onset_frame"] == res["length"]["frames"][20] and res["status"] == "emerged_within"
    assert res["onset_interval"] == [res["length"]["frames"][19], res["length"]["frames"][20]]
    px = np.asarray(res["length"]["px"])
    assert np.all(px[:20] == 0) and px[20] > 0
    assert np.all(np.diff(px[20:]) >= -1e-9)  # the moved onset keeps the series non-decreasing
    assert any(f.startswith("onset_tip_detector:15") for f in res["flags"])


def test_an_onset_only_a_little_later_is_kept():
    res = reading(onset_i=28, lengths=[0.0] * 28 + [3.0 + 0.5 * k for k in range(NB - 28)])
    before = [list(res["length"]["px"]), res["onset_frame"]]
    tipdet.apply(res, FakeMaps(growing_tip(20)), POS, None, META, ON)  # 8 bins earlier: not more than 10
    assert [res["length"]["px"], res["onset_frame"]] == before


def test_no_tube_stays_no_tube_unless_asked():
    res = reading()
    tipdet.apply(res, FakeMaps(growing_tip(20)), POS, None, META, ON)
    assert res["status"] == "no_emergence_by_end" and not any(res["length"]["px"])
    res = reading()
    p = replace(ON, tipdet_onset="later_or_missing", tipdet_young=True)
    tipdet.apply(res, FakeMaps(growing_tip(20)), POS, lambda th: 0.0, META, p)
    assert res["status"] == "emerged_within" and res["onset_frame"] == res["length"]["frames"][20]
    px = np.asarray(res["length"]["px"])
    assert np.all(np.diff(px) >= -1e-9) and px[-1] > 0
    assert len(res["path"]) == 2 and res["exit_xy"] == res["path"][0]


def test_young_lengths_come_from_the_tips_distance_to_the_visible_edge():
    lengths = [0.0] * 20 + [1.0 + 0.2 * k for k in range(NB - 20)]  # a reading that under-reads a young tube
    res = reading(onset_i=20, lengths=lengths)
    p = replace(ON, tipdet_onset="off", tipdet_young=True, tipdet_young_k=2.0, tipdet_young_px=20.0)
    tipdet.apply(res, FakeMaps(growing_tip(20, speed=0.5)), POS, lambda th: -1.0, META, p)
    px = np.asarray(res["length"]["px"])
    # at bin 30 the tip is r + 2 + 5 px from the centre: length = 17 - (r - 1) + 2 = 10
    assert px[30] == pytest.approx(10.0, abs=0.01)
    assert res["tip"]["xy"][30] == [117.0, 100.0]
    assert np.all(np.diff(px) >= -1e-9)
    assert res["onset_frame"] == res["length"]["frames"][20]  # onset untouched with tipdet_onset off


def test_young_tips_are_in_the_reading_frame_of_a_drifting_grain():
    lengths = [0.0] * 20 + [1.0] * (NB - 20)
    res = reading(onset_i=20, lengths=lengths)
    res["drift"] = {"frames": res["length"]["frames"], "xy": [[3.0, -2.0]] * NB}
    pos = np.tile([103.0, 98.0], (NB, 1))
    tips = {b: [(103.0 + R0 + 4.0, 98.0, 0.7)] for b in range(20, NB)}
    p = replace(ON, tipdet_onset="off", tipdet_young=True)
    tipdet.apply(res, FakeMaps(tips), pos, None, META, p)
    assert res["tip"]["xy"][25] == [100.0 + R0 + 4.0, 100.0]  # the field position less the drift


def test_weak_or_far_peaks_are_not_tips():
    lengths = [0.0] * 20 + [1.0] * (NB - 20)
    res = reading(onset_i=20, lengths=lengths)
    tips = {b: [(100.0 + R0 + 4.0, 100.0, 0.1), (100.0 + R0 + 40.0, 100.0, 0.9)] for b in range(20, NB)}
    p = replace(ON, tipdet_onset="off", tipdet_young=True, tipdet_young_min=0.2)
    tipdet.apply(res, FakeMaps(tips), POS, None, META, p)
    assert res["length"]["px"] == lengths  # 0.1 is too weak, 40 px out is no young tube


def test_crop_maps_follow_the_grain():
    torch = pytest.importorskip("torch")
    from sparsetrack import learned
    from sparsetrack.render import Renderer

    torch.manual_seed(0)
    net = learned._unet((4, 8, 16, 32), "batch").eval()
    rng = np.random.default_rng(1)
    nb, size = 12, 256
    bins = (120 + 3 * rng.standard_normal((nb, size, size))).astype(np.float32)
    for b in range(nb):  # a dark blob growing at a fixed place
        bins[b, 120:124, 130:130 + 2 * b] -= 30
    meta = {"frames_per_bin": FPB, "n_bins": nb, "ref_start": 0, "shifts": [[0.0, 0.0]] * nb}
    R = Renderer(bins, meta)
    a = tipdet.GrainTipMaps(R, net, np.tile([128.0, 128.0], (nb, 1)), half=16)
    b = tipdet.GrainTipMaps(R, net, np.tile([132.0, 126.0], (nb, 1)), half=16)
    assert a.maps.shape == (nb, 32, 32) and not a.maps[:a.lo].any() and not a.maps[a.hi + 1:].any()
    # the same reference pixel read from both crops
    for k in (a.lo, a.hi):
        ja, ia = a.to_pix(k, 135.5, 122.5)
        jb, ib = b.to_pix(k, 135.5, 122.5)
        assert abs(float(a.maps[k][int(ia), int(ja)]) - float(b.maps[k][int(ib), int(jb)])) < 0.02
