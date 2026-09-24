"""adaptive_crop: a grain whose path ends at its crop edge is read again with a bigger crop; others are untouched."""

import numpy as np
import pytest

pytest.importorskip("torch")  # evaluate.py imports the model

from sparsetrack import analyze as A  # noqa: E402


def test_adaptive_crop_reads_edge_paths_again(monkeypatch):
    calls = []

    def fake(renderer, meta, grain, others, p, _settled=False):
        calls.append((grain["id"], p.half))
        side = 2 * p.half
        # "long" runs to the edge of any crop; "short" ends 40 px from the grain
        end = (side - 2.0, p.half) if grain["id"] == "long" else (p.half + 40.0, p.half)
        pts = np.array([[p.half, p.half], end], float)
        return {"id": grain["id"], "flags": [],
                "_diag": (np.zeros((side, side), np.float32), None, None, pts, None, None, p.half - 0.5)}

    monkeypatch.setattr(A, "analyze_grain", fake)
    from prototypes.learned_evidence.evaluate import adaptive_crop

    with adaptive_crop(big=300):
        long_ = A.analyze_grain(None, {}, {"id": "long"}, [], A.Params())
        short = A.analyze_grain(None, {}, {"id": "short"}, [], A.Params())
    assert A.analyze_grain is fake
    assert A.matched_kymograph.__module__ == "sparsetrack.analyze"
    assert long_["flags"] == ["crop_grown:300"]
    assert short["flags"] == []
    # read again once, at the bigger crop, even though that path still ends at the edge
    assert calls == [("long", 150), ("long", 300), ("short", 150)]


def test_chunked_matched_kymograph_reads_long_paths():
    import cv2

    from prototypes.learned_evidence.evaluate import _chunked

    rng = np.random.default_rng(0)
    signed = rng.normal(0, 1, (2, 700, 700)).astype(np.float32)
    pts = np.stack([np.linspace(50, 650, 1200), np.full(1200, 350.0)], axis=1)  # a 600 px path, 0.5 px steps
    normal = np.tile([0.0, 1.0], (len(pts), 1))
    across = np.arange(-3.5, 3.5 + 1e-9, 0.5)
    template = rng.normal(0, 1, (len(pts), len(across))).astype(np.float32)
    angles = np.arange(-45, 45 + 1e-9, 1.5)  # SparseTrack's 61 rotation angles
    with pytest.raises(cv2.error):
        A.matched_kymograph(signed, template, pts, normal, 350.0, across, angles)
    out = _chunked(A.matched_kymograph)(signed, template, pts, normal, 350.0, across, angles)
    assert out.shape == (2, len(angles), len(pts))
    ref = A.matched_kymograph(signed, template, pts, normal, 350.0, across, angles[20:26], lateral_offset=0.0)
    np.testing.assert_array_equal(out[:, 20:26], ref)
