"""Prefix decoder, per-bin frame-edge mask (``Params.edge_mask``): a grain whose tube is visible
while the stage has not yet moved keeps it, instead of losing every pixel that leaves the frame in some bin."""
import numpy as np
import pytest

from sparsetrack.render import Renderer

from prototypes.learned_evidence import prefix as PF

T, H, W = 40, 160, 160
GX, GY, GR = 60.0, 80.0, 10.0


def edge_movie(jump_at=30, jump=-40.0):
    """A grain 60 px from the left edge with a thin dark tube growing left from its rim (1.5 px per bin from bin 5);
    from ``jump_at`` the stage moves ``jump`` px in x, so the tube's far part has no source in the frame from then on."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    shifts = np.zeros((T, 2))
    shifts[jump_at:, 0] = jump
    img = np.full((T, H, W), 130.0, np.float32)
    prob = np.zeros((T, H, W), np.float32)
    for t in range(T):
        L = float(np.clip((t - 5) * 1.5, 0, 45))
        tube = (xx <= GX - GR) & (xx >= GX - GR - L) & (np.abs(yy - GY) <= 1.5)
        prob[t] = np.where(tube, 16.0, 0.0)
        fx = xx - shifts[t, 0]  # frame coordinates: content at reference x appears at x + shift
        frg = np.hypot(fx - GX, yy - GY)
        img[t][frg < GR] = 90.0
        img[t][(frg >= GR - 2) & (frg < GR)] = 60.0
        img[t][(fx <= GX - GR) & (fx >= GX - GR - L) & (np.abs(yy - GY) <= 1.5)] -= 30.0
        img[t] += rng.normal(0, 1.5, (H, W)).astype(np.float32)
    meta_i = {"shifts": shifts.tolist(), "n_bins": T, "frames_per_bin": 1, "ref_start": 0}
    meta_p = {"shifts": np.zeros((T, 2)).tolist(), "n_bins": T, "frames_per_bin": 1, "ref_start": 0}
    return Renderer(prob, meta_p), Renderer(img, meta_i), meta_p


@pytest.mark.parametrize("mode", ["movie", "bin"])
def test_prefix_edge_mask(mode):
    RP, RI, meta = edge_movie()
    p = PF.Params(half=60, big=60, edge_mask=mode, image_term=False)
    res = PF.decode_grain(RP, RI, meta, {"id": "g001", "x": GX, "y": GY, "r": GR}, [], p)
    L = np.array(res["length"]["px"])
    if mode == "movie":  # the whole movie's mask cuts the tube ~10 px from the rim for good
        assert L.max() <= 14.0
    else:  # per bin: the tube is read while it is in the frame (36 px by bin 29)
        assert L[29] >= 28.0
