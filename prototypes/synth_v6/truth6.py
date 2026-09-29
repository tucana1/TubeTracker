"""Truth rasters for v6 scenes: ``prototypes.learned_flood.truth.frame_truth`` with the v6 tip - the tube
body widens towards the apex (the bulb) and ends in a rounded dome, as ``Scene.render`` draws it."""

from __future__ import annotations

import numpy as np

from prototypes.learned_flood.truth import HALF_WIDTH, _tube_maps, tube_point


def frame_truth(scene, k: int, scored_only: bool = False) -> dict:
    """``body`` (uint8), ``instance`` (int16, 1 + tube index) and ``tips`` [(x, y, i, L)] at frame ``k``."""
    body = np.zeros((scene.h, scene.w), np.uint8)
    inst = np.zeros((scene.h, scene.w), np.int16)
    tips = []
    v6 = scene.cfg.tip_round or scene.cfg.tip_bulb[1] > 0
    for i, t in enumerate(scene.tubes):
        if scored_only and not t.scored:
            continue
        L = float(t.length(np.array([float(k)]))[0])
        if L <= 0:
            continue
        s, d, lo, hi = _tube_maps(scene, i, k)
        if v6:
            near = s <= L
            w = np.full(s.shape, t.width, np.float64)
            w[near] = scene.tip_width(t, t.bright, L, s[near].astype(np.float64))
            dist = d.astype(np.float64)
            dist[near] = scene.tip_distance(t, t.bright, L, s[near].astype(np.float64), dist[near], w[near])
            on = near & (dist <= HALF_WIDTH[t.bright] * w)
            xx, yy = scene.maps[i][2], scene.maps[i][3]
            on &= scene.outside_grain(t, k, xx, yy) >= 0.5  # not over its own grain (as drawn)
        else:
            on = (s <= L) & (d <= HALF_WIDTH[t.bright] * t.width)
        win = (slice(lo[1], hi[1] + 1), slice(lo[0], hi[0] + 1))
        body[win][on] = 1
        inst[win][on] = i + 1
        tips.append((*tube_point(t, k, L), i, L))
    return {"body": body, "instance": inst, "tips": tips}
