"""Grain-centred temporal emergence evidence (WO3 arm 1, WO4 runtime).

Registered grain-boundary/appearance-change detector: the posed native
tile is compared against a pre-emergence reference built from that
grain's earliest scheduled frames; the emergence ring (0.9r..2.5r) is
scored by deviation beyond ring-background noise, weighted by angular
concentration (a nascent tube is a localized protrusion; focus/drift
shifts scatter around the whole rim).

Controlled comparison (same data/split, posed queries): this unweighted
detector separates the cf70 timeline exactly (0/300 -> 0.0, 6000/9000 ->
1.0) while the frozen-trunk temporal head scores every cf70 frame
visible (~2.0) and the scalar presence head is all-visible. Selected as
the first installed event model for that reason.

Runtime reference = median of the first two scheduled frames, flagged
unreviewed_reference with a left-censor warning: a grain emerged at
schedule start blinds its own reference. Reviewed absent endpoints
(WO2 queue) replace the default once available. Never paint a later
tube backward: the reference always predates the scored frame.
"""

from __future__ import annotations

import numpy as np

RING_INNER_R = 0.9
RING_OUTER_R = 2.5
K_NOISE = 4.0
TILE = 288
# Calibration (development): cf70 absent training frames score exactly 0.0
# (no deviating ring pixel); the faintest visible panel case scores 0.098.
# 0002's validated absence is held OUT of calibration pending its scoped
# WO2 control (its panel reference is itself).
CAL_ABSENT_MAX = 0.0
CAL_VISIBLE_MIN = 0.098
CAL_THRESHOLD = 0.05
# Veto emission: strongest rim-route total on the audit run is ~11.2
# (cf70 frame 300: cap 2.81 + route 8.36). 16 beats any observed rim
# route with margin, while real tubes never face the veto (their scores
# are ~1.0, far above admission). Pinned by test, not tuned per movie.
VETO_EMISSION = 16.0

EVENT_CONTRACT = {
    "version": "tubetracker.event_evidence.v1",
    "ring_radii_r": [RING_INNER_R, RING_OUTER_R],
    "k_noise": K_NOISE,
    "reference_policy": "median_of_first_two_scheduled_frames",
    "cal_threshold": CAL_THRESHOLD,
    "veto_emission": VETO_EMISSION,
    "calibration_basis": "cf70 frames 0/300 absent (train), panel visible min 0.098",
}


def emergence_score(tile, dist, radius, ref):
    """(frac, noise, concentration); score = concentration * (1 + frac)."""
    ring = (dist >= RING_INNER_R * radius) & (dist <= RING_OUTER_R * radius)
    noise = float(np.std(np.asarray(ref)[ring])) + 1.0
    dev = np.abs(np.asarray(tile, float) - np.asarray(ref, float)) > K_NOISE * noise
    cand = dev & ring
    frac = float(cand.sum() / max(1, int(ring.sum())))
    yy, xx = np.mgrid[:tile.shape[0], :tile.shape[1]]
    gy, gx = float(np.mean(xx[ring])), float(np.mean(yy[ring]))
    ang = (np.arctan2(yy - gy, xx - gx) * 180.0 / np.pi).astype(np.int16) % 360
    hist = np.bincount(ang[cand], minlength=360).astype(np.float64)
    hist = hist.reshape(36, 10).sum(axis=1)
    concentration = float(hist.max() / max(1.0, hist.sum()))
    return frac, noise, concentration


def score_from_parts(frac, concentration):
    return float(concentration * (1.0 + frac))


def veto_emission(score, threshold=CAL_THRESHOLD):
    """Confident-absence emission; 0 above the admission threshold."""
    if score is None or score > threshold:
        return 0.0
    return VETO_EMISSION * max(0.0, 1.0 - score / threshold)
