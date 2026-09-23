"""Grain-centred rim-profile emergence evidence (WO3 winner, WO4 runtime).

Learned rim-profile detector: the annulus 0.8r..1.1r is sampled into 72
5-degree angular bins and median-subtracted, so a uniform brightness/focus
shift cancels and only a localized disruption of the grain's circular rim
survives. A logistic head on those 72 deviations separates the reviewed
absent/visible pairs where both whole-crop heads and the frame-difference
change-detector failed (H483).

Reference-free: unlike the change-detector (which needs a pre-emergence
reference frame and goes blind when a grain is emerged at schedule start),
this scores a single (frame, grain, radius) query. That is the whole point —
the few-pixel rim bump is a within-frame feature, not a frame-to-frame change.

Runtime scores the posed native tile with the same 288 crop / query contract
as training (WO1-parity). The trained artifact carries its own
standardization contract; a runtime feature that cannot be reproduced from
the artifact is an error, never a silent default.
"""
from __future__ import annotations

import json
import math

import numpy as np

TILE = 288
RIM_INNER_R = 0.8
RIM_OUTER_R = 1.1
DEG_PER_BIN = 5
N_BINS = 360 // DEG_PER_BIN
CONTRACT = "tubetracker.rim_profile_evidence.v1"


class RimProfileEvidence:
    """Scores owned-emergence from a grain's circular rim disruption.

    Reference-free: unlike the change-detector (which needs a pre-emergence
    reference frame and goes blind when a grain is emerged at schedule start),
    this scores a single (frame, grain, radius) query. That is the whole point —
    the few-pixel rim bump is a within-frame feature, not a frame-to-frame
    change. The learned weights act on rotation-invariant disruption stats, so
    any tube orientation is scored, not just the training grains' angles.
    """

    N_FEATURES = 6

    def __init__(self, *, mu, sd, weight, bias, contract=CONTRACT, feature_names=None):
        if contract != CONTRACT:
            raise ValueError(f"rim-profile contract mismatch: {contract!r}")
        mu = np.asarray(mu, float)
        sd = np.asarray(sd, float)
        w = np.asarray(weight, float)
        n = self.N_FEATURES
        if not (mu.shape == sd.shape == w.shape == (n,)):
            raise ValueError(f"rim disruption model expects {n} features, "
                             f"got mu={mu.shape} sd={sd.shape} weight={w.shape}")
        if not np.all(np.isfinite(np.concatenate([mu, sd, w, [bias]]))):
            raise ValueError("rim-profile parameters must be finite")
        self.mu, self.sd, self.weight = mu, sd, w
        self.bias = float(bias)
        self.feature_names = list(feature_names) if feature_names else None

    @classmethod
    def from_json(cls, path):
        doc = json.loads(open(path).read())
        return cls(mu=doc["mu"], sd=doc["sd"], weight=doc["weight"],
                   bias=doc["bias"], contract=doc.get("contract", CONTRACT),
                   feature_names=doc.get("feature_names"))

    @staticmethod
    def rim_profile_deviation(tile, grain_xy, radius):
        """72-bin median-subtracted rim-profile deviation for one posed tile.

        `grain_xy` is in TILE coordinates (the runtime crop origin is already
        subtracted by the caller, matching training's posed-tile convention).
        """
        tile = np.asarray(tile, float)
        h, w = tile.shape[:2]
        if (h, w) != (TILE, TILE):
            raise ValueError(f"rim-profile expects a {TILE}x{TILE} posed tile, got {(h, w)}")
        radius = float(radius)
        if not (math.isfinite(radius) and radius > 0):
            raise ValueError("rim-profile requires a positive finite grain radius")
        gx, gy = float(grain_xy[0]), float(grain_xy[1])
        if not (math.isfinite(gx) and math.isfinite(gy)):
            raise ValueError("rim-profile requires a finite grain query")
        yy, xx = np.mgrid[:h, :w]
        dx, dy = xx - gx, yy - gy
        dist = np.hypot(dx, dy)
        band = (dist >= RIM_INNER_R * radius) & (dist <= RIM_OUTER_R * radius)
        if not band.any():
            raise ValueError("rim band is empty for this query (grain outside crop?)")
        ang = ((np.arctan2(dy, dx) * 180.0 / np.pi).astype(int) % 360) // DEG_PER_BIN
        prof = np.zeros(N_BINS)
        for b in range(N_BINS):
            sel = band & (ang == b)
            prof[b] = tile[sel].mean() if sel.any() else 0.0
        return prof - np.median(prof)

    @classmethod
    def disruption_feature_names(cls):
        return ["max_out", "max_in", "mean_abs", "conc_out", "conc_all",
                "max_run_frac"]

    @staticmethod
    def disruption_features(tile, grain_xy, radius):
        """Rotation-invariant rim-disruption summary (the emergence feature).

        A nascent tube is a LOCALIZED protrusion that breaks the grain's
        circular rim at ONE angle. These stats describe how disrupted and how
        localized the rim is, with no dependence on WHICH angle the tube
        points — so the feature generalizes to any tube orientation instead of
        memorizing the training grains' angles (the failure a 72-bin
        angle-specific head showed: weight sum negative, argmax at one angle).

        Returns 6 angle-free numbers:
          max_out      strongest outward rim push (bump crest)
          max_in       strongest inward dip
          mean_abs     average rim roughness
          conc_out     angular concentration of the outward push (localized ->
                       high; a uniform rough rim / focus shift -> low)
          conc_all     angular concentration of total |deviation|
          max_run_frac longest contiguous disrupted arc / 72 (a tube is one
                       contiguous arc, not scattered speckle)
        """
        dev = RimProfileEvidence.rim_profile_deviation(tile, grain_xy, radius)
        max_out = float(dev.max())
        max_in = float(-dev.min())
        mean_abs = float(np.abs(dev).mean())
        # concentration = share of total deviation in the single widest bin
        tot_out = float(np.clip(dev, 0, None).sum())
        conc_out = float(dev.max() / tot_out) if tot_out > 1e-9 else 0.0
        tot_all = float(np.abs(dev).sum())
        conc_all = float(np.abs(dev).max() / tot_all) if tot_all > 1e-9 else 0.0
        # longest contiguous run of bins with |dev| above its own half-max,
        # circularly (a tube arc can wrap the 0/360 seam)
        thresh = 0.5 * np.abs(dev).max()
        hot = (np.abs(dev) >= thresh) if thresh > 0 else np.zeros(N_BINS, bool)
        run = 0
        if hot.any():
            double = np.concatenate([hot, hot])
            cur = 0
            for v in double:
                cur = cur + 1 if v else 0
                run = max(run, cur)
            run = min(run, N_BINS)
        max_run_frac = run / N_BINS
        return np.array([max_out, max_in, mean_abs, conc_out, conc_all,
                         max_run_frac], float)

    def score(self, tile, grain_xy, radius):
        """Emergence logit (higher = visible rim outgrowth / rim disruption)."""
        feats = self.disruption_features(tile, grain_xy, radius)
        x = (feats - self.mu) / self.sd
        return float(x @ self.weight + self.bias)
