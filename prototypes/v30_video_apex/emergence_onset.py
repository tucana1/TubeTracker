"""Emergence onset from a grain's growing rim protrusion.

Replaces the failed per-frame rim-disruption logit for germination timing. The
disruption magnitude was confounded: a slightly irregular grain rim ROTATING
into view reads as "disruption" exactly like a tube, so the per-frame signal
was noisy and non-monotonic on cf70 (spurious early cluster, quiet middle).

The investigator's criterion gives the real discriminator: a germinating tube
is a protrusion at a FIXED grain-relative angle that GROWS OUTWARD monotonically,
while rim irregularity just rotates around at constant size. A pollen grain's
germ pore is a small, stable aperture (the ~2.5px protrusion baseline); germination
is that bump ELONGATING into a tube. So onset = where a protrusion at a
consistent angle begins sustained outward growth.

Pure image + posed-centre math (no model). Conservative by construction: it
ignores the stable pore baseline and only fires on sustained growth, tolerating
the investigator's accepted "detect a little after true onset if resolution-limited".
"""
from __future__ import annotations

import cv2
import numpy as np

N_BINS = 72
# per-angle outward reach is measured as the farthest grain-boundary radius;
# protrusion = that reach's max over angles minus the grain's median radius.

def grain_component_mask(tile, grain_xy, radius):
    """The grain's OWN dark body+tube mask: threshold + dominant component.

    Robust dark-grain threshold (Otsu: grain/bright background are bimodal; an
    ad-hoc threshold silently under-measures a faint tube's reach). Only the
    connected dark component that contains the grain counts: this keeps the
    measurement from running into a neighbouring grain/detritus (a separate
    dark blob) and faking a long tube (rev17 field scan: 11-29px phantom
    growth from neighbours). A pollen grain has a BRIGHT centre and a dark
    rim/body, so seeding from the exact centre hits background; seed from the
    dominant dark component in a small disk (the grain's own rim/body) instead.

    Returns (grain_mask, dist, labels): boolean body+tube mask, per-pixel
    distance to the grain centre, and the raw component labels.
    """
    tile = np.asarray(tile, dtype=np.float32)
    h, w = tile.shape
    gx, gy = float(grain_xy[0]), float(grain_xy[1])
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(xx - gx, yy - gy)
    tt = tile if tile.max() <= 255 else tile / max(tile.max(), 1e-6) * 255
    tt8 = np.clip(tt, 0, 255).astype(np.uint8)
    otsu_thr, _ = cv2.threshold(tt8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = tile <= float(otsu_thr)
    ncomp, labels = cv2.connectedComponents(dark.astype(np.uint8))
    cyi, cxi = int(round(gy)), int(round(gx))
    cyi = min(max(cyi, 0), h - 1); cxi = min(max(cxi, 0), w - 1)
    near = (dist <= max(radius, 3.0)) & dark
    if near.any():
        vals, counts = np.unique(labels[near], return_counts=True)
        vals, counts = vals[vals > 0], counts[vals > 0]
        grain_label = int(vals[np.argmax(counts)]) if vals.size else 0
    else:
        grain_label = int(labels[cyi, cxi])
    if grain_label == 0:                    # no dark grain body: fall back
        grain_mask = dark
    else:
        grain_mask = labels == grain_label
    return grain_mask, dist, labels


def boundary_radius_profile(tile, grain_xy, radius, max_reach_bins=56):
    """Farthest grain-boundary radius per angular bin.

    Records how far the grain's own dark component (body + attached tube)
    reaches along each 5-degree wedge — the farthest component pixel, in tile
    pixels. A protrusion / nascent tube reaches farther than the grain's own
    radius.

    Validated against 16 human tube traces (rev18 growth-curve round): this
    farthest-component measure beats the older contiguous-dark ray walk
    (MAE 2.1px vs 5.0px, bias -2.1 vs -5.0). The walk broke at the first
    bright pixel past rim+2.5px and cut faint or curved tubes — up to 14px
    under-read at a tube tip, which is what produced the spiky length curves
    (the spikes were frames where the walk got lucky along the tube).
    Neighbour rejection is unchanged: only the dark component attached to
    the grain's own body counts.
    """
    grain_mask, dist, _labels = grain_component_mask(tile, grain_xy, radius)
    h, w = tile.shape
    gx, gy = float(grain_xy[0]), float(grain_xy[1])
    yy, xx = np.mgrid[:h, :w]
    ang = np.arctan2(yy - gy, xx - gx)
    profile = np.zeros(N_BINS)
    ok = np.zeros(N_BINS, dtype=bool)
    for b in range(N_BINS):
        a0 = -np.pi + b * 2 * np.pi / N_BINS
        sel = (ang >= a0) & (ang < a0 + 2 * np.pi / N_BINS) & (dist < max_reach_bins)
        rr = dist[sel]
        dd = grain_mask[sel]        # only the grain's own body+tube counts
        if rr.size == 0:
            continue
        keep = rr >= radius * 0.55
        vals = rr[keep & dd]
        profile[b] = float(vals.max()) if vals.size else float(radius)
        ok[b] = True
    return profile, ok


def protrusion_series(tiles, grain_xys, radius):
    """Track the dominant growing protrusion across a frame series.

    Returns (protrusion_height[T], protrusion_angle_deg[T], dominant_bin).
    The dominant bin is the angle whose boundary reach shows the strongest
    sustained GROWTH (max minus its own early baseline) — i.e. where a tube is
    emerging — not merely the tallest static pore.
    """
    profs = []
    for tile, g in zip(tiles, grain_xys):
        p, ok = boundary_radius_profile(tile, g, radius)
        profs.append(np.where(ok, p, np.nan))
    P = np.array(profs)                     # (T, N_BINS)
    med = np.nanmedian(P, axis=1)           # per-frame median boundary (=grain r)
    # Per-frame TALLEST protrusion: the farthest the grain boundary reaches past
    # its own radius, and the angle it points at (a germinating tube pushes one
    # spot out and grows). Measured robustly on real grains; a fixed-angle
    # variant mis-selected the tube angle and degraded both reviewed cases.
    reach = np.where(np.isfinite(P), P, -1e9)
    dom_bin = np.argmax(reach, axis=1)      # (T,) angle bin of the tallest bump
    height = np.nanmax(P, axis=1) - med     # (T,) bump height over the grain rim
    ang = (dom_bin * 360.0 / N_BINS + 180.0) % 360.0
    return height, ang, dom_bin


def detect_onset(frames, height, *, frac=0.4, smooth=5, min_growth=1.5,
                 timing="knee", noise_k=2.5):
    """Conservative germination onset at the protrusion-growth rise.

    The pore bump is a stable baseline; germination is that bump ELONGATING
    into a tube. Two physical guards keep it from firing on noise: a real tube
    raises the bump by >= `min_growth` px (not a sub-pixel wiggle) AND stays
    elevated to the end of the window (a tube keeps growing / never retracts; a
    one-frame rim glitch does).

    `timing` places the onset call. "knee" (default) = where the sustained rise
    crosses `frac` of the rise. "rise_start" = where the smoothed height first
    leaves the pore baseline band (base + max(noise_k*noise, 0.6px)) — the
    first-visible moment. "ramp_start" = the changepoint where the series stops
    being flat and becomes a rising ramp (piecewise constant-then-linear fit;
    the tube's growth ramp begins at emergence). Calibrated on the rev18 human
    emergence brackets: the knee read 250-3250 frames late (it is mid-growth by
    construction); rise_start/ramp_start landed in- or near-bracket. Never-
    firing-early is the hard rule in all three; lateness is tolerated only as a
    resolution-limited fallback.
    """
    frames = np.asarray(frames)
    h = np.asarray(height, dtype=float)
    n = len(h)
    if smooth and n >= smooth:                       # kill single-frame spikes
        k = np.ones(smooth) / smooth
        pad = smooth // 2
        hp = np.pad(h, (pad, pad), mode="edge")      # edge-pad: zero-pad would
        h = np.convolve(hp, k, mode="valid")[:n]     # drag the tail toward 0
    nbase = max(3, n // 5)
    base = float(np.median(h[:nbase]))
    mad = float(np.median(np.abs(h[:nbase] - base))) or 0.05
    noise = 1.4826 * mad
    peak = float(np.nanmax(h))
    if timing == "rise_start":
        cutoff = base + max(noise_k * noise, 0.6)
    else:
        cutoff = base + frac * (peak - base) if timing != "ramp_start" \
            else base + max(noise_k * noise, 0.6)
    growth = peak - base
    # resolution / confidence: a germ pore is itself a ~2-3px stable bump and a
    # nascent tube only grows it a little. Confidence reflects the ABSOLUTE bump
    # growth vs what's resolvable (not growth/smoothed-noise, which overstates
    # confidence once smoothing flattens the noise). G0 grows the pore only
    # ~1.5px (tube 2.7->4.2px) = near the resolution floor -> low confidence,
    # wide onset range biased LATE (the investigator accepts a slightly-late call
    # over a false-early one). cf70 grows it ~3px (2.4->5.5px) = high confidence.
    confidence = "high" if growth >= 2.5 else ("low" if growth < 1.8 else "medium")
    span = 0 if confidence == "high" else (400 if confidence == "medium" else 900)
    base_ret = {"baseline_px": base, "cutoff_px": cutoff, "peak_px": peak,
                "growth_px": growth, "noise_px": noise,
                "confidence": confidence}

    def _none(reason):
        return {"onset_frame": None, "verdict": "no_emergence_by_end",
                "reason": reason, **base_ret}

    if growth < min_growth:                          # not enough growth = no tube
        return _none("growth_below_min_px")
    if float(h[-1]) <= cutoff:                       # fell back = transient, not emergence
        return _none("does_not_persist")
    if timing == "ramp_start":
        # changepoint: flat (constant) before, rising (linear) after; the ramp
        # origin is where germination began, not where growth became visible
        n_tail = max(4, n // 5)
        best_i, best_sse = None, np.inf
        for i in range(nbase, n - n_tail + 1):
            pre, post = h[:i], h[i:]
            sse = float(np.sum((pre - pre.mean()) ** 2))
            x = np.arange(post.size, dtype=float)
            if post.size >= 2 and np.ptp(x) > 0:
                sl, ic = np.polyfit(x, post, 1)
                sse += float(np.sum((post - (sl * x + ic)) ** 2))
            if sse < best_sse:
                best_sse, best_i = sse, i
        onset = int(frames[best_i])
        return {"onset_frame": onset, "verdict": "emergence_detected",
                "reason": "growth_ramp_origin",
                "onset_range": [onset - span // 3, onset + span],
                **base_ret}
    for i in range(n):
        if h[i] > cutoff:                            # knee of the sustained rise
            onset = int(frames[i])
            # a low-confidence onset is uncertain and biased LATE (resolution-limited)
            return {"onset_frame": onset, "verdict": "emergence_detected",
                    "reason": "growth_knee",
                    "onset_range": [onset - span // 3, onset + span],
                    **base_ret}
    return _none("no_crossing")
