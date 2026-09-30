"""Census check: features of each census "grain" that tell real grains from things the census should not count.

The census (``grains.detect``: Hough circles with a dark rim or body) also finds things an annotator excludes: dark
debris or dead matter (``not_a_grain``), grains touching others or carrying attached bits (``clump``), blurred or
passing grains (``out_of_focus``). Across the three labelled movies the annotator judged 110 census grains and
excluded 25. A "grain" that is debris never germinates, so false grains pull the germination share down.

``features`` measures each census grain on the census's own reference image (flat-fielded as the census was) and on
the first bins after it. Contrasts are divided by the movie's median ring contrast over its whole census, so the
features mean the same in movies filmed at other exposures. ``likely_false`` scores them with a logistic model fitted
on the annotator's exclusions (``MODEL``; prototypes/census_check/ has how it was fitted and judged leave one movie
out). It is an option (``Params.census_check``, off by default): see prototypes/census_check/README.md for why.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from . import grains as G
from . import stack

# the features in the order the model takes them
FEATURES = ("ring_rel", "body_rel", "disc_dark", "fill", "rim_cov", "rim_cv", "outer_dark", "sharp",
            "mass_area", "mass_extent", "mass_offset", "nn_r", "stay", "move")


def reference_image(bins: np.ndarray, meta: dict, census: dict) -> np.ndarray:
    """The census's reference: the registered mean of its reference bins, flat-fielded if the census was."""
    rs = int(meta.get("ref_start", 0))
    which = census.get("reference_bins") or list(range(rs, rs + 3))
    h, w = bins.shape[1:]
    acc = np.zeros((h, w), np.float64)
    for b in which:
        dx, dy = meta["shifts"][b]
        acc += cv2.warpAffine(np.asarray(bins[b], np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]), (w, h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ref = acc / len(which)
    return G.flat_field(ref) if census.get("flatfield") else ref


def _crop(img: np.ndarray, x: float, y: float, half: int) -> np.ndarray:
    return cv2.getRectSubPix(np.ascontiguousarray(img, np.float32), (2 * half, 2 * half), (x - 0.5, y - 0.5))


def _rays(crop: np.ndarray, half: int, radii: np.ndarray, n_ang: int = 36) -> np.ndarray:
    """(n_ang, len(radii)) samples along rays from the crop's centre."""
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    mx = (half + radii[None] * np.cos(ang)[:, None] - 0.5).astype(np.float32)
    my = (half + radii[None] * np.sin(ang)[:, None] - 0.5).astype(np.float32)
    return cv2.remap(crop, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def grain_features(ref: np.ndarray, g: dict, scale: float, later: np.ndarray | None = None) -> dict:
    """One grain's features on the reference image ``ref`` (and ``later``, the mean of a few bins after it, for
    ``stay`` / ``move``); ``scale`` = the movie's median ring contrast."""
    x, y, r = float(g["x"]), float(g["y"]), float(g["r"])
    half = int(math.ceil(2 * r + 12))
    c = _crop(ref, x, y, half)
    jj, ii = np.meshgrid(np.arange(2 * half), np.arange(2 * half))
    d = np.hypot(jj + 0.5 - half, ii + 0.5 - half)
    bg = float(np.median(c[(d >= r + 6) & (d <= r + 14)]))
    s = max(float(scale), 1.0)
    disc = d < r - 1.5
    f = {"ring_rel": float(g.get("ring_contrast", 0.0)) / s, "body_rel": float(g.get("body_contrast", 0.0)) / s}
    f["disc_dark"] = (bg - float(c[d < 0.6 * r].mean())) / s
    f["fill"] = float((c[disc] < bg - 0.5 * s).mean())
    rim = _rays(c, half, np.arange(r - 3.0, r + 2.01, 0.5))
    rim_dark = bg - rim.min(axis=1)
    f["rim_cov"] = float((rim_dark > 0.4 * s).mean())
    f["rim_cv"] = float(np.std(rim_dark) / max(np.mean(np.abs(rim_dark)), 1.0))
    f["outer_dark"] = float((c[(d >= r + 3) & (d <= r + 10)] < bg - 0.5 * s).mean())
    prof = _rays(cv2.GaussianBlur(c, (0, 0), 0.8), half, np.arange(max(r - 4.0, 1.0), r + 3.01, 0.5))
    f["sharp"] = float(np.abs(np.diff(prof, axis=1)).max(axis=1).mean() / 0.5 / s)
    # the dark mass the circle sits on: the connected dark pixels that touch the disc
    dark = (c < bg - 0.4 * s).astype(np.uint8)
    n, lab = cv2.connectedComponents(dark, connectivity=8)
    ids = np.unique(lab[(d < r) & (dark > 0)])
    mass = np.isin(lab, ids[ids > 0])
    if mass.any():
        f["mass_area"] = float(mass.sum() / (math.pi * r * r))
        f["mass_extent"] = float(d[mass].max() / r)
        my, mx = np.nonzero(mass)
        f["mass_offset"] = float(math.hypot(mx.mean() + 0.5 - half, my.mean() + 0.5 - half) / r)
    else:
        f["mass_area"], f["mass_extent"], f["mass_offset"] = 0.0, 0.0, 0.0
    nn = g.get("nn_dist")
    f["nn_r"] = float(min(nn / r, 20.0)) if nn else 20.0
    if later is not None:
        h2 = int(math.ceil(r + 6))
        a, b = _crop(ref, x, y, h2), _crop(later, x, y, h2)
        a0, b0 = a - a.mean(), b - b.mean()
        f["stay"] = float((a0 * b0).sum() / max(math.sqrt((a0 * a0).sum() * (b0 * b0).sum()), 1e-6))
        win = cv2.createHanningWindow((2 * h2, 2 * h2), cv2.CV_32F)
        (sx, sy), _ = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), win)
        f["move"] = float(min(math.hypot(sx, sy), 2 * r) / r)
    else:
        f["stay"], f["move"] = 1.0, 0.0
    return f


def features(cache_dir: str | Path, census: dict | None = None, later_bins: tuple[int, int] = (12, 18),
             grains: list[dict] | None = None) -> dict:
    """grain id -> features for every grain of a cache's census (``census``: a grains.json document; default the
    cache's own), or only for ``grains`` (census entries, e.g. from a labels file). ``later_bins``: the bins (after
    the reference start) averaged for ``stay`` / ``move``. Reads the cache only."""
    bins, meta = stack.load(cache_dir)
    census = census or json.loads((Path(cache_dir) / "grains.json").read_text())
    gl = list(census["grains"].values()) if isinstance(census["grains"], dict) else census["grains"]
    ref = reference_image(bins, meta, census)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    lo, hi = min(rs + later_bins[0], nb - 1), min(rs + later_bins[1], nb)
    later = reference_image(bins, meta, {"reference_bins": list(range(lo, max(hi, lo + 1))),
                                         "flatfield": census.get("flatfield")})
    scale = float(np.median([max(float(g.get("ring_contrast", 0.0)), 1.0) for g in gl])) if gl else 1.0
    return {g["id"]: grain_features(ref, g, scale, later) for g in (grains if grains is not None else gl)}


def likely_false(feats: dict, model: dict | None = None) -> float:
    """P(not a grain the annotator would count) from ``features`` (one grain's), by ``model`` (default ``MODEL``)."""
    m = model or MODEL
    z = float(m["intercept"])
    for k, w, mu, sd in zip(m["features"], m["coef"], m["mean"], m["scale"]):
        z += w * (float(feats[k]) - mu) / sd
    return 1.0 / (1.0 + math.exp(-z))


def check(cache_dir: str | Path, grains: list[dict] | None = None, model: dict | None = None) -> dict[str, float]:
    """grain id -> P(likely not a grain) for the cache's census grains (or ``grains``)."""
    return {gid: likely_false(f, model) for gid, f in features(cache_dir, grains=grains).items()}


def flagged(cache_dir: str | Path, grains: list[dict] | None = None, model: dict | None = None) -> dict[str, float]:
    """The grains the census check flags (P >= the model's threshold): grain id -> P."""
    m = model or MODEL
    return {gid: p for gid, p in check(cache_dir, grains, m).items() if p >= float(m["threshold"])}


FLAG = "likely_not_a_grain"  # the analysis flag (grains.csv / predictions.json): "likely_not_a_grain:<P>"

# fitted by prototypes/census_check/fit.py on the three labelled movies' judgements (SMALL features, standardised over
# the training grains, balanced class weights; the threshold maximises F1 in sample); judged leave one movie out in
# prototypes/census_check/README.md
MODEL: dict = {
    "features": ["disc_dark", "stay", "mass_area", "outer_dark", "ring_rel"],
    "coef": [1.0755, -0.5057, 0.3389, 0.1554, -0.1865],
    "intercept": -0.3483,
    "mean": [0.852, 0.8587, 1.2117, 0.0321, 0.9815],
    "scale": [0.5332, 0.2971, 0.415, 0.0725, 0.3063],
    "threshold": 0.7487,
    "fitted_on": "103 judged census-isolated grains of ld, m2, m1 (21 excluded)",
}


def annotate(cache_dir: str | Path) -> dict[str, float]:
    """Write each census grain's P into the cache's grains.json (field ``likely_not_a_grain``; nothing else changes)
    and return every grain's P."""
    path = Path(cache_dir) / "grains.json"
    doc = json.loads(path.read_text())
    p = check(cache_dir)
    for g in (doc["grains"].values() if isinstance(doc["grains"], dict) else doc["grains"]):
        g[FLAG] = round(p[g["id"]], 3)
    doc["census_check"] = {"threshold": MODEL["threshold"], "fitted_on": MODEL.get("fitted_on")}
    path.write_text(json.dumps(doc, indent=1))
    return p


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="census check: the census grains that are likely not grains")
    ap.add_argument("cache")
    ap.add_argument("--write", action="store_true", help=f"store each grain's P in the cache's grains.json ({FLAG})")
    a = ap.parse_args(argv)
    p = annotate(a.cache) if a.write else check(a.cache)
    fl = {g: v for g, v in p.items() if v >= float(MODEL["threshold"])}
    print(f"{len(fl)} of {len(p)} census grains flagged {FLAG} (P >= {MODEL['threshold']:.2f}): "
          + ", ".join(f"{g} {v:.2f}" for g, v in sorted(fl.items())))


if __name__ == "__main__":
    main()
