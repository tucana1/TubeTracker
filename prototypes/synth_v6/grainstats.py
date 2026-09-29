"""How much does a grain itself change in the network's input? Measured per grain and bin on a
cache (real or synthetic), relative to the movie's "before" image (first reference bins), in the
globally registered crops the network sees:

- disp: the grain's own displacement (phase correlation of the grain window, px): crescents;
- blur: signed Gaussian sigma (px) that best maps the before image onto the bin after aligning
  (> 0: the bin is blurrier, < 0: sharper): focus / halo rings;
- interior: mean change inside the grain (r - 4) after aligning, as a share of the grain's
  interior contrast against the background: cytoplasm moving;
- rim_raw / rim_res: RMS change in the rim annulus before / after aligning and deblurring;
- bg: RMS change 10-20 px outside the rim (noise floor).

Only bins before the grain's onset (minus a margin) are used for disp/blur/rim, so tubes do not
contaminate them; the interior change is also reported around germination.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from .sheet import View, real_onsets, synth_onsets


def _hp(a: np.ndarray, s: float = 6.0) -> np.ndarray:
    return a - cv2.GaussianBlur(a, (0, 0), s)


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), img.shape[::-1], flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _blur(img: np.ndarray, s: float) -> np.ndarray:
    return img if s <= 0 else cv2.GaussianBlur(img, (0, 0), s)


def measure(view: View, x: float, y: float, r: float, bins: list[int], half: int = 32) -> list[dict]:
    early = np.mean([view.R.crop(b, x, y, half) for b in range(view.rs, view.rs + 3)], axis=0)
    yy, xx = np.mgrid[0:2 * half, 0:2 * half]
    d = np.hypot(xx - half + 0.5, yy - half + 0.5)
    win = np.clip((r + 10 - d) / 3.0, 0, 1).astype(np.float32)
    inner, rim = d < r - 4, (d >= r - 3) & (d <= r + 5)
    bgr = (d >= r + 10) & (d <= r + 20)
    bg_level = float(np.median(early[bgr]))
    contrast = float(np.mean(early[inner]) - bg_level)
    sig = np.arange(0, 2.01, 0.25)
    out = []
    for b in bins:
        cur = view.R.crop(b, x, y, half)
        if not np.all(np.isfinite(cur)):
            continue
        sh, _, _ = phase_cross_correlation(_hp(early) * win, _hp(cur) * win, upsample_factor=20, normalization=None)
        dy, dx = float(sh[0]), float(sh[1])  # shift that moves cur onto early
        al = _shift(cur, dx, dy)
        # signed blur: bin blurrier (early blurred matches it) or sharper (bin blurred matches early)
        e1 = [np.mean(((_blur(early, s) - al)[rim | inner]) ** 2) for s in sig]
        e2 = [np.mean(((early - _blur(al, s))[rim | inner]) ** 2) for s in sig]
        i1, i2 = int(np.argmin(e1)), int(np.argmin(e2))
        blur = float(sig[i1]) if e1[i1] <= e2[i2] else -float(sig[i2])
        res = (_blur(early, blur) - al) if blur > 0 else (early - _blur(al, -blur))
        out.append({"bin": int(b), "dx": dx, "dy": dy, "disp": float(np.hypot(dx, dy)), "blur": blur,
                    "interior": float(np.mean((al - early)[inner]) / contrast) if abs(contrast) > 3 else float("nan"),
                    "interior_gl": float(np.mean((al - early)[inner])),
                    "rim_raw": float(np.sqrt(np.mean((cur - early)[rim] ** 2))),
                    "rim_al": float(np.sqrt(np.mean((al - early)[rim] ** 2))),
                    "rim_res": float(np.sqrt(np.mean(res[rim] ** 2))),
                    "bg": float(np.sqrt(np.mean((cur - early)[bgr] ** 2))), "contrast": contrast})
    return out


def run(cache: str, labels: str | None = None, truth: str | None = None, step: int = 4, margin: int = 2,
        max_grains: int = 60) -> dict:
    view = View(cache)
    ons = real_onsets(labels) if labels else synth_onsets(truth)
    doc = json.loads(Path(labels or truth).read_text())
    grains = doc["grains"]
    controls = []
    for gid, lab in doc["labels"].items():
        g = grains[gid]
        if g.get("excluded") or not g.get("isolated", True):
            continue
        if lab.get("onset", {}).get("verdict") == "no_emergence_by_end":
            controls.append((gid, g["x"], g["y"], g["r"]))
    rows = []
    for gid, (x, y, r, on) in sorted(ons.items())[:max_grains]:
        bins = list(range(view.rs + 3, min(on - margin, view.nb - 1), step))
        for m in measure(view, x, y, r, bins):
            rows.append({"gid": gid, "kind": "pre", "rel": m["bin"] - on, **m})
        around = [on + o for o in (-6, -3, 0, 5, 10, 20, 30, 45) if view.rs + 3 <= on + o < view.nb - 1]
        for m in measure(view, x, y, r, around):
            rows.append({"gid": gid, "kind": "around", "rel": m["bin"] - on, **m})
    for gid, x, y, r in controls[:max_grains]:
        for m in measure(view, x, y, r, list(range(view.rs + 3, view.nb - 1, step))):
            rows.append({"gid": gid, "kind": "control", "rel": None, **m})
    return {"cache": cache, "rows": rows}


def summary(res: dict) -> str:
    rows = res["rows"]
    lines = []
    for kind in ("pre", "control"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        a = lambda k: np.array([r[k] for r in sub], float)
        q = lambda v: " ".join(f"{p:.2f}" for p in np.nanpercentile(v, [10, 50, 90]))
        lines.append(f"{kind:8s} n={len(sub):4d} grains={len({r['gid'] for r in sub}):3d} | disp p10/50/90 {q(a('disp'))} | "
                     f"blur {q(a('blur'))} | |interior| {q(np.abs(a('interior')))} | rim_raw {q(a('rim_raw'))} | "
                     f"rim_al {q(a('rim_al'))} | rim_res {q(a('rim_res'))} | bg {q(a('bg'))}")
    # bin-to-bin displacement steps (jitter) within pre-onset series
    steps = []
    for gid in {r["gid"] for r in rows if r["kind"] in ("pre", "control")}:
        s = sorted((r for r in rows if r["gid"] == gid and r["kind"] in ("pre", "control")), key=lambda r: r["bin"])
        for u, v in zip(s[:-1], s[1:]):
            steps.append(np.hypot(v["dx"] - u["dx"], v["dy"] - u["dy"]))
    if steps:
        lines.append(f"displacement change between samples 4 bins apart: p10/50/90 "
                     f"{' '.join(f'{p:.2f}' for p in np.percentile(steps, [10, 50, 90]))}")
    ar = [r for r in rows if r["kind"] == "around"]
    if ar:
        by = {}
        for r in ar:
            by.setdefault(r["gid"], {})[r["rel"]] = r["interior"]
        for rel in (10, 20, 30, 45):
            dv = [v[rel] - v[-3] for v in by.values() if rel in v and -3 in v and np.isfinite(v[rel]) and np.isfinite(v[-3])]
            if dv:
                lines.append(f"interior change onset-3 -> onset+{rel}: n={len(dv)} |d| p50 {np.median(np.abs(dv)):.3f} "
                             f"p90 {np.percentile(np.abs(dv), 90):.3f}; signed {' '.join(f'{x:+.2f}' for x in sorted(dv))}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--labels")
    ap.add_argument("--truth")
    ap.add_argument("--out")
    ap.add_argument("--step", type=int, default=4)
    a = ap.parse_args()
    res = run(a.cache, a.labels, a.truth, a.step)
    if a.out:
        Path(a.out).write_text(json.dumps(res))
    print(summary(res))
