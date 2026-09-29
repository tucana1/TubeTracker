"""Contact sheets of young tubes: rows alternate the registered bin and its change from the
movie's "before" image (the first reference bins: what the tube network is shown), at bins
around germination. Works on real caches (onsets from human labels) and synthetic ones
(onsets from the truth file), so the two can be compared side by side.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sparsetrack import stack
from sparsetrack.render import Renderer

REPO = Path(__file__).resolve().parents[2]


class View:
    def __init__(self, cache_dir: str | Path):
        self.bins, self.meta = stack.load(cache_dir)
        self.R = Renderer(self.bins, self.meta)
        self.rs = int(self.meta.get("ref_start", 0))
        self.nb = int(self.meta["n_bins"])

    def crops(self, x: float, y: float, bins: list[int], half: int = 24):
        early = np.mean([self.R.crop(b, x, y, half) for b in range(self.rs, self.rs + 3)], axis=0)
        imgs = [self.R.crop(int(np.clip(b, 0, self.nb - 1)), x, y, half) for b in bins]
        return early, imgs, [im - early for im in imgs]


def to_u8(img: np.ndarray, lo: float, hi: float, zoom: int) -> np.ndarray:
    u = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.resize(u, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)


def grain_rows(view: View, x: float, y: float, bins: list[int], label: str, half: int = 24, zoom: int = 4,
               chg_window: float | None = 30.0) -> np.ndarray:
    early, imgs, chg = view.crops(x, y, bins, half)
    lo, hi = np.percentile(np.concatenate([early.ravel()] + [i.ravel() for i in imgs]), [0.5, 99.5])
    c = chg_window or float(np.percentile(np.abs(np.concatenate([d.ravel() for d in chg])), 99.5))
    top = np.concatenate([to_u8(i, lo, hi, zoom) for i in imgs], axis=1)
    bot = np.concatenate([to_u8(d, -c, c, zoom) for d in chg], axis=1)
    sheet = cv2.cvtColor(np.concatenate([top, bot], axis=0), cv2.COLOR_GRAY2BGR)
    side = 2 * half * zoom
    for j, b in enumerate(bins):
        cv2.putText(sheet, f"{label} b{b}", (j * side + 3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1,
                    cv2.LINE_AA)
    cv2.putText(sheet, f"change +/-{c:.0f}", (3, side + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1,
                cv2.LINE_AA)
    return sheet


def real_onsets(labels_path: str | Path) -> dict[str, tuple[float, float, float, int]]:
    L = json.loads(Path(labels_path).read_text())
    out = {}
    for gid, lab in L["labels"].items():
        g = L["grains"][gid]
        on = lab.get("onset", {})
        if g.get("excluded") or on.get("verdict") != "emerged_within":
            continue
        out[gid] = (g["x"], g["y"], g["r"], int(on["first_visible_bin"]))
    return out


def synth_onsets(truth_path: str | Path, fpb: int = 25) -> dict[str, tuple[float, float, float, int]]:
    T = json.loads(Path(truth_path).read_text())
    out = {}
    for gid, lab in T["labels"].items():
        g = T["grains"][gid]
        t = lab.get("truth", {})
        if "onset_frame" not in t or not g.get("isolated"):
            continue
        out[gid] = (g["x"], g["y"], g["r"], int(t["onset_frame"] // fpb))
    return out


def sheet(view: View, onsets: dict, gids: list[str], offsets=(-3, 0, 3, 6, 10, 20), out: str | Path = None,
          half: int = 24, zoom: int = 4, chg_window: float | None = 30.0, tag: str = "") -> np.ndarray:
    rows = []
    for gid in gids:
        x, y, r, on = onsets[gid]
        rows.append(grain_rows(view, x, y, [on + o for o in offsets], f"{tag}{gid}", half, zoom, chg_window))
        rows.append(np.full((4, rows[-1].shape[1], 3), 255, np.uint8))
    img = np.concatenate(rows[:-1], axis=0)
    if out:
        cv2.imwrite(str(out), img)
    return img


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--labels", help="human labels (real movie)")
    ap.add_argument("--truth", help="synthetic truth file")
    ap.add_argument("--gids", nargs="*")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--offsets", default="-3,0,3,6,10,20")
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v = View(a.cache)
    ons = real_onsets(a.labels) if a.labels else synth_onsets(a.truth)
    gids = a.gids or sorted(np.random.default_rng(a.seed).choice(sorted(ons), size=min(a.n, len(ons)),
                                                                   replace=False).tolist())
    sheet(v, ons, gids, [int(o) for o in a.offsets.split(",")], a.out, chg_window=a.window or None)
    print(a.out, gids)
