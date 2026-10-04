"""Where the body starts (diagnosis + post-pass): a tube that leaves its grain tangentially runs along the rim from where
it emerged before it parts from the grain; the cheapest route from the rim starts where it parts (and is short by the
hugging stretch). Post-pass on a reading: at each bin, from the start of the chosen body, follow the tube map's band
along the rim (r - 2 .. r + ``BAND_OUT`` px, P >= ``P_BAND``, connected to the start) away from the tip's side, and add
``w_back`` x its length (at most ``MAX_BACK`` px). The band's length is measured geodesically along it.

    python -m prototypes.tip_trajectory.backrim ld --w 1.0
"""
from __future__ import annotations

import argparse
import json
import math

import cv2
import numpy as np
from skimage.graph import MCP_Geometric

from . import dp
from .cands import HALF, crop_u8
from .common import OUT, PROB, baseline, cache_dir, labels

BAND_OUT, P_BAND, MAX_BACK = 7.0, 0.5, 30.0


def back_lengths(doc: dict, movie: str, choice_by_grain: dict, prob=None) -> dict:
    """Per grain, per bin with a chosen candidate: the length of the map's rim band behind the body's start."""
    prob = prob if prob is not None else np.load(cache_dir(movie) / PROB / "bins.npy", mmap_mode="r")
    H = 48
    S = 2 * H
    jj, ii = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    out = {}
    for gid, G in doc["grains"].items():
        choice = choice_by_grain[gid]
        r = G["r"]
        res = {}
        for i, k in enumerate(choice):
            if k < 0:
                continue
            b = doc["rs"] + i
            B = G["bins"][b]
            body = B["bodies"][k] / 10.0  # grain frame
            cx, cy = G["x"] + G["drift"][b][0], G["y"] + G["drift"][b][1]
            ix, iy = int(round(cx)), int(round(cy))
            gx, gy = cx - (ix - H) - 0.5, cy - (iy - H) - 0.5
            to_crop = np.array([G["drift"][b][0] - (ix - H) - 0.5, G["drift"][b][1] - (iy - H) - 0.5])
            s0 = body[0] + to_crop
            tip = body[-1] + to_crop
            Pc = crop_u8(np.asarray(prob[b]), ix, iy, H).astype(np.float32) / 250.0
            rg = np.hypot(jj - gx, ii - gy)
            band = (rg >= r - 2) & (rg <= r + BAND_OUT) & (Pc >= P_BAND)
            si, sj = int(round(s0[1])), int(round(s0[0]))
            if not (0 <= si < S and 0 <= sj < S):
                continue
            win = band[max(si - 2, 0):si + 3, max(sj - 2, 0):sj + 3]
            if not win.any():
                res[b] = 0.0
                continue
            n, lab = cv2.connectedComponents(band.astype(np.uint8), connectivity=8)
            ids = np.unique(lab[max(si - 2, 0):si + 3, max(sj - 2, 0):sj + 3][win])
            comp = np.isin(lab, ids[ids > 0])
            a_s = math.atan2(s0[1] - gy, s0[0] - gx)
            a_t = math.atan2(tip[1] - gy, tip[0] - gx)
            side = math.remainder(a_t - a_s, 2 * math.pi)
            ang = np.arctan2(ii - gy, jj - gx)
            rel = np.angle(np.exp(1j * (ang - a_s)))
            behind = comp & ((rel * (1 if side >= 0 else -1)) < -0.05)  # on the other side of the start from the tip
            if not behind.any():
                res[b] = 0.0
                continue
            cost = np.where(comp, 1.0, np.inf)
            cost[max(si - 2, 0):si + 3, max(sj - 2, 0):sj + 3] = 1.0
            mcp = MCP_Geometric(cost, fully_connected=True)
            cum, _ = mcp.find_costs([(si, sj)])
            vals = np.where(behind & np.isfinite(cum), cum, -1.0)
            res[b] = float(min(max(vals.max(), 0.0), MAX_BACK))
        out[gid] = res
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--w", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--cands", default="cands_{movie}.pkl")
    a = ap.parse_args(argv)
    p = {**dp.DEFAULT, **json.loads((OUT / "tune_ld_c.json").read_text())["best"]}
    doc = dp.load(a.movie, a.cands.format(movie=a.movie))
    q = dp.with_speed(doc, dp.with_scale(doc, p))
    choices = {gid: dp.viterbi(G, doc["rs"], doc["nb"], q)[0] for gid, G in doc["grains"].items()}
    back = back_lengths(doc, a.movie, choices)
    nz = [v for g in back.values() for v in g.values() if v > 0]
    print(f"{a.movie}: rim band behind the start at {len(nz)} of {sum(len(g) for g in back.values())} chosen bins, median "
          f"{np.median(nz) if nz else 0:.1f} px", flush=True)
    lab, base = labels(a.movie), baseline(a.movie)
    pg_b = dp.per_grain(dp.score(lab, base))
    ref = None
    for w in [0.0] + list(a.w):
        dp.BACK = {gid: {b: w * v for b, v in g.items()} for gid, g in back.items()} if w > 0 else None
        ev = dp.evaluate(a.movie, doc, p)
        pg = dp.per_grain(ev["rep"])
        ref = pg if ref is None else ref
        pb, pr = dp.paired(pg_b, pg), dp.paired(ref, pg)
        print(f"  w_back {w}: lengths {ev['lengths']}/{ev['n']} l&t {ev['lt']} onsets {ev['onsets']} | vs 0.8.8 len "
              f"{pb[1][0]:+d} [{pb[1][1]:+.0f}, {pb[1][2]:+.0f}] | vs w 0: len {pr[1][0]:+d} [{pr[1][1]:+.0f}, {pr[1][2]:+.0f}]"
              f" l&t {pr[2][0]:+d}", flush=True)
    dp.BACK = None


if __name__ == "__main__":
    main()
