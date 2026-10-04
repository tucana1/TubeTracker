"""The tip-trajectory reader's predictions (a candidates file + a setting) with, optionally, the tip detector's
young-tube lengths on top (``sparsetrack.tipdet.apply``, Params.tipdet_young as on main), read offline from the stored
full-frame detector maps instead of running the network round each grain (``FullFrameTips`` places its crop where
``tipdet.GrainTipMaps`` would); scored and paired against 0.8.8 and against the reader alone.

    python -m prototypes.tip_trajectory.combo m1 --cands cands_m1.pkl cands_m1_st.pkl [--young]
"""
from __future__ import annotations

import argparse
import copy
import json
import math

import cv2
import numpy as np

from sparsetrack import stack, tipdet
from sparsetrack.analyze import Params, exit_edge
from sparsetrack.render import Renderer

from . import dp
from .common import OUT, baseline, cache_dir, labels


class FullFrameTips:
    """``tipdet.GrainTipMaps``'s interface (lo, hi, all_peaks) over the stored full-frame tip maps (OUT/det/<movie>)."""

    def __init__(self, movie: str, pos: np.ndarray, rs: int, nb: int, half: int = 64, sticky_px: float = 8.0):
        self.movie, self.half = movie, int(half)
        self.lo, self.hi = rs + tipdet.K + 1, nb - 2
        self.centre = np.zeros((nb, 2), int)
        c = np.round(pos[0]).astype(int)
        for b in range(nb):
            if np.hypot(*(pos[b] - c)) > sticky_px:
                c = np.round(pos[b]).astype(int)
            self.centre[b] = c

    def all_peaks(self, b: int, min_value: float = 0.03) -> list[tuple[float, float, float]]:
        f = OUT / "det" / self.movie / f"b{b:03d}.npz"
        if not f.exists():
            return []
        T = np.load(f)["tip"]
        cx, cy = self.centre[b]
        h = self.half
        H, W = T.shape
        m = np.zeros((2 * h, 2 * h), np.float32)
        y0, x0 = cy - h, cx - h
        a0, a1, b0, b1 = max(0, y0), min(H, cy + h), max(0, x0), min(W, cx + h)
        if a1 > a0 and b1 > b0:
            m[a0 - y0:a1 - y0, b0 - x0:b1 - x0] = T[a0:a1, b0:b1] / 250.0
        pk = (m == cv2.dilate(m, np.ones((9, 9), np.uint8))) & (m >= min_value)
        ys, xs = np.nonzero(pk)
        out: list = []
        for o in np.argsort(-m[ys, xs], kind="stable"):
            x, y = int(xs[o]), int(ys[o])
            if all(math.hypot(x - a, y - c_) >= 4 for a, c_, _ in out):
                out.append((x, y, float(m[y, x])))
        return [(float(cx - h + 0.5 + x), float(cy - h + 0.5 + y), v) for x, y, v in out]


def with_young(movie: str, pred: dict, ids: list[str], p: Params) -> dict:
    """``pred`` with ``tipdet.apply`` run on the given grains' readings."""
    bins, meta = stack.load(cache_dir(movie))
    R = Renderer(bins, meta)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    out = dict(pred)
    out["grains"] = [copy.deepcopy(g) if g["id"] in ids else g for g in pred["grains"]]
    for res in out["grains"]:
        if res["id"] not in ids or res.get("status") == "unobservable":
            continue
        pos = tipdet.grain_positions(res, rs, nb)
        gm = FullFrameTips(movie, pos, rs, nb, p.tipdet_half)
        early = np.mean([np.nan_to_num(R.crop(k, float(pos[rs][0]), float(pos[rs][1]), 64)) for k in range(rs, rs + 3)],
                        axis=0)
        edges: dict = {}

        def edge(theta, early=early, res=res, edges=edges):
            k = int(round(math.degrees(theta) % 360 / 5.0)) % 72
            if k not in edges:
                edges[k] = exit_edge(early, 63.5, float(res["r"]), math.radians(5.0 * k))
            return edges[k]

        tipdet.apply(res, gm, pos, edge, meta, p)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--setting", default="tune_ld_c.json")
    ap.add_argument("--young", action="store_true")
    ap.add_argument("--onset", default="off")
    ap.add_argument("--mid", action="store_true", help="also with lengths along the middle of the tube (centred.py)")
    ap.add_argument("--prob", help="tube maps the candidates were built on (for --mid; default the shipped ones)")
    a = ap.parse_args(argv)
    m = a.movie
    lab, base = labels(m), baseline(m)
    pg_b = dp.per_grain(dp.score(lab, base))
    p = {**dp.DEFAULT, **json.loads((OUT / a.setting).read_text())["best"]}
    tp = Params(tipdet_young=True, tipdet_onset=a.onset)
    ref = None
    for name in a.cands:
        doc = dp.load(m, name)
        dp.BACK = None
        ev = dp.evaluate(m, doc, p)
        variants = [("reader", ev["pred"])]
        if a.young:
            variants.append((f"reader + tipdet_young (onset {a.onset})", with_young(m, ev["pred"], list(doc["grains"]), tp)))
        if a.mid:
            from .centred import mid_corrections
            q = dp.with_speed(doc, dp.with_scale(doc, p))
            choices = {gid: dp.viterbi(G, doc["rs"], doc["nb"], q)[0] for gid, G in doc["grains"].items()}
            prob_dir = a.prob if (a.prob and "_st" in name) else None
            dp.BACK = mid_corrections(doc, m, choices, prob_dir)
            evm = dp.evaluate(m, doc, p)
            variants.append(("reader + mid", evm["pred"]))
            if a.young:
                variants.append(("reader + mid + tipdet_young", with_young(m, evm["pred"], list(doc["grains"]), tp)))
            dp.BACK = None
        for label, pred in variants:
            rep = dp.score(lab, pred)
            pg = dp.per_grain(rep)
            ref = pg if ref is None else ref
            pb, pr = dp.paired(pg_b, pg), dp.paired(ref, pg)
            f = lambda x: f"{x[0]:+d} [{x[1]:+.0f}, {x[2]:+.0f}]"
            print(f"{m} {name} {label}: onsets {rep['onset']['hits']}/{rep['onset']['n_timed']} lengths "
                  f"{rep['length_full']['within_tolerance']}/{rep['length_full']['n']} l&t {rep['tips']['length_and_tip']} | "
                  f"vs 0.8.8 on {f(pb[0])} len {f(pb[1])} l&t {f(pb[2])} | vs first row on {f(pr[0])} len {f(pr[1])} "
                  f"l&t {f(pr[2])}", flush=True)


if __name__ == "__main__":
    main()
