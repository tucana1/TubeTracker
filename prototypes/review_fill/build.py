"""Build the carried routes and kymographs the simulated reviews read (evaluate.py).

Per scored grain (isolated, not excluded) with >= 2 FULL human traces: the two traces a simulated reviewer gives,
the LATEST full trace (protocol P1) and the EARLIEST full trace with a tube >= 8 px (P2; P3 = both), each made a
route (``fill.anchor_route``: the trace, on along 0.8.8's route at that bin if the apex lies on it, then straight
40 px) and carried through the whole movie (``fill.carry_all``). P3's route joins the two (``fill.join``).
Saved per grain: the kymographs K (tube map maxed over +/- w px across the route, w = 1, 2) of the early (e), late
(l) and joined (j) routes, and those routes at every traced bin (for tips and route checks).

    python -m prototypes.review_fill.build ld [--only g001,g002]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from prototypes.review_fill.fill import (GrainFlow, anchor_route, arclen, carry_all, dense, forward_join, join,
                                         kymograph)
from sparsetrack import stack
from sparsetrack.render import Renderer
from sparsetrack.report import turned_path

DATA = Path("/Users/joshjiang/Documents/TubeTracker")
WORK = Path(__file__).resolve().parents[2]
OUT = WORK / "runs/research/review_fill"
BASE = Path("/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/eaa2b708-9f78-4565-aa0c-9af3763da8fd/"
            "scratchpad/bt/base088")
PROB = "prob_tubes_bn_real_ld_m2"
MIN_EARLY_PX = 8.0
WS = (1.0, 2.0)

cv2.setNumThreads(1)  # the machine is shared


def labels(movie: str) -> dict:
    return json.loads((WORK / f"benchmark/labels/{movie}_v1.json").read_text())


def baseline(movie: str) -> dict:
    return json.loads((BASE / f"{movie}_real_0/predictions.json").read_text())


def full_traces(lab: dict, gid: str) -> list[tuple[int, dict]]:
    tr = (lab["labels"].get(gid) or {}).get("traces") or {}
    return sorted(((int(b), t) for b, t in tr.items() if t["state"] == "full"), key=lambda x: x[0])


def grains(lab: dict) -> list[str]:
    """Scored grains (isolated, not excluded) with at least two FULL traces."""
    return [gid for gid, g in sorted(lab["grains"].items())
            if not g.get("excluded") and g.get("isolated", True) and len(full_traces(lab, gid)) >= 2]


def anchors_of(traces: list) -> tuple:
    """(early, late): the earliest full trace with a tube >= 8 px (None if none) and the latest."""
    late = traces[-1]
    early = next(((b, t) for b, t in traces if float(t["length_px"]) >= MIN_EARLY_PX), None)
    return early, late


def series_index(res: dict, b: int, fpb: int) -> int:
    frames = res["length"]["frames"]
    return int(min(max(b - frames[0] // fpb, 0), len(frames) - 1))


def model_route(res: dict | None, pred: dict, b: int) -> np.ndarray | None:
    """0.8.8's drawn route at bin b, reference coordinates where the tube lay then (None if it drew none)."""
    if not res or not res.get("path") or len(res["path"]) < 2:
        return None
    fpb = int(pred.get("frames_per_bin", 300))
    t = series_index(res, b, fpb)
    try:
        path = turned_path(res, t, pred)
    except Exception:  # noqa: BLE001
        return None
    drift = (res.get("drift") or {}).get("xy")
    if drift:
        path = path + np.asarray(drift[min(t, len(drift) - 1)], float)
    return path if len(path) >= 2 and arclen(path)[-1] >= 3.0 else None


def build_grain(R: Renderer, P, lab: dict, pred: dict, gid: str) -> tuple[dict, dict]:
    g = lab["grains"][gid]
    traces = full_traces(lab, gid)
    res = next((x for x in pred["grains"] if x["id"] == gid), None)
    early, late = anchors_of(traces)
    rs, nb = int(R.ref_start), int(R.n_bins)
    picks = {"l": late}
    if early is not None and early[0] != late[0]:
        picks["e"] = early
    routes, info = {}, {}
    for k, (b, t) in picks.items():
        F, L, joined = anchor_route(t["path_xy_ref"], model_route(res, pred, b))
        routes[k] = F
        info[k] = {"bin": b, "L": L, "joined_model": joined, "N": len(F)}
    t0 = time.time()
    gf = GrainFlow(R, P, [dense(t["path_xy_ref"]) for _, t in traces] + list(routes.values()), (g["x"], g["y"]))
    carried = {k: carry_all(gf, routes[k], info[k]["bin"]) for k in routes}
    n_flows = len(gf._flow)
    t_carry = time.time() - t0
    if "e" in carried:
        ie, il = min(max(info["e"]["bin"], rs), nb - 1) - rs, min(max(info["l"]["bin"], rs), nb - 1) - rs
        carried["j"] = join(carried["e"], carried["l"], ie, il, info["e"]["L"])
        # P2 alternative: beyond the early trace's carried apex, 0.8.8's own route at each later bin
        carried["f"] = forward_join(carried["e"], ie, info["e"]["L"],
                                    [model_route(res, pred, rs + i) for i in range(nb - rs)])
    gf._flow.clear()
    t1 = time.time()
    arrs = {}
    for k, Rk in carried.items():
        for w, K in kymograph(gf, Rk, WS).items():
            arrs[f"K_{k}_{w:g}"] = K.astype(np.float16)
    t_kymo = time.time() - t1
    bins = [b for b, _ in traces]
    idx = [min(max(b, rs), nb - 1) - rs for b in bins]
    for k, Rk in carried.items():
        arrs[f"X_{k}"] = Rk[idx].astype(np.float32)  # routes at the traced bins (reference coordinates)
    gf.clear()
    meta = {"gid": gid, "rs": rs, "nb": nb, "traced_bins": bins, "anchors": info, "half": gf.half,
            "flows": n_flows, "t_carry": round(t_carry, 2), "t_kymo": round(t_kymo, 2)}
    return arrs, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--only")
    a = ap.parse_args(argv)
    lab, pred = labels(a.movie), baseline(a.movie)
    bins, meta = stack.load(DATA / f"runs/sparsetrack/{a.movie}")
    R = Renderer(bins, meta)
    P = np.load(DATA / f"runs/sparsetrack/{a.movie}/{PROB}/bins.npy", mmap_mode="r")
    od = OUT / a.movie
    od.mkdir(parents=True, exist_ok=True)
    gids = grains(lab)
    if a.only:
        gids = [g for g in gids if g in a.only.split(",")]
    for gid in gids:
        if (od / f"{gid}.json").exists():
            continue
        t0 = time.time()
        arrs, m = build_grain(R, P, lab, pred, gid)
        np.savez_compressed(od / f"{gid}.npz", **arrs)
        (od / f"{gid}.json").write_text(json.dumps(m))
        an = " ".join(f"{k}:b{v['bin']}/{v['L']:.0f}px/N{v['N']}{'/model' if v['joined_model'] else ''}"
                      for k, v in m["anchors"].items())
        print(f"{a.movie} {gid} half {m['half']} flows {m['flows']} carry {m['t_carry']:.1f}s kymo {m['t_kymo']:.1f}s "
              f"total {time.time() - t0:.1f}s  {an}", flush=True)


if __name__ == "__main__":
    main()
