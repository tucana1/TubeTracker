"""How well grains are followed: against the truth of a synthetic movie (``synth_moves``), or label-free on a real
movie (how many grains are followed, and for how long).

    python -m prototypes.grain_tracking.evaluate_tracks synth runs/grain_tracking/synth/moves_m2_s31_cache \
        runs/grain_tracking/synth/moves_m2_s31_truth.json
    python -m prototypes.grain_tracking.evaluate_tracks real runs/sparsetrack/m1

Trackers: "phase" (0.6.0: ``local_shifts`` on the analysis crop, then ``checked_drift``, which drops an implausible
track for zero drift) and "follow" (``analyze.followed_drift``, Params.grain_track "follow").
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sparsetrack import stack
from sparsetrack.analyze import Params, checked_drift, followed_drift, local_shifts
from sparsetrack.render import Renderer


def phase_drift(renderer: Renderer, meta: dict, g: dict, p: Params) -> tuple[np.ndarray, str | None]:
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    crops = np.stack([renderer.crop(b, g["x"], g["y"], p.half) for b in range(rs, nb)])
    if np.isnan(crops).any():
        crops = np.nan_to_num(crops, nan=float(np.nanmedian(crops)))
    ls = local_shifts(crops, p.half - 0.5, g["r"], p.reg_pad, p.ref_bins)
    return checked_drift(ls, p)


def synth(cache: Path, truth_path: Path, n_static: int = 12, seed: int = 0) -> dict:
    bins, meta = stack.load(cache)
    r = Renderer(bins, meta)
    census = json.loads((cache / "grains.json").read_text())["grains"]
    truth = json.loads(truth_path.read_text())
    tracks = truth["grain_tracks"]["tracks"]
    by_census = {}
    for tid, tg in truth["grains"].items():
        by_census.setdefault(tg["census_id"], tg)
    p = Params()
    pf = Params(grain_track="follow")
    rows = []
    rng = np.random.default_rng(seed)
    static = [cid for cid in by_census if cid not in tracks]
    chosen = [cid for cid, tr in tracks.items() if tr["kind"] in ("drift", "push")] + \
        list(rng.choice(static, min(n_static, len(static)), replace=False))
    for cid in chosen:
        tg = by_census.get(cid)
        if tg is None:
            continue
        d = [np.hypot(g["x"] - tg["x"], g["y"] - tg["y"]) for g in census]
        k = int(np.argmin(d))
        if d[k] > 6:
            continue
        g = census[k]
        others = [o for o in census if o["id"] != g["id"]]
        nb = int(meta["n_bins"])
        if cid in tracks:
            t_xy = np.asarray(tracks[cid]["xy"], float)[:nb]
            kind, vanish = tracks[cid]["kind"], tracks[cid]["vanish_bin"]
        else:
            t_xy, kind, vanish = np.zeros((nb, 2)), "static", None
        t_rel = t_xy - t_xy[:3].mean(axis=0)
        # the census finds the grain at its reference-bin place; drifts are measured from there
        t0 = time.time()
        ph, flag = phase_drift(r, meta, g, p)
        t_phase = time.time() - t0
        t0 = time.time()
        fd = followed_drift(r, meta, g, others, pf)
        t_follow = time.time() - t0
        fo = fd["drift"]
        seen = np.arange(nb) < (vanish if vanish is not None else nb)
        e_ph = np.hypot(*(ph - t_rel).T)[seen]
        fin = np.isfinite(fo[:, 0])
        e_fo = np.hypot(*(np.where(fin[:, None], fo, 0.0) - t_rel).T)[seen & fin]
        rows.append({"grain": g["id"], "truth": tg["id"], "kind": kind, "reach": float(np.hypot(*t_rel.T).max()),
                     "vanish_bin": vanish, "phase_flag": flag, "follow_lost_from": fd["lost_from"],
                     "follow_lost_reason": fd["lost_reason"], "followed_bins": int(fin.sum()),
                     "seen_bins": int(seen.sum()),
                     "phase_err_median": float(np.median(e_ph)), "phase_err_p90": float(np.percentile(e_ph, 90)),
                     "phase_within2": float(np.mean(e_ph <= 2.0)),
                     "follow_err_median": float(np.median(e_fo)) if len(e_fo) else None,
                     "follow_err_p90": float(np.percentile(e_fo, 90)) if len(e_fo) else None,
                     "follow_within2_of_seen": float(np.sum(e_fo <= 2.0) / max(seen.sum(), 1)),
                     "t_phase_s": round(t_phase, 2), "t_follow_s": round(t_follow, 2)})
        print(json.dumps(rows[-1]), flush=True)
    return {"cache": str(cache), "rows": rows}


def summary(rows: list[dict]) -> str:
    out = []
    for kind in ("static", "drift", "push", "all moving"):
        sel = [x for x in rows if (x["kind"] == kind if kind != "all moving" else x["kind"] in ("drift", "push"))]
        if not sel:
            continue
        seen = sum(x["seen_bins"] for x in sel)
        ph = sum(x["phase_within2"] * x["seen_bins"] for x in sel) / seen
        fo = sum(x["follow_within2_of_seen"] * x["seen_bins"] for x in sel) / seen
        out.append(f"{kind:12s} grains {len(sel):3d}: bins within 2 px of the truth: phase {100 * ph:5.1f}%  follow "
                   f"{100 * fo:5.1f}% | median error phase {np.median([x['phase_err_median'] for x in sel]):.2f} "
                   f"follow {np.median([x['follow_err_median'] or 0 for x in sel]):.2f} px | phase drift_rejected "
                   f"{sum(x['phase_flag'] == 'drift_rejected' for x in sel)}")
    van = [x for x in rows if x["vanish_bin"] is not None]
    if van:
        lag = [x["follow_lost_from"] - x["vanish_bin"] for x in van if x["follow_lost_from"] is not None]
        out.append(f"vanishing grains {len(van)}: lost by follow {len(lag)} (lost bin - vanish bin: "
                   f"{sorted(lag)}); never lost {len(van) - len(lag)}")
    kept = [x for x in rows if x["vanish_bin"] is None]
    false = [x for x in kept if x["follow_lost_from"] is not None]
    out.append(f"grains that stay: {len(kept)}, lost by follow anyway: {len(false)} "
               f"{[(x['grain'], x['kind'], x['follow_lost_from'], x['follow_lost_reason']) for x in false]}")
    out.append(f"time per grain: phase {np.mean([x['t_phase_s'] for x in rows]):.1f} s, follow "
               f"{np.mean([x['t_follow_s'] for x in rows]):.1f} s")
    return "\n".join(out)


def real(cache: Path, only: list[str] | None = None) -> dict:
    """Label-free: every census grain followed; how long, how far, where lost."""
    bins, meta = stack.load(cache)
    r = Renderer(bins, meta)
    census = json.loads((cache / "grains.json").read_text())["grains"]
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    pf, p = Params(grain_track="follow"), Params()
    rows = []
    for g in census:
        if only and g["id"] not in only:
            continue
        others = [o for o in census if o["id"] != g["id"]]
        fd = followed_drift(r, meta, g, others, pf)
        ph, flag = phase_drift(r, meta, g, p)
        fin = np.isfinite(fd["drift"][:, 0])
        rows.append({"grain": g["id"], "x": g["x"], "y": g["y"], "border": g.get("border"),
                     "isolated": g.get("isolated"), "lost_from": fd["lost_from"], "lost_reason": fd["lost_reason"],
                     "followed_bins": int(fin.sum()), "n_bins": nb - rs,
                     "reach_px": round(float(np.nanmax(np.hypot(*fd["drift"].T))) if fin.any() else 0.0, 1),
                     "phase_flag": flag,
                     "drift": np.round(np.nan_to_num(fd["drift"], nan=np.nan), 2).tolist()})
        x = rows[-1]
        print(f"{x['grain']} lost_from {x['lost_from']} ({x['lost_reason']}) followed {x['followed_bins']}/"
              f"{x['n_bins']} reach {x['reach_px']} phase {flag}", flush=True)
    return {"cache": str(cache), "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("synth", "real"))
    ap.add_argument("cache")
    ap.add_argument("truth", nargs="?")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.mode == "synth":
        res = synth(Path(args.cache), Path(args.truth))
        print(summary(res["rows"]))
    else:
        res = real(Path(args.cache))
    if args.out:
        Path(args.out).write_text(json.dumps(res))
