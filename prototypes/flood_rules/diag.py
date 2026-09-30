"""Why the flood reads (or misses) a grain's tube: run in a ``bench serve`` process on its maps in memory.

    python -m prototypes.flood_rules.bench submit QUEUE NAME diag=prototypes.flood_rules.diag:grains [max_len=8]

``grains``: for every grain the flood reads, its start history (each bin before it starts, or restarts, at which a
new piece touches the start band, and why the piece was refused: an arc round the rim, or joined to older material
far out) and, per FULL trace, the flood's length there against the human's, with what the map marks at the trace
(P along it, how far beyond the rim the marks in its direction reach, and when they arrived).
"""

from __future__ import annotations

import math

import numpy as np


def _capture(M, g, p) -> dict:
    """read_grain with its arrival map and flood captured."""
    from sparsetrack import learned
    cap = {}
    orig_flood, orig_arr = learned.flood, learned.arrivals

    def arrivals(present, blocked, *a, **k):
        cap["present_any"] = present.any(axis=0)
        cap["present"] = present
        return orig_arr(present, blocked, *a, **k)

    def flood(arr, rg, ang, blocked, gr, *a, **k):
        out = orig_flood(arr, rg, ang, blocked, gr, *a, **k)
        cap.update(arr=arr, rg=rg, ang=ang, blocked=blocked, gr=gr, fl=out)
        return out

    learned.flood, learned.arrivals = flood, arrivals
    try:
        others = [o for o in M.physical if o["id"] != g["id"]]
        cap["res"] = learned.read_grain(M.renderer, M.prob, M.meta, g, others, p)
    finally:
        learned.flood, learned.arrivals = orig_flood, orig_arr
    return cap


def _starts(cap: dict, p, until: int) -> list[dict]:
    """Every new piece touching the start band before bin ``until``, with the start rules' verdict."""
    import cv2
    arr, rg, ang, blocked, gr = cap["arr"], cap["rg"], cap["ang"], cap["blocked"], cap["gr"]
    blocked = blocked | (rg < gr + p.flood_halo)
    start = (rg >= gr + p.flood_halo) & (rg <= gr + p.flood_halo + p.flood_start_band)
    out = []
    for b in range(min(until, int(arr.max()))):
        new = (arr == b) & ~blocked
        if not (new & start).any():
            continue
        nl, lab = cv2.connectedComponents(new.astype(np.uint8), connectivity=8)
        _, lab_all = cv2.connectedComponents(((arr <= b) & ~blocked).astype(np.uint8), connectivity=8)
        old_far = (arr < b - p.flood_old_far_bins) & (rg > gr + p.flood_old_far_px)
        for l in range(1, nl):
            c = lab == l
            if not (c & start).any():
                continue
            a = np.angle(np.exp(1j * (ang[c] - np.angle(np.mean(np.exp(1j * ang[c]))))))
            arc = float(np.rad2deg(np.ptp(a)))
            old = bool((np.isin(lab_all, np.unique(lab_all[c])) & old_far).any())
            out.append({"bin": b, "px": int(c.sum()), "reach": round(float(rg[c].max() - gr), 1),
                        "deg": round(float(np.rad2deg(np.angle(np.mean(np.exp(1j * ang[c]))))), 0),
                        "arc": round(arc, 0), "old_far": old,
                        "verdict": "arc" if arc > p.flood_arc_deg else "old_far" if old else "start"})
    return out


def flood_events(arr, rg, ang, blocked, gr, p) -> tuple[dict, list]:
    """``learned.flood`` (same arguments as ``read_grain`` passes) with its history: every start, every reset (give
    up) and the bins it grew at, and the length then. The flood's output is checked against ``learned.flood``'s."""
    import cv2
    from sparsetrack import learned
    from sparsetrack.learned import _extend_dist
    recent, bridge, start_band = p.flood_recent, p.flood_bridge, p.flood_start_band
    min_len, give_up, halo = p.flood_min_len, p.flood_give_up, p.flood_halo
    n = int(arr.max())
    blocked = blocked | (rg < gr + halo)
    start = (rg >= gr + halo) & (rg <= gr + halo + start_band)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge + 1, 2 * bridge + 1))
    tube = np.zeros(arr.shape, bool)
    t_in = np.full(arr.shape, -1, np.int32)
    dist = np.full(arr.shape, np.inf)
    length = np.zeros(n)
    emerge = grew = None
    ev = []
    for b in range(n):
        if emerge is not None and b - grew > give_up and not (tube & (rg > gr + min_len)).any():
            fin = tube & np.isfinite(dist)
            ev.append({"bin": b, "event": "reset", "had_px": int(tube.sum()),
                       "reach": round(float(rg[tube].max() - gr), 1) if tube.any() else 0.0,
                       "len": round(float(dist[fin].max()), 1) if fin.any() else 0.0, "started": emerge})
            tube[:], t_in[:], dist[:], emerge = False, -1, np.inf, None
            length[:b] = 0.0
        new = (arr == b) & ~blocked
        if new.any():
            nl, lab = cv2.connectedComponents(new.astype(np.uint8), connectivity=8)
            if not tube.any():
                _, lab_all = cv2.connectedComponents(((arr <= b) & ~blocked).astype(np.uint8), connectivity=8)
                old_far = (arr < b - p.flood_old_far_bins) & (rg > gr + p.flood_old_far_px)
                for l in range(1, nl):
                    c = lab == l
                    if not (c & start).any():
                        continue
                    a = np.angle(np.exp(1j * (ang[c] - np.angle(np.mean(np.exp(1j * ang[c]))))))
                    if np.ptp(a) > np.deg2rad(p.flood_arc_deg):
                        ev.append({"bin": b, "event": "refused:arc", "px": int(c.sum()),
                                   "arc": round(float(np.rad2deg(np.ptp(a))), 0)})
                        continue
                    if (np.isin(lab_all, np.unique(lab_all[c])) & old_far).any():
                        ev.append({"bin": b, "event": "refused:old_far", "px": int(c.sum()),
                                   "reach": round(float(rg[c].max() - gr), 1)})
                        continue
                    tube |= c
                    t_in[c] = b
                    dist[c & start] = rg[c & start] - gr
                    _extend_dist(dist, c, c & start, 1)
                    ev.append({"bin": b, "event": "start", "px": int(c.sum()),
                               "deg": round(float(np.rad2deg(np.angle(np.mean(np.exp(1j * ang[c]))))), 0),
                               "reach": round(float(rg[c].max() - gr), 1)})
                if tube.any():
                    emerge = grew = b
            else:
                tip = tube & (t_in >= t_in.max() - recent)
                seeds = cv2.dilate(tip.astype(np.uint8), ker).astype(bool)
                for l in range(1, nl):
                    c = lab == l
                    if (c & seeds).any():
                        _extend_dist(dist, c, tip, bridge)
                        tube |= c
                        t_in[c] = b
                        grew = b
        fin = tube & np.isfinite(dist)
        length[b] = float(dist[fin].max()) if fin.any() else 0.0
    out = {"tube": tube, "t_in": t_in, "dist": dist, "emerge": emerge, "length": np.maximum.accumulate(length)}
    ref = learned.flood(arr, rg, ang, blocked, gr, recent, bridge, start_band, min_len, give_up, halo,
                        p.flood_arc_deg, p.flood_old_far_px, p.flood_old_far_bins)
    assert np.array_equal(ref["length"], out["length"]) and ref["emerge"] == out["emerge"]
    return out, ev


def exits(M, grain: str, bins: str, **overrides) -> dict:
    """The flood's reach and its length from the exit (``from_exit``) at some bins, with the centreline's points."""
    from sparsetrack import learned
    from sparsetrack.analyze import Params
    p = Params(**{"model": M.model, **overrides})
    g = next(x for x in M.grains if x["id"] == grain)
    rs = int(M.meta.get("ref_start", 0))
    cap = _capture(M, g, p)
    fl, gr = cap["fl"], cap["gr"]
    tube, t_in, dist = fl["tube"], fl["t_in"], fl["dist"]
    centre = p.flood_half - 0.5
    zone = gr + p.flood_halo + p.flood_start_band
    out = {"reported": {}, "rows": []}
    for b in (int(x) for x in (bins.split(",") if isinstance(bins, str) else np.atleast_1d(bins))):
        t = b - rs
        sel = tube & (t_in <= t) & np.isfinite(dist)
        if not sel.any():
            out["rows"].append({"bin": b, "reach": 0.0})
            continue
        y, x = np.unravel_index(int(np.argmax(np.where(sel, dist, -1.0))), dist.shape)
        line = learned.centreline(sel, dist, (int(y), int(x)), centre, gr, p.flood_bridge + 0.5)
        cut, L = learned.from_exit(line, centre, gr, zone)
        out["rows"].append({"bin": b, "reach": float(fl["length"][t]), "exit_len": L,
                            "line_r": [round(float(np.hypot(q[0] - centre, q[1] - centre) - gr), 1) for q in line],
                            "cut_r": [round(float(np.hypot(q[0] - centre, q[1] - centre) - gr), 1) for q in cut],
                            "tip_r": round(float(np.hypot(y - centre, x - centre) - gr), 1), "tip_dist": float(dist[y, x])})
        out["reported"][b] = cap["res"]["length"]["px"][t]
    return out


def picture(M, grain: str, bin: int, size: int = 14, at_tube: bool = True, **overrides) -> dict:
    """ASCII views round a trace's exit at one bin: the arrival bin of each pixel relative to that bin (digits = bins
    before it, '+' later / never, '#' blocked), which pixels the flood holds then ('T'), P x 10 at that bin, and the
    human trace's pixels ('*')."""
    from sparsetrack.analyze import Params
    p = Params(**{"model": M.model, **overrides})
    g = next(x for x in M.grains if x["id"] == grain)
    rs = int(M.meta.get("ref_start", 0))
    cap = _capture(M, g, p)
    arr, rg, gr, fl = cap["arr"], cap["rg"], cap["gr"], cap["fl"]
    k = bin - rs
    half = p.flood_half
    centre = half - 0.5
    t = ((M.labels["labels"][grain].get("traces") or {}).get(str(bin))) or {}
    path = np.asarray(t.get("path_xy_ref") or [[g["x"], g["y"]]], float)
    ax_, ay_ = (path[-1] + path[0]) / 2 - [g["x"], g["y"]]
    if at_tube:  # centre the view on the flood's tube pixels in the trace's direction (a drifting grain's trace is off)
        sector = np.abs(np.angle(np.exp(1j * (cap["ang"] - math.atan2(ay_, ax_))))) <= np.deg2rad(45)
        sel = fl["tube"] & (fl["t_in"] <= k) & sector & (rg < gr + 15)
        if sel.any():
            yy_, xx_ = np.nonzero(sel)
            ay_, ax_ = yy_.mean() - centre, xx_.mean() - centre
    cy, cx = int(round(centre + ay_)), int(round(centre + ax_))
    ys, xs = range(cy - size, cy + size + 1), range(cx - size, cx + size + 1)
    blocked = cap["blocked"] | (rg < gr + p.flood_halo)
    held = fl["tube"] & (fl["t_in"] <= k)
    # human trace pixels (census place; the view offset of a moving grain is ignored)
    trace = np.zeros(arr.shape, bool)
    if len(path) >= 2:
        for a_, b_ in zip(path[:-1], path[1:]):
            for s in np.linspace(0, 1, 40):
                q = a_ + s * (b_ - a_) - [g["x"], g["y"]]
                trace[int(round(centre + q[1])), int(round(centre + q[0]))] = True
    pres = cap["present"][k]  # P >= flood_p at that bin, registered as the flood read it
    rows_a, rows_p = [], []
    for y in ys:
        la, lp = "", ""
        for x in xs:
            if held[y, x]:
                ch = "T"
            elif blocked[y, x]:
                ch = "#"
            elif arr[y, x] > k:
                ch = "+"
            else:
                ch = str(min(9, k - int(arr[y, x]))) if k - arr[y, x] >= 0 else "+"
            la += ("*" if trace[y, x] and ch in "+#" else ch)
            lp += ("o" if pres[y, x] else ".") if not (np.hypot(y - centre, x - centre) < gr) else "g"
        rows_a.append(la)
        rows_p.append(lp)
    return {"grain": grain, "bin": bin, "emerge_abs": None if fl["emerge"] is None else fl["emerge"] + rs,
            "human": t.get("length_px"), "flood_len_then": float(fl["length"][k]),
            "legend": "arrival: T held by the flood, digit = arrived that many bins ago (9: 9 or more), + not yet, # blocked"
                      " (halo/other grain), * trace pixel not marked; P: digit = P x 10, . < 0.05",
            "arrival": rows_a, "P": rows_p}


def grains(M, max_len: float = 1e9, only: str = "", **overrides) -> dict:
    from sparsetrack.analyze import Params
    p = Params(**{"model": M.model, **overrides})
    fpb = int(M.meta["frames_per_bin"])
    rs = int(M.meta.get("ref_start", 0))
    ids = set(only.split(",")) if only else None
    out = {}
    for g in M.grains:
        gid = g["id"]
        if ids is not None and gid not in ids:
            continue
        lab = M.labels["labels"].get(gid) or {}
        full = {int(b): t for b, t in (lab.get("traces") or {}).items() if t["state"] == "full" and not t.get("contact")}
        if ids is None and not any(t["length_px"] <= max_len for t in full.values()):
            continue
        cap = _capture(M, g, p)
        fl, res = cap["fl"], cap["res"]
        arr, rg, ang, gr = cap["arr"], cap["rg"], cap["ang"], cap["gr"]
        half = p.flood_half
        centre = half - 0.5
        emerge = fl["emerge"]
        rows = []
        for b, t in sorted(full.items()):
            k = b - rs
            L = res["length"]["px"][k] if 0 <= k < len(res["length"]["px"]) else None
            path = np.asarray(t.get("path_xy_ref") or [], float)
            mark = {}
            if len(path) >= 2:
                # the trace's exit direction and apex in crop pixels (the grain's census place: a moved grain is off)
                ex, ey = path[0] - [g["x"], g["y"]]
                ax_, ay_ = path[-1] - [g["x"], g["y"]]
                a0 = math.atan2(ay_, ax_) if math.hypot(ax_, ay_) > 1 else math.atan2(ey, ex)
                sector = np.abs(np.angle(np.exp(1j * (ang - a0)))) <= np.deg2rad(25)
                arrived = sector & (arr <= k)
                mark = {"apex_r": round(float(math.hypot(ax_, ay_) - gr), 1),
                        "marks_reach": round(float(rg[arrived].max() - gr), 1) if arrived.any() else None,
                        "arrived_px": int(arrived.sum()),
                        "first_arrival_in_sector": int(arr[sector & (rg >= gr + 1)].min()) if (sector & (rg >= gr + 1)).any() else None}
            rows.append({"bin": b, "human": round(t["length_px"], 1), "model": L,
                         "hit": None if L is None else abs(L - t["length_px"]) <= max(2.0, 0.1 * t["length_px"]),
                         **mark})
        on = lab.get("onset") or {}
        _, ev = flood_events(arr, rg, ang, cap["blocked"], gr, p)
        refused = [e for e in ev if e["event"].startswith("refused")]
        out[gid] = {"emerge": emerge, "onset_bin": None if res.get("onset_frame") is None else res["onset_frame"] // fpb,
                    "status": res["status"], "flags": res["flags"], "human_fv": on.get("first_visible_bin"),
                    "human_la": on.get("last_absent_bin"), "gr": gr,
                    "starts": _starts(cap, p, (emerge if emerge is not None else int(arr.max())) + 1)[:40],
                    "events": [e for e in ev if not e["event"].startswith("refused")][:30] + refused[:12],
                    "n_refused": len(refused),
                    "flood_len": [round(float(v), 1) for v in fl["length"][::5]],
                    "read_len": [round(float(v), 1) for v in res["length"]["px"][::5]],
                    "traces": rows}
    return out
