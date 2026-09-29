"""Kymographs of the labelled real grains, with partial labels from the human traces.

    python -m prototypes.kymo_reader.real_data ld m2

Per included grain of ``benchmark/labels/{movie}_v1.json`` (read from a copy in
runs/kymo_reader/labels_copy, as ``Bench`` may write next to its labels file) two kymographs:

- ``st``: along SparseTrack 0.5.3's own path (runs/lab_checks_2026-09-29 predictions), turned per bin
  by its rotation track about the exit (change-read grains; flood-read grains carry none);
- ``route``: along the human's latest FULL trace (the oracle route), turned by SparseTrack's rotation
  track along that route (``strot.read_route``).

Partial labels (``lab_y``, ``lab_m``: target and mask over (bin, arc)) along the human route, from the
FULL/PARTIAL/no-tube traces and the onset bracket, extended by monotonicity (see ``partial_labels``).
Along SparseTrack's path the same labels are only used where that path agrees with the human route.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import kymo, store, strot

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/kymo_reader"
CACHES = {"ld": REPO / "runs/sparsetrack/ld", "m2": REPO / "runs/sparsetrack/m2"}
ST053 = {"ld": REPO / "runs/lab_checks_2026-09-29/ld_st053.json",
         "m2": REPO / "runs/lab_checks_2026-09-29/m2_st053_predictions.json"}
VMAX = {"ld": 4.0, "m2": 4.5}  # st053's automatic front caps on these movies


def labels_path(movie: str) -> Path:
    p = OUT / "labels_copy" / f"{movie}_v1.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text((REPO / f"benchmark/labels/{movie}_v1.json").read_text())
    return p


def human_traces(lab: dict, rs: int) -> list[tuple[int, str, float, list]]:
    """(observed bin index, state, length, view path) for every trace, oldest first."""
    out = []
    for k, tr in sorted((lab.get("traces") or {}).items(), key=lambda kv: int(kv[0])):
        out.append((int(k) - rs, tr["state"], float(tr.get("length_px") or 0.0), tr.get("path_xy_view") or []))
    return out


def with_origins(traces, path: kymo.Path1, default: float = 0.0) -> list[tuple[int, str, float, float]]:
    """Each trace's own first click (where its length starts) as an arc position on ``path``
    (negative inside the path's start); ``default`` for traces without a clicked path."""
    out = []
    for t_k, state, L_k, pv in traces:
        o = default
        if len(pv) >= 2:
            d, a = dist_to_polyline(np.asarray(pv[0], float), path.pts)
            o = float(path.s[0] + a)
        out.append((t_k, state, L_k, o))
    return out


def latest_route(lab: dict) -> tuple[int, list] | None:
    full = [(int(k), tr) for k, tr in (lab.get("traces") or {}).items()
            if tr["state"] == "full" and len(tr.get("path_xy_view") or []) >= 2]
    if not full:
        return None
    k, tr = max(full, key=lambda kv: kv[0])
    return k, tr["path_xy_view"]


def partial_labels(T: int, s: np.ndarray, traces, onset: dict | None, rs: int, origin: float = 0.0,
                   m_abs: float = 2.0, m_rel: float = 0.10, rim_px: float = 2.0, stub_px: float = 1.0,
                   inside_px: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Y(t, s) = "arc s is covered by bin t" (grain body or tube up to its front), known in part:
    (target, mask). The front is F(t) = start + length, the start being where the annotator's trace
    begins (``traces`` carry each trace's own start as an arc position; ``origin`` is the grain's).

    FULL trace (t_k, L_k, o_k): Y = 1 for s < o_k + L_k - m at t >= t_k; Y = 0 for s > o_k + L_k + m at
    t <= t_k (m = max(m_abs, m_rel L_k); a tube neither shrinks nor un-grows). PARTIAL: the first part
    only. No tube at t_k, or before the onset bracket: Y = 0 for s > origin + rim_px up to that bin;
    Y = 1 for s < origin + stub_px from the first visible bin; always Y = 1 for s < origin - inside_px
    (the grain). Cells claimed both ways are left out.
    """
    pos = np.zeros((T, len(s)), bool)
    neg = np.zeros((T, len(s)), bool)
    pos[:, s < origin - inside_px] = True
    for t_k, state, L_k, o_k in traces:
        if not 0 <= t_k < T:
            continue
        m = max(m_abs, m_rel * L_k)
        if state in ("full", "partial") and L_k > 0:
            pos[t_k:, s < o_k + L_k - m] = True
            if state == "full":
                neg[:t_k + 1, s > o_k + L_k + m] = True
        elif state == "no_tube":
            neg[:t_k + 1, s > origin + rim_px] = True
    on = onset or {}
    v = on.get("verdict")
    if v == "emerged_within":
        la, fv = on.get("last_absent_bin"), on.get("first_visible_bin")
        if la is not None and la - rs >= 0:
            neg[:min(la - rs, T - 1) + 1, s > origin + rim_px] = True
        if fv is not None:
            pos[max(fv - rs, 0):, s < origin + stub_px] = True
    elif v == "no_emergence_by_end":
        neg[:, s > origin + rim_px] = True
    elif v == "emerged_at_start":
        pos[:, s < origin + stub_px] = True
    both = pos & neg
    return (pos & ~both).astype(np.uint8), ((pos | neg) & ~both).astype(np.uint8)


def dist_to_polyline(p: np.ndarray, poly: np.ndarray) -> tuple[float, float]:
    """(distance, arc position of the nearest point) of point p to a polyline."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    l2 = np.maximum((ab ** 2).sum(1), 1e-12)
    u = np.clip(((p - a) * ab).sum(1) / l2, 0, 1)
    q = a + u[:, None] * ab
    d = np.hypot(*(q - p).T)
    i = int(np.argmin(d))
    cum = np.r_[0.0, np.cumsum(np.sqrt(l2))]
    return float(d[i]), float(cum[i] + u[i] * np.sqrt(l2[i]))


def st_agreement(traces, path: kymo.Path1, rot: np.ndarray | None, pivot, tol_abs: float = 4.0,
                 tol_rel: float = 0.10) -> list[tuple[int, str, float, list]]:
    """The traces whose apex lies on SparseTrack's (turned) path, with their length kept; the rest
    are dropped (their labels would describe a different route)."""
    keep = []
    for t_k, state, L_k, pv in traces:
        if state not in ("full", "partial") or len(pv) < 2:
            keep.append((t_k, state, L_k, pv))
            continue
        apex = np.asarray(pv[-1], float)
        poly = path.pts if rot is None else kymo.turn(path.pts, np.asarray(pivot, float), float(rot[t_k]))
        d, _ = dist_to_polyline(apex, poly)
        if d <= max(tol_abs, tol_rel * L_k):
            keep.append((t_k, state, L_k, pv))
    return keep


def extract_movie(movie: str, log=print) -> Path:
    from sparsetrack import stack
    from sparsetrack.bench.server import Bench
    from sparsetrack.render import Renderer
    out = OUT / "data" / f"real_{movie}.npz"
    bins, meta = stack.load(CACHES[movie])
    r = Renderer(bins, meta)
    rs = r.ref_start
    T = r.n_bins - rs
    lp = labels_path(movie)
    labels = json.loads(lp.read_text())
    bench = Bench(CACHES[movie], lp)
    preds = {p["id"]: p for p in json.loads(ST053[movie].read_text())["grains"]}
    census = list(labels["grains"].values())
    physical = [g for g in census if g.get("exclude_reason") != "not_a_grain"]
    prm = strot.params(vmax_px=VMAX[movie])
    fp = OUT / "data" / f"flood_paths_{movie}.json"   # flood grains' centrelines (see flood_paths.py)
    flood_paths = json.loads(fp.read_text()) if fp.exists() else {}
    samples = []
    t0 = time.time()
    for gid, g in sorted(labels["grains"].items()):
        if g.get("excluded"):
            continue
        lab = labels["labels"].get(gid) or {}
        follow = bench.follow(gid)
        others = [o for o in physical if o["id"] != gid]
        traces = human_traces(lab, rs)
        info = {"movie": movie, "grain": gid, "isolated": bool(g.get("isolated", True)),
                "verdict": (lab.get("onset") or {}).get("verdict")}
        lr = latest_route(lab)
        p = preds.get(gid)
        if p and "reader:flood" in p.get("flags", []):
            fl = flood_paths.get(gid)
            if fl is None or not fl.get("same_lengths") or len(fl.get("path") or []) < 2:
                log(f"{movie} {gid}: flood grain without a checked centreline: no 'st' sample")
                p = None
            else:
                p = {**p, "path": fl["path"]}
        if p and p.get("path") and len(p["path"]) >= 2:
            path = kymo.resample(p["path"], back=kymo.BACK)
            rot = np.asarray(p.get("rotation_deg") or [], float)
            rot = rot if len(rot) == T else None
            ky = kymo.extract(r, g, path, follow, others, rot_deg=rot, pivot=p.get("exit_xy"))
            agree = st_agreement(traces, path, rot, p.get("exit_xy") or path.pts[path.i0])
            origin = (dist_to_polyline(np.asarray(lr[1][0], float), path.pts)[1] + path.s[0]) if lr else 0.0
            ky["lab_y"], ky["lab_m"] = partial_labels(T, path.s, with_origins(agree, path, origin), lab.get("onset"),
                                                      rs, origin=origin)
            ky["origin"] = np.float32(origin)
            ky["st_len"] = np.asarray(p["length"]["px"], np.float32)
            samples.append(store.pack(ky, {**info, "variant": "st", "reader": "flood" if "reader:flood" in p["flags"]
                                           else "change", "st_status": p["status"],
                                           "traces_agree": len([a for a in agree if a[1] in ("full", "partial")]),
                                           "traces_all": len([a for a in traces if a[1] in ("full", "partial")])}))
        if lr is not None:
            kl, route = lr
            rr = strot.read_route(r, meta, g, others, route, prm)
            path = kymo.resample(route, back=kymo.BACK)
            ky = kymo.extract(r, g, path, follow, others, rot_deg=None if rr is None else rr["rotation_deg"],
                              pivot=None if rr is None else rr["pivot"])
            ky["lab_y"], ky["lab_m"] = partial_labels(T, path.s, with_origins(traces, path, 0.0), lab.get("onset"),
                                                      rs, origin=0.0)
            ky["origin"] = np.float32(0.0)
            ky["st_len"] = (np.zeros(T, np.float32) if rr is None else rr["length"].astype(np.float32))
            if rr is not None and rr.get("tip"):
                ky["st_tip"] = np.asarray(rr["tip"]["xy"], np.float32)
            samples.append(store.pack(ky, {**info, "variant": "route", "route_bin": kl,
                                           "st_status": None if rr is None else rr["status"],
                                           "st_onset_frame": None if rr is None else rr["onset_frame"],
                                           "st_flags": [] if rr is None else rr["flags"]}))
        log(f"{movie} {gid}: {len(samples)} samples, {time.time() - t0:.0f} s")
    store.save(out, samples)
    log(f"{movie}: {len(samples)} samples -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return out


if __name__ == "__main__":
    import sys
    for mv in sys.argv[1:]:
        extract_movie(mv, log=lambda *x: print(*x, flush=True))
