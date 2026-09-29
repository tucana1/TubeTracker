"""SparseTrack's own rotation track (and change-evidence front) along a given route.

``sparsetrack.analyze.analyze_grain`` is run with the route as its only centreline candidate
(the trick of ``scripts/oracle_route.py``), with the change reader. What comes back is the
per-bin rotation about the tube exit that SparseTrack would use to read that route, and its
own front along it (the "SparseTrack on this route" baseline). Nothing in sparsetrack/ is
modified: the candidate function is swapped for the duration of one call.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

import sparsetrack.analyze as A


OFF_SINCE_053 = {"exit_edge": False, "exit_edge_onset": False, "flood_compete": False, "flood_fallback": False,
                 "flood_from_exit": False, "cand_contact_tips": False}


def params(**over) -> "A.Params":
    base = dict(reader="change")
    names = {f.name for f in dataclasses.fields(A.Params)}
    # options added after 0.5.3 (the working tree moves on): keep 0.5.3 behaviour
    for k, v in OFF_SINCE_053.items():
        if k in names:
            base[k] = v
    base.update(over)
    return A.Params(**{k: v for k, v in base.items() if k in names})


def route_yx(route_xy: np.ndarray, g: dict, half: int) -> np.ndarray | None:
    """A view-frame route as integer crop pixels (y, x), starting where it leaves the grain circle
    (as ``scripts/oracle_route.py``)."""
    centre = half - 0.5
    hp = np.asarray(route_xy, float) - [g["x"] - centre, g["y"] - centre]
    seg = np.diff(hp, axis=0)
    cum = np.concatenate([[0], np.cumsum(np.hypot(*seg.T))])
    if cum[-1] < 2:
        return None
    s = np.arange(0, cum[-1] + 1e-9, 1.0)
    d = np.stack([np.interp(s, cum, hp[:, 0]), np.interp(s, cum, hp[:, 1])], 1)
    out = np.nonzero(np.hypot(d[:, 0] - centre, d[:, 1] - centre) >= g["r"])[0]
    if not len(out):
        return None
    yx = np.clip(np.round(d[out[0]:, ::-1]).astype(int), 0, 2 * half - 1)
    keep = np.ones(len(yx), bool)
    keep[1:] = np.any(np.diff(yx, axis=0) != 0, axis=1)
    yx = yx[keep]
    return yx if len(yx) >= 3 else None


def read_route(renderer, meta: dict, g: dict, others: list[dict], route_xy, p=None) -> dict | None:
    """SparseTrack (change reader) along ``route_xy`` only. Returns rotation_deg (T,), pivot (the
    rotation pivot, reference coords), SparseTrack's length series and its flags; None if the
    grain's change map has no region at the rim (SparseTrack never reads a path then)."""
    p = p or params()
    yx = route_yx(route_xy, g, p.half)
    if yx is None:
        return None
    orig = A.candidate_paths
    A.candidate_paths = lambda *a, **k: [yx]
    try:
        res = A.analyze_grain(renderer, meta, g, others, p)
    finally:
        A.candidate_paths = orig
    rot = res.get("rotation_deg")
    if not rot or not res.get("path"):
        return None
    # the rotation pivot is the path's first point on the census circle (read_path), in the frame the
    # grain was analysed in (a settled grain may have been re-centred)
    gx, gy = res["x"], res["y"]
    first = np.asarray(res["path"][0], float)
    v = first - [gx, gy]
    v = v / (np.linalg.norm(v) + 1e-9)
    pivot = np.array([gx, gy]) + v * g["r"]
    if res.get("exit_edge_px") is None and res.get("exit_xy"):
        pivot = np.asarray(res["exit_xy"], float)
    return {"rotation_deg": np.asarray(rot, float), "pivot": pivot, "length": np.asarray(res["length"]["px"], float),
            "status": res["status"], "onset_frame": res.get("onset_frame"), "flags": res.get("flags", []),
            "path": res["path"], "tip": res.get("tip")}
