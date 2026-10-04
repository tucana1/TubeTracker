"""Why a reading misses a traced length: per FULL trace (not in contact) of the scored grains, one class -
hit; no_tube (nothing read); tip_ok_len_off (tip within max(5 px, 10%) of the apex, length off: where the trace starts
or the route's shape); short (tip on the human polyline, short of the apex); long (tip past the apex, within 6 px of
the trace's straight extension) ; off_route (tip elsewhere: another tube or branch, or a moved tube).
Also: young (< 15 px) / mid / long (>= 60 px) and whether the grain moved (0.8.8 drift > 10 px by then)."""
from __future__ import annotations

import numpy as np

from .cands import seg_dist
from .common import drift_per_bin, length_class


def classify(lab: dict, pred: dict, gids: list[str], rs: int, nb: int, fpb: int) -> list[dict]:
    by = {g["id"]: g for g in pred["grains"]}
    rows = []
    for gid in gids:
        p = by.get(gid)
        if p is None:
            continue
        drift = drift_per_bin(p, rs, nb)
        frames = np.asarray(p["length"]["frames"])
        for b, t in sorted(((int(b), t) for b, t in (lab["labels"][gid].get("traces") or {}).items()), key=lambda x: x[0]):
            if t["state"] != "full" or t.get("contact"):
                continue
            h = float(t["length_px"])
            i = int(np.argmin(np.abs(frames - (t.get("source_frame") or b * fpb + fpb // 2))))
            L = float(p["length"]["px"][i])
            poly = np.asarray(t["path_xy_ref"], float)
            apex = poly[-1]
            tol = max(2.0, 0.1 * h)
            moved = bool(np.hypot(*drift[min(b, nb - 1)]) > 10)
            row = {"gid": gid, "bin": b, "h": h, "pred": L, "cls": length_class(h), "moved": moved}
            tips = (p.get("tip") or {}).get("xy")
            if abs(L - h) <= tol:
                row["why"] = "hit"
            elif L <= 0 or not tips:
                row["why"] = "no_tube"
            else:
                tip = np.asarray(tips[i], float) + drift[min(b, nb - 1)]
                te = float(np.hypot(*(tip - apex)))
                row["tip_err"] = te
                if te <= max(5.0, 0.1 * h):
                    row["why"] = "tip_ok_len_off"
                else:
                    d_poly = float(seg_dist(tip[None], poly)[0])
                    u = poly[-1] - poly[-2] if len(poly) >= 2 else np.zeros(2)
                    nu = float(np.hypot(*u))
                    ext = np.stack([apex, apex + (u / nu) * 60.0]) if nu > 1e-6 else apex[None]
                    if d_poly <= 3.0:
                        row["why"] = "short"
                    elif float(seg_dist(tip[None], ext)[0]) <= 6.0:
                        row["why"] = "long"
                    else:
                        row["why"] = "off_route"
            rows.append(row)
    return rows


def table(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        for key in (r["cls"], "moved" if r["moved"] else "still", "all"):
            d = out.setdefault(key, {})
            d[r["why"]] = d.get(r["why"], 0) + 1
    return out
