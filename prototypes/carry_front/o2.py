"""O2: why SparseTrack 0.8.8 misses the traces it misses. Per scored FULL trace (not in contact) whose length is out of
tolerance: the model read no tube then (onset/germination), or its drawn route then (sparsetrack.report.turned_path at
the bin's series index, moved by the grain's drift, cut to the shorter of the model's and the human's length) lies on
average > 3 px from the human polyline (route), or the route is right and the front short or long.

    python -m prototypes.carry_front.o2
"""
from __future__ import annotations

import json
import sys

import numpy as np

from prototypes.carry_front.carry import OUT, REPO, dense
from prototypes.carry_front.o1 import BASE

sys.path.insert(0, str(REPO))
from sparsetrack.evaluate import length_at, match_grains  # noqa: E402
from sparsetrack.report import turned_path  # noqa: E402
from sparsetrack.routes import cut  # noqa: E402

ROUTE_PX = 3.0


def mean_dist(route: np.ndarray, human: np.ndarray) -> float:
    d = np.hypot(route[:, None, 0] - human[None, :, 0], route[:, None, 1] - human[None, :, 1]).min(axis=1)
    return float(d.mean())


def drawn(res: dict, pred: dict, i: int, L: float) -> np.ndarray | None:
    try:
        path = turned_path(res, i, pred)
    except Exception:
        return None
    if len(path) < 2:
        return None
    drift = (res.get("drift") or {}).get("xy")
    d = np.asarray(drift[i], float) if drift and i < len(drift) else np.zeros(2)
    return dense(cut(path, L), 1.0) + d


def classify(movie: str) -> dict:
    labels = json.loads((REPO / f"benchmark/labels/{movie}_v1.json").read_text())
    pred = json.loads((BASE / f"{movie}_real_0/predictions.json").read_text())
    fpb = labels["frames_per_bin"]
    grains = {g: v for g, v in labels["grains"].items() if not v.get("excluded") and v.get("isolated", True)}
    match = match_grains({"grains": grains}, pred["grains"])
    rows = []
    for gid in sorted(grains):
        res = match.get(gid)
        lab = labels["labels"].get(gid) or {}
        on = lab.get("onset") or {}
        frames = (res or {}).get("length", {}).get("frames") or []
        for key, t in sorted((lab.get("traces") or {}).items(), key=lambda kv: int(kv[0])):
            if t["state"] != "full" or t.get("contact") or res is None:
                continue
            b = int(key)
            frame = t.get("source_frame") or b * fpb + fpb // 2
            L = length_at(res, frame)
            if L is None:
                continue
            h = float(t["length_px"])
            tol = max(2.0, 0.1 * h)
            i = int(np.argmin(np.abs(np.asarray(frames) - frame))) if frames else 0
            hum = dense(t["path_xy_ref"], 0.5)
            dist = None
            if L >= 2.0:
                r = drawn(res, pred, i, min(L, h))
                dist = mean_dist(r, hum) if r is not None and len(r) else None
            hit = abs(L - h) <= tol
            if hit:
                why = "hit"
            elif L < 2.0:
                st, of = res.get("status"), res.get("onset_frame")
                why = ("onset:never" if st == "no_emergence_by_end" else
                       "onset:late" if of is not None and of > frame else "onset:zero")
            elif dist is not None and dist > ROUTE_PX:
                why = "route"
            else:
                why = "short" if L < h else "long"
            rows.append({"grain": gid, "bin": b, "h": round(h, 1), "pred": round(L, 1), "dist": None if dist is None else
                         round(dist, 2), "why": why, "reader": "flood" if res.get("path_by_bin") is not None or
                         any("flood" in f for f in res.get("flags", [])) else "change",
                         "human_fv": on.get("first_visible_bin"), "model_onset_bin": None if res.get("onset_frame") is None
                         else int(res["onset_frame"]) // fpb, "status": res.get("status")})
    return rows


def main():
    out = {}
    for movie in ("ld", "m2", "m1"):
        rows = classify(movie)
        out[movie] = rows
        n = len(rows)
        cnt = {}
        for r in rows:
            k = r["why"].split(":")[0]
            cnt[k] = cnt.get(k, 0) + 1
        sub = {}
        for r in rows:
            if r["why"].startswith("onset"):
                sub[r["why"]] = sub.get(r["why"], 0) + 1
        hd = [r["dist"] for r in rows if r["why"] == "hit" and r["dist"] is not None]
        print(f"{movie}: {n} traces; " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) +
              f" | onset detail {sub} | drawn-route distance on hits: median {np.median(hd):.2f}, "
              f"> 3 px {np.mean(np.asarray(hd) > ROUTE_PX):.0%}")
        for r in rows:
            if r["why"] != "hit":
                print(f"   {r['grain']} b{r['bin']}: human {r['h']} model {r['pred']} dist {r['dist']} -> {r['why']} "
                      f"(onset model {r['model_onset_bin']} human {r['human_fv']}, {r['status']})")
    (OUT / "o2.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
