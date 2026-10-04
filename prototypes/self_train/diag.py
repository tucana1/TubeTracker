"""Diagnosis only: how right the training crops made from pseudo-labels are, judged on the human traces of the same
grain at the same bin (never used to select or train).

    python -m prototypes.self_train.diag m1_r1 --movie m1

Per crop of the pseudo-traced and propagated shards that has a human trace (FULL or PARTIAL, with a route, not
touching anything) of its grain at its bin: the share of its tube pixels within 4 px of the human route (as far as
the human route goes: tube pixels beyond a PARTIAL trace's end are not judged) and the share of its scored
background within 2 px of the human route (background on a tube the person traced).
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from prototypes.self_train.select import OUT
from prototypes.tube_adapt.common import labels


def judge(shard, H: dict, half: int = 48, tol: int = 0) -> dict:
    z = np.load(shard)
    B, W, info, grain = z["body"], z["w"], z["info"], z["grain"]
    kinds = z["kind"] if "kind" in z.files else np.array(["?"] * len(B))
    res = {}
    for k in range(len(B)):
        gid, b = str(grain[k]), int(info[k][0])
        tr = (H["labels"].get(gid) or {}).get("traces") or {}
        t = next((tr[str(b + j)] for j in sorted(range(-tol, tol + 1), key=abs) if str(b + j) in tr), None)
        if not t or t["state"] not in ("full", "partial") or len(t.get("path_xy_ref") or []) < 2 or t.get("contact"):
            continue
        cx, cy = float(info[k][1]), float(info[k][2])
        q = np.asarray(t["path_xy_ref"], float) - [cx - half, cy - half] - 0.5
        line = np.zeros((2 * half, 2 * half), np.uint8)
        cv2.polylines(line, [np.round(q * 4).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1, cv2.LINE_8, 2)
        d = cv2.distanceTransform(1 - line, cv2.DIST_L2, 5)
        body, w = B[k] > 0, W[k] > 0
        judged = body.copy()
        if t["state"] == "partial":  # beyond the traced end the tube is not known: judge only near the route
            judged &= d <= 12.0
        r = res.setdefault(str(kinds[k]), {"crops": 0, "tube_px": 0, "tube_on": 0, "bg_px": 0, "bg_on_tube": 0})
        r["crops"] += 1
        r["tube_px"] += int(judged.sum())
        r["tube_on"] += int((judged & (d <= 4.0)).sum())
        bg = w & ~body
        r["bg_px"] += int(bg.sum())
        r["bg_on_tube"] += int((bg & (d <= 2.0)).sum())
    for r in res.values():
        r["tube_on_share"] = round(r["tube_on"] / max(r["tube_px"], 1), 3)
        r["bg_on_tube_share"] = round(r["bg_on_tube"] / max(r["bg_px"], 1), 4)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--movie", default="m1")
    ap.add_argument("--tol", type=int, default=1, help="a human trace up to this many bins away counts")
    a = ap.parse_args(argv)
    H = labels(a.movie)
    out = {}
    for kind in ("trace", "prop"):
        f = OUT / "shards" / f"{kind}_{a.name}.npz"
        if f.exists():
            out[kind] = judge(f, H, tol=a.tol)
            for k, r in out[kind].items():
                print(f"{a.name} {kind}/{k}: {r['crops']} crops within {a.tol} bin(s) of a human trace; tube px on the human route "
                      f"{100 * r['tube_on_share']:.0f}% ({r['tube_on']}/{r['tube_px']}); scored background on it "
                      f"{100 * r['bg_on_tube_share']:.2f}% ({r['bg_on_tube']}/{r['bg_px']})", flush=True)
    (OUT / "pseudo" / f"{a.name}_crop_diagnosis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
