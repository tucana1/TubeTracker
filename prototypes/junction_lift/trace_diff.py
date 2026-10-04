"""Per trace (every state, contact traces included): 0.8.8's length and a variant's, against the human's.

    trace_diff.py TAG m2 m1 ld
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MAIN, SP, movie  # noqa: E402
from sparsetrack.evaluate import length_at, tip_error  # noqa: E402

tag = sys.argv[1]
for m in sys.argv[2:]:
    path = SP + f"jt/eval/{tag}/{m}_predictions.json"
    if not os.path.exists(path):
        continue
    new = {g["id"]: g for g in json.load(open(path))["grains"]}
    base = movie(m)["G"]
    lab = movie(m)["lab"]
    fpb = lab["frames_per_bin"]
    better = worse = same = 0
    for gid, L in sorted(lab["labels"].items()):
        if gid not in new or gid not in base:
            continue
        for key, tr in sorted((L.get("traces") or {}).items(), key=lambda kv: int(kv[0])):
            if tr["state"] not in ("full", "partial"):
                continue
            frame = tr.get("source_frame") or int(key) * fpb + fpb // 2
            a, b = length_at(base[gid], frame), length_at(new[gid], frame)
            if a is None or b is None or abs(a - b) < 0.05:
                continue
            h = tr["length_px"]
            tol = max(2.0, 0.1 * h)
            if tr["state"] == "full":
                ea, eb = abs(a - h), abs(b - h)
                ok_a, ok_b = ea <= tol, eb <= tol
                ta, tb = tip_error(base[gid], frame, tr), tip_error(new[gid], frame, tr)
                verdict = "closer" if eb < ea - 0.5 else ("farther" if eb > ea + 0.5 else "same")
            else:  # a lower bound
                ok_a, ok_b = a >= h - 2.0, b >= h - 2.0
                ta = tb = None
                verdict = "lower bound " + ("kept" if ok_a == ok_b else ("now met" if ok_b else "now missed"))
            better += verdict in ("closer", "lower bound now met")
            worse += verdict in ("farther", "lower bound now missed")
            same += verdict not in ("closer", "farther", "lower bound now met", "lower bound now missed")
            print(f"{m} {gid}@{key:<4s} {tr['state']:7s}{' contact' if tr.get('contact') else '        '} human {h:6.1f}  "
                  f"0.8.8 {a:6.1f}{'*' if ok_a else ' '}  {tag} {b:6.1f}{'*' if ok_b else ' '}  "
                  f"tip {'-' if ta is None else f'{ta:.0f}'}->{'-' if tb is None else f'{tb:.0f}'} px  {verdict}")
    print(f"== {m}: traces whose length changed: closer/met {better}, farther/missed {worse}, neither {same}")
