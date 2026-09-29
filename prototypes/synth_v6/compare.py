"""The comparison sheet: real young tubes (human onsets) next to synthetic v6 ones (truth onsets), both shown
as the tube network sees them - the registered bin and its change from the movie's "before" image - at the
same bins around the first visible bin (human: first_visible_bin; synthetic: first bin with >= 2 px of tube).

    python -m prototypes.synth_v6.compare
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .sheet import View, grain_rows

REPO = Path(__file__).resolve().parents[2]
OFFSETS = (-3, 0, 3, 6, 10, 20)
REAL = [("m2", "g052"), ("m2", "g033"), ("m2", "g005"), ("m2", "g060"), ("ld", "g031"), ("ld", "g038")]
SYNTH = [("runs/synth_v6/preview/pv_m2_v6_s1_look_cache", "s007"), ("runs/synth_v6/preview/pv_m2_v6_s1_look_cache", "s005"),
         ("runs/synth_v6/preview/pv_m2_v6_s1_look_cache", "s004"), ("runs/synth_v6/preview/pv_m2_v6_s1_look_cache", "s008"),
         ("runs/synth_v6/preview/pv_ld_v6_s2_cache", "s005"), ("runs/synth_v6/preview/pv_ld_v6_s2_cache", "s002")]


def banner(text: str, width: int, color=(0, 0, 160)) -> np.ndarray:
    b = np.full((18, width, 3), 255, np.uint8)
    cv2.putText(b, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return b


def main(out: str = "runs/synth_v6/compare_young_tubes.png") -> None:
    rows = []
    views = {}
    for (movie, gid), (cache, sid) in zip(REAL, SYNTH):
        L = json.loads((REPO / f"benchmark/labels/{movie}_v1.json").read_text())
        g, on = L["grains"][gid], L["labels"][gid]["onset"]
        v = views.setdefault(movie, View(REPO / f"runs/sparsetrack/{movie}"))
        fb = int(on["first_visible_bin"])
        # the human traced this grain following its drift: centre the crops on it too (same pixels, re-centred)
        tr = sorted(((int(b), t) for b, t in L["labels"][gid]["traces"].items() if t["state"] == "full"))[0][1]
        off = tr.get("view_offset") or [0.0, 0.0]
        r = grain_rows(v, g["x"] + off[0], g["y"] + off[1], [fb + o for o in OFFSETS], f"{gid}", zoom=3)
        T = json.loads((REPO / cache / "truth.json").read_text())
        sg, st = T["grains"][sid], T["labels"][sid]
        fv = int(st["onset"]["first_visible_frame"]) // 25
        sv = views.setdefault(cache, View(REPO / cache))
        s = grain_rows(sv, sg["x"], sg["y"], [fv + o for o in OFFSETS], f"{sid}", zoom=3)
        t = st["truth"]
        look = ("light-cored" if t.get("bright_core") and not t.get("evolves") else
                "light-cored, young dark" if t.get("bright_core") else "dark")
        left = np.concatenate([banner(f"REAL {movie} {gid} (first visible bin {fb})", r.shape[1]), r], axis=0)
        right = np.concatenate([banner(f"SYNTH v6 {'m2' if 'm2' in cache else 'ld'} field {sid}: {look}, "
                                       f"bulb {t.get('bulb', 0):.2f}, body {t.get('body_change')}", s.shape[1],
                                       (0, 120, 0)), s], axis=0)
        gap = np.full((left.shape[0], 12, 3), 255, np.uint8)
        rows += [np.concatenate([left, gap, right], axis=1), np.full((8, left.shape[1] * 2 + 12, 3), 255, np.uint8)]
    head = banner("columns: -3, 0, +3, +6, +10, +20 bins from the first visible bin; rows: registered bin / its change "
                  "from the movie's 'before' image (+/-30 grey levels) - as the tube network sees them", rows[0].shape[1])
    cv2.imwrite(str(REPO / out), np.concatenate([head] + rows, axis=0))
    print(out)


if __name__ == "__main__":
    main()
