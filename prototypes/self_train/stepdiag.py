"""How right the step's pseudo-labels are, judged on the human labels (reported only; nothing is chosen with it): the
readings of ``sparsetrack.selftrain.select`` (every confident bin, before the spacing and the cap) at the human FULL
traces that fall on confident bins, against the readings at the other FULL traces (``select.diagnose``'s measures:
length within max(2 px, 10%), apex within max(5 px, 10%) of the human apex, the centred route's apex).

    python -m prototypes.self_train.stepdiag m1 PRED.json PROB_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sparsetrack import stack
from sparsetrack.selftrain import Maps, Settings, select, series

from prototypes.tip_trajectory.common import cache_dir, labels


def main(movie: str, pred_path: str, prob_dir: str) -> dict:
    lab = labels(movie)
    pred = json.loads(Path(pred_path).read_text())
    bins, meta = stack.load(cache_dir(movie))
    nb, (h, w) = int(meta["n_bins"]), bins.shape[1:]
    census = {g["id"]: g for g in json.loads((cache_dir(movie) / "grains.json").read_text())["grains"]}
    every = Settings(spacing=1, cap=10 ** 6)  # every confident bin, centred as the pseudo-traces are
    pseudo, _ = select(pred, census, Maps.load(prob_dir), nb, every, w, h, log=lambda *a: None)
    fpb = int(pred.get("frames_per_bin", 300))
    preds = {r["id"]: r for r in pred["grains"]}
    rows = {"confident": [], "other": []}
    for gid, L in lab["labels"].items():
        if lab["grains"][gid].get("excluded") or gid not in preds:
            continue
        Lb, _, _, _ = series(preds[gid], fpb, nb)
        conf = pseudo.get(gid, {}).get("traces", {})
        for b, t in (L.get("traces") or {}).items():
            b = int(b)
            if t["state"] != "full" or t.get("contact") or len(t.get("path_xy_ref") or []) < 2:
                continue
            hl = float(t["length_px"])
            ok_len = abs(float(Lb[b]) - hl) <= max(2.0, 0.1 * hl)
            if b in conf:
                apex = np.asarray(conf[b]["apex_xy_ref"])
                ok_tip = float(np.hypot(*(apex - np.asarray(t["path_xy_ref"][-1])))) <= max(5.0, 0.1 * hl)
                rows["confident"].append((gid, b, hl, float(Lb[b]), ok_len, ok_tip))
            else:
                rows["other"].append((gid, b, hl, float(Lb[b]), ok_len, None))
    c, o = rows["confident"], rows["other"]
    out = {"confident": {"n": len(c), "len_ok": sum(r[4] for r in c), "tip_ok": sum(r[5] for r in c),
                         "median_err": float(np.median([r[3] - r[2] for r in c])) if c else None},
           "other": {"n": len(o), "len_ok": sum(r[4] for r in o)}}
    print(f"{movie}: human FULL traces on confident bins {out['confident']}; on other bins {out['other']}")
    return out


if __name__ == "__main__":
    main(*sys.argv[1:4])
