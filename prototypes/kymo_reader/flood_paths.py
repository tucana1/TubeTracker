"""Rim-to-tip centrelines for SparseTrack 0.5.3's flood-read grains.

The frozen st053 prediction files (runs/lab_checks_2026-09-29, made at 03:16 and 03:54 on 29 Sep)
store a flood grain's ``path`` as its tube pixels in rim-distance order (every third pixel), not
as a line: the centreline fix (de251b0, "Flood reader: its reported path is a centreline") landed
while they were being made. This re-runs ``sparsetrack.learned.read_grain`` with st053's own
parameters for those grains and keeps its centreline, after checking that its length series is
the frozen one (same flood, same tube).

    python -m prototypes.kymo_reader.flood_paths ld m2
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/kymo_reader/data"


def run(movie: str, log=print) -> Path:
    import sparsetrack.analyze as A
    from sparsetrack import learned, stack
    from sparsetrack.render import Renderer
    from .real_data import CACHES, ST053, labels_path
    base = json.loads(ST053[movie].read_text())
    names = {f.name for f in dataclasses.fields(A.Params)}
    prm = {k: (tuple(v) if isinstance(v, list) else v) for k, v in base["params"].items() if k in names}
    from .strot import OFF_SINCE_053
    for k, v in OFF_SINCE_053.items():
        if k in names:
            prm[k] = v
    p = A.Params(**prm)
    bins, meta = stack.load(CACHES[movie])
    r = Renderer(bins, meta)
    prob = Renderer(*stack.load(learned.prob_cache(CACHES[movie], p.model or learned.MODEL, log)))
    labels = json.loads(labels_path(movie).read_text())
    physical = [g for g in labels["grains"].values() if g.get("exclude_reason") != "not_a_grain"]
    out = {}
    for g in base["grains"]:
        if "reader:flood" not in g.get("flags", []) or g["id"] not in labels["grains"]:
            continue
        lg = labels["grains"][g["id"]]
        if lg.get("excluded"):
            continue
        others = [o for o in physical if o["id"] != g["id"]]
        res = learned.read_grain(r, prob, meta, lg, others, p)
        frozen = np.asarray(g["length"]["px"], float)
        mine = np.asarray(res["length"]["px"], float)
        both = (mine > 0) & (frozen > 0)  # the hybrid may zero lengths before the change reader's onset
        same = bool(np.allclose(mine[both], frozen[both], atol=0.05)) if both.any() else not frozen.any()
        line = np.asarray(res.get("path") or [], float)
        seg = float(np.hypot(*np.diff(line, axis=0).T).sum()) if len(line) > 1 else 0.0
        out[g["id"]] = {"path": res.get("path") or [], "same_lengths": same, "exit_xy": res.get("exit_xy"),
                        "centreline_px": round(seg, 1), "path_length_px": res.get("path_length_px"),
                        "final_frozen": g.get("final_length_px"), "final_rerun": res.get("final_length_px")}
        log(f"{movie} {g['id']}: centreline {seg:.0f} px (flood reach {res.get('path_length_px')}), "
            f"lengths as frozen: {same} (final {g.get('final_length_px')} vs {res.get('final_length_px')})")
    path = OUT / f"flood_paths_{movie}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return path


if __name__ == "__main__":
    for mv in sys.argv[1:]:
        run(mv, log=lambda *a: print(*a, flush=True))
