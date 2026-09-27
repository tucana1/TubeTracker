"""Truth documents for a synthetic movie as the labelling tool would score it (from round 1's red team).

``docs_for(truth, scene)`` gives "orig" (the generator's truth: traces every 12 bins, onset bracket from the physical
start to the first bin at 2 px) and "human_t<X>" for X = 2, 3, 4, 6 px: first visible = the first bin where the tube
is X px long, a one-bin bracket, and traces only at the bins the tool asks for (``trace_bins``); a tube that never
reaches X px counts as no emergence. Scores use ``sparsetrack.evaluate.score(doc, pred, onset_tol=50)``.
"""

from __future__ import annotations

import json

import numpy as np


def cfg_of(c: dict):
    from sparsetrack.synth import SynthConfig
    return SynthConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()})


def docs_for(truth: dict, scene) -> dict:
    from sparsetrack.bench.server import trace_bins
    fpb, nb = truth["frames_per_bin"], truth["n_bins"]
    centres = np.array([b * fpb + fpb // 2 for b in range(nb)], float)
    out = {"orig": truth}
    tubes = {f"s{i + 1:03d}": t for i, t in enumerate(scene.tubes) if t.scored}
    for x in (2.0, 3.0, 4.0, 6.0):
        d = {**truth, "labels": {}}
        for gid, lab in truth["labels"].items():
            if gid not in tubes:
                d["labels"][gid] = lab  # controls and foreign tubes as written
                continue
            length = tubes[gid].length(centres)
            if not np.any(length >= x):
                d["labels"][gid] = {"onset": {"verdict": "no_emergence_by_end"}, "traces": {}}
                continue
            fv = int(np.argmax(length >= x))
            tr = {str(b): {"state": "full", "length_px": round(float(length[b]), 3), "source_frame": int(centres[b])}
                  for b in trace_bins(fv, nb)}
            d["labels"][gid] = {"onset": {"verdict": "emerged_within", "last_absent_frame": int(centres[max(fv - 1, 0)]),
                                          "first_visible_frame": int(centres[fv])}, "traces": tr}
        out[f"human_t{x:g}"] = d
    return out


def load_docs(truth_path, field) -> dict:
    from sparsetrack.synth import Scene
    truth = json.loads(open(truth_path).read())
    return docs_for(truth, Scene(str(field), cfg_of(truth["config"])))
