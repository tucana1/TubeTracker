"""Thresholded behaviour at the pixel level: on the synthetic truth's built-tube body (truth.frame_truth, scored tubes),
the share of tube pixels with P > 0.5, and the pixels with P > 0.5 more than 3 px from any tube (all tubes), for the
plain v2 map and TTA variants, every 8th bin. Also the tube pixels that cross 0.5 either way.

    python pixel_check.py MOVIE VARIANT[,VARIANT...] [--core 0.6]   (variants as tta.combine: id, t2p, t4l, ...)
--core: score only the tube's core (points within 0.6 x width px of its centreline) instead of its visible body.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/tta")
sys.path.insert(0, str(ME))
import numpy as np  # noqa: E402
import tta  # noqa: E402


def main(movie: str, variants: list[str], core: float | None = None) -> dict:
    import cv2
    if core is not None:  # only the tube's core: points within `core` px of the centreline (x width factor)
        from prototypes.learned_evidence import truth as T
        T.HALF_WIDTH = {False: core, True: core}
    from common5 import FIELD, truth_path
    from rescore import cfg_of
    from sparse import Sparse
    from sparsetrack.synth import Scene
    from prototypes.learned_evidence.truth import frame_truth
    truth = json.loads(truth_path(movie).read_text())
    scene = Scene(str(FIELD), cfg_of(truth["config"]))
    fpb = int(truth["frames_per_bin"])
    maps = {v: Sparse(tta.combine("v2", v, movie)) for v in variants}
    nb = maps[variants[0]].n_bins
    tot = {v: {"tube_px": 0, "tube_on": 0, "far_on": 0, "gain": 0, "loss": 0} for v in variants}
    for b in range(8, nb - 1, 8):
        ft = frame_truth(scene, b * fpb + fpb // 2, scored_only=True)
        body = ft["body"].astype(bool)
        allb = frame_truth(scene, b * fpb + fpb // 2)["body"]
        far = cv2.dilate(allb, np.ones((7, 7), np.uint8)) == 0
        on_id = maps["id"].bin(b) > 8.0 if "id" in maps else None
        for v, s in maps.items():
            on = s.bin(b) > 8.0  # P > 0.5 (P x 16 > 8)
            t = tot[v]
            t["tube_px"] += int(body.sum())
            t["tube_on"] += int((on & body).sum())
            t["far_on"] += int((on & far).sum())
            if on_id is not None:
                t["gain"] += int((on & ~on_id & body).sum())
                t["loss"] += int((~on & on_id & body).sum())
    for v, t in tot.items():
        print(f"{movie} {v:5s}: tube px with P>0.5 {t['tube_on']}/{t['tube_px']} = {100 * t['tube_on'] / max(t['tube_px'], 1):.1f}% "
              f"(vs id: +{t['gain']} -{t['loss']}) | P>0.5 more than 3 px from any tube: {t['far_on']}", flush=True)
    return tot


if __name__ == "__main__":
    core = float(sys.argv[sys.argv.index("--core") + 1]) if "--core" in sys.argv else None
    out = main(sys.argv[1], sys.argv[2].split(","), core)
    (ME / "logs" / f"pixel_{sys.argv[1]}{'_core' if core else ''}.json").write_text(json.dumps(out, indent=1))
