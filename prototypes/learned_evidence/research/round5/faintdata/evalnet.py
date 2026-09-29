"""A model's network pass on development movies (or the real movie), stored sparse in my folder, then fused with B3 and
decoded by the shared round-5 decoder; predictions in my folder (preds/<movie>/<tag>.json).

    python evalnet.py TAG MODEL_PT MOVIE[,MOVIE...]     (1 thread per process)

The network pass is R5/sparse5.py's build (the same functions as evaluate.prob_cache, in the same order) with a model
path instead of a model name; the decode is R5/decode5.decode with B3's shared sparse cache (common5.sparse_path).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/faintdata")
R5 = ME.parent.parent / "round5"
sys.path.insert(0, str(R5))
from common5 import image_cache, sparse_path  # noqa: E402  (also puts the repo and agents/fusion on sys.path)

SP = ME / "sparse"
PREDS = ME / "preds"


def build(tag: str, model_pt: str, movie: str) -> Path:
    import torch
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    from sparse import _save  # agents/fusion/sparse.py: the shared format and floor
    from sparsetrack import stack
    from prototypes.learned_evidence.data import normalise
    from prototypes.learned_evidence.evaluate import SCALE_P, fingerprint, registered_frame
    from prototypes.learned_evidence.model import load, predict
    out = SP / f"{tag}_{movie}.npz"
    if out.exists():
        return out
    SP.mkdir(parents=True, exist_ok=True)
    ic = image_cache(movie)
    net = load(str(model_pt), "cpu")
    bins, meta = stack.load(ic)
    shifts = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    early = np.mean([registered_frame(bins, shifts, b) for b in range(rs, rs + 3)], axis=0)
    late = np.mean([registered_frame(bins, shifts, b) for b in range(nb - 4, nb - 1)], axis=0)

    def gen():
        for b in range(nb):
            yield (predict(net, normalise(registered_frame(bins, shifts, b), early, late))[0] * SCALE_P).astype(np.float16)

    m = {**meta, "shifts": [[0.0, 0.0]] * nb, "raw_shifts": [[0.0, 0.0]] * nb,
         "evidence": f"learned tube probability x {SCALE_P} (from {ic})", "model_sha1": fingerprint(net),
         "model_path": str(model_pt)}
    _save(out, gen(), bins.shape, m, json.loads((ic / "grains.json").read_text()))
    return out


def decode(tag: str, movie: str) -> Path:
    import decode5
    decode5.TMP = ME / "tmp"  # its transient dense fused cache in my folder, not round5's
    out = PREDS / movie / f"{tag}.json"
    if out.exists():
        return out
    thick = sparse_path("B3", movie)
    if thick is None:
        raise SystemExit(f"no B3 sparse cache for {movie} yet")
    return decode5.decode(SP / f"{tag}_{movie}.npz", thick, movie, out)


if __name__ == "__main__":
    import cv2
    cv2.setNumThreads(1)
    tag, model_pt, movies = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    for mv in movies:
        t = time.time()
        p = build(tag, model_pt, mv)
        t1 = time.time()
        while sparse_path("B3", mv) is None:  # thick s31 and seeds 7, 8, 13-15: B3 still being built by round5
            time.sleep(30)
        decode(tag, mv)
        print(f"EVAL_DONE {tag} {mv} net {t1 - t:.0f}s decode {time.time() - t1:.0f}s "
              f"({os.path.getsize(p) / 1e6:.1f} MB)", flush=True)
