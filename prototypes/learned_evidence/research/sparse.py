"""Sparse probability maps: a network's P x 16 (float16) per bin, with values under P = 0.001 dropped and every other
value kept exactly, so many networks' evidence on many movies fits on disk (about 3-10 MB a movie instead of 450).

The per-bin decoder thresholds P at 0.5 after resampling, so the dropped values can only matter within 0.001 of the
threshold: synthetic predictions decoded from these maps match the dense caches' exactly; on the real sample movie
3 of 35 grains moved by about 1 px.

    python -m prototypes.learned_evidence.research.sparse MODEL MOVIE [MOVIE ...]
        MODEL: v2, B3 (common.MODELS) or a path to a .pt file (stored under its file name's stem)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .common import MODELS, image_cache, sparse_path

FLOOR = np.float16(0.016)  # P x 16 units: P >= 0.001 is kept exactly


class Sparse:
    """Dense float16 bins (P x 16) on demand from a sparse npz."""

    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=False)
        self.idx, self.val, self.ptr = z["idx"], z["val"], z["ptr"]
        self.shape = tuple(int(v) for v in z["shape"])
        self.meta = json.loads(str(z["meta"]))
        self.grains = json.loads(str(z["grains"]))
        self.n_bins = self.shape[0]

    def bin(self, b: int) -> np.ndarray:
        out = np.zeros(self.shape[1] * self.shape[2], np.float16)
        s, e = self.ptr[b], self.ptr[b + 1]
        out[self.idx[s:e]] = self.val[s:e]
        return out.reshape(self.shape[1:])


def save(path: Path, bins_iter, shape, meta: dict, grains: dict) -> None:
    idx, val, ptr = [], [], [0]
    for x in bins_iter:
        f = np.asarray(x, np.float16).ravel()
        k = np.flatnonzero(f >= FLOOR).astype(np.int32)
        idx.append(k)
        val.append(f[k])
        ptr.append(ptr[-1] + len(k))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, idx=np.concatenate(idx), val=np.concatenate(val), ptr=np.asarray(ptr, np.int64),
                        shape=np.asarray(shape, np.int64), meta=json.dumps(meta), grains=json.dumps(grains))
    tmp.replace(path)


def build(model: str | Path, movie: str, out: Path | None = None, sealed: bool = False, threads: int = 2) -> Path:
    """The network pass of ``evaluate.prob_cache`` (the same functions, in the same order), stored sparse."""
    import torch

    from sparsetrack import stack

    from ..data import normalise
    from ..evaluate import SCALE_P, fingerprint, registered_frame
    from ..model import load, predict
    torch.set_num_threads(threads)
    path = Path(MODELS.get(str(model), model))
    out = out or sparse_path(path.stem if str(model) not in MODELS else str(model), movie)
    if out.exists():
        return out
    ic = image_cache(movie, sealed)
    net = load(str(path), "cpu")
    bins, meta = stack.load(ic)
    shifts = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    early = np.mean([registered_frame(bins, shifts, b) for b in range(rs, rs + 3)], axis=0)
    late = np.mean([registered_frame(bins, shifts, b) for b in range(nb - 4, nb - 1)], axis=0)

    def gen():
        for b in range(nb):
            x = normalise(registered_frame(bins, shifts, b), early, late)
            yield (predict(net, x)[0] * SCALE_P).astype(np.float16)

    m = {**meta, "shifts": [[0.0, 0.0]] * nb, "raw_shifts": [[0.0, 0.0]] * nb,
         "evidence": f"learned tube probability x {SCALE_P} (from {ic})", "model_sha1": fingerprint(net)}
    save(out, gen(), bins.shape, m, json.loads((ic / "grains.json").read_text()))
    return out


if __name__ == "__main__":
    import cv2
    cv2.setNumThreads(1)
    for mv in sys.argv[2:]:
        t = time.time()
        p = build(sys.argv[1], mv)
        print(f"SPARSE_DONE {sys.argv[1]} {mv} {p.name} {p.stat().st_size / 1e6:.1f} MB {time.time() - t:.0f} s", flush=True)
