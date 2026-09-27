"""Decode a movie from sparse maps as ``pipeline.py`` does: a base network's map fused with a thick one's by the
repository's ``fuse.fuse`` (rule ``fuse.RULE``), read by ``reach.analyze`` at its current defaults with continuity
(big 300, burst on, vmax 4 on synthetic movies and 16 on the real one).

    python -m prototypes.learned_evidence.research.decode TAG BASE THICK|none MOVIE[,MOVIE...] [--no-continuity]
        -> preds/<movie>/<TAG>.json; BASE and THICK name sparse maps <model>_<movie>.npz (sparse.py)
The current default is ``decode default v2 B3 ...``.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .common import PREDS, TMP, check, image_cache, sparse_path, vmax_of, write_json
from .sparse import Sparse


def dense_fused(base_npz: Path, thick_npz: Path | None, out: Path) -> Path:
    """A probability cache (sparsetrack.stack format) of the fused maps."""
    from .. import fuse
    sb = Sparse(base_npz)
    st = Sparse(thick_npz) if thick_npz else None
    if st is not None and st.shape != sb.shape:
        raise ValueError(f"{base_npz} and {thick_npz} differ in shape")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    arr = np.lib.format.open_memmap(out / "bins.npy", mode="w+", dtype=np.float16, shape=sb.shape)
    for b in range(sb.n_bins):
        pb = sb.bin(b)
        arr[b] = fuse.fuse(pb, st.bin(b)) if st is not None else pb
    arr.flush()
    del arr
    (out / "grains.json").write_text(json.dumps(sb.grains))
    (out / "meta.json").write_text(json.dumps({k: v for k, v in sb.meta.items() if k != "model_sha1"}))
    return out


def decode(base_npz: Path, thick_npz: Path | None, movie: str, out_json: Path, continuity: bool = True,
           sealed: bool = False, log=print) -> Path:
    from .. import reach
    t0 = time.time()
    pc = dense_fused(Path(base_npz), Path(thick_npz) if thick_npz else None,
                     TMP / f"{movie}_{out_json.stem}_{int(t0 * 1000) % 10 ** 9}")
    try:
        kw = dict(big=300, burst=True, vmax=vmax_of(movie))
        if continuity:
            kw["continuity"] = "path"
        pred = reach.analyze(pc, image_cache(movie, sealed), log=lambda *a: None, **kw)
        pred["decoder"] = kw | {"base": str(base_npz), "thick": str(thick_npz) if thick_npz else None}
        write_json(out_json, pred)
    finally:
        shutil.rmtree(pc, ignore_errors=True)
    log(f"{movie} -> {out_json}: {time.time() - t0:.0f} s")
    return out_json


def main(argv=None, sealed: bool = False):
    import cv2
    cv2.setNumThreads(1)
    argv = sys.argv[1:] if argv is None else argv
    tag, base, thick, movies = argv[0], argv[1], argv[2], argv[3].split(",")
    for mv in movies:
        if mv != "real":
            check(mv, sealed)
        out = PREDS / mv / f"{tag}.json"
        if out.exists():
            print(f"DECODE_SKIP {tag} {mv} (exists)", flush=True)
            continue
        b, t = sparse_path(base, mv), None if thick == "none" else sparse_path(thick, mv)
        missing = [p for p in (b, t) if p is not None and not p.exists()]
        if missing:
            print(f"DECODE_MISSING {tag} {mv}: {', '.join(map(str, missing))}", flush=True)
            continue
        decode(b, t, mv, out, continuity="--no-continuity" not in argv, sealed=sealed)
        print(f"DECODE_DONE {tag} {mv}", flush=True)


if __name__ == "__main__":
    main()
