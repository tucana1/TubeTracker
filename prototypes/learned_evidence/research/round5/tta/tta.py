"""Test-time augmentation (TTA) of the evidence networks, stored sparse in the round-5 format.

One pass = the repository's own ``model.predict`` (256 px tiles, 48 px linear ramps) on a dihedral transform of the
network input (the bin, "before" and "after" registered images transformed together), its output transformed back.
Registration and normalisation are evaluate.prob_cache's (the code path of R5/sparse5.py's ``build``); each pass is
stored as P x 16 in float16 with P < 0.001 dropped (agents/fusion/sparse.py's format):
    sparse/<MODEL>_x<T>_<MOVIE>.npz        (T in TRANSFORMS; "e", the identity, only for the exactness check)
The identity pass of every variant is the existing plain cache (common5.sparse_path), the one the baseline decoded.
A TTA variant averages the passes of its transforms, as probabilities ("p") or logits ("l"; P clipped to
[0.001, 1 - 1e-6]: values under the storage floor are taken as P = 0.001) -- see ``combine``.

    python tta.py worker QUEUE_FILE [--threads 1]       run jobs from a queue (atomic claims; several workers)
    python tta.py pass MODEL T MOVIE [--threads 1]       one pass
Queue lines: "pass MODEL T MOVIE" or "decode TAG MOVIE" (TAG in tags.TAGS; runs once its passes exist).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

if __name__ == "__main__":  # threads fixed before numpy/torch load
    _thr = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "1"
    os.environ["OMP_NUM_THREADS"] = _thr
    os.environ["MKL_NUM_THREADS"] = _thr
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

SCR = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad")
REPO = Path("/home/user/TubeTracker")
ME = SCR / "agents" / "tta"
SP = ME / "sparse"
WORK = ME / "work"
PREDS = ME / "preds"
LOGS = ME / "logs"
for p in (REPO, SCR / "round5", SCR / "agents" / "fusion"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MODELS = {"v2": REPO / "prototypes/learned_evidence/models/unet_v2_sample_field.pt",
          "B3": REPO / "prototypes/learned_evidence/models/unet_thick_b3.pt"}
FLOOR = np.float16(0.016)  # P x 16 units, as agents/fusion/sparse.py
LO, HI = 1e-3, 1.0 - 1e-6  # logit clip


def _t(x):  # transpose of the last two axes
    return np.swapaxes(x, -1, -2)


# name -> (forward, inverse) on the last two axes (H, W)
TRANSFORMS = {
    "e": (lambda x: x, lambda x: x),
    "h": (lambda x: np.flip(x, -1), lambda x: np.flip(x, -1)),
    "v": (lambda x: np.flip(x, -2), lambda x: np.flip(x, -2)),
    "r180": (lambda x: np.flip(x, (-2, -1)), lambda x: np.flip(x, (-2, -1))),
    "t": (_t, _t),
    "r90": (lambda x: np.rot90(x, 1, (-2, -1)), lambda x: np.rot90(x, -1, (-2, -1))),
    "r270": (lambda x: np.rot90(x, -1, (-2, -1)), lambda x: np.rot90(x, 1, (-2, -1))),
    "at": (lambda x: np.flip(_t(x), (-2, -1)), lambda x: np.flip(_t(x), (-2, -1))),
}
SETS = {"t2": ["e", "r180"], "t4": ["e", "h", "v", "r180"], "t4r": ["e", "r90", "r180", "r270"],
        "t8": ["e", "h", "v", "r180", "t", "r90", "r270", "at"]}


def check_transforms() -> None:
    x = np.arange(2 * 5 * 7, dtype=np.float32).reshape(2, 5, 7)
    for name, (f, inv) in TRANSFORMS.items():
        assert np.array_equal(inv(f(x)), x), name
    assert len({f(x).tobytes() for f, _ in TRANSFORMS.values()}) == 8


def image_cache_of(movie: str) -> Path:
    from common5 import image_cache  # refuses the fresh held-out seeds
    return image_cache(movie)


def pass_path(model: str, t: str, movie: str) -> Path:
    if t == "e":
        from common5 import sparse_path
        return sparse_path(model, movie)
    return SP / f"{model}_x{t}_{movie}.npz"


def run_pass(model: str, t: str, movie: str, threads: int = 1, chunk: int = 16, log=print) -> Path:
    """One transformed pass over every bin, stored sparse; resumable by chunks of bins."""
    import torch
    torch.set_num_threads(threads)
    import cv2
    cv2.setNumThreads(1)
    from sparse import _save  # noqa: F401  (format reference: agents/fusion/sparse.py)
    from sparsetrack import stack
    from prototypes.learned_evidence.data import normalise
    from prototypes.learned_evidence.evaluate import SCALE_P, fingerprint, registered_frame
    from prototypes.learned_evidence.model import load, predict
    check_transforms()
    out = SP / f"{model}_x{t}_{movie}.npz"  # also for "e" (the check), never over the shared cache
    if out.exists():
        return out
    ic = image_cache_of(movie)
    net = load(str(MODELS[model]), "cpu")
    bins, meta = stack.load(ic)
    shifts = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    early = np.mean([registered_frame(bins, shifts, b) for b in range(rs, rs + 3)], axis=0)
    late = np.mean([registered_frame(bins, shifts, b) for b in range(nb - 4, nb - 1)], axis=0)
    f, inv = TRANSFORMS[t]
    wdir = WORK / f"{model}_x{t}_{movie}"
    wdir.mkdir(parents=True, exist_ok=True)
    tpath = wdir / "timing.json"
    timing = json.loads(tpath.read_text()) if tpath.exists() else {}
    for c0 in range(0, nb, chunk):
        cpath = wdir / f"chunk_{c0:04d}.npz"
        if cpath.exists():
            continue
        w0, c_0 = time.time(), time.process_time()
        idx, val, cnt = [], [], []
        for b in range(c0, min(nb, c0 + chunk)):
            x = normalise(registered_frame(bins, shifts, b), early, late)
            p = predict(net, x if t == "e" else np.ascontiguousarray(f(x)))[0]
            p = p if t == "e" else np.ascontiguousarray(inv(p))
            v = (p * SCALE_P).astype(np.float16).ravel()  # exactly evaluate.prob_cache's value for "e"
            k = np.flatnonzero(v >= FLOOR).astype(np.int32)
            idx.append(k)
            val.append(v[k])
            cnt.append(len(k))
        tmp = cpath.with_suffix(".tmp.npz")
        np.savez(tmp, idx=np.concatenate(idx), val=np.concatenate(val), cnt=np.asarray(cnt, np.int64))
        tmp.replace(cpath)
        timing[str(c0)] = {"wall_s": time.time() - w0, "cpu_s": time.process_time() - c_0,
                           "bins": min(nb, c0 + chunk) - c0, "threads": threads}
        tpath.write_text(json.dumps(timing, indent=1))
    chunks = sorted(wdir.glob("chunk_*.npz"))
    zs = [np.load(c) for c in chunks]
    idx = np.concatenate([z["idx"] for z in zs]).astype(np.int32)
    val = np.concatenate([z["val"] for z in zs]).astype(np.float16)
    cnt = np.concatenate([z["cnt"] for z in zs])
    assert len(cnt) == nb, (len(cnt), nb)
    ptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    m = {**meta, "shifts": [[0.0, 0.0]] * nb, "raw_shifts": [[0.0, 0.0]] * nb,
         "evidence": f"learned tube probability x {SCALE_P} (from {ic}); input transformed by {t}, output back",
         "model_sha1": fingerprint(net), "tta_transform": t}
    grains = json.loads((Path(ic) / "grains.json").read_text())
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, idx=idx, val=val, ptr=ptr, shape=np.asarray(bins.shape, np.int64),
                        meta=json.dumps(m), grains=json.dumps(grains))
    tmp.replace(out)
    for z in zs:
        z.close()
    wall = sum(v["wall_s"] for v in timing.values())
    cpu = sum(v["cpu_s"] for v in timing.values())
    rec = {"model": model, "t": t, "movie": movie, "bins": nb, "wall_s": wall, "cpu_s": cpu, "threads": threads,
           "finished": time.strftime("%H:%M:%S")}
    with open(LOGS / "passes.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    for c in chunks:
        c.unlink()
    log(f"PASS_DONE {model} {t} {movie}: {nb} bins, wall {wall:.0f} s, cpu {cpu:.0f} s")
    return out


# ------------------------------------------------------------------------------------------------ variants
def combine(model: str, variant: str, movie: str) -> Path:
    """<model>_<variant>_<movie>.npz from the stored passes. variant = <set><p|l>, e.g. t4p; "id" = the plain cache."""
    from sparse import Sparse, _save
    if variant == "id":
        return pass_path(model, "e", movie)
    out = SP / f"{model}_{variant}_{movie}.npz"
    if out.exists():
        return out
    ts, mode = SETS[variant[:-1]], variant[-1]
    srcs = [Sparse(pass_path(model, t, movie)) for t in ts]
    s0 = srcs[0]

    def gen():
        for b in range(s0.n_bins):
            ps = np.stack([s.bin(b) for s in srcs]).astype(np.float64) / 16.0
            if mode == "p":
                p = ps.mean(axis=0)
            elif mode == "l":
                q = np.clip(ps, LO, HI)
                p = 1.0 / (1.0 + np.exp(-np.log(q / (1.0 - q)).mean(axis=0)))
                p[ps.max(axis=0) < LO] = 0.0  # every pass under the storage floor: stays under it
            elif mode == "m":  # exploratory: the union (max) of the passes
                p = ps.max(axis=0)
            else:
                raise ValueError(variant)
            yield (p * 16.0).astype(np.float16)

    meta = dict(s0.meta, tta={"transforms": ts, "average": {"p": "prob", "l": "logit", "m": "max"}[mode]},
                model_sha1=s0.meta.get("model_sha1"))
    _save(out, gen(), s0.shape, meta, s0.grains)
    return out


def decode_tag(tag: str, movie: str, log=print) -> Path:
    """tags.TAGS[tag] = (base variant, thick variant or None): decode5.decode on those caches."""
    import decode5
    from tags import TAGS
    out = PREDS / movie / f"{tag}.json"
    if out.exists():
        return out
    bv, tv = TAGS[tag]
    base = combine("v2", bv, movie)
    thick = combine("B3", tv, movie) if tv else None
    decode5.TMP = ME / "tmp"  # the dense fused cache lives in my folder while decoded
    t0, c0 = time.time(), time.process_time()
    decode5.decode(base, thick, movie, out, log=lambda *a: None)
    rec = {"tag": tag, "movie": movie, "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0,
           "finished": time.strftime("%H:%M:%S")}
    with open(LOGS / "decodes.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    log(f"DECODE_DONE {tag} {movie}: wall {rec['wall_s']:.0f} s")
    return out


def deps_ready(tag: str, movie: str) -> bool:
    from tags import TAGS
    bv, tv = TAGS[tag]
    need = [("v2", t) for t in (SETS[bv[:-1]] if bv != "id" else ["e"])]
    need += [("B3", t) for t in (SETS[tv[:-1]] if tv and tv != "id" else (["e"] if tv else []))]
    return all((p := pass_path(m, t, movie)) is not None and p.exists() for m, t in need)


def worker(queue: Path, threads: int = 1, log=print) -> None:
    claims = ME / "claims"
    claims.mkdir(exist_ok=True)
    while True:
        jobs = [ln.split() for ln in queue.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        todo = None
        for j in jobs:
            key = claims / "_".join(j)
            if key.exists():
                continue
            if j[0] == "decode" and not deps_ready(j[1], j[2]):
                continue
            try:
                key.mkdir()
            except FileExistsError:
                continue
            todo = j
            break
        if todo is None:
            pending = [j for j in jobs if not (claims / "_".join(j)).exists()]
            if not pending:
                log("QUEUE_EMPTY")
                return
            time.sleep(30)  # decodes waiting on passes other workers are running
            continue
        try:
            if todo[0] == "pass":
                run_pass(todo[1], todo[2], todo[3], threads, log=log)
            elif todo[0] == "timing":  # time_pipeline.py in a subprocess (one thread)
                import subprocess
                with open(LOGS / f"timing_pipeline_{todo[1]}.log", "w") as fh:
                    subprocess.run([sys.executable, str(ME / "time_pipeline.py"), todo[1]], stdout=fh,
                                   stderr=subprocess.STDOUT, check=True)
                log(f"TIMING_DONE {todo[1]}")
            else:
                decode_tag(todo[1], todo[2], log=log)
        except Exception as e:  # keep the queue going; the claim stays (marked FAILED) until removed by hand
            import traceback
            log(f"JOB_FAILED {' '.join(todo)}: {e!r}\n{traceback.format_exc()}")
            (claims / "_".join(todo) / "FAILED").write_text(traceback.format_exc())


if __name__ == "__main__":
    what = sys.argv[1]
    thr = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 1
    say = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731
    if what == "worker":
        worker(Path(sys.argv[2]), thr, log=say)
    elif what == "pass":
        run_pass(sys.argv[2], sys.argv[3], sys.argv[4], thr, log=say)
    elif what == "decode":
        decode_tag(sys.argv[2], sys.argv[3], log=say)
    elif what == "same":  # two sparse caches hold the same values?
        za, zb = np.load(sys.argv[2]), np.load(sys.argv[3])
        print("; ".join(f"{k} {'equal' if np.array_equal(za[k], zb[k]) else 'DIFFERENT'}"
                        for k in ("shape", "ptr", "idx", "val")))
