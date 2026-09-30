"""``scripts/synth_bench.py --real`` for one tube network, with its probability movie held in memory.

    python -m prototypes.tube_net.bench MODEL.pt MOVIE [MOVIE ...] --out runs/tube_net/e2e_X.json \
        [--baseline runs/lab_checks_2026-09-29/final_070.json] [--tile 256]

SparseTrack (the current defaults, only ``model`` changed) reads each movie exactly as the bench does, but
``learned.prob_cache`` is replaced by the same computation kept in memory (``pixels.prob_movie`` over every bin),
so no 0.2-0.5 GB probability movie is written per network. One process per movie, as the bench does. The dump has
the bench's ``--dump-real`` format; ``--baseline`` prints the bench's paired comparison.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))


def _one(movie: str, model: str, tile: int, work: str, bg: int = 0) -> dict:
    import synth_bench as sb
    from sparsetrack import learned, stack
    from sparsetrack.analyze import Params
    from prototypes.tube_net.pixels import prob_movie

    mem = {}
    load = stack.load

    def prob_cache(cache_dir, model_path, log=print):
        return ("in-memory", str(cache_dir))

    def patched_load(p):
        if isinstance(p, tuple) and p and p[0] == "in-memory":
            if "pm" not in mem:
                _, meta = load(REPO / sb.REAL[movie][0])
                mem["pm"] = prob_movie(model, movie, list(range(int(meta["n_bins"]))), tile=tile, bg=bg, log=print)
            return mem["pm"]
        return load(p)

    learned.prob_cache, stack.load = prob_cache, patched_load
    return sb.score_real(movie, Params(model=model), "0", Path(work) / movie)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("movies", nargs="+", choices=("ld", "m2"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--tile", type=int, default=0, help="override the checkpoint's tile size")
    ap.add_argument("--bg", type=int, default=0, help="override the checkpoint's local-background scale")
    ap.add_argument("--work", default="/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/"
                                      "eaa2b708-9f78-4565-aa0c-9af3763da8fd/scratchpad/tubenet/bench")
    a = ap.parse_args(argv)
    import synth_bench as sb
    tag = (f"_t{a.tile}" if a.tile else "") + (f"_bg{a.bg}" if a.bg else "")
    work = str(Path(a.work) / (Path(a.model).stem + tag))
    with ProcessPoolExecutor(max_workers=len(a.movies)) as ex:
        jobs = {mv: ex.submit(_one, mv, str(Path(a.model).resolve()), a.tile, work, a.bg) for mv in a.movies}
        res = {mv: j.result() for mv, j in jobs.items()}
    print(sb.line(Path(a.model).stem + tag, {"synthetic": {
        "on_hit": 0, "on_truth": 0, "errs": [], "early": 0, "late": 0, "missed": 0, "len_hit": 0, "len_n": 0,
        "abs_ok": 0, "abs_n": 0, "ctrl_fp": 0, "ctrl_n": 0}, "real": res}), flush=True)
    for mv, g in res.items():
        worst = sorted(g["errs"], key=lambda e: -abs(e[0]))[:8]
        print(f"  worst {mv} length errors: " + ", ".join(f"{gid}@{fr} {e:+.1f}/{h:.0f}" for e, h, gid, fr in worst))
    if a.baseline:
        print(sb.paired(json.loads(Path(a.baseline).read_text()), res), flush=True)
    Path(a.out).write_text(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
