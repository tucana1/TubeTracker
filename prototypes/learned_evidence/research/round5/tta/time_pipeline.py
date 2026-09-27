"""Time the pieces of an analysis (pipeline.py without labels, as Analyze_Movie_Learned.command runs it) on one
movie, at one thread: v2 pass, B3 pass, fusion, SparseTrack's two runs (default mode) or the speed cap probe
(--only-perbin), the per-bin decoder, the review pictures and reports. Wall and CPU seconds per piece.

    python time_pipeline.py MOVIE [--passes]   -> logs/timing_pipeline_<MOVIE>.json (dense caches deleted after)
Without --passes the v2 and B3 caches are dense copies of the sparse ones (the passes are timed by tta.py).
"""
from __future__ import annotations

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/tta")
sys.path.insert(0, str(ME))
import tta  # noqa: E402,F401  (paths)


def run(movie: str, passes: bool = False) -> dict:
    import torch
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    from prototypes.learned_evidence import evaluate, fuse, reach, review
    from prototypes.learned_evidence.finetune import speed_cap
    from prototypes.learned_evidence.model import load as load_model
    from sparsetrack.report import write_growth_curves, write_population
    field = tta.image_cache_of(movie)
    work = ME / "tmp" / f"timing_{movie}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    T = {}

    def timed(name, fn):
        w, c = time.time(), time.process_time()
        r = fn()
        T[name] = {"wall_s": round(time.time() - w, 1), "cpu_s": round(time.process_time() - c, 1)}
        print(time.strftime("%H:%M:%S"), name, T[name], flush=True)
        return r

    try:
        if passes:
            net = load_model(str(tta.MODELS["v2"]), "cpu")
            pcache = timed("v2_pass", lambda: evaluate.prob_cache(field, net, work / "prob_v2", log=lambda *a: None))
            tnet = load_model(str(tta.MODELS["B3"]), "cpu")
            tcache = timed("B3_pass", lambda: evaluate.prob_cache(field, tnet, work / "prob_thick",
                                                                  log=lambda *a: None))
        else:  # the network passes are timed by tta.py's passes (logs/passes.jsonl); dense copies of the caches
            import decode5
            from common5 import sparse_path
            pcache = decode5.dense_fused(sparse_path("v2", movie), None, work / "prob_v2")
            tcache = decode5.dense_fused(sparse_path("B3", movie), None, work / "prob_thick")
        ecache = timed("fuse", lambda: fuse.fused_cache(pcache, tcache, work / "prob_fused", log=lambda *a: None))
        vmax = timed("speed_cap (only-perbin)", lambda: speed_cap(pcache, field, field / "grains.json"))
        with evaluate.adaptive_crop():
            timed("sparsetrack_baseline (default mode)", lambda: evaluate.run_baseline(field, work))
            timed("sparsetrack_learned (default mode)",
                  lambda: evaluate.run_on_prob_cache(pcache, field, work, "learned"))
        kw = dict(big=300, burst=True, vmax=vmax, continuity="path")
        perbin = timed("perbin_decoder", lambda: reach.analyze(ecache, field, log=lambda *a: None, **kw))
        timed("review", lambda: review.write_review(ecache, field, perbin, work / "perbin", **kw))
        ids = [g["id"] for g in perbin["grains"] if g.get("status") == "emerged_within"]
        timed("reports", lambda: (write_population(perbin, work / "perbin"),
                                  write_growth_curves(perbin, work / "perbin", ids)))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out = {"movie": movie, "threads": 1, "pieces": T, "passes_timed_here": passes,
           "finished": time.strftime("%H:%M:%S")}
    (ME / "logs" / f"timing_pipeline_{movie}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run(sys.argv[1], "--passes" in sys.argv)
