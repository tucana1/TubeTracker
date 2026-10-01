"""Sample-rate experiments on the sparse movie: build a cache per (sampling, frames per bin), its tube maps, then
analyse the labelled grains and score them. Timings of each stage are logged.

    python run_rate.py NAME SAMPLE FPB
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/joshjiang/Documents/TubeTracker")
from sparsetrack import evaluate, learned, stack
from sparsetrack.analyze import analyze

REPO = Path("/Users/joshjiang/Documents/TubeTracker")
MOVIE = "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"
name, sample, fpb = sys.argv[1], sys.argv[2], int(sys.argv[3])
sample = sample if sample in ("keyframes", "all") else int(sample)
root = Path(__file__).parent / name
cache = root / "cache"
t = {}
t0 = time.time()
if not (cache / "meta.json").exists():
    stack.prepare(MOVIE, cache, frames_per_bin=fpb, ref_bins=3, ref_start="auto", sample=sample, log=print)
t["prepare"] = time.time() - t0
t0 = time.time()
learned.prob_cache(cache, learned.MODEL, log=print)
t["maps"] = time.time() - t0
labels = Path(__file__).parent / "labels_shifted.json"
labels = labels if labels.exists() else REPO / "benchmark/labels/ld_v1.json"
t0 = time.time()
pred = analyze(cache, root / "analysis", grains_path=labels, log=lambda *a: None)
t["analyze"] = time.time() - t0
r = evaluate.score(json.loads((REPO / "benchmark/labels/ld_v1.json").read_text()), pred)
out = {"name": name, "sample": str(sample), "fpb": fpb, "n_bins": stack.load(cache)[1]["n_bins"], "times_s": t,
       "onsets": f"{r['onset']['hits']}/{r['onset']['n_timed']}", "onset_median_abs_frames": r["onset"]["median_abs_error"],
       "lengths": f"{r['length_full']['within_tolerance']}/{r['length_full']['n']}",
       "length_median_abs": r["length_full"]["median_abs_error"], "length_bias": r["length_full"]["bias"],
       "tips": f"{r['tips']['within']}/{r['tips']['n']}", "tip_median": r["tips"]["median_px"],
       "length_and_tip": r["tips"]["length_and_tip"], "t50": [r["population"].get("t50_human"), r["population"].get("t50_model")],
       "growth_within": r["growth"].get("rate_within"), "growth_r": r["growth"].get("rate_pearson")}
(root / "score.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
