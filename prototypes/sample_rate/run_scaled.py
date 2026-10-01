"""The 150-frames-per-bin sparse movie with the bin-counted settings doubled (the same windows in time)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/Users/joshjiang/Documents/TubeTracker")
from sparsetrack import evaluate
from sparsetrack.analyze import Params, analyze
REPO = Path("/Users/joshjiang/Documents/TubeTracker")
k = 2
p = Params()
p = Params(**{**p.__dict__, **{n: getattr(p, n) * k for n in (
    "wedge_base_bins", "wedge_hold", "persist_bins", "mf_back_bins", "flood_recent", "flood_persist", "flood_old_far_bins",
    "flood_give_up", "cont_after_bins", "cont_margin_bins", "cont_gate_bins", "settle_bins", "late_bins", "centre_late")},
    "max_turn": p.max_turn / k})
root = Path(__file__).parent / "key_150"
t0 = time.time()
pred = analyze(root / "cache", root / "analysis_scaled", grains_path=REPO / "benchmark/labels/ld_v1.json", params=p,
               log=lambda *a: None)
r = evaluate.score(json.loads((REPO / "benchmark/labels/ld_v1.json").read_text()), pred)
print(json.dumps({"analyze_s": round(time.time() - t0), "onsets": f"{r['onset']['hits']}/{r['onset']['n_timed']}",
                  "onset_median_abs_frames": r["onset"]["median_abs_error"],
                  "lengths": f"{r['length_full']['within_tolerance']}/{r['length_full']['n']}",
                  "length_median_abs": r["length_full"]["median_abs_error"], "tips": f"{r['tips']['within']}/{r['tips']['n']}",
                  "length_and_tip": r["tips"]["length_and_tip"], "t50": r["population"].get("t50_model"),
                  "growth_within": r["growth"].get("rate_within")}))
