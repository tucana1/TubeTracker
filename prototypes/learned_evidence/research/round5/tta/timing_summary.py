"""Analysis CPU time (1 thread) and the ratio each TTA recipe gives, from logs/passes.jsonl and logs/timing_pipeline_*.json."""
import json
import statistics as st
from pathlib import Path

ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/tta")
rows = [json.loads(line) for line in open(ME / "logs" / "passes.jsonl")]


def med(model, real):
    xs = [r["cpu_s"] for r in rows if r["model"] == model and (r["movie"] == "real") == real]
    return st.median(xs), len(xs), min(xs), max(xs)


for movie, real in (("v5faints30", False), ("real", True)):
    P = json.loads((ME / "logs" / f"timing_pipeline_{movie}.json").read_text())["pieces"]
    c = {k.split(" ")[0]: v["cpu_s"] for k, v in P.items()}
    v2, n2, lo2, hi2 = med("v2", real)
    b3, n3, lo3, hi3 = med("B3", real)
    rest_default = c["fuse"] + c["sparsetrack_baseline"] + c["sparsetrack_learned"] + c["perbin_decoder"] + c["review"] + c["reports"]
    rest_only = c["fuse"] + c["speed_cap"] + c["perbin_decoder"] + c["review"] + c["reports"]
    print(f"{movie}: v2 pass {v2:.0f} s (n {n2}, {lo2:.0f}-{hi2:.0f}), B3 pass {b3:.0f} s (n {n3}, {lo3:.0f}-{hi3:.0f}); "
          f"rest default mode {rest_default:.0f} s, only-perbin {rest_only:.0f} s; pieces {c}")
    for mode, rest in (("default", rest_default), ("only-perbin", rest_only)):
        base = v2 + b3 + rest
        out = [f"{mode}: plain {base:.0f} s (networks {100 * (v2 + b3) / base:.0f}%)"]
        for name, k2, k3 in (("t2 v2", 2, 1), ("t2 both", 2, 2), ("t4 v2", 4, 1), ("t4 both", 4, 4), ("t8 v2", 8, 1)):
            out.append(f"{name} x{(k2 * v2 + k3 * b3 + rest) / base:.2f}")
        print("   " + " | ".join(out))
