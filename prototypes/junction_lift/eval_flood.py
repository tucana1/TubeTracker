"""A flood variant scored against the 0.8.8 baseline in one process: only the flood-read grains are read again (with
the variant's Params; the change reader's grains and the hybrid's choice of reader do not depend on flood options),
merged into the baseline predictions and scored as scripts/synth_bench.py scores them (paired bootstrap over grains).

    eval_flood.py TAG m2 ld m1 -- flood_lift_px=20.0
"""
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MAIN, SP, WT, movie  # noqa: E402
import reflood  # noqa: E402

sys.path.insert(0, WT + "/scripts")
from synth_bench import _within, paired  # noqa: E402
from sparsetrack.evaluate import score  # noqa: E402

BASE = json.load(open(SP + "bt/base088.json"))


def parse(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def per_grain(rep, pred_path):
    grains = {}
    for r in rep["rows"]:
        full = r.get("full", [])
        grains[r["grain"]] = {"onset_hit": (abs(r["onset_error"]) <= 600) if "onset_error" in r else None,
                              "len_hit": sum(_within(f["error"], f["human"]) for f in full), "len_n": len(full),
                              "both_hit": sum(_within(f["error"], f["human"]) and f.get("tip_error", 1e9)
                                              <= max(5.0, 0.1 * f["human"]) for f in full)}
    return {"on_hit": rep["onset"]["hits"], "on_n": rep["onset"]["n_timed"],
            "len_hit": rep["length_full"]["within_tolerance"], "len_n": rep["length_full"]["n"],
            "len_med": rep["length_full"]["median_abs_error"], "len_bias": rep["length_full"]["bias"],
            "both": rep["tips"]["length_and_tip"],
            "errs": [(f["error"], f["human"], r["grain"], f["frame"]) for r in rep["rows"] for f in r.get("full", [])],
            "grains": grains, "pred": pred_path}


if __name__ == "__main__":
    tag = sys.argv[1]
    k = sys.argv.index("--")
    movies, kv = sys.argv[2:k], sys.argv[k + 1:]
    kw = {a.split("=")[0]: parse(a.split("=")[1]) for a in kv}
    out_dir = SP + f"jt/eval/{tag}/"
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for m in movies:
        t0 = time.time()
        mv = movie(m)
        pred = copy.deepcopy(mv["pred"])
        changed = []
        for j, g in enumerate(pred["grains"]):
            if "reader:flood" not in g["flags"]:
                continue
            st = reflood.run(m, g["id"], p=reflood.base_params(m, **kw), use_cache=False)
            r = st["res"]
            if not np.allclose(r["length"]["px"], g["length"]["px"]):
                changed.append(g["id"])
            pred["grains"][j] = r
            print(f"  {m} {g['id']}: final {g.get('final_length_px')} -> {r.get('final_length_px')}", flush=True)
        path = out_dir + f"{m}_predictions.json"
        json.dump(pred, open(path, "w"))
        rep = score(json.load(open(MAIN + f"benchmark/labels/{m}_v1.json")), pred)
        res = per_grain(rep, path)
        results[m] = res
        b = BASE[m]
        print(f"== {m} ({time.time() - t0:.0f} s; lengths changed on {len(changed)} grains: {' '.join(changed)})")
        print(f"   onset {res['on_hit']}/{res['on_n']} (base {b['on_hit']}/{b['on_n']}), lengths {res['len_hit']}/{res['len_n']} "
              f"(base {b['len_hit']}/{b['len_n']}), length and tip {res['both']} (base {b['both']}), "
              f"median |err| {res['len_med']:.2f} (base {b['len_med']:.2f}), bias {res['len_bias']:+.2f} (base {b['len_bias']:+.2f})")
        print(paired(BASE, {m: res}), flush=True)
        for gid in sorted(res["grains"]):
            a, c = b["grains"].get(gid), res["grains"][gid]
            if a and (a["len_hit"] != c["len_hit"] or a.get("both_hit") != c.get("both_hit") or a["onset_hit"] != c["onset_hit"]):
                print(f"   {gid}: lengths {a['len_hit']} -> {c['len_hit']} of {c['len_n']}, length and tip "
                      f"{a.get('both_hit')} -> {c.get('both_hit')}, onset {a['onset_hit']} -> {c['onset_hit']}")
        json.dump(results, open(out_dir + "results.json", "w"), default=float)
