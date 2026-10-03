"""Follow-ups to O1 (3 Oct): O3 (0.8.8's own routes, o3.py kymographs) and the soft recency term, paired against
0.8.8 over grains, all in one process (the machine is short of memory: one kymograph set loaded at a time).

- O3b / O3c with the O1 front settings tuned on ld (oracle routes), and re-tuned on ld on O3's own kymographs (0.8.8's
  routes start at 0.8.8's exit, not the annotator's); onsets from the front ("front") or 0.8.8's kept ("0.8.8"); m2-tuned
  front settings for sensitivity.
- Soft recency (carry.dp_recency), alone and with the void cost, its settings tuned on ld with the front settings
  fixed (m2-tuned for sensitivity), on O1's b/c and on O3b/O3c.
- O1 b/c with the human onset forced, re-run after fixing the infeasible-onset bug (vmax = 1 had been voided).

    python -m prototypes.carry_front.followup     # -> runs/research/carry_front/followup.json, o3/pred/*.json
"""
from __future__ import annotations

import json
import time

from prototypes.carry_front.carry import OUT
from prototypes.carry_front.o1 import grid_search, load, predict
from prototypes.carry_front.o3eval import PRED, SOFT_GRID, fmt, hits, report, run

OUT_JSON = OUT / "followup.json"


def soft_table(grains, v, prm) -> list[tuple]:
    return [(hits(grains, v, prm, s), s) for s in SOFT_GRID]


def best(table, void_ok=True):
    cand = [(sc, s) for sc, s in table if void_ok or s["void"] == 0]
    return max(cand, key=lambda x: x[0])[1]


def main():
    out = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    PRED.mkdir(parents=True, exist_ok=True)
    o1ld = json.loads((OUT / "o1" / "tune_ld" / "summary.json").read_text())
    o1m2 = json.loads((OUT / "o1" / "tune_m2" / "summary.json").read_text())

    def record(name, label, m, mv, pred, rows, prm, soft, onset, save):
        r = report(mv, pred, rows)
        out[f"{name} | {label} | {m}"] = {**r, "params": prm, "soft": soft, "onset": onset}
        print(fmt(f"{name} {label}", m, r), flush=True)
        if save:
            fn = PRED / f"{name}_{label.replace(' ', '_').replace(',', '').replace('/', '-').replace('+', 'plus')}_{m}.json"
            fn.write_text(json.dumps(pred))
        OUT_JSON.write_text(json.dumps(out, default=float))

    for s, root in (("O3", OUT / "o3"), ("O1", OUT)):
        data = {m: load(m, root, only={"b", "c"}, all_scored=(s == "O3")) for m in ("ld", "m2", "m1")}
        for v in ("b", "c"):
            name = f"{s}{v}"
            t0 = time.time()
            ld_full = [g for g in data["ld"][1] if g.traces]
            m2_full = [g for g in data["m2"][1] if g.traces]
            o1_ld, o1_m2 = o1ld[f"ld/{v}"]["params"], o1m2[f"m2/{v}"]["params"]
            if s == "O3":
                prm_ld = grid_search(ld_full, v, False)[0][3]
                prm_m2 = grid_search(m2_full, v, False)[0][3]
            else:
                prm_ld, prm_m2 = o1_ld, o1_m2
            tab_ld = soft_table(ld_full, v, prm_ld)
            tab_m2 = soft_table(m2_full, v, prm_ld)
            soft_ld, soft_ld_nv, soft_m2 = best(tab_ld), best(tab_ld, False), best(tab_m2)
            none_ld = hits(ld_full, v, prm_ld, None)
            print(f"{name}: front settings (ld) {prm_ld}, (m2) {prm_m2}; ld hits without soft {none_ld}, best soft "
                  f"{max(t[0] for t in tab_ld)}; soft (ld) {soft_ld}, no void {soft_ld_nv}; soft (m2) {soft_m2} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            configs = []
            if s == "O3":
                configs += [("O1 ld settings, front onsets", o1_ld, "dp", None),
                            ("re-tuned ld, front onsets", prm_ld, "dp", None),
                            ("re-tuned ld, 0.8.8 onsets", prm_ld, "088", None),
                            ("re-tuned m2, front onsets", prm_m2, "dp", None),
                            ("re-tuned m2, 0.8.8 onsets", prm_m2, "088", None)]
            configs += [("ld settings, no soft", prm_ld, "dp", None),
                        ("ld + soft recency, no void", prm_ld, "dp", soft_ld_nv),
                        ("ld + soft recency + void", prm_ld, "dp", soft_ld),
                        ("ld + soft recency + void tuned on m2", prm_ld, "dp", soft_m2)]
            if s == "O3":
                configs += [("re-tuned ld + soft + void, 0.8.8 onsets", prm_ld, "088", soft_ld)]
            for label, prm, onset, soft in configs:
                for m in ("ld", "m2", "m1"):
                    mv, gs = data[m]
                    pred, rows = run(mv, gs, v, prm, onset, soft, f"carry_front {name} {label}")
                    record(name, label, m, mv, pred, rows, prm, soft, onset, s == "O3")
            if s == "O1":  # the human onset forced, re-tuned on ld after the infeasible-onset fix
                res = grid_search(ld_full, v, True)
                prm_f = res[0][3]
                print(f"{name} forced (fixed): best on ld {res[0][:3]} {prm_f}", flush=True)
                for m in ("ld", "m2", "m1"):
                    mv, gs = data[m]
                    gs_full = [g for g in gs if g.traces]
                    pred, rows = predict(mv, gs_full, v, prm_f, True, f"carry_front {name} forced")
                    record(name, "human onset forced (fixed)", m, mv, pred, rows, prm_f, None, "human", False)
        del data


if __name__ == "__main__":
    main()
