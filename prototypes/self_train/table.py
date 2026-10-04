"""Tables of the self-training experiment: each self-trained network on all of a movie's labelled grains against the
starting network's 0.8.8 reading (paired over grains, 95% bootstrap intervals), and on movie 1 against the
human-label fine-tunes of ``prototypes/tube_adapt`` on the same grains (each grain judged by the fold that did not
see its traces).

    python -m prototypes.self_train.table m1 --nets m1_r1_sb m1_r1_sa [--human a b c]
    python -m prototypes.self_train.table m2 --nets m2_r1_sb --base-e2e e2e_m2_base_ldm1 --base-pix pix_m2_base_ldm1
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from prototypes.self_train.select import OUT
from prototypes.tube_adapt.common import BASE_DUMP, OUT as ADAPT
from prototypes.tube_adapt.evaluate import paired
from prototypes.tube_adapt.table import fmt_pix, fmt_pp, pix_cells
from prototypes.tube_net.pixpair import compare


def jload(p):
    return json.loads(p.read_text()) if p.exists() else None


def pix_of(d: dict) -> tuple[list, tuple, list]:
    r = d["summary"]["rim_before_onset"]
    return d["rows"], (r["marked"], r["grain_bins"]), d["tip_extent"]


def human_pooled(v: str):
    rows, te, rim, e2e = [], [], [0, 0], {}
    for f in ("k2f0", "k2f1"):
        p = jload(ADAPT / "eval" / f"pix_{f}_{v}_m1.json")
        e = jload(ADAPT / "eval" / f"e2e_{f}_{v}.json")
        if p is None or e is None:
            return None
        rw, rc, t = pix_of(p)
        rows += rw; te += t; rim[0] += rc[0]; rim[1] += rc[1]
        e2e.update(e["grains"])
    return rows, tuple(rim), te, e2e


def e2e_line(base: dict, new: dict, grains: list[str]) -> str:
    p = paired(base, new, grains)
    on, ln, bo = p["onset_hit"], p["len_hit"], p["both_hit"]
    n_on = sum(1 for g in grains if g in new and new[g]["onset_hit"] is not None)
    return (f"lengths {ln[4]}/{p['len_n']} ({ln[0]:+d}, {ln[1]:+.0f}..{ln[2]:+.0f}) | length+tip {bo[4]} ({bo[0]:+d}, "
            f"{bo[1]:+.0f}..{bo[2]:+.0f}) | onsets {on[4]}/{n_on} ({on[0]:+d}, {on[1]:+.0f}..{on[2]:+.0f})")


def onset_detail(e2e: dict, grains) -> str:
    errs = [e2e[g]["onset_error"] for g in grains if g in e2e and e2e[g].get("onset_error") is not None]
    early = sum(1 for x in errs if x < -600)
    late = sum(1 for x in errs if x > 600)
    return f"onset errors: {early} early, {late} late (beyond 2 bins) of {len(errs)} timed"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--nets", nargs="+", required=True)
    ap.add_argument("--human", nargs="*", default=["a", "b", "c"])
    ap.add_argument("--base-e2e", help="e2e json of the starting network (default: the 0.8.8 dump)")
    ap.add_argument("--base-pix", help="pix json of the starting network (default: tube_adapt's base on m1)")
    a = ap.parse_args(argv)
    if a.base_e2e:
        base_e2e = jload(OUT / "eval" / f"{a.base_e2e}.json")["grains"]
    else:
        base_e2e = json.loads(BASE_DUMP.read_text())[a.movie]["grains"]
    if a.base_pix:
        bp = pix_of(jload(OUT / "eval" / f"{a.base_pix}.json"))
    else:
        b0 = jload(ADAPT / "eval" / f"pix_base_{a.movie}.json")
        bp = pix_of(b0)
    nets = {n: (jload(OUT / "eval" / f"pix_{n}.json"), jload(OUT / "eval" / f"e2e_{n}.json")) for n in a.nets}
    grains = sorted(base_e2e)
    if nets and next(iter(nets.values()))[1]:
        grains = sorted(next(iter(nets.values()))[1]["grains"])
    print(f"== {a.movie}: {len(grains)} labelled grains; pixel check columns: traced marked | long marked | stubs, "
          f"tips | young stubs, tips | beside | rim before onset | marks end vs apex, within 2 px")
    print(f"  start  pix: {fmt_pix(pix_cells(*bp))}")
    print(f"  start  e2e: " + e2e_line(base_e2e, base_e2e, grains).replace("(+0, +0..+0)", "") + " | "
          + onset_detail(base_e2e, grains))
    for n, (p, e) in nets.items():
        if p:
            print(f"  {n} pix: {fmt_pix(pix_cells(*pix_of(p)))}\n      vs start: {fmt_pp(compare(bp[0], p['rows']))}")
        if e:
            print(f"  {n} e2e vs start: {e2e_line(base_e2e, e['grains'], grains)} | {onset_detail(e['grains'], grains)}")
    if a.movie != "m1":
        return
    print("\n== movie 1: human-label fine-tunes (tube_adapt, pooled over its two folds: each grain judged by the network "
          "that did not see its traces)")
    for v in a.human:
        h = human_pooled(v)
        if h is None:
            continue
        rows, rim, te, e2e = h
        print(f"  human {v} pix: {fmt_pix(pix_cells(rows, rim, te))}\n      vs start: {fmt_pp(compare(bp[0], rows))}")
        print(f"  human {v} e2e vs start: {e2e_line(base_e2e, e2e, grains)} | {onset_detail(e2e, grains)}")
        for n, (p, e) in nets.items():
            if p:
                print(f"    {n} minus human {v}: pix {fmt_pp(compare(rows, p['rows']))}")
            if e:
                print(f"    {n} minus human {v}: e2e {e2e_line(e2e, e['grains'], grains)}")


if __name__ == "__main__":
    main()


def detail(movie: str, net: str, base_e2e_name: str | None = None) -> None:
    """Per grain: onset error (frames) and each FULL trace's length error (model - human, px), start -> network."""
    if base_e2e_name:
        base = jload(OUT / "eval" / f"{base_e2e_name}.json")["grains"]
    else:
        base = json.loads(BASE_DUMP.read_text())[movie]["grains"]
    new = jload(OUT / "eval" / f"e2e_{net}.json")["grains"]
    for g in sorted(new):
        b, n = base.get(g, {}), new[g]
        bf = {f["frame"]: f for f in b.get("full", [])}
        nf = {f["frame"]: f for f in n.get("full", [])}
        cells = []
        for fr in sorted(set(bf) | set(nf)):
            x, y = bf.get(fr), nf.get(fr)
            h = (x or y)["human"]
            fe = lambda f: f"{f['error']:+.1f}" if f else "-"
            cells.append(f"b{fr // 300} {h:.0f}px {fe(x)}->{fe(y)}")
        print(f"  {g}: onset {b.get('onset_error')} -> {n.get('onset_error')} ({b.get('onset_hit')}->{n.get('onset_hit')}) | "
              + "; ".join(cells))


def score_pred(movie: str, pred_path: str, out_name: str) -> dict:
    """An existing predictions file scored as ``tube_adapt.evaluate.e2e`` scores a fresh read (per-grain fields), into
    eval/<out_name>.json (e.g. 0.8.8's own m1 reading, whose dump keeps no onset errors)."""
    from prototypes.tube_adapt.common import labels, subset
    from prototypes.tube_adapt.evaluate import within
    from sparsetrack.evaluate import load as load_pred, score
    L = labels(movie)
    grains = sorted(g for g in L["labels"] if not L["grains"][g].get("excluded"))
    rep = score(subset(L, grains), load_pred(pred_path))
    out = {}
    for r in rep["rows"]:
        full = r.get("full", [])
        out[r["grain"]] = {"onset_hit": (abs(r["onset_error"]) <= 600) if "onset_error" in r else None,
                           "len_hit": sum(within(f["error"], f["human"]) for f in full), "len_n": len(full),
                           "both_hit": sum(within(f["error"], f["human"]) and f.get("tip_error", 1e9)
                                           <= max(5.0, 0.1 * f["human"]) for f in full),
                           "full": full, "pred_status": r.get("pred"), "onset_error": r.get("onset_error")}
    res = {"on_hit": rep["onset"]["hits"], "on_n": rep["onset"]["n_timed"],
           "len_hit": rep["length_full"]["within_tolerance"], "len_n": rep["length_full"]["n"],
           "both": rep["tips"]["length_and_tip"], "grains": out, "pred": str(pred_path)}
    (OUT / "eval" / f"{out_name}.json").write_text(json.dumps(res, default=float))
    return res
