"""The sweep's table: every setting run in the two servers' queues, paired against the default rules on the same maps
(95% bootstrap intervals over grains), with the accidental length hits.

    python -m prototypes.flood_rules.sweep Q_M2 Q_LD [--json OUT]

Rows are settings (``sNN_*`` in both queues; on ld also ``fNN_*``, the same setting with the flood reading every
grain). Columns: m2 hybrid, ld flood everywhere, ld hybrid - each "hits (paired change vs default, CI)" for
lengths and length-and-tip, and onsets; "acc" = length hits on grains whose flood first claimed material (before
the onset look-back) more than 10 bins before the annotator's last-absent bin; "acc_lb" the same from the reported
onset (after the look-back).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .bench import EARLY, paired

REPO = Path(__file__).resolve().parents[2]
LABELS = {"ld": "benchmark/labels/ld_v1.json", "m2": "benchmark/labels/m2_v1.json"}


def accidental(movie: str, r: dict, labels: dict) -> tuple[int, int, list[str]]:
    """(hits by the flood's first claim, hits by the reported flood onset, grains early by first claim)."""
    fpb = int(labels["frames_per_bin"])
    by_claim, by_onset, early = 0, 0, []
    for gid, o in (r.get("own") or {}).items():
        on = (labels["labels"].get(gid) or {}).get("onset") or {}
        if o.get("onset_frame") is None or on.get("verdict") != "emerged_within" or on.get("last_absent_bin") is None:
            continue
        lb = next((int(f.split(":")[1]) for f in o["flags"] if f.startswith("onset_lookback:")), 0)
        onset = o["onset_frame"] // fpb
        hits = r["grains"].get(gid, {}).get("len_hit", 0)
        if onset + lb < on["last_absent_bin"] - EARLY:
            by_claim += hits
            early.append(gid)
        if onset < on["last_absent_bin"] - EARLY:
            by_onset += hits
    return by_claim, by_onset, early


def cell(movie: str, r: dict, base: dict | None, labels: dict) -> dict:
    out = {"len": r["len_hit"], "both": r["both"], "on": r["on_hit"], "on_n": r["on_n"],
           "rate_within": (r.get("growth") or {}).get("rate_within"),
           "rate_n": (r.get("growth") or {}).get("grains_with_rate"),
           "rate_r": (r.get("growth") or {}).get("rate_pearson")}
    out["acc"], out["acc_lb"], out["early_grains"] = accidental(movie, r, labels)
    if base is not None:
        for key, lab in (("len_hit", "d_len"), ("both_hit", "d_both"), ("onset_hit", "d_on")):
            out[lab] = paired(base[movie], {**r, "grains": r["grains"]}, key) if movie in base else None
    return out


def fmt(c: dict | None) -> str:
    if c is None:
        return f"{'-':>38s}"
    s = f"{c['len']:3d} {c['both']:3d} {c['on']:2d} a{c['acc']}/{c['acc_lb']}"
    if c.get("d_len"):
        (dl, lo, hi), (db, blo, bhi) = c["d_len"], c["d_both"]
        s += f" {dl:+3d}({lo:+.0f},{hi:+.0f}) {db:+3d}({blo:+.0f},{bhi:+.0f})"
    return f"{s:>38s}"


def load(q: Path, name: str, movie: str) -> dict | None:
    f = q / "done" / f"{name}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d if movie in d else None


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("q_m2")
    ap.add_argument("q_ld")
    ap.add_argument("--json")
    ap.add_argument("--base", default="d02_default_reload", help="the default rules' run in each queue")
    ap.add_argument("--flood-base", default="f02_default_reload")
    a = ap.parse_args(argv)
    qm, ql = Path(a.q_m2), Path(a.q_ld)
    labels = {mv: json.loads((REPO / p).read_text()) for mv, p in LABELS.items()}
    bm, bl, bf = load(qm, a.base, "m2"), load(ql, a.base, "ld"), load(ql, a.flood_base, "ld")
    names = sorted({f.stem for f in (qm / "done").glob("s*.json")} | {f.stem for f in (ql / "done").glob("s*.json")})
    rows = {}
    print(f"{'setting':22s} {'m2 hybrid: len both on acc | d_len d_both':>38s} {'ld flood everywhere':>38s} "
          f"{'ld hybrid':>38s}")
    for name in [a.base] + names:
        m = load(qm, name, "m2")
        lf = load(ql, a.flood_base if name == a.base else "f" + name[1:], "ld")
        lh = load(ql, name, "ld")
        rows[name] = {"m2": cell("m2", m["m2"], bm, labels["m2"]) if m else None,
                      "ld_flood": cell("ld", lf["ld"], bf, labels["ld"]) if lf else None,
                      "ld_hybrid": cell("ld", lh["ld"], bl, labels["ld"]) if lh else None,
                      "set": (m or lh or lf or {}).get("m2" if m else "ld", {}).get("set")}
        print(f"{name:22s} {fmt(rows[name]['m2'])} {fmt(rows[name]['ld_flood'])} {fmt(rows[name]['ld_hybrid'])}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
