"""Real-movie audit: a candidate's predictions against R5 default, grain by grain, with the audit's visual estimates
(SCR/agents/audit/verdicts.csv) judged as SCR/agents/simreal/real_check.py does (onset within 6 bins; lengths within
[0.75 lo, 1.25 hi]; a no-tube estimate is met under 8 px; '>= x' has no upper bound; '?', 'unclear' skipped).

    python audit_tta.py TAG            (my preds/real/<TAG>.json vs R5/preds/real/default.json)
Lists every grain whose onset, b64 or end reading changes, marks fixes (+) and breaks (-) against the audit.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/tta")
SCR = ME.parent.parent


def parse(s: str):  # as real_check.parse
    s = (s or "").strip()
    if not s or s.startswith("?") or "unclear" in s or "can't" in s:
        return None
    if s.startswith("no tube") or s == "0":
        return (0.0, 0.0)
    m = re.match(r"^(>=|>)\s*(\d+)", s)
    if m:
        return (float(m.group(2)), math.inf)
    m = re.match(r"^~?(\d+)\s*-\s*(\d+)", s)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = re.match(r"^~?(\d+)(\+?)", s)
    if m:
        v = float(m.group(1))
        return (v, math.inf) if m.group(2) == "+" else (v, v)
    return None


def within(pred, est):  # as real_check.within
    lo, hi = est
    if hi == 0:
        return pred < 8.0
    return 0.75 * lo <= pred <= 1.25 * hi


def readings(g):  # as real_check.readings
    if g is None:
        return None, 0.0, 0.0
    frames, px = g["length"]["frames"], g["length"]["px"]
    b64 = px[frames.index(64)] if 64 in frames else None
    onset = g["onset_frame"] if g["status"] != "no_emergence_by_end" else None
    return onset, b64, g["final_length_px"]


def load(p: Path):
    return {g["id"]: g for g in json.loads(p.read_text())["grains"]}


def audit(tag: str, base_path: Path | None = None, new_path: Path | None = None, out=print) -> dict:
    base = load(base_path or SCR / "round5" / "preds" / "real" / "default.json")
    new = load(new_path or ME / "preds" / "real" / f"{tag}.json")
    rows = list(csv.DictReader(open(SCR / "agents" / "audit" / "verdicts.csv")))
    tot = {"base": [0, 0, 0, 0, 0, 0], "new": [0, 0, 0, 0, 0, 0]}
    fixes, breaks, changed = [], [], []
    for r in rows:
        gid = r["grain"]
        ests = (parse(r["est_onset"]), parse(r["est_b64"]), parse(r["est_end"]))
        ra, rb = readings(base.get(gid)), readings(new.get(gid))
        marks = {}
        for who, rd in (("base", ra), ("new", rb)):
            on, b64, end = rd
            ok = [None, None, None]
            if ests[0] is not None:
                ok[0] = (on is None) if ests[0] == (0.0, 0.0) else (on is not None and abs(on - ests[0][0]) <= 6)
            if ests[1] is not None and b64 is not None:
                ok[1] = within(b64, ests[1])
            if ests[2] is not None:
                ok[2] = within(end, ests[2])
            for i, o in enumerate(ok):
                if o is not None:
                    tot[who][2 * i] += o
                    tot[who][2 * i + 1] += 1
            marks[who] = ok
        diff = any((x != y) for x, y in zip(ra, rb))
        if not diff:
            continue
        verdict = []
        for i, name in enumerate(("onset", "b64", "end")):
            a, b = marks["base"][i], marks["new"][i]
            if a is not None and b is not None and a != b:
                (fixes if b else breaks).append(f"{gid}:{name}")
                verdict.append(("+" if b else "-") + name)
        fmt = lambda rd: (f"on {rd[0] if rd[0] is not None else '-':>4} b64 "  # noqa: E731
                          f"{rd[1] if rd[1] is not None else '-':>6} end {rd[2]:6.1f}")
        changed.append(gid)
        out(f"{gid} audit on {r['est_onset'][:8]:>8} b64 {r['est_b64'][:10]:>10} end {r['est_end'][:16]:>16} | "
            f"default {fmt(ra)} | {tag} {fmt(rb)} | {' '.join(verdict) or 'no change in verdict'}")
    for who in ("base", "new"):
        a = tot[who]
        out(f"{'default' if who == 'base' else tag:14s} onsets within 6 bins {a[0]}/{a[1]} | b64 within 25% "
            f"{a[2]}/{a[3]} | end within 25% {a[4]}/{a[5]}")
    out(f"grains changed {len(changed)}; fixes {len(fixes)} {fixes}; breaks {len(breaks)} {breaks}")
    return {"changed": changed, "fixes": fixes, "breaks": breaks, "totals": tot}


if __name__ == "__main__":
    audit(sys.argv[1])
