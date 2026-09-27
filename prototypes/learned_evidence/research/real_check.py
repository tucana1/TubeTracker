"""Gross-failure check on the real sample movie against a visual audit of its 35 grains (audit/verdicts.csv, made on
25 Sep 2026 from panels of the movie; estimates are good to about 10-20%).

    python -m prototypes.learned_evidence.research.real_check BASE_TAG NEW_TAG     (preds/real/<TAG>.json)
A length counts as right when it lies in [0.75 lo, 1.25 hi] of the audit's estimate (a value, a range, or ">= x");
a no-tube estimate is met by a reading under 8 px; onsets within 6 bins. Estimates marked "?", "unclear" or "can't
tell" are skipped. Per grain: a gross fix gains a mark and loses none, a gross break loses one and gains none.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

from .common import PREDS

AUDIT = Path(__file__).parent / "audit" / "verdicts.csv"


def parse(s: str):
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


def within(pred: float, est) -> bool:
    lo, hi = est
    return pred < 8.0 if hi == 0 else 0.75 * lo <= pred <= 1.25 * hi


def readings(g: dict | None):
    if g is None:  # not analysed: judged a ghost disc (not a grain, so no tube)
        return None, 0.0, 0.0
    frames, px = g["length"]["frames"], g["length"]["px"]
    b64 = px[frames.index(64)] if 64 in frames else None
    onset = g["onset_frame"] if g["status"] != "no_emergence_by_end" else None
    return onset, b64, g["final_length_px"]


def marks(g: dict | None, row: dict) -> dict:
    eo, e64, eend = parse(row["est_onset"]), parse(row["est_b64"]), parse(row["est_end"])
    on, b64, end = readings(g)
    out = {}
    if eo is not None:
        out["onset"] = (on is None) if eo == (0.0, 0.0) else (on is not None and abs(on - eo[0]) <= 6)
    if e64 is not None and b64 is not None:
        out["mid"] = within(b64, e64)
    if eend is not None:
        out["end"] = within(end, eend)
    return out


def check(a: str, b: str, preds=PREDS) -> dict:
    rows = list(csv.DictReader(open(AUDIT)))
    pa = {g["id"]: g for g in json.loads((preds / "real" / f"{a}.json").read_text())["grains"]}
    pb = {g["id"]: g for g in json.loads((preds / "real" / f"{b}.json").read_text())["grains"]}
    tot = [{"onset": [0, 0], "mid": [0, 0], "end": [0, 0]} for _ in (a, b)]  # by side: a tag may be compared with itself
    fixes, breaks, mixed, lines = [], [], [], []
    for r in rows:
        gid = r["grain"]
        ma, mb = marks(pa.get(gid), r), marks(pb.get(gid), r)
        for side, m in ((0, ma), (1, mb)):
            for k, ok in m.items():
                tot[side][k][0] += int(ok)
                tot[side][k][1] += 1
        gained = [k for k in mb if mb[k] and not ma.get(k, False)]
        lost = [k for k in ma if ma[k] and not mb.get(k, False)]
        if gained and not lost:
            fixes.append(gid)
        elif lost and not gained:
            breaks.append(gid)
        elif gained and lost:
            mixed.append(gid)
        if gained or lost:
            ra, rb = readings(pa.get(gid)), readings(pb.get(gid))
            lines.append(f"  {gid}: +{gained} -{lost} | onset/mid/end {ra} -> {rb} (audit {r['est_onset']}/"
                         f"{r['est_b64']}/{r['est_end']})")
    for t, x in zip((a, b), tot):
        print(f"{t:14s} onsets within 6 bins {x['onset'][0]}/{x['onset'][1]} | mid-movie lengths within 25% "
              f"{x['mid'][0]}/{x['mid'][1]} | end lengths within 25% {x['end'][0]}/{x['end'][1]}")
    print(f"{a} -> {b}: gross fixes {len(fixes)} {fixes}, gross breaks {len(breaks)} {breaks}, mixed {len(mixed)} {mixed}")
    print("\n".join(lines))
    return {"totals": dict(zip(("base", "new"), tot)), "fixes": fixes, "breaks": breaks, "mixed": mixed}


if __name__ == "__main__":
    check(sys.argv[1], sys.argv[2])
