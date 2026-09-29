"""Real-movie audit, grain by grain, of a prediction against the round-5 default (R5/preds/real/default.json), with the
audit's estimates (SCR/agents/audit/verdicts.csv) and the shared tolerances (SCR/agents/simreal/real_check.py: onsets
within 6 bins; lengths within [0.75 lo, 1.25 hi], '>= x' without upper bound; a no-tube estimate is met by < 8 px).
Classification as round 3/4 (agents/fusion/real_check2.py):
  gross fix   = a mark gained (onset now within 6 bins; a length now within +-25% of the audit);
  gross break = an onset now more than 6 bins off (was within), a new tubeless call (< 8 px, was >= 8, audit > 0),
                or a length that newly runs onto a foreign tube (now above 1.25 x the audit's upper bound, was not);
  "short"     = a length that fell from within to below the tolerance (listed, neither of the above).
Also listed: every grain whose readings change, onsets that newly come more than 6 bins early, and totals.

    python real_audit.py NEW_JSON [BASE_JSON]
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

SCR = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad")
BASE = SCR / "round5" / "preds" / "real" / "default.json"
ROWS = list(csv.DictReader(open(SCR / "agents" / "audit" / "verdicts.csv")))


# ---- parse / within / readings: verbatim from SCR/agents/simreal/real_check.py
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


def within(pred, est):
    lo, hi = est
    if hi == 0:
        return pred < 8.0
    return 0.75 * lo <= pred <= 1.25 * hi


def readings(g):
    if g is None:  # not analysed: the current decoder judged the census disc a ghost (not a grain, so no tube)
        return None, 0.0, 0.0
    frames, px = g["length"]["frames"], g["length"]["px"]
    b64 = px[frames.index(64)] if 64 in frames else None
    onset = g["onset_frame"] if g["status"] != "no_emergence_by_end" else None
    return onset, b64, g["final_length_px"]
# ----


def load(p):
    return {g["id"]: g for g in json.loads(Path(p).read_text())["grains"]}


def onset_ok(on, eo):
    return (on is None) if eo == (0.0, 0.0) else (on is not None and abs(on - eo[0]) <= 6)


def classify(base, new):
    pr, pn = load(base), load(new)
    fixes, breaks, shorts, lines, tot = [], [], [], [], {"o": [0, 0, 0], "m": [0, 0, 0], "e": [0, 0, 0]}
    early = [0, 0]
    for r in ROWS:
        gid = r["grain"]
        ests = [parse(r["est_onset"]), parse(r["est_b64"]), parse(r["est_end"])]
        vo, vn = readings(pr.get(gid)), readings(pn.get(gid))
        eo = ests[0]
        if eo is not None:
            tot["o"][0] += onset_ok(vo[0], eo)
            tot["o"][1] += onset_ok(vn[0], eo)
            tot["o"][2] += 1
            if eo != (0.0, 0.0):
                early[0] += vo[0] is not None and vo[0] < eo[0] - 6
                early[1] += vn[0] is not None and vn[0] < eo[0] - 6
        for k, key in ((1, "m"), (2, "e")):
            if ests[k] is not None and vo[k] is not None and vn[k] is not None:
                tot[key][0] += within(vo[k], ests[k])
                tot[key][1] += within(vn[k], ests[k])
                tot[key][2] += 1
        if vo == vn:
            continue
        f, b, s = [], [], []
        if eo is not None:
            ok_o, ok_n = onset_ok(vo[0], eo), onset_ok(vn[0], eo)
            if ok_n and not ok_o:
                f.append("onset")
            if ok_o and not ok_n:
                b.append("onset" + (" early" if vn[0] is not None and vn[0] < eo[0] - 6 else ""))
        for k, name in ((1, "b64"), (2, "end")):
            e = ests[k]
            if e is None or vo[k] is None or vn[k] is None:
                continue
            wo, wn = within(vo[k], e), within(vn[k], e)
            lo, hi = e
            over_o, over_n = (hi > 0 and vo[k] > 1.25 * hi), (hi > 0 and vn[k] > 1.25 * hi)
            if hi == 0:  # a no-tube grain: a reading of 8 px or more is a foreign tube
                over_o, over_n = vo[k] >= 8.0, vn[k] >= 8.0
            if wn and not wo:
                f.append(name)
            if over_n and not over_o:
                b.append(f"{name} foreign")
            elif lo > 0 and vn[k] < 8.0 <= vo[k]:
                b.append(f"{name} tubeless")
            elif wo and not wn:
                s.append(f"{name} short")
        fixes += [(gid, x) for x in f]
        breaks += [(gid, x) for x in b]
        shorts += [(gid, x) for x in s]
        fmt = lambda v: "-" if v is None else f"{v:.0f}"  # noqa: E731
        tagline = ("FIX " if f and not b else "BREAK " if b and not f else "MIXED " if f and b else "") + \
                  ("short " if s else "")
        lines.append(f"  {gid} [{tagline.strip() or 'no mark change'}] on/b64/end {fmt(vo[0])}/{fmt(vo[1])}/{fmt(vo[2])} -> "
                     f"{fmt(vn[0])}/{fmt(vn[1])}/{fmt(vn[2])}  (audit {r['est_onset']}/{r['est_b64']}/{r['est_end']})"
                     f"  fix {f} break {b} short {s}")
    gf = len({g for g, _ in fixes} - {g for g, _ in breaks})
    gb = len({g for g, _ in breaks} - {g for g, _ in fixes})
    gm = len({g for g, _ in fixes} & {g for g, _ in breaks})
    print(f"--- {base} -> {new}")
    print("\n".join(lines))
    print(f"readings fixed {len(fixes)}, broken {len(breaks)}, short {len(shorts)} | grains fix-only {gf}, break-only {gb},"
          f" both {gm}")
    print(f"marks base -> new: onsets within 6 bins {tot['o'][0]} -> {tot['o'][1]} /{tot['o'][2]} | b64 within +-25% "
          f"{tot['m'][0]} -> {tot['m'][1]} /{tot['m'][2]} | end within +-25% {tot['e'][0]} -> {tot['e'][1]} /{tot['e'][2]}"
          f" | onsets > 6 bins early {early[0]} -> {early[1]}")
    return {"fixes": fixes, "breaks": breaks, "shorts": shorts, "grains_fix": gf, "grains_break": gb, "grains_both": gm,
            "marks": tot, "early": early}


if __name__ == "__main__":
    classify(sys.argv[2] if len(sys.argv) > 2 else BASE, sys.argv[1])
