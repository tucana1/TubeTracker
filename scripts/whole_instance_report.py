"""One-command whole-instance report for a checkpoint (rev8 step 3).

Runs the real evaluator twice — the held-out tube's distal fit and the
clump's query-swap matrix — and prints a single comparison block plus a
combined JSON. Used for A/B comparisons between training recipes so the
numbers are produced by the same code path every time:

  python scripts/whole_instance_report.py --checkpoint <ckpt> --base 4

Readouts, and what each one can and cannot show:
  * held-out distal IoU — a DISJOINT read (the tube is in dev).
  * swap diag/off-diag/spread — the query-selectivity structure. Rows
    identical (spread 0) means the query is ignored.
  * control row — a query disc on the emptiest crop point; a non-zero
    response there means the model fires on tube-like pixels regardless
    of the query.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MOVIE = ("ld=/Users/joshjiang/Downloads/"
         "test1lowdensjoshua-28c-hz.mp4 .mp4")


def _run(ckpt: str, snapshot: str, frame: str, prefix: str, base: int,
         out: Path, query_at: str = "") -> dict:
    cmd = [sys.executable, str(REPO / "scripts" / "eval_whole_instance.py"),
           "--checkpoint", ckpt, "--snapshot", snapshot, "--frame", frame,
           "--movie", MOVIE, "--multiscale", "--base", str(base),
           "--owner-prefix", prefix, "--out", str(out)]
    if query_at:
        cmd += ["--query-at", query_at]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if r.returncode != 0:
        print(f"  (eval failed: {r.stderr.strip().splitlines()[-1:]})")
        return {}
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--snapshot", default="runs/prototypes/v30/snapshots/snap18")
    ap.add_argument("--base", type=int, default=4)
    ap.add_argument("--clump-frame", default="ld:42000")
    ap.add_argument("--held-frame", default="ld:49350")
    ap.add_argument("--held-owner", default="ld|obs-r4-p05")
    ap.add_argument("--clump-owner", default="ld|rev8p")
    ap.add_argument("--tag", default="report")
    a = ap.parse_args()

    tmp = Path("runs/prototypes/v30")
    tmp.mkdir(parents=True, exist_ok=True)
    held = _run(a.checkpoint, a.snapshot, a.held_frame, a.held_owner,
                a.base, tmp / f"_held_{a.tag}.json")
    swap = _run(a.checkpoint, a.snapshot, a.clump_frame, a.clump_owner,
                a.base, tmp / f"_swap_{a.tag}.json")
    ctl = _run(a.checkpoint, a.snapshot, a.clump_frame, a.clump_owner,
               a.base, tmp / f"_ctl_{a.tag}.json", query_at="auto")

    out = {"checkpoint": a.checkpoint, "base": a.base,
           "held_out_tube": held.get("iou", {}),
           "swap": {k: swap.get(k) for k in
                    ("diag_mean_iou", "offdiag_mean_iou",
                     "rows_peaking_on_own_tube", "max_row_spread_iou",
                     "degenerate")},
           "control_row": (ctl.get("iou", {}) or {}).get("<control>", {}) if ctl
           else {}}
    dest = tmp / f"whole_instance_{a.tag}.json"
    dest.write_text(json.dumps(out, indent=1, default=str))

    print(f"\n=== whole-instance report: {a.checkpoint} (base {a.base}) ===")
    hi = (held.get("iou", {}) or {}).get(a.held_owner.split("|")[-1], {})
    prof = held.get("prob_profile", {})
    print(f"held-out tube {a.held_owner}: IoU={hi.get(a.held_owner.split('|')[-1])}")
    for k, v in list(prof.items())[:6]:
        print(f"    {k:>9}: mean_p={v['mean_p']:.3f} paint={v['paint_frac']:.2f}")
    sw = out["swap"]
    # a collapsed model must not read as a pass: all-zero rows used to
    # score 3/3 because 0 >= 0 holds for every column
    _peak = ("inconclusive(model predicts nothing)"
             if sw.get("degenerate")
             else f"{sw.get('rows_peaking_on_own_tube')}/3")
    print(f"clump swap: diag={sw.get('diag_mean_iou')} "
          f"off-diag={sw.get('offdiag_mean_iou')} "
          f"self-peaked={_peak} "
          f"spread={sw.get('max_row_spread_iou')}")
    if ctl:
        print(f"control row: {ctl.get('iou', {}).get('<control>')}")
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
