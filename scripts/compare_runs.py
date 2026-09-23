"""Equal-updates comparison of two (or more) WP-B arms.

The review asks for objectives compared "at equal total updates, same
decisive set". This reads each run's `history.json`, aligns by epoch,
and prints the trajectory side by side, plus the config differences
that must be absent for the comparison to mean anything.

usage: python scripts/compare_runs.py --run A:runs/... --run B:runs/... \
       [--out compare.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(run: Path):
    hf = run / "history.json"
    if not hf.exists():
        raise SystemExit(
            f"{run}: no history.json yet — the arm has not finished its "
            "first epoch (the trainer writes it every epoch), or the run "
            "predates that change")
    h = json.loads(hf.read_text())
    man = {}
    mf = run / "run_manifest.json"
    if mf.exists():
        man = json.loads(mf.read_text())
    return h, man


def _dev_iou(rec: dict) -> float | None:
    vals = [d.get("distal_body_iou") for d in (rec.get("dev") or [])
            if d.get("distal_body_iou") is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="LABEL:path/to/run-dir")
    ap.add_argument("--at", default="0,10,20,30,40,50,60")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    runs = []
    for spec in a.run:
        label, _, path = spec.partition(":")
        h, man = _load(Path(path))
        runs.append((label, path, h, man))
    n = min(len(h) for _, _, h, _ in runs)
    print(f"equal-updates window: {n} epochs "
          f"(runs: {', '.join(f'{l}={len(h)}' for l, _, h, _ in runs)})")

    # the decisive-set check: same number of updates per epoch?
    for label, _p, h, _m in runs:
        ups = [r.get("updates_this_epoch") for r in h[:n]]
        print(f"{label}: updates/epoch min {min(ups)} max {max(ups)}")

    want = [int(v) for v in a.at.split(",") if int(v) < n]
    cols = ["epoch"] + [f"{l} {k}" for l, _, _, _ in runs
                        for k in ("body", "dice", "devIoU")]
    print((" | ".join(f"{c:>12s}" for c in cols)))
    rows = []
    for e in want:
        row: dict = {"epoch": e}
        line = [f"{e:>12d}"]
        for label, _p, h, _m in runs:
            rec = h[e]
            parts = rec.get("parts") or {}
            body = rec.get("mean_body_loss") or rec.get("train_loss")
            dice = parts.get("mean_body_dice")
            di = _dev_iou(rec)
            row[label] = {"body": body, "dice": dice, "dev_iou": di,
                          "updates": rec.get("updates_this_epoch")}
            line += [f"{(body if body is not None else float('nan')):>12.4f}",
                     f"{(dice if dice is not None else float('nan')):>12.4f}",
                     f"{(di if di is not None else float('nan')):>12.4f}"]
        rows.append(row)
        print(" | ".join(line))

    # what differs in the manifests (must be nothing but the objective)
    keys = set()
    for _l, _p, _h, m in runs:
        keys |= set(m)
    print("\nmanifest differences:")
    diffs = {}
    for k in sorted(keys):
        vals = [str(m.get(k))[:60] for _l, _p, _h, m in runs]
        if len(set(vals)) > 1:
            diffs[k] = dict(zip([l for l, _, _, _ in runs], vals))
    for k, v in diffs.items():
        print(f"  {k}: {v}")
    if not diffs:
        print("  none — only the objective/run name differ")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"runs": [{"label": l, "path": p,
                       "epochs": len(h)} for l, p, h, _ in runs],
             "window": n, "rows": rows, "manifest_differences": diffs},
            indent=1, default=str))
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
