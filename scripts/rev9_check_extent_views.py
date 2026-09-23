"""rev9 WP-A.1 exit evidence: the exact learn3 training views.

The reviewer's contracts audit found four of the fourteen learn3 mask
views whose target used a bad historical extent (1300 / 644 / 1080 /
2475 unreviewed pixels supervised as background). This script runs the
CURRENT target code over those exact crops and counts how many still
license anything from an unusable extent. Expected after the fix: 0.

Usage:
    .venv/bin/python scripts/rev9_check_extent_views.py [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.targets import (  # noqa: E402
    body_mask_channels_from_raster, samples_from_snapshot)

SNAP = REPO / "runs/prototypes/v30/snapshots/snap20"
AUDIT = REPO / "runs/prototypes/v30/rev9-audit-repro/contracts-audit.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(AUDIT),
                    help="contracts-audit.json carrying the view list")
    args = ap.parse_args()
    audit = json.loads(Path(args.json).read_text())
    views = audit["invalid_extent_training_views"]
    samples = {s.obs_uuid: s for s in samples_from_snapshot(str(SNAP))
               if s.kind == "body_mask"}
    by_short = {}
    for s in samples.values():
        by_short.setdefault(s.obs_uuid.replace("mask-", ""), s)
    rows = []
    licensed = 0
    for v in views:
        key = v["mask"].replace("mask-", "")
        s = by_short.get(key)
        if s is None or not s.mask_raster:
            rows.append({**v, "checked": False})
            continue
        x, y, w, h = v["crop"]
        ch = body_mask_channels_from_raster(
            int(h), int(w), (float(x), float(y)), s.mask_raster,
            complete=bool(s.complete), review_region=s.review_region)
        n_lic = int(ch["bg_reviewed"].sum())
        if n_lic:
            licensed += 1
        rows.append({**v, "checked": True, "extent_used": ch["extent_used"],
                     "quarantine_reason": ch["quarantine_reason"],
                     "bg_reviewed_px": n_lic})
    out = {"n_views": len(views), "n_licensed_by_bad_extent": licensed,
           "verdict": ("PASS: no unreviewed pixel is supervised"
                       if licensed == 0 else "FAIL"),
           "rows": rows}
    print(json.dumps(out, indent=2)[:4000])
    return 0 if licensed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
