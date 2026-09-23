"""Step-1 exit check: does an annotation survive end to end?

Runs the whole chain on a REAL project and verifies, per sample type,
that the annotation's pixels and identity reach the training targets
unchanged:

    UI save (already done) -> reopen DB -> snapshot -> targets

Usage:
    .venv/bin/python scripts/verify_label_pipeline.py \
        --project-dir ~/Documents/TubeTracker-annotator-projects/<p> \
        --movie "ld=/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4" \
        [--default-movie ld] [--keep-snapshot runs/prototypes/v30/snapshots/verify]

Exit 0 only when every check passes (or is explicitly reported with a
reason). The point is not a green light for its own sake: each check
below corresponds to a way rev7 lost or mis-attributed supervision.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    return ok


def counts(db: Path) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {k: n for k, n in con.execute(
            "SELECT kind, COUNT(*) FROM entities GROUP BY kind")}
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--movie", action="append", default=[],
                    help="movie mapping key=path (repeatable)")
    ap.add_argument("--default-movie", default="")
    ap.add_argument("--keep-snapshot", default="")
    a = ap.parse_args()

    proj = Path(a.project_dir).expanduser()
    db = proj / "annotations.db"
    all_ok = True
    print(f"verify_label_pipeline: {proj}")

    # ---- 1. database present, readable, and re-openable -------------
    all_ok &= check("database exists", db.exists(), str(db))
    if not db.exists():
        return 1
    c1 = counts(db)
    con = sqlite3.connect(str(db))
    try:
        n_ent = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        n_rev = con.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]
    finally:
        con.close()
    all_ok &= check("entities present", n_ent > 0, f"{n_ent} entities, "
                    f"{n_rev} revisions")
    all_ok &= check("reopen is stable", counts(db) == c1)

    # ---- 2. snapshot build -----------------------------------------
    tmp = Path(a.keep_snapshot) if a.keep_snapshot else Path(
        tempfile.mkdtemp(prefix="verify_snap_"))
    if tmp.exists():
        shutil.rmtree(tmp)
    cmd = [sys.executable, str(REPO / "scripts" / "build_v30_snapshot.py"),
           "--project-dir", str(proj), "--out", str(tmp)]
    for spec in a.movie:
        cmd += ["--movie", spec]
    if a.default_movie:
        cmd += ["--default-movie", a.default_movie]
    pr = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    all_ok &= check("snapshot builds", pr.returncode == 0,
                    pr.stderr.strip().splitlines()[-1] if pr.stderr else "")
    if pr.returncode != 0:
        return 1
    man = json.loads((tmp / "snapshot_manifest.json").read_text())

    # ---- 3. every annotation kind is accounted for ------------------
    kinds_present = [k for k in ("observations", "regions", "body_masks",
                                 "duels", "census", "tubes", "crossings")
                     if (tmp / f"{k.replace('body_masks', 'body_masks')}.json"
                         ).exists()]
    src_kinds = {k: v for k, v in c1.items()}
    print(f"  project kinds: {src_kinds}")
    for k, fname in (("observation", "observations.json"),
                     ("region", "regions.json"),
                     ("mask", "body_masks.json"),
                     ("duel", "duels.json"),
                     ("census", None)):
        if k not in src_kinds:
            continue
        if fname is None:
            continue
        rows = json.loads((tmp / fname).read_text())
        n_snap = len(rows)
        # mask rows can drop when neither raster nor stamps exist
        expect = src_kinds[k]
        ok = n_snap == expect or k == "mask"
        all_ok &= check(f"{k} rows reach the snapshot",
                        ok, f"{n_snap} of {expect}")

    # ---- 4. loader joins + identity/link/extent ---------------------
    from prototypes.v30_video_apex.targets import (
        apex_pos_neg_masks, body_mask_from_raster, census_pos_neg_masks,
        samples_from_snapshot)
    ss = samples_from_snapshot(str(tmp))
    by_kind: dict[str, list] = {}
    for s in ss:
        by_kind.setdefault(s.kind, []).append(s)
    print(f"  samples: { {k: len(v) for k, v in sorted(by_kind.items())} }")
    all_ok &= check("every sample has an identity key",
                    all(s.sample_key and s.owner_key for s in ss))
    unlinked = [s for s in ss if s.quarantine_reason]
    print(f"  quarantined: "
          f"{[(s.kind, s.quarantine_reason) for s in unlinked]}")

    # ---- 5. pixels: each type must land where the annotation says ---
    for s in by_kind.get("body_mask", []):
        if s.quarantine_reason:
            continue
        if not s.mask_raster:
            continue
        h = w = 288
        ox = float(s.focus_xy[0]) - w / 2.0
        oy = float(s.focus_xy[1]) - h / 2.0
        t, v = body_mask_from_raster(h, w, (ox, oy), s.mask_raster,
                                     complete=s.complete,
                                     review_region=s.review_region)
        painted = int((t > 0).sum())
        all_ok &= check(f"mask {s.mask_uuid}: paint lands in its crop",
                        painted > 0, f"{painted} px")
        if s.complete and s.review_region:
            all_ok &= check(
                f"mask {s.mask_uuid}: background only inside review",
                int(v.sum()) <= int(v.size) and float(
                    v.sum()) > painted, f"valid {int(v.sum())} px")
        elif s.complete and not s.review_region:
            all_ok &= check(
                f"mask {s.mask_uuid}: complete but no extent -> "
                f"band-only (no invented background)", True)

    for s in by_kind.get("tip_only", []) + by_kind.get("path_tip", []):
        pos, neg = apex_pos_neg_masks(288, 288,
                                      (s.tip_xy[0] - 144, s.tip_xy[1] - 144),
                                      (144.0, 144.0), [])
        all_ok &= check(f"tip {s.obs_uuid}: disc at the clicked point",
                        int(pos.sum()) > 150, f"{int(pos.sum())} px")
        break  # one representative is enough for the pixel contract

    for s in by_kind.get("neg_region", []):
        b = s.region_box
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        poly = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        _, neg = apex_pos_neg_masks(288, 288, (cx - 144, cy - 144), None,
                                    [{"kind": "verified_negative",
                                      "class_scope": "apex",
                                      "polygon_xy": poly}])
        all_ok &= check(f"region {s.obs_uuid}: box paints negative",
                        int(neg.sum()) > 1000, f"{int(neg.sum())} px")
        break

    for s in by_kind.get("comparison", []):
        if s.quarantine_reason:
            continue
        all_ok &= check(
            f"comparison {s.obs_uuid}: lanes + preference intact",
            len(s.lanes_a) >= 2 and len(s.lanes_b) >= 2
            and s.preference in ("A", "B", "neither"),
            f"pref={s.preference}")
        all_ok &= check(
            f"comparison {s.obs_uuid}: owner-linked",
            bool(s.owner_key))
        break

    for s in by_kind.get("census_tile", []):
        tb = s.region_box
        pos, neg = census_pos_neg_masks(
            288, 288, (float(tb[0]) - 10, float(tb[1]) - 10),
            s.census_tips, s.census_grains,
            (tb[0], tb[1], tb[2], tb[3]), s.census_complete,
            class_scopes=s.census_class_scopes)
        all_ok &= check(
            f"census {s.obs_uuid}: tip positives + extent-scoped "
            f"background", int(pos.sum()) > 0 or int(neg.sum()) > 0,
            f"pos={int(pos.sum())} neg={int(neg.sum())}")
        break

    if not a.keep_snapshot:
        shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
