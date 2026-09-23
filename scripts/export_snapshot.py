"""Export a versioned training snapshot from an annotation project (P0A).

Runs fold + export audit, then writes an immutable snapshot manifest
with file hashes, split membership, schema version and annotation
provenance. The training code loads the snapshot dir; the live project
db is never read by training. Refuses to export when the audit fails.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--movie", required=True)
    ap.add_argument("--movie-tag", default="ld")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args()

    snap = Path(a.snapshot_dir)
    if snap.exists():
        print(f"refusing to overwrite {snap}")
        return 1
    work = snap.parent / (snap.name + ".build")
    if work.exists():
        shutil.rmtree(work)

    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "fold_human_labels.py"),
         "--project-dir", a.project_dir, "--movie", a.movie,
         "--dataset-dir", a.dataset_dir, "--out-dir", str(work),
         "--movie-tag", a.movie_tag],
        capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1
    audit_csv = work / "export_audit.csv"
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "audit_export.py"),
         "--project-dir", a.project_dir, "--dataset-dir", str(work),
         "--movie-tag", a.movie_tag, "--seed", str(a.split_seed),
         "--out", str(audit_csv)],
        capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr[-2000:])
        shutil.rmtree(work, ignore_errors=True)
        return 1

    from tubetracker.annotation_schema import SCHEMA_VERSION
    files = {}
    for p in sorted(work.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(work))] = sha256_file(p)
    labels = list(csv.DictReader(open(work / "labels.csv")))
    manifest = {
        "snapshot_version": 1,
        "schema_version": SCHEMA_VERSION,
        "project_dir": str(a.project_dir),
        "movie": str(a.movie),
        "movie_tag": a.movie_tag,
        "split_seed": a.split_seed,
        "n_label_rows": len(labels),
        "provenance_counts": {},
        "files": files,
    }
    for row in labels:
        k = (row["label_type"], row["provenance"])
        manifest["provenance_counts"][f"{k[0]}:{k[1]}"] = \
            manifest["provenance_counts"].get(f"{k[0]}:{k[1]}", 0) + 1
    (work / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    work.rename(snap)
    print(f"snapshot -> {snap} ({len(labels)} labels, "
          f"{len(files)} files hashed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
