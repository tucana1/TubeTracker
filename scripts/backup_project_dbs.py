"""Project-DB backup with restore verification (rev8 step 1).

The rev7 loss happened because the only copies of an annotation project
lived in /tmp. This tool makes a backup that is *proved* restorable
before new labeling starts:

  1. sqlite backup API copy (consistent under an active writer),
  2. integrity_check on the copy,
  3. content hash of the entity table (uuid, kind, data, revision),
  4. restore check: reopen the copy, recount every kind, rehash,
  5. drift report against any previous manifest in the backup dir.

Usage:
    .venv/bin/python scripts/backup_project_dbs.py \
        --backup-dir runs/prototypes/annotation_backups \
        --project-dir ~/Documents/TubeTracker-annotator \
        [--project-dir ...] [--json out.json]

Exit 1 if any project fails integrity, restore or hash agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path


def entity_digest(db: Path) -> str:
    """Content hash over the entity table (order-independent per row)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT uuid, kind, data, revision FROM entities").fetchall()
    finally:
        con.close()
    h = hashlib.sha256()
    for r in sorted(rows):
        h.update(repr(tuple(r)).encode())
    return h.hexdigest()


def kind_counts(db: Path) -> dict[str, int]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        out = {k: n for k, n in con.execute(
            "SELECT kind, COUNT(*) FROM entities GROUP BY kind")}
        rev = con.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]
    finally:
        con.close()
    out["_revisions"] = int(rev)
    return out


def backup_one(project: Path, backup_dir: Path) -> dict:
    src = project / "annotations.db"
    # parent-qualified name: two projects called "v30w2" from different
    # parents must never overwrite each other's backup
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_"
                   for ch in project.parent.name)[-40:]
    name = f"{safe}__{project.name}"
    dest = backup_dir / f"{name}.db"
    rec: dict = {"project": str(project), "source": str(src),
                 "backup": str(dest)}
    if not src.exists():
        rec["status"] = "MISSING-SOURCE"
        return rec
    backup_dir.mkdir(parents=True, exist_ok=True)
    # 1. consistent copy through the backup API
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        d = sqlite3.connect(str(dest))
        try:
            s.backup(d)
        finally:
            d.close()
    finally:
        s.close()
    # 2. integrity on the copy
    con = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
    try:
        integ = con.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        con.close()
    rec["integrity_check"] = integ
    live_counts = kind_counts(src)
    rec["live_counts"] = live_counts
    # 3. content hash + 4. restore check from a fresh copy of the backup
    live_digest = entity_digest(src)
    rec["live_digest"] = live_digest
    with tempfile.TemporaryDirectory() as td:
        restored = Path(td) / "restored.db"
        shutil.copy2(dest, restored)
        rec["restored_counts"] = kind_counts(restored)
        rec["restored_digest"] = entity_digest(restored)
    rec["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
    ok = (integ == "ok"
          and rec["restored_counts"] == live_counts
          and rec["restored_digest"] == live_digest)
    rec["status"] = "OK" if ok else "FAIL"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", action="append", required=True)
    ap.add_argument("--backup-dir", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    backup_dir = Path(a.backup_dir).expanduser()
    manifest_path = backup_dir / "backup_manifest.json"
    prev: dict = {}
    if manifest_path.exists():
        try:
            prev = {r["project"]: r for r in json.loads(
                manifest_path.read_text()).get("records", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            prev = {}

    records = []
    failed = 0
    for spec in a.project_dir:
        rec = backup_one(Path(spec).expanduser(), backup_dir)
        # 5. drift against the previous manifest: a changed digest means
        # the live project moved since the last verified backup (normal
        # after labeling) — recorded, never silently overwritten.
        old = prev.get(rec["project"])
        if old and old.get("live_digest") and rec.get("live_digest"):
            rec["drift_vs_previous"] = (
                "changed" if old["live_digest"] != rec["live_digest"]
                else "identical")
        records.append(rec)
        if rec["status"] != "OK":
            failed += 1
        print(f"  {rec['status']:<15} {Path(rec['project']).name:<16} "
              f"counts={rec.get('live_counts', {})}")

    manifest = {"records": records,
                "n_ok": sum(1 for r in records if r["status"] == "OK"),
                "n_failed": failed}
    backup_dir.mkdir(parents=True, exist_ok=True)
    out = Path(a.json) if a.json else manifest_path
    out.write_text(json.dumps(manifest, indent=1, default=str))
    print(f"\nwrote {out}: {manifest['n_ok']} ok, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
