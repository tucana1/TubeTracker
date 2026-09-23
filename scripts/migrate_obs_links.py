"""Migrate random-ID observations to task-derived links (P0 repair).

For each observation whose uuid is not task-derived, find completed
tasks with matching (source_frame, owner_uuid) and no task-derived
observation yet. Unambiguous matches are copied to obs-<task_uuid>
with migrated_from + task_uuid fields (original rows preserved).
Ambiguous/unmatched rows are reported for re-review, never guessed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tubetracker.annotation_store import AnnotationStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--actor", default="migrate")
    a = ap.parse_args()
    store = AnnotationStore(Path(a.project_dir) / "annotations.db")
    try:
        tasks = store.unfinished_tasks(limit=100000)
        # unfinished_tasks only returns pending; read all via entities
        import sqlite3
        con = sqlite3.connect(str(Path(a.project_dir) / "annotations.db"))
        try:
            trows = con.execute(
                "SELECT uuid, data FROM entities WHERE kind='task'").fetchall()
            orows = con.execute(
                "SELECT uuid, data FROM entities WHERE kind='observation'"
            ).fetchall()
        finally:
            con.close()
        tasks = [(u, json.loads(d)) for u, d in trows]
        obs = [(u, json.loads(d)) for u, d in orows]
        linked = {o.get("task_uuid") for _, o in obs if o.get("task_uuid")}
        needy = [(u, t) for u, t in tasks
                 if t.get("completed") and u not in linked
                 and f"obs-{u}" not in {o_uid for o_uid, _ in obs}
                 and f"obs-{u}-0" not in {o_uid for o_uid, _ in obs}]
        orphans = [(u, o) for u, o in obs if not u.startswith("obs-")
                   and not o.get("task_uuid")]
        migrated, ambiguous, unmatched = [], [], []
        for ouid, o in orphans:
            cands = [(tu, t) for tu, t in needy
                     if int(t.get("query_frames", [None])[0])
                     == int(o.get("source_frame", -1))
                     and str(t.get("owner_uuid", ""))
                     == str(o.get("owner_uuid", ""))]
            if len(cands) == 1:
                tu, t = cands[0]
                new_uid = f"obs-{tu}"
                store.save("observation", new_uid, {
                    **o, "task_uuid": tu, "migrated_from": ouid,
                }, actor=a.actor)
                migrated.append({"from": ouid, "to": new_uid})
                needy.remove((tu, t))
            elif not cands:
                unmatched.append(ouid)
            else:
                ambiguous.append({"obs": ouid,
                                  "tasks": [tu for tu, _ in cands]})
        report = {"migrated": migrated, "ambiguous": ambiguous,
                  "unmatched": unmatched}
        out = Path(a.project_dir) / "migration_obs_links.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"migrated={len(migrated)} ambiguous={len(ambiguous)} "
              f"unmatched={len(unmatched)} -> {out}", flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
