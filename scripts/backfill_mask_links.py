"""rev7 one-off: backfill source_obs_uuid on legacy mask tasks.

Matches each body_mask task's guide_path exactly against snapshot-11
observation paths (same movie + frame, bit-exact geometry). Additive
field only; the store keeps every revision.
"""

import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SNAP = Path("/tmp/v30snap11/observations.json")
PROJECTS = ["/tmp/annotator_v30mask", "/tmp/annotator_v30next"]


def main() -> int:
    import numpy as np

    from tubetracker.annotation_store import AnnotationStore

    obs = json.loads(SNAP.read_text())
    paths = [(o["obs_uuid"], str(o.get("movie", "")),
              int(o.get("source_frame", -1)), o["path_xy"])
             for o in obs
             if o.get("path_xy") and len(o["path_xy"]) >= 2]
    for proj in PROJECTS:
        db = str(Path(proj) / "annotations.db")
        c = sqlite3.connect(db)
        try:
            rows = c.execute(
                "select uuid, data from entities where kind='task'"
            ).fetchall()
        finally:
            c.close()
        store = AnnotationStore(db)
        try:
            for uuid, data in rows:
                t = json.loads(data)
                if t.get("task_type") != "body_mask":
                    continue
                if t.get("source_obs_uuid"):
                    print(proj, uuid, "already linked")
                    continue
                g = np.asarray(t.get("guide_path", []), float)
                best = None
                for ouuid, movie, fr, p in paths:
                    if str(t.get("movie", "")) != movie:
                        continue
                    if int(t["query_frames"][0]) != fr:
                        continue
                    p = np.asarray(p, float)
                    if g.shape == p.shape and np.abs(g - p).max() < 1e-6:
                        best = ouuid
                        break
                if best:
                    t["source_obs_uuid"] = best
                    store.save("task", uuid, t, actor="rev7-backfill")
                    print(proj, uuid, "->", best)
                else:
                    print(proj, uuid, "NO MATCH")
        finally:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
