"""rev12 P1.1: the UI correction → restart → recompute → export receipt.

Executes the full chain on a WORKING COPY of the real annotation store
(the same store the annotation UI writes):

1. run the inferred interval (v1) and record its cache key + export;
2. apply a correction THROUGH the store API (lane endpoint edit, actor
   'ui-correction') — this is the UI write path;
3. bridge the store edit into corrections.json (owner/frame tip);
4. RE-RUN the interval (the "restart"): the correction changes the
   cache key, the cache is invalidated, inference recomputes, the
   export is regenerated and carries the human-corrected row with its
   revision.

Writes runs/prototypes/v30/rev12_interval/correction_receipt.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

X_DB = Path("/Users/joshjiang/Documents/TubeTracker-annotator-projects/"
            "rev10round/annotations.db")
OUT = REPO / "runs/prototypes/v30/rev12_interval"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from tubetracker.annotation_store import AnnotationStore

    # ---- working copy of the real store (never the live one) --------
    work_db = out / "store_work.db"
    if work_db.exists():
        work_db.unlink()
    shutil.copy2(X_DB, work_db)

    # ---- v1: run the interval (cache miss or hit, recorded) ---------
    def _run(extra=()):
        cmd = [sys.executable, str(REPO /
               "scripts/rev12_build_inferred_interval.py"),
               "--out-dir", str(out),
               "--crossing-store", str(work_db), *extra]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    rc1, log1 = _run()
    assert rc1 == 0, log1
    exp1 = json.loads((out / "export_inferred.json").read_text())
    key1 = exp1["cache_key"]

    # ---- the correction, through the store API (UI path) ------------
    con_uuid = None
    s1 = AnnotationStore(work_db)
    import sqlite3
    con = sqlite3.connect(work_db)
    con_uuid = con.execute(
        "select uuid from entities where kind='crossing' "
        "order by uuid limit 1").fetchone()[0]
    con.close()
    unit = s1.load(con_uuid)
    assert unit is not None, f"crossing {con_uuid} missing"
    rev_before = unit["revision"]
    data = json.loads(json.dumps(unit["data"]))
    lane = sorted(data["lanes"])[0]
    tip0 = list(map(float, data["lanes"][lane][-1]))
    data["lanes"][lane][-1] = [tip0[0] + 1.5, tip0[1] - 0.75]
    rev_after = s1.save("crossing", con_uuid, data,
                        actor="ui-correction")
    s1.close()
    # restart: a FRESH store handle must see the edit (persistence)
    s2 = AnnotationStore(work_db)
    got = s2.load(con_uuid)
    assert got is not None
    persisted = (list(map(float, got["data"]["lanes"][lane][-1]))
                 == data["lanes"][lane][-1])
    hist = s2.history(con_uuid)
    s2.close()

    # ---- bridge the store edit to the corrections file --------------
    links = json.loads((REPO / "runs/prototypes/v30/rev11own_verify"
                        "/crossing_owner_links.json").read_text())
    frame = int(got["data"].get("source_frame", -1))
    owner = None
    for lk in links:
        if int(lk["frame"]) == frame and lk["lane"] == lane:
            owner = lk.get("owner")
    corrections = {}
    if owner:
        corrections[owner] = {str(frame): {
            "tip": [float(data["lanes"][lane][-1][0]),
                    float(data["lanes"][lane][-1][1])],
            "path": [[float(q[0]), float(q[1])]
                     for q in data["lanes"][lane]],
            "actor": "ui-correction", "revision": rev_after}}
    cfile = out / "corrections.json"
    cfile.write_text(json.dumps(corrections, indent=1))

    # ---- v2: restart + recompute ------------------------------------
    rc2, log2 = _run(("--corrections-file", str(cfile)))
    assert rc2 == 0, log2
    exp2 = json.loads((out / "export_inferred.json").read_text())
    key2 = exp2["cache_key"]
    corr_rows = [r for r in exp2["rows"]
                 if r["state"] == "human-corrected"]
    receipt = {
        "store": str(work_db), "unit": con_uuid,
        "rev_before": rev_before, "rev_after": rev_after,
        "correction_persisted_after_restart": bool(persisted),
        "n_revisions": len(hist),
        "actors": [h.get("actor") for h in hist],
        "cache_key_v1": key1, "cache_key_v2": key2,
        "cache_invalidated": key1 != key2,
        "recomputed": "cache MISS" in log2,
        "n_human_corrected_rows": len(corr_rows),
        "correction_in_export": bool(corr_rows and owner and
                                     corr_rows[0]["owner"] == owner and
                                     corr_rows[0]["frame"] == frame),
        "correction_revision_recorded": bool(
            corr_rows and corr_rows[0].get("correction", {})
            .get("revision") == rev_after),
        "export_regenerated": (exp1.get("cache_key") !=
                               exp2.get("cache_key")),
    }
    (out / "correction_receipt.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))
    ok = all([receipt["correction_persisted_after_restart"],
              receipt["cache_invalidated"], receipt["recomputed"],
              receipt["correction_in_export"],
              receipt["correction_revision_recorded"]])
    print("receipt:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
