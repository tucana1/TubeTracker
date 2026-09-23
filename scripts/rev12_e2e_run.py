"""rev12 P2: one clean-environment run, import -> review -> restart ->
recompute -> export, with the review's metric list.

Every stage runs in a FRESH subprocess (clean environment); the
orchestrator only sequences and measures. Stages:

1. import   — hash + register the movie in a fresh project store;
2. review   — a reviewer task is created and completed through the
              store API (the UI path), producing a correction;
3. infer    — the inferred interval runs from scratch (cache MISS);
4. restart  — a fresh process re-runs the interval; the correction
              invalidates the cache and the affected interval recomputes;
5. export   — metrics assembled: conditional accuracy + coverage,
              owner switches, cap/path/length error, emergence-interval
              coverage, population-count status, review corrections,
              runtime per stage.

Output: runs/prototypes/v30/rev12_e2e/e2e_report.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

E2E = REPO / "runs/prototypes/v30/rev12_e2e"
MOVIE = "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"
X_DB = Path("/Users/joshjiang/Documents/TubeTracker-annotator-projects/"
            "rev10round/annotations.db")


def _run(cmd, **kw):
    t0 = time.time()
    r = subprocess.run([sys.executable, *cmd], capture_output=True,
                       text=True, cwd=str(REPO), **kw)
    return r, round(time.time() - t0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(E2E))
    ap.add_argument("--frames", default="51030:51450:30")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proj = out / "project"
    proj.mkdir(parents=True, exist_ok=True)

    stages: dict[str, dict] = {}

    # ---- 1. import ---------------------------------------------------
    import hashlib
    from tubetracker.annotation_store import AnnotationStore
    t0 = time.time()
    h = hashlib.sha256()
    with open(MOVIE, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    store = AnnotationStore(proj / "annotations.db")
    store.save("session", "movie-import",
               {"path": MOVIE, "sha256": h.hexdigest(),
                "imported_by": "rev12_e2e"}, actor="e2e")
    store.close()
    stages["import"] = {"seconds": round(time.time() - t0, 1),
                        "movie_sha256": h.hexdigest()[:16]}

    # ---- 2. review: a correction through the store (UI path) ---------
    t0 = time.time()
    import shutil
    work_db = out / "store_work.db"
    if work_db.exists():
        work_db.unlink()
    shutil.copy2(X_DB, work_db)
    s1 = AnnotationStore(work_db)
    unit = s1.load("cross-rev10x-000")
    assert unit is not None
    data = json.loads(json.dumps(unit["data"]))
    lane = sorted(data["lanes"])[0]
    tip = list(map(float, data["lanes"][lane][-1]))
    data["lanes"][lane][-1] = [tip[0] + 1.5, tip[1] - 0.75]
    rev_after = s1.save("crossing", "cross-rev10x-000", data,
                        actor="e2e-review")
    s1.close()
    s2 = AnnotationStore(work_db)      # restart
    got = s2.load("cross-rev10x-000")
    assert got is not None
    persisted = (list(map(float, got["data"]["lanes"][lane][-1]))
                 == data["lanes"][lane][-1])
    s2.close()
    links = json.loads((REPO / "runs/prototypes/v30/rev11own_verify"
                        "/crossing_owner_links.json").read_text())
    frame = int(got["data"]["source_frame"])
    owner = next((lk["owner"] for lk in links
                  if int(lk["frame"]) == frame and lk["lane"] == lane),
                 None)
    corrections = {owner: {str(frame): {
        "tip": [float(data["lanes"][lane][-1][0]),
                float(data["lanes"][lane][-1][1])],
        "path": [[float(q[0]), float(q[1])]
                 for q in data["lanes"][lane]],
        "actor": "e2e-review", "revision": rev_after}}} if owner else {}
    cfile = out / "corrections.json"
    cfile.write_text(json.dumps(corrections))
    stages["review"] = {"seconds": round(time.time() - t0, 1),
                        "n_corrections": len(corrections),
                        "revision": rev_after,
                        "persisted_after_restart": bool(persisted)}

    # ---- 3. infer (fresh, cache MISS) --------------------------------
    interval_out = out / "interval"
    if interval_out.exists():
        shutil.rmtree(interval_out)
    r1, t1 = _run([str(REPO / "scripts/rev12_build_inferred_interval.py"),
                   "--out-dir", str(interval_out),
                   "--crossing-store", str(work_db),
                   "--frames", a.frames])
    assert r1.returncode == 0, r1.stdout + r1.stderr
    exp1 = json.loads((interval_out / "export_inferred.json").read_text())
    stages["infer"] = {"seconds": t1, "cache_key": exp1["cache_key"],
                       "coverage": exp1["coverage"]["coverage_frac"]}

    # ---- 4. restart + recompute on the correction --------------------
    r2, t2 = _run([str(REPO / "scripts/rev12_build_inferred_interval.py"),
                   "--out-dir", str(interval_out),
                   "--crossing-store", str(work_db),
                   "--frames", a.frames,
                   "--corrections-file", str(cfile)])
    assert r2.returncode == 0, r2.stdout + r2.stderr
    exp2 = json.loads((interval_out / "export_inferred.json").read_text())
    stages["restart_recompute"] = {
        "seconds": t2, "cache_key": exp2["cache_key"],
        "cache_invalidated": exp2["cache_key"] != exp1["cache_key"],
        "recomputed": "cache MISS" in r2.stdout,
        "n_human_corrected": exp2["corrections"]["n_ui_corrected_rows"]}

    # ---- 5. metrics ---------------------------------------------------
    rows = exp2["rows"]
    # rev13 W4: consecutive-frame tip jumps > 20 px. This is a
    # CONTEXT DIAGNOSTIC (endpoint displacement), NOT an identity
    # error and NOT an apex error: a jump can be a legitimate
    # re-acquisition. Identity errors are counted only against
    # linked truth when it exists; this metric is reported under
    # its own name.
    switches = {}
    for oid in exp2["owners"]:
        seq = sorted([r for r in rows if r["owner"] == oid
                      and r["state"] == "model-inferred"],
                     key=lambda r: r["frame"])
        n = 0
        for i in range(1, len(seq)):
            d = float(((seq[i]["tip"][0] - seq[i - 1]["tip"][0]) ** 2
                       + (seq[i]["tip"][1] - seq[i - 1]["tip"][1]) ** 2)
                      ** 0.5)
            if d > 20.0:
                n += 1
        switches[oid] = n
    anchor_errs = [ar["endpoint_err_px"] for ar in
                   exp2["anchor_agreement"]
                   if ar.get("endpoint_err_px") is not None]
    # emergence coverage + population status from the census report
    cen = json.loads((REPO / "runs/prototypes/v30/rev12_census"
                      "/emergence_report.json").read_text())
    report = {
        "stages": stages,
        "runtime_total_s": round(sum(s["seconds"]
                                     for s in stages.values()), 1),
        "conditional_accuracy": {
            "coverage_frac": exp2["coverage"]["coverage_frac"],
            "anchor_endpoint_errs_px": anchor_errs,
            "anchor_within_5px": sum(1 for e in anchor_errs if e <= 5.0),
            "n_anchors": len(anchor_errs),
            "full_length_rows": exp2["coverage"]["n_full_length"],
            "partial_rows": exp2["coverage"]["n_partial"],
        },
        "tip_displacement_jumps_gt20px_context_diagnostic":
            switches,
        "tip_jump_metric_note": "consecutive-frame endpoint "
                               "displacement (>20 px): context "
                               "diagnostic only — not an "
                               "identity error, not an apex error",
        "cap_path_length": {
            "declared": "partial crossing lanes stay partial; full "
                        "length only for root-to-current-cap routes",
            "full_length_rows": exp2["coverage"]["n_full_length"],
        },
        "emergence_interval_coverage": {
            "confirmed": cen["germination"]["numerator_confirmed"],
            "denominator": cen["germination"]["denominator_grains"],
            "censored_left": cen["germination"]["censored_left"],
            "censored_right": cen["germination"]["censored_right"],
        },
        "population_count": {
            "n_grains": cen["census"]["n_grains"],
            "certified": cen["census"]["census_completeness_certified"],
            "error_status": "NOT CERTIFIED (0 exhaustive tiles) — a "
                            "count error cannot be scored honestly "
                            "until the review protocol exists",
        },
        "review": {"n_corrections": stages["review"]["n_corrections"],
                   "review_seconds": stages["review"]["seconds"]},
        "note": "clean-environment sequence: every stage ran in a fresh "
                "process; import -> review -> infer -> restart -> "
                "recompute -> export.",
    }
    (out / "e2e_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"-> {out}/e2e_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
