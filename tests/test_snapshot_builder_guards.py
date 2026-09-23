"""The snapshot builder must refuse a project passed twice (rev8).

Measured mistake this pins: snap20 was first built with rev8masks2 in
the predecessor's project list AND passed again as an extra, so its two
masks appeared twice in body_masks.json — every row from that project
double-counted. A snapshot is provenance; it must fail loudly instead.
"""
from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts/build_v30_snapshot.py"


def test_builder_refuses_a_duplicate_project_dir(tmp_path):
    proj = tmp_path / "p"
    proj.mkdir()
    out = tmp_path / "snap"
    r = subprocess.run(
        [sys.executable, str(BUILDER),
         "--project-dir", str(proj),
         "--project-dir", str(proj),
         "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 2, (r.returncode, r.stdout[-400:], r.stderr[-400:])
    assert "passed more than once" in r.stdout
    # and it wrote nothing: refusal happens before any work
    assert not (out / "body_masks.json").exists()


def test_builder_accepts_the_same_dir_once(tmp_path):
    sys.path.insert(0, str(REPO))
    from tubetracker.annotation_store import AnnotationStore
    proj = tmp_path / "p"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    try:
        store.save("task", "t1", {"uuid": "t1", "movie": "ld",
                                  "query_frames": [48300],
                                  "task_type": "body_mask",
                                  "owner_uuid": "o1", "completed": False},
                   actor="t")
    finally:
        store.close()
    out = tmp_path / "snap2"
    r = subprocess.run(
        [sys.executable, str(BUILDER),
         "--project-dir", str(proj),
         "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout[-400:], r.stderr[-400:])
    assert (out / "body_masks.json").exists()


def test_builder_fails_closed_on_a_missing_db(tmp_path):
    """No db means no snapshot: silence would read as success."""
    proj = tmp_path / "empty"
    proj.mkdir()
    out = tmp_path / "snap3"
    r = subprocess.run(
        [sys.executable, str(BUILDER),
         "--project-dir", str(proj),
         "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode != 0, (r.returncode, r.stdout[-300:])
    assert "missing db" in r.stdout
    assert not out.exists()


def test_parent_validation_role_survives_region_and_mask_export(tmp_path):
    import numpy as np
    from tubetracker.annotation_store import AnnotationStore
    from prototypes.v30_video_apex.targets import encode_mask_raster
    from prototypes.v30_video_apex.cap_evidence import CapLabelIndex
    project=tmp_path/'project';project.mkdir()
    store=AnnotationStore(project/'annotations.db')
    store.save('task','validation-task',{'movie':'ld','query_frames':[10],
        'task_type':'body_mask','annotation_role':'validation','review_origin':'human',
        'owner_uuid':'grain','target_xy':[8,8],'target_r':4})
    store.save('mask','mask-validation',{'task_uuid':'validation-task','source_frame':10,
        'movie_uuid':'ld','owner_uuid':'grain','painted_xy':[[8,8]],'brush_px':3,
        'mask_raster':encode_mask_raster(np.ones((16,16),bool)),
        'review_region':[[0,0],[16,16]],'complete':True,'review_origin':'human'})
    store.save('region','region-validation',{'task_uuid':'validation-task','movie_uuid':'ld',
        'source_frame':10,'kind':'verified_negative','confirmed':True,'class_scope':'cap',
        'polygon_xy':[[30,30],[40,30],[40,40],[30,40]],'review_origin':'human'})
    store.close()
    out=tmp_path/'snapshot'
    result=subprocess.run([sys.executable,str(BUILDER),'--project-dir',str(project),
                           '--out',str(out)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    masks=json.loads((out/'body_masks.json').read_text())
    regions=json.loads((out/'regions.json').read_text())
    assert masks[0]['annotation_role']==regions[0]['annotation_role']=='validation'
    truth=CapLabelIndex.from_snapshot(out)
    assert truth.negatives and not truth.for_training().negatives


def test_canvas_provenance_survives_database_snapshot_and_training_loader(tmp_path):
    import numpy as np
    from tubetracker.annotation_store import AnnotationStore
    from prototypes.v30_video_apex.targets import encode_mask_raster, samples_from_snapshot
    from prototypes.v30_video_apex.batch_builder import own_mask_target, finalize_body_supervision
    project=tmp_path/'project';project.mkdir()
    region=[[0.,4.],[24.,12.]]
    provenance={'schema':'tubetracker.review_geometry.v1','kind':'native_canvas',
        'canvas_size_wh':[120,40],'camera_center_xy':[12,8],'camera_zoom':5,
        'movie':'ld','source_frame':10}
    paint=np.zeros((16,24),bool);paint[7:10,8:13]=True
    store=AnnotationStore(project/'annotations.db')
    store.save('task','body',{'uuid':'body','movie':'ld','owner_uuid':'grain',
        'task_type':'body_mask','query_frames':[10],'target_xy':[8,8],'target_r':4},actor='test-fixture')
    store.save('mask','mask-body',{'task_uuid':'body','movie_uuid':'ld','owner_uuid':'grain',
        'source_frame':10,'complete':True,'painted_xy':[[8,8],[12,8]],'brush_px':3,
        'mask_raster':encode_mask_raster(paint),'target_xy':[8,8],'target_r':4,
        'review_region':region,'review_region_provenance':provenance},actor='test-fixture')
    store.close()
    movie=tmp_path/'identity.mp4';movie.write_bytes(b'identity fixture; no frame decoding')
    out=tmp_path/'snapshot'
    result=subprocess.run([sys.executable,str(BUILDER),'--project-dir',str(project),
        '--out',str(out),'--movie',f'ld={movie}'],capture_output=True,text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    records=json.loads((out/'body_masks.json').read_text())
    assert len(records)==1 and records[0]['review_region_provenance']==provenance
    sample=next(s for s in samples_from_snapshot(out) if s.kind=='body_mask')
    assert sample.review_region_provenance==provenance
    target=finalize_body_supervision(own_mask_target(sample,16,24,(0,0)))
    assert target.sel_bg[8,2]  # Genuine width must survive, rather than falling back to a square.
    assert not target.sel_bg[2,2]


def test_snapshot_preserves_independent_tip_and_quarantines_workflow_dependants(tmp_path):
    from tubetracker.annotation_store import AnnotationStore
    from prototypes.v30_video_apex.cap_evidence import CapLabelIndex
    from prototypes.v30_video_apex.targets import samples_from_snapshot
    proj = tmp_path / "reviews"
    proj.mkdir()
    store = AnnotationStore(proj / "annotations.db")
    for uid, origin in [("human", "human"), ("receipt", "workflow_test")]:
        store.save("task", uid, {"uuid": uid, "movie": "ld", "owner_uuid": uid,
            "query_frames": [10], "task_type": "centerline", "review_origin": origin,
            "target_xy": [10, 10], "target_r": 4, "distinct_cap_evidence": "two-lanes",
            "grains": {"A": {"xy": [10, 10], "root_xy": [14, 10]}}},
            actor="test-fixture")
        store.save("observation", "obs-" + uid, {"task_uuid": uid, "source_frame": 10,
            "owner_uuid": uid, "direct_state": "direct_visible", "direct_xy": [25, 10],
            "tip_source": "explicit_point", "path_xy": [[10, 10], [15, 10]],
            "path_complete": False}, actor="test-fixture")
    store.save("region", "receipt-region", {"task_uuid": "receipt",
        "kind": "verified_negative", "confirmed": True, "class_scope": "cap",
        "polygon_xy": [[50, 50], [60, 50], [60, 60], [50, 60]]}, actor="test-fixture")
    store.save("ruling", "human-ruling", {"grain_owner": "human", "class": "germinated",
        "rule": "fixture rule", "reviewed_frames": [10]}, actor="test-fixture")
    store.close()
    out = tmp_path / "snapshot"
    movie = tmp_path / "identity-only.mp4"
    movie.write_bytes(b"identity fixture; no decoding in this test")
    run = subprocess.run([sys.executable, str(BUILDER), "--project-dir", str(proj),
        "--out", str(out), "--movie", f"ld={movie}"], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    observations = json.loads((out / "observations.json").read_text())
    assert len(observations) == 1
    assert observations[0]["tip_source"] == "explicit_point"
    assert observations[0]["distinct_cap_evidence"] == "two-lanes"
    assert observations[0]["review_origin"] == "human"
    audit = json.loads((out / "workflow_review_records.json").read_text())
    assert {r["uuid"] for r in audit} == {"receipt", "obs-receipt", "receipt-region"}
    manifest = json.loads((out / "snapshot_manifest.json").read_text())
    assert manifest["n_observations"] == 1 and manifest["n_workflow_records_excluded"] == 3
    assert len(CapLabelIndex.from_snapshot(out).points[("ld", 10)]) == 1
    samples = samples_from_snapshot(str(out))
    assert len(samples) == 1 and samples[0].kind == "tip_only"
    assert json.loads((out / "tubes.json").read_text())[0]["source_frame"] == 10
    ruling = json.loads((out / "rulings.json").read_text())[0]
    assert ruling["uuid"] == "human-ruling" and ruling["revision"] == 1
