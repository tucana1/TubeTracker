import copy
import json
from contextlib import closing

import numpy as np
import pytest

from tubetracker.analysis_contracts import (
    AnalysisRequest, measurement_rows, resolve_review_constraints)
from tubetracker.annotation_store import AnnotationStore
from tubetracker.movie_analysis import MovieAnalysisService, export_analysis
from prototypes.v30_video_apex.cap_evidence import CapLabelIndex, emit_cap_candidates
from prototypes.v30_video_apex.route_evidence import generate_owned_routes


def observation(owner="a", **extra):
    return dict(movie="m", owner_uuid=owner, source_frame=30,
                direct_state="direct_visible", direct_xy=[50., 32.],
                source="review", source_id="obs", source_revision=1, **extra)


def test_scoped_reassignment_and_distinct_cap_provenance():
    request = AnalysisRequest("", "m", [30], owners=[{"id": "a"}, {"id": "b"}])
    old = observation()
    foreign = dict(observation("b"), movie="other", live_review=True, reassign_from_owner="a")
    rows, _ = resolve_review_constraints([old, foreign], request, request.owners)
    assert ("a", 30) in rows
    local = dict(foreign, movie="m", source_id="new", distinct_cap_evidence="reviewed-two-tips")
    rows, _ = resolve_review_constraints([old, local], request, request.owners)
    assert ("a", 30) not in rows
    assert rows[("b", 30)]["distinct_cap_evidence"] == "reviewed-two-tips"
    assert rows[("b", 30)]["tip_xy"] == [50., 32.]


def test_partial_endpoint_never_becomes_point_truth_or_constraint():
    o = observation(path_xy=[[10, 32], [50, 32]], path_complete=False,
                    task_type="review_path", obs_uuid="obs", obs_revision=1)
    index = CapLabelIndex([o], [])
    assert not index.points
    request = AnalysisRequest("", "m", [30], owners=[{"id": "a"}])
    constraints, ignored = resolve_review_constraints([o], request, request.owners)
    assert not constraints and "licensed" in ignored[0]["reason"]
    explicit = dict(o, tip_source="explicit_point")
    assert CapLabelIndex([explicit], []).points


@pytest.mark.parametrize("complete", [False, True])
def test_workflow_geometry_cannot_be_laundered_by_a_genuine_tip(complete):
    genuine = observation(task_type="review_tip", tip_source="explicit_point")
    trace = dict(observation(), task_type="centerline", source_id="workflow-path",
        review_origin="workflow_test", path_xy=[[10, 32], [50, 32]], path_complete=complete,
        tip_source="unobserved" if not complete else "reviewed_full_path")
    req = AnalysisRequest("", "m", [30], owners=[{"id": "a"}])
    constraints, _ = resolve_review_constraints([trace, genuine], req, req.owners)
    c = constraints[("a", 30)]
    assert c["tip_evidence"]["review_origin"] == "human"
    assert c["geometry_source"]["review_origin"] == c["review_origin"] == "workflow_test"
    assert c["path_complete"] == complete


def test_curved_route_wraps_around_grain_without_certifying_unreviewed_length():
    import cv2
    body = np.full((100, 100), .02, np.float32)
    centre = [50, 50]
    angles = np.linspace(-np.pi/2, np.pi, 160)
    path = np.column_stack((50 + 24*np.cos(angles), 50 + 24*np.sin(angles))).astype(np.int32)
    cv2.polylines(body, [path], False, .99, 7)
    owner = {"id": "a", "grain_native": centre, "grain_radius_px": 25,
             "attachment_native": path[0].tolist(), "attachment_verified": True}
    caps = emit_cap_candidates([.99], [path[-1].tolist()], [True], movie="m", source_frame=30)
    routes = generate_owned_routes(body, (0, 0), owner, caps)
    assert routes
    route = routes[0]
    p = np.asarray(route["current_path_xy"])
    assert np.linalg.norm(np.diff(p, axis=0), axis=1).sum() > 90
    assert np.min(np.linalg.norm(p-centre, axis=1)) > .72*25
    assert route["route_probability"] > .8
    assert route["path_complete"] is False and route["path_certificate"] is None
    assert np.linalg.norm(p[-1]-path[-1]) < 1e-6


def test_negative_crop_origin_keeps_native_coordinates():
    from types import SimpleNamespace
    from prototypes.v30_video_apex.targets import load_clip_pixels
    frame = np.arange(36, dtype=np.uint8).reshape(6, 6)
    reader = SimpleNamespace(read=lambda _: SimpleNamespace(frame=frame, exact=True))
    crop = load_clip_pixels(reader, [0], (-2, -1, 6, 5), [0])
    assert crop.shape == (1, 5, 6)
    np.testing.assert_allclose(crop[0, 1:, 2:], frame[:4, :4]/255)
    assert not crop[0, :1].any() and not crop[0, :, :2].any()


def service_fixture(tmp_path):
    movie, cap, body = [tmp_path / p for p in ("movie.bin", "cap.pt", "body.pt")]
    for p in (movie, cap, body):
        p.write_bytes(b"fixture-v1")
    calls = []
    def provider(request, owners):
        calls.append(request.to_dict())
        frames, arrays = {}, {}
        for frame in request.frames:
            pool = emit_cap_candidates([.99], [[50, 32]], [True], movie="m", source_frame=frame)
            by_owner = {}
            for owner in owners:
                k = f"{owner['id']}-{frame}"
                arrays[k] = np.full((64, 64), .99, dtype=np.float32)
                by_owner[owner["id"]] = {"body_key": k, "origin": [0, 0]}
            frames[str(frame)] = {"caps": pool, "owners": by_owner}
        return {"frames": frames, "image_size": [64, 64]}, arrays
    service = MovieAnalysisService(cap, body_checkpoint=body, cache_dir=tmp_path/"cache",
                                   pixel_provider=provider)
    request = AnalysisRequest(str(movie), "m", [0, 30], owners=[{
        "id": "a", "grain_native": [8, 32], "grain_radius_px": 3,
        "attachment_native": [10, 32], "attachment_verified": True}])
    return service, request, calls


def test_real_store_restart_hard_review_and_cache_layers(tmp_path):
    service, request, calls = service_fixture(tmp_path)
    initial = service.analyze(request)
    assert len(calls) == 1 and not initial["cache"]["pixel_hit"]
    assert all(r["length_px"] is None for r in initial["rows"])
    db = tmp_path / "annotations.db"
    store = AnnotationStore(db)
    store.save("task", "review", {"movie": "m", "owner_uuid": "a", "task_type": "review_tip"})
    point = {"task_uuid": "review", "owner_uuid": "a", "source_frame": 30,
             "direct_state": "direct_visible", "direct_xy": [53.125, 32.25],
             "tip_source": "explicit_point"}
    store.save("observation", "obs-review", point)
    store.close()
    store = AnnotationStore(db)
    try:
        revised = service.analyze(request, review_entities=store.entities(), review_source=str(db))
        assert len(calls) == 1 and revised["cache"]["pixel_hit"] and not revised["cache"]["inference_hit"]
        last = revised["rows"][-1]
        assert last["tip_xy"] == point["direct_xy"] and last["as_inference_constraint"]
        assert last["length_px"] is None  # a corrected point still isn't a full path
        request.acquisition = {"seconds_per_source_frame": 2., "cadence_source": "fixture acquisition"}
        measured = service.analyze(request, review_entities=store.entities(), review_source=str(db))
        assert len(calls) == 1 and measured["cache"]["inference_hit"]
        assert not measured["cache"]["measurement_hit"]
        assert measured["rows"][-1]["elapsed_time_s"] == 60
        paths = export_analysis(measured, tmp_path/"export")
        assert json.loads(open(paths["json"]).read())["rows"][-1]["tip_xy"] == point["direct_xy"]
        hit = service.analyze(request, review_entities=store.entities(), review_source=str(db))
        assert hit["cache"]["measurement_hit"]
        hidden = dict(point, direct_state="not_directly_visible", direct_xy=None)
        store.save("observation", "obs-review", hidden)
        hidden_result = service.analyze(request, review_entities=store.entities(), review_source=str(db))
        assert hidden_result["rows"][-1]["state"] == "occluded"
        assert hidden_result["rows"][-1]["tip_xy"] is None
        assert len(calls) == 1
    finally:
        store.close()
    changed = copy.deepcopy(request)
    changed.owners[0]["grain_native"][0] += 1
    service.analyze(changed)
    assert len(calls) == 2
    (tmp_path/"movie.bin").write_bytes(b"different movie, same path")
    service.analyze(changed)
    assert len(calls) == 3
    (tmp_path/"body.pt").write_bytes(b"changed weights")
    service.analyze(changed)
    assert len(calls) == 4


def test_measurement_units_require_full_paths_and_acquisition_provenance():
    rows = [{"owner_id": "a", "source_frame": f, "state": "present",
             "path_complete": full, "length_px": length}
            for f, full, length in [(0, True, 10), (30, True, 16), (60, False, 100), (90, True, 22)]]
    metadata = {"seconds_per_source_frame": 2, "cadence_source": "acquisition log",
                "micrometres_per_pixel": .5, "calibration_source": "stage micrometer"}
    result = measurement_rows(rows, metadata, first_frame=0)
    assert result[1]["growth_px_per_s"] == .1
    assert result[1]["growth_um_per_s"] == .05
    assert result[2]["length_px"] is None and result[3]["growth_px_per_s"] is None
    assert all(r["growth_px_per_s"] is None for r in measurement_rows(rows, {}, first_frame=0))
    with pytest.raises(ValueError, match="cadence_source"):
        measurement_rows(rows, {"seconds_per_source_frame": 2}, first_frame=0)


def test_exact_component_receipt_survives_service_cache_and_export(tmp_path):
    service, request, _ = service_fixture(tmp_path)
    request.solver = {'exact_single_owner': True}
    result = service.analyze(request)
    searches = result['inference']['component_search']
    assert searches[0]['owners'] == ['a']
    assert searches[0]['algorithm'] == 'exact_single_owner_forward_backward'
    assert searches[0]['states'] > 0
    assert result['inference']['temporal_search_truncated'] is False
    cached = service.analyze(request)
    assert cached['cache']['measurement_hit']
    paths = export_analysis(cached, tmp_path/'exact-export')
    assert json.loads(open(paths['json']).read())['inference']['component_search'] == searches


@pytest.mark.parametrize('marker', [
    {'provenance': 'workflow-test'},
    {'review_origin': 'synthetic_test'},
    {'constraint': {'review_origin': 'human', 'geometry_source': {'review_origin': 'workflow_test'}}},
    {'path_certificate': {'tip_evidence': {'review_origin': 'workflow_test'}}},
])
def test_workflow_geometry_cannot_supply_lengths_or_bridge_growth_rates(marker):
    rows = [dict(owner_id='a', source_frame=f, state='present', path_complete=True,
                 length_px=length) for f, length in [(0,20),(1,1000),(2,22),(3,24)]]
    rows[1].update(marker)
    measured = measurement_rows(rows, {'seconds_per_source_frame':2., 'cadence_source':'test log',
                                      'micrometres_per_pixel':.5, 'calibration_source':'test scale'}, first_frame=0)
    assert rows[1]['length_px'] == 1000  # the diagnostic geometry remains auditable
    assert measured[1]['length_px'] is None and measured[1]['length_um'] is None
    assert measured[1]['measurement_domain'] == 'workflow_verification'
    assert measured[1]['measurement_provenance']['exclusion'] == 'workflow_verification'
    assert measured[1]['growth_px_per_s'] is None and measured[2]['growth_px_per_s'] is None
    assert measured[3]['growth_px_per_s'] == 1.0


def test_workflow_full_trace_cannot_enter_native_grain_summary_or_measurement_csv(tmp_path):
    import csv
    service, request, _ = service_fixture(tmp_path)
    entities = []
    for task_id, frame, origin in [('real',0,'human'), ('synthetic',30,'workflow_test')]:
        entities.append({'uuid':task_id,'kind':'task','revision':1,'data':{
            'movie':'m','owner_uuid':'a','task_type':'centerline','review_origin':origin}})
        entities.append({'uuid':'obs-'+task_id,'kind':'observation','revision':1,'data':{
            'task_uuid':task_id,'owner_uuid':'a','source_frame':frame,
            'direct_state':'direct_visible','direct_xy':[50.,32.],
            'path_xy':[[10.,32.],[50.,32.]],'path_complete':True,'tip_source':'reviewed_full_path'}})
    result = service.analyze(request, review_entities=entities, review_source=str(tmp_path/'annotations.db'))
    summary = result['population']['grain_summaries'][0]
    assert summary['complete_measurement_count'] == 1
    assert summary['last_complete_length_frame'] == 0 and summary['last_complete_length_px'] == 40.
    assert summary['latest_length_withheld_reasons'] == ['workflow_verification']
    workflow = result['rows'][-1]
    assert workflow['path_complete'] and workflow['current_path_xy'] == [[10.,32.],[50.,32.]]
    assert workflow['length_px'] is None and workflow['provenance'] == 'workflow-test'
    paths = export_analysis(result, tmp_path/'workflow-export')
    exported = list(csv.DictReader(open(paths['csv'])))
    assert exported[-1]['length_px'] == '' and exported[-1]['length_um'] == ''
    assert exported[-1]['measurement_domain'] == 'workflow_verification'


def test_reference_context_is_explicit_and_invalidates_pixel_and_route_caches(tmp_path):
    from prototypes.v30_video_apex.route_quality import geometry_fingerprint
    service, request, calls = service_fixture(tmp_path)
    request.body_normalization_context = 'grain_reference'
    with pytest.raises(ValueError, match='analysis_roi'):
        request.validate()
    request.body_extent, request.roi_xyxy = 'analysis_roi', [0, 0, 64, 64]
    request.body_normalization_context = 'per_tile'
    first = service.analyze(request)
    reference = copy.deepcopy(request)
    reference.body_normalization_context = 'grain_reference'
    changed = service.analyze(reference)
    assert len(calls) == 2 and calls[-1]['body_normalization_context'] == 'grain_reference'
    assert changed['inputs']['pixel']['body_normalization_context'] == 'grain_reference'
    assert not any(changed['cache'][k] for k in ('pixel_hit', 'inference_hit', 'measurement_hit'))
    assert first['inputs']['inference']['pixel_key'] != changed['inputs']['inference']['pixel_key']
    assert geometry_fingerprint(request.owners, request.roi_xyxy, request.body_extent) != geometry_fingerprint(
        request.owners, request.roi_xyxy, request.body_extent, 'grain_reference')
    restored = service.analyze(request)
    assert len(calls) == 2 and restored['cache']['pixel_hit'] and restored['cache']['inference_hit']


def test_area_tip_survives_restart_as_presence_without_a_fabricated_point(tmp_path):
    service, request, calls = service_fixture(tmp_path)
    db = tmp_path / 'annotations.db'
    area = [[48., 28.], [56., 28.], [56., 36.], [48., 36.]]
    with closing(AnnotationStore(db)) as store:
        store.save('task', 'area-task', {'movie': 'm', 'owner_uuid': 'a', 'task_type': 'review_tip'})
        store.save('observation', 'area-tip', {
            'task_uuid': 'area-task', 'owner_uuid': 'a', 'source_frame': 30,
            'direct_state': 'direct_visible', 'direct_xy': None, 'direct_region': area,
            'tip_source': 'explicit_region'})
    with closing(AnnotationStore(db)) as store:
        result = service.analyze(request, review_entities=store.entities(), review_source=str(db))
        row = result['rows'][-1]
        assert row['state'] == 'imprecise'
        assert row['tip_xy'] is None and row['tip_region_xy'] == area
        assert not row['path_complete'] and row['length_px'] is None
        summary = result['population']['grain_summaries'][0]
        assert summary['onset_by_frame'] == 30
        assert summary['biological_class'] is None


def test_corrected_selection_keeps_frame_root_when_reviewed_tip_already_has_a_cap(tmp_path, monkeypatch):
    from prototypes.v30_video_apex import state_solver
    service, request, calls = service_fixture(tmp_path)
    request.frames = [30]
    db = tmp_path/'annotations.db'
    with closing(AnnotationStore(db)) as store:
        store.save('task', 'full', {'movie': 'm', 'owner_uuid': 'a', 'task_type': 'centerline'})
        store.save('observation', 'full-path', {
            'task_uuid': 'full', 'owner_uuid': 'a', 'source_frame': 30,
            'direct_state': 'direct_visible', 'direct_xy': [50., 32.],
            'path_xy': [[11., 32.], [30., 32.], [50., 32.]],
            'path_complete': True, 'tip_source': 'reviewed_full_path',
            'review_origin': 'human'})
        roots = []
        exits = []
        solve = state_solver.solve_joint_states
        def observe_candidates(candidates, *args, **kwargs):
            roots.append(candidates['a'][30][0]['current_path_xy'][0])
            exits.append(candidates['a'][30][0].get('exit_contract'))
            return solve(candidates, *args, **kwargs)
        monkeypatch.setattr(state_solver, 'solve_joint_states', observe_candidates)
        result = service.analyze(request, review_entities=store.entities(), review_source=str(db))
    # Length work (rev16): exit estimation applies only to provisional rim
    # roots. This owner carries a declared attachment, so every candidate
    # keeps its root verbatim and no exit is stamped; the reviewed frame
    # root is still kept on the human-constrained rows.
    assert roots == [[10., 32.], [11., 32.], [11., 32.]]
    assert exits == [None, None, None]
    assert result['rows'][0]['path_complete']
    assert result['model_without_reviews'][0]['current_path_xy'][0] == [10., 32.]
