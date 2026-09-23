import copy
import json
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest

from tubetracker.analysis_workbench import AnalysisSession
from tubetracker.annotation_app import AnnotatorController
from tubetracker.annotation_store import AnnotationStore
from test_movie_analysis import service_fixture


def test_changing_cache_or_registry_recreates_the_actual_service(tmp_path):
    service, request, _ = service_fixture(tmp_path)
    store = AnnotationStore(tmp_path/'annotations.db')
    config = {'cap_checkpoint': service.cap_checkpoint, 'body_checkpoint': service.body_checkpoint,
              'request': request.to_dict()}
    session = AnalysisSession(store, config, service=service)
    config['cache_dir'] = str(tmp_path/'new-cache')
    session.configure(config)
    assert session.service is not service and session.service.cache_dir == tmp_path/'new-cache'
    previous = session.service
    config['registry_path'] = str(tmp_path/'new-registry.sqlite')
    session.configure(config)
    assert session.service is not previous and session.service.registry_path == tmp_path/'new-registry.sqlite'
    store.close()


def test_reopening_a_result_refreshes_its_guide_without_changing_human_answers_or_drafts(tmp_path):
    service, request, _ = service_fixture(tmp_path)
    with closing(AnnotationStore(tmp_path/'annotations.db')) as store:
        session = AnalysisSession(store, {'cap_checkpoint': service.cap_checkpoint,
            'body_checkpoint': service.body_checkpoint, 'request': request.to_dict()}, service=service)
        c = AnnotatorController(store, SimpleNamespace(native_size=(64,64)))
        session.recompute()
        session.open_row(c, 'a', 30)
        c.save_apex(50, 32)
        c._region_pts = [[40, 30], [45, 30], [45, 35]]
        c.persist_draft()
        draft = copy.deepcopy(c.current['_drawing_draft'])
        answers = copy.deepcopy(store.entities('observation'))
        row = next(r for r in session.result['rows'] if r['source_frame'] == 30)
        row.update(tip_xy=[60,40], current_path_xy=[[10,32],[30,45],[60,40]])
        task = session.open_row(c, 'a', 30)
        assert task['focus_xy'] == task['draft_xy'] == [60,40]
        assert task['guide_path'] == row['current_path_xy']
        assert task['_drawing_draft'] == draft
        assert store.entities('observation') == answers


def test_same_frame_number_in_another_movie_clears_analysis_overlays(tmp_path):
    service, request, _ = service_fixture(tmp_path)
    store = AnnotationStore(tmp_path/'annotations.db')
    session = AnalysisSession(store, {'cap_checkpoint': service.cap_checkpoint,
        'body_checkpoint': service.body_checkpoint, 'request': request.to_dict()}, service=service)
    session.recompute()
    session._layers = ['stale-layer']
    controller = SimpleNamespace(viewer=SimpleNamespace(layers=['stale-layer']), current={'movie':'other-movie'})
    session.show_frame(controller, 0)
    assert not controller.viewer.layers and not session._layers
    store.close()


def test_context_frame_blocks_direct_annotation_writes_before_any_mutation(tmp_path):
    import pytest
    store = AnnotationStore(tmp_path / "annotations.db")
    controller = AnnotatorController(store, SimpleNamespace())
    controller.current = {"uuid": "p", "owner_uuid": "a", "query_frames": [30],
                          "task_type": "centerline", "completed": False}
    controller._path_pts = [[10, 20], [18, 22]]
    controller._context_offset = 1
    before = copy.deepcopy(controller.current)
    actions = [lambda: controller.save_apex(18, 22),
               lambda: controller.save_state("not_directly_visible"),
               lambda: controller.save_cant_tell(),
               lambda: controller.save_path(True),
               lambda: controller.save_region([[0, 0], [1, 0], [1, 1]]),
               lambda: controller.grain_click("A", 10, 20),
               lambda: controller.root_click("A", 11, 20),
               lambda: controller.save_mask(True),
               lambda: controller.save_mask_none(),
               lambda: controller.commit_mask_labels(True),
               lambda: controller.finish_census(),
               lambda: controller.skip_unresolvable(),
               lambda: controller.save_not_a_ball(),
               lambda: controller.save_and_next()]
    for action in actions:
        with pytest.raises(ValueError, match="task frame"):
            action()
        assert controller.current == before
        assert store.entities() == []
    store.close()


def test_census_review_changes_inventory_and_population_then_survives_restart(tmp_path):
    import pytest
    service, request, calls = service_fixture(tmp_path)
    store = AnnotationStore(tmp_path / "annotations.db")
    config = {"cap_checkpoint": service.cap_checkpoint, "body_checkpoint": service.body_checkpoint,
              "request": request.to_dict()}
    session = AnalysisSession(store, config, service=service)
    c = AnnotatorController(store, SimpleNamespace(native_size=(64, 64)))
    session.recompute()
    session.open_census(c, 0)
    with pytest.raises(ValueError, match="Confirm or remove"):
        c.finish_census()
    c.edit_census_grain(0)
    c.census_click(42, 48)
    c.current["census_membership_resolved"] = True
    c.finish_census()
    assert session.stale
    result = session.recompute()
    population = result["population"]
    assert len(result["owners"]) == population["census"]["n_grains"] == 2
    assert population["census"]["census_completeness_certified"]
    assert population["germination"]["population_fraction"] is None
    calls_before = len(calls)
    c.record_census_ruling(0, "germinated", "fixture protocol and observed tube", [0])
    classified = session.recompute()
    assert len(calls) == calls_before  # biological classification does not rerun pixel inference
    assert classified["population"]["germination"]["numerator_germinated"] == 1
    assert classified["cache"]["export_key"] != result["cache"]["export_key"]
    task_id = c.current["uuid"]
    identifiers = [g["grain_id"] for g in c.current["census_instances"]]
    store.close()
    store = AnnotationStore(tmp_path / "annotations.db")
    resumed = AnalysisSession(store, service=service)
    assert not resumed.stale
    c = AnnotatorController(store, SimpleNamespace(native_size=(64, 64)))
    resumed.open_census(c, 0)
    assert c.current["uuid"] == task_id
    assert [g["grain_id"] for g in c.current["census_instances"]] == identifiers
    c.edit_census_grain(1, remove=True)
    assert resumed.stale and not c.current["census_complete"]
    store.close()


def test_workflow_census_can_exercise_inventory_but_never_certify_biological_counts(tmp_path):
    service, request, calls = service_fixture(tmp_path)
    store = AnnotationStore(tmp_path / "annotations.db")
    config = {"cap_checkpoint": service.cap_checkpoint, "body_checkpoint": service.body_checkpoint,
              "verification_only": True, "request": request.to_dict()}
    session = AnalysisSession(store, config, service=service)
    c = AnnotatorController(store, SimpleNamespace(native_size=(64, 64)), actor="workflow-test")
    session.open_census(c, 0)
    c.edit_census_grain(0)
    c.current["census_membership_resolved"] = True
    c.finish_census()
    c.record_census_ruling(0, "germinated", "workflow test only", [0])
    result = session.recompute()
    assert not result["population"]["census"]["census_completeness_certified"]
    assert result["population"]["germination"]["numerator_germinated"] == 0
    store.close()


def test_viewing_results_after_census_does_not_invalidate_export_or_lose_undo(tmp_path):
    service, request, _ = service_fixture(tmp_path)
    store = AnnotationStore(tmp_path/'annotations.db')
    session = AnalysisSession(store, {'cap_checkpoint': service.cap_checkpoint,
        'body_checkpoint': service.body_checkpoint, 'request': request.to_dict()}, service=service)
    c = AnnotatorController(store, SimpleNamespace(native_size=(64,64)))
    session.open_census(c, 0)
    c.edit_census_grain(0)
    c.census_click(42,48)
    c.current['census_membership_resolved'] = True
    c.finish_census()
    uid = c.current['uuid']
    session.recompute()
    revision = store.load(uid)['revision']
    session.open_row(c, 'a', 0)
    assert not session.stale and store.load(uid)['revision'] == revision
    session.export()
    session.open_census(c, 0)
    c.restore_draft()
    assert c.undo_last_dot()
    c.persist_draft()
    assert len(c.current['census_instances']) == 1
    assert session.stale and not c.current['census_complete']
    saved = store.load(uid)
    c.load_tasks([saved['data']])
    assert store.load(uid)['revision'] == saved['revision']
    store.close()


def test_controller_edit_restart_recompute_export_uses_same_service(tmp_path):
    service, request, calls = service_fixture(tmp_path)
    config = {"cap_checkpoint": service.cap_checkpoint, "body_checkpoint": service.body_checkpoint,
              "request": request.to_dict()}
    store = AnnotationStore(tmp_path/"annotations.db")
    reader = SimpleNamespace()
    session = AnalysisSession(store, config, service=service)
    controller = AnnotatorController(store, reader)
    controller.analysis_session = session
    result = session.recompute()
    assert len(result["rows"]) == 2 and not session.stale
    task = session.open_row(controller, "a", 30)
    assert task["draft_xy"] is not None
    assert not session.stale  # simply viewing a result is not a correction
    controller.save_apex(51.234567, 32.123456)
    assert session.stale
    store.close()
    store = AnnotationStore(tmp_path/"annotations.db")
    resumed = AnalysisSession(store, service=service)
    assert resumed.stale and resumed.result is not None
    new = resumed.recompute()
    assert new["rows"][-1]["tip_xy"] == [51.234567, 32.123456]
    assert len(calls) == 1 and not resumed.stale
    exported = resumed.export()
    assert json.loads(open(exported["json"]).read())["rows"][-1]["as_inference_constraint"]
    store.close()


def test_partial_path_keeps_independent_point_and_cannot_upgrade_length(tmp_path):
    service, request, _ = service_fixture(tmp_path)
    store = AnnotationStore(tmp_path/"annotations.db")
    config = {"cap_checkpoint": service.cap_checkpoint, "body_checkpoint": service.body_checkpoint,
              "request": request.to_dict()}
    session = AnalysisSession(store, config, service=service)
    controller = AnnotatorController(store, SimpleNamespace())
    session.recompute()
    session.open_row(controller, "a", 30)
    controller.save_apex(50,32)
    session.start_path(controller)
    controller.path_click(10,32)
    controller.path_click(20,32)
    controller.save_path(False)
    result = session.recompute()
    row = result["rows"][-1]
    assert row["tip_xy"] == [50,32]
    assert row["length_px"] is None
    assert row["current_path_xy"] == [[10,32],[20,32]]
    store.close()


def test_full_paths_enable_actual_rate_after_acquisition_metadata(tmp_path):
    service, request, calls = service_fixture(tmp_path)
    store = AnnotationStore(tmp_path/"annotations.db")
    config = {"cap_checkpoint": service.cap_checkpoint, "body_checkpoint": service.body_checkpoint,
              "request": request.to_dict()}
    session = AnalysisSession(store, config, service=service)
    controller = AnnotatorController(store, SimpleNamespace())
    session.recompute()
    for frame, tip in [(0, [50,32]),(30,[56,32])]:
        session.open_row(controller,"a",frame)
        session.start_path(controller)
        controller.path_click(10,32)
        controller.path_click(*tip)
        controller.save_path(True)
    result = session.recompute()
    assert [r["length_px"] for r in result["rows"]] == [40,46]
    assert result["rows"][-1]["growth_px_per_s"] is None
    config = copy.deepcopy(config)
    config["request"]["acquisition"] = {"seconds_per_source_frame":2,"cadence_source":"test acquisition"}
    session.configure(config)
    result = session.recompute()
    assert result["rows"][-1]["growth_px_per_s"] == .1
    # rev14 P1: frame roots belong to the route stage only; this body model
    # takes grain centre/radius, so review changes never re-run its pixels.
    assert result["cache"]["inference_hit"] and len(calls) == 1
    store.close()


def test_saved_partial_trace_reopens_as_editable_geometry_without_relabeling(tmp_path):
    service, request, _ = service_fixture(tmp_path)
    store = AnnotationStore(tmp_path / "annotations.db")
    config = {"request": request.to_dict(), "cap_checkpoint": service.cap_checkpoint,
              "body_checkpoint": service.body_checkpoint}
    session = AnalysisSession(store, config, service=service)
    controller = AnnotatorController(store, SimpleNamespace())
    session.recompute()
    session.open_row(controller, "a", 30)
    session.start_path(controller)
    controller.path_click(10, 32)
    controller.path_click(20, 32)
    controller.save_path(False)
    uid = controller.current["uuid"]
    revision = store.load("obs-" + uid)["revision"]
    partial = session.recompute()["rows"][-1]
    assert partial["state"] == "unreviewed_tip" and partial["tip_xy"] is None
    assert partial["partial_path_xy"] == [[10, 32], [20, 32]]
    assert partial["length_px"] is None
    store.close()
    store = AnnotationStore(tmp_path / "annotations.db")
    session = AnalysisSession(store, service=service)
    assert session.selection["kind"] == "path"
    controller = AnnotatorController(store, SimpleNamespace())
    session.open_row(controller, "a", 30)
    session.start_path(controller)
    assert controller._path_pts == [[10, 32], [20, 32]]
    assert store.load("obs-" + uid)["revision"] == revision
    controller.undo_last_dot()
    controller.path_click(25, 35)
    controller.save_path(False)
    assert store.load("obs-" + uid)["data"]["path_xy"] == [[10, 32], [25, 35]]
    assert store.load("obs-" + uid)["revision"] == revision + 1
    store.close()


def test_visibility_review_preserves_partial_path_after_reopen(tmp_path):
    store = AnnotationStore(tmp_path / "annotations.db")
    controller = AnnotatorController(store, SimpleNamespace())
    controller.current = {"uuid": "p", "owner_uuid": "a", "query_frames": [30],
                          "task_type": "centerline"}
    controller.path_click(10, 20)
    controller.path_click(18, 22)
    controller.save_path(False)
    controller.save_state("not_directly_visible")
    store.close()
    store = AnnotationStore(tmp_path / "annotations.db")
    row = store.load("obs-p")["data"]
    assert row["path_xy"] == [[10, 20], [18, 22]]
    assert row["direct_state"] == "not_directly_visible"
    assert not row["path_complete"] and row.get("direct_xy") is None
    store.close()


def test_context_navigation_hides_task_marks_and_refreshes_analysis_frame(tmp_path):
    class Reader:
        def __len__(self):
            return 100
        def read(self, frame):
            return SimpleNamespace(frame=frame)
    store = AnnotationStore(tmp_path / "annotations.db")
    controller = AnnotatorController(store, Reader())
    raw = SimpleNamespace(name="raw", data=5, visible=True)
    mark = SimpleNamespace(name="apex", visible=True)
    hidden = SimpleNamespace(name="draft-hidden", visible=False)
    controller.viewer = SimpleNamespace(layers=[raw, mark, hidden])
    controller._image_layer = raw
    controller.current = {"uuid": "t", "query_frames": [5]}
    shown = []
    controller.analysis_session = SimpleNamespace(show_frame=lambda c, f: shown.append(f))
    controller.step_frame(1)
    assert raw.data == 6 and not mark.visible and not hidden.visible and shown == [6]
    controller.step_frame(1)
    controller.back_to_task_frame()
    assert raw.data == 5 and mark.visible and not hidden.visible and shown == [6, 7, 5]
    store.close()


@pytest.mark.parametrize('task_type,movie,marks', [
    ('census', 'm', ['census-balls', 'census-tile']),
    ('neg_region', 'other-movie', ['neg-boxes', 'known-tip']),
])
def test_opening_result_and_path_replaces_previous_task_markup(tmp_path, task_type, movie, marks):
    class Reader:
        native_size = (64, 64)

        def __len__(self):
            return 100

        def read(self, frame):
            return SimpleNamespace(frame=np.full((64, 64), frame, np.uint8))

    class Layers(list):
        def __init__(self):
            super().__init__()
            self.selection = SimpleNamespace(active=None)

        def __getitem__(self, key):
            if isinstance(key, str):
                for layer in self:
                    if layer.name == key:
                        return layer
                raise KeyError(key)
            return super().__getitem__(key)

    class Layer:
        def __init__(self, data, name):
            self.data, self.name, self.visible = data, name, True

    class Viewer:
        def __init__(self):
            self.layers = Layers()
            self.camera = SimpleNamespace(center=(0, 0), zoom=1.)

        def add_image(self, data, name, **kwargs):
            layer = Layer(data, name)
            self.layers.append(layer)
            return layer

        add_points = add_shapes = add_image

    service, request, _ = service_fixture(tmp_path)
    with closing(AnnotationStore(tmp_path/'annotations.db')) as store:
        session = AnalysisSession(store, {'cap_checkpoint': service.cap_checkpoint,
            'body_checkpoint': service.body_checkpoint, 'request': request.to_dict()}, service=service)
        session.recompute()
        controller = AnnotatorController(store, Reader(), viewer=Viewer())
        controller.analysis_session = session
        controller._readers.update({key: controller.reader for key in (movie, 'm')})
        previous = {'uuid': 'old-task', 'task_type': task_type, 'movie': movie,
            'query_frames': [0], 'focus_xy': [32, 32], 'suppress_analysis_overlays': True,
            'review_region': [[0, 0], [64, 0], [64, 64], [0, 64]],
            'census_grains': [[8, 32]], 'neg_boxes': [[5, 5, 15, 15]],
            'known_tip_xy': [50, 32]}
        controller.current = previous
        controller._show_current()
        controller.draw_task_markup()  # Existing census/region workspace entry point.
        assert all(controller.viewer.layers[name].visible for name in marks)
        controller.step_frame(1)
        assert all(not controller.viewer.layers[name].visible for name in marks)

        session.open_row(controller, 'a', 30)
        assert int(controller._image_layer.data[0, 0]) == 30
        assert all(not controller.viewer.layers[name].visible for name in marks)
        session.start_path(controller)
        assert all(not controller.viewer.layers[name].visible for name in marks)

        controller.current = previous
        controller._context_offset = 0
        controller._show_current()
        assert all(controller.viewer.layers[name].visible for name in marks)
