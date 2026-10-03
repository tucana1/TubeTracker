"""Corrections made in the app: saved through the labelling tool's store, shown at once, exported."""

import csv
import json

import pytest

from app_fixture import frame, make_run
from tubetracker.app import overlay
from tubetracker.app.corrections import Reviewer, ReviewError
from tubetracker.app.exports import export_movie
from tubetracker.app.model import RunData
from tubetracker.app.runfolder import RunFolder


@pytest.fixture
def reviewer(tmp_path):
    data = RunData(RunFolder(make_run(tmp_path / "run", {"duration_s": 24000.0, "um_per_px": 0.5})))
    r = Reviewer(data)
    r.start(background=False)
    assert r.ready() and r.error is None
    return r


def test_the_review_file_is_prefilled_with_the_models_answers_and_the_grains_drift(reviewer):
    f = reviewer.folder
    doc = json.loads(f.review_labels.read_text())
    assert f.review_labels.with_suffix(".model.json").exists()
    assert doc["labels"]["g001"]["onset"]["first_visible_bin"] == 10
    assert doc["prefill"]["observed_until"] == {"g003": frame(25)}
    g3 = doc["labels"]["g003"]["traces"]
    t = next(iter(g3.values()))
    assert t["view_offset"][0] == pytest.approx(0.5 * min(t["bin"], 25))  # the analysis' own drift
    assert reviewer.data.grain("g001")["review"]["state"] == "model"


def test_confirm_marks_the_grain_checked_and_unchanged(reviewer):
    out = reviewer.act("g001", "confirm")
    assert out["gid"] == "g001" and "confirmed" in out["message"]
    g = reviewer.data.grain("g001")
    assert g["review"]["state"] == "checked" and g["review"]["onset"] == "checked" and g["done"]
    assert g["review"]["traces_changed"] == 0 and g["review"]["traces_open"] == 0


def test_onset_here_moves_the_onset_and_the_curve(reviewer):
    reviewer.act("g001", "onset", b=14)
    g = reviewer.data.grain("g001")
    assert g["onset"] == 14 and g["review"]["onset"] == "changed" and g["review"]["state"] == "corrected"
    assert g["L"][13] == 0.0 and g["L"][20] > 0 and g["Lm"][10] == 2.0
    reviewer.act("g002", "no_onset")
    assert reviewer.data.grain("g002")["review"]["onset"] == "checked"


def test_a_tip_click_reads_the_length_along_the_route(reviewer):
    d = reviewer.data
    g = d.grain("g001")
    route, _ = overlay.route_at(g, 30)
    tip = overlay.to_length(route, 25.0)[-1]
    reviewer.act("g001", "tip", b=30, x=tip[0] + 0.0, y=tip[1] + 3.0)  # 3 px off the route: snapped onto it
    g = d.grain("g001")
    assert g["L"][30] == pytest.approx(25.0, abs=0.1) and g["review"]["state"] == "corrected"
    assert any(t["bin"] == 30 and t["L"] == pytest.approx(25.0, abs=0.1) for t in g["human"])
    end = route[-1]
    reviewer.act("g001", "tip", b=35, x=end[0] + 12.0, y=end[1])  # past the route's end: carried on to the click
    assert d.grain("g001")["L"][35] == pytest.approx(overlay.path_length(route) + 12.0, abs=0.2)
    with pytest.raises(ReviewError):
        reviewer.act("g001", "tip", b=30, x=tip[0], y=tip[1] + 40.0)  # nowhere near the tube
    with pytest.raises(ReviewError):
        reviewer.act("g002", "tip", b=30, x=150.0, y=50.0)  # never germinated: set its onset first


def swaying_reviewer(tmp_path) -> Reviewer:
    """The fixture's movie with g001's tube swinging round its exit, 1 degree per bin up to straight at bin 39."""
    run = make_run(tmp_path / "run", {"duration_s": 24000.0})
    pred_path = run / "analysis" / "predictions.json"
    pred = json.loads(pred_path.read_text())
    g1 = next(g for g in pred["grains"] if g["id"] == "g001")
    g1["rotation_deg"] = [float(f // 300 - 39) for f in g1["length"]["frames"]]
    pred_path.write_text(json.dumps(pred))
    r = Reviewer(RunData(RunFolder(run)))
    r.start(background=False)
    return r


def test_far_from_a_drawn_tube_a_tip_click_snaps_to_the_route_drawn_at_that_time(tmp_path):
    """A tube drawn at bin 39 is drawn along the person's route near bin 39; at bin 15 the tube follows the model's
    route as it lay then (turned 24 degrees), and a tip clicked on it there is read along that route."""
    import math

    r = swaying_reviewer(tmp_path)
    d = r.data
    exit_ = (68.0, 50.0)
    r.act("g001", "path", b=39, points=[list(exit_), [exit_[0] + 60.0, exit_[1]]])  # the model's 60 px, straight
    route, mine = d.route("g001", 15)
    assert not mine and route == overlay.model_route(d.grain("g001"), 15)
    assert d.route("g001", 30) == ([list(exit_), [exit_[0] + 60.0, exit_[1]]], True)  # near the drawn time
    th = math.radians(-24.0)
    click = (exit_[0] + 10.0 * math.cos(th), exit_[1] + 10.0 * math.sin(th))  # 10 px along the route at bin 15
    tube = overlay.tube_at(d.grain("g001"), 15)
    assert tube[-1] == pytest.approx([exit_[0] + 12.0 * math.cos(th), exit_[1] + 12.0 * math.sin(th)], abs=0.05)
    r.act("g001", "tip", b=15, x=click[0], y=click[1])
    g = d.grain("g001")
    assert g["L"][15] == pytest.approx(10.0, abs=0.1)  # on the person's straight route it would read 9.1
    assert overlay.tube_at(g, 15)[-1] == pytest.approx(list(click), abs=0.05)


def test_confirm_far_from_a_drawn_tube_saves_the_route_drawn_at_that_time(tmp_path):
    """Confirm keeps the model's own proposals where they show what is drawn; where it draws a length itself, it is
    along the route drawn at that time: far from a drawn tube, the model's route then."""
    import math

    r = swaying_reviewer(tmp_path)
    exit_ = (68.0, 50.0)
    r.act("g001", "path", b=39, points=[list(exit_), [exit_[0] + 60.0, exit_[1]]])
    for doc in (r.data.model_doc, r.bench.doc):  # no model proposal at bin 16 (far from 39): confirm draws one
        doc["labels"]["g001"]["traces"].pop("16")
    r.act("g001", "confirm")  # the tool's trace times: 16, 27 and 38
    traces = json.loads(r.folder.review_labels.read_text())["labels"]["g001"]["traces"]
    apex = traces["16"]["path_xy_view"][-1]
    assert traces["16"]["view"] == "app-confirm" and traces["16"]["length_px"] == pytest.approx(14.0, abs=0.05)
    assert math.degrees(math.atan2(apex[1] - exit_[1], apex[0] - exit_[0])) == pytest.approx(-23.0, abs=0.2)
    assert traces["27"]["view"] == "model"  # the model's own answer there already showed the length drawn


def test_a_followed_grains_points_are_stored_in_its_own_frame(reviewer):
    d = reviewer.data
    b = 20
    dx = d.drift_at("g003", b)[0]
    pts = [[68.0 + dx, 115.0], [88.0 + dx, 115.0]]  # clicked where the grain is at bin 20
    reviewer.act("g003", "path", b=b, points=pts)
    t = json.loads(d.folder.review_labels.read_text())["labels"]["g003"]["traces"][str(b)]
    assert t["path_xy_view"][0] == [68.0, 115.0] and t["length_px"] == pytest.approx(20.0)
    assert overlay.route_at(d.grain("g003"), b) == ([[68.0, 115.0], [88.0, 115.0]], True)


def test_burst_exclude_revert_and_undo(reviewer):
    d = reviewer.data
    reviewer.act("g001", "burst", b=22)
    g = d.grain("g001")
    assert g["lost"] == 22 and g["lost_why"] == "burst" and len(g["L"]) == d.n_bins
    reviewer.act("g002", "exclude", reason="not_a_grain")
    assert d.grain("g002")["excluded"] == "not_a_grain"
    assert "g002" not in {c["gid"] for c in d.checks(d.grains())}
    reviewer.act("g002", "include")
    assert d.grain("g002")["excluded"] is None
    undone = reviewer.act(None, "undo")
    assert undone["gid"] == "g002" and d.grain("g002")["excluded"] == "not_a_grain"
    reviewer.act("g001", "revert")
    assert d.grain("g001")["review"]["state"] == "model" and d.grain("g001")["lost"] is None
    with pytest.raises(ReviewError):
        reviewer.act("g001", "exclude", reason="because")
    with pytest.raises(ReviewError):
        reviewer.act("g999", "confirm")


def test_export_writes_the_tables_in_the_movies_units_with_corrections(reviewer):
    d = reviewer.data
    route, _ = overlay.route_at(d.grain("g001"), 30)
    tip = overlay.to_length(route, 30.0)[-1]
    reviewer.act("g001", "tip", b=30, x=tip[0], y=tip[1])
    out = export_movie(d)
    results = d.folder.results
    assert {"grains.csv", "growth.csv", "germination.png", "growth_curves.png", "summary.txt"} <= {
        p.name for p in results.iterdir()}
    rows = {r["grain"]: r for r in csv.DictReader(open(results / "grains.csv"))}
    assert rows["g001"]["source"] == "reviewed (corrected)" and rows["g002"]["status"] == "not germinated"
    assert rows["g003"]["lost_after_frame"] == str(frame(25)) and rows["g004"]["isolated"] == "no"
    assert float(rows["g001"]["onset_by_min"]) == pytest.approx(frame(10) * 24000.0 / 12000 / 60, abs=0.1)
    growth = [r for r in csv.DictReader(open(results / "growth.csv")) if r["grain"] == "g001"]
    at30 = next(r for r in growth if r["bin"] == "30")
    assert float(at30["length_um"]) == pytest.approx(15.0, abs=0.1) and at30["checked_here"] == "yes"
    reviewed = [r for r in csv.DictReader(open(d.folder.review_labels.parent / "reviewed_growth.csv"))
                if r["grain"] == "g001" and r["bin"] == "30"]
    assert float(reviewed[0]["length_px"]) == pytest.approx(30.0, abs=0.1)  # the review export has it too
    assert out["folder"] == str(results)


def test_confirm_does_not_count_a_grain_the_model_could_not_read(tmp_path):
    """The model pre-fills an unreadable grain as never germinated: Enter must not turn debris into a counted grain."""
    run = make_run(tmp_path / "run", {"duration_s": 24000.0})
    pred_path = run / "analysis" / "predictions.json"
    pred = json.loads(pred_path.read_text())
    g2 = next(g for g in pred["grains"] if g["id"] == "g002")
    g2["status"], g2["flags"] = "unobservable", ["no_grain"]
    pred_path.write_text(json.dumps(pred))
    r = Reviewer(RunData(RunFolder(run)))
    r.start(background=False)
    with pytest.raises(ReviewError, match="could not be read"):
        r.act("g002", "confirm")
    assert r.data.grain("g002")["status"] == "unobservable"  # still not counted
    r.act("g002", "exclude", reason="not_a_grain")  # what it says to do instead
    assert r.data.grain("g002")["excluded"] == "not_a_grain"
    r.act("g002", "include")
    r.act("g002", "no_onset")  # a person who says it is a grain that never germinated is believed
    r.act("g002", "confirm")
    assert r.data.grain("g002")["status"] == "no_emergence_by_end"
