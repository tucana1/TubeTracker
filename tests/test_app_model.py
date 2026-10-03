"""The app's plain-Python layer: units, run folders and setup, an analysis as the window shows it."""

import json

import pytest

from app_fixture import FPB, N_BINS, REF_START, frame, make_run
from tubetracker.app import overlay
from tubetracker.app.model import RunData, stall_bin
from tubetracker.app.runfolder import RunFolder, find_runs, folder_for, safe_name
from tubetracker.app.units import Units, duration_seconds, format_duration, seconds_per_frame


# ---------------------------------------------------------------------------------------------- units
def test_seconds_per_frame_comes_from_how_long_the_movie_ran():
    assert seconds_per_frame(7 * 3600, 25200) == pytest.approx(1.0)
    assert seconds_per_frame(None, 100) is None and seconds_per_frame(100, 0) is None
    assert duration_seconds("7", "18", "") == 7 * 3600 + 18 * 60
    assert duration_seconds("", "", "") is None and duration_seconds(0, 0) is None
    with pytest.raises(ValueError):
        duration_seconds("-1")
    assert format_duration(26280) == "7 h 18 min" and format_duration(59) == "59 s"


def test_units_convert_frames_lengths_and_rates():
    u = Units(s_per_frame=0.5, um_per_px=0.65)
    assert u.minutes(1200) == pytest.approx(10.0) and u.um(10) == pytest.approx(6.5)
    assert u.rate(0.01) == pytest.approx(0.01 * 0.65 * 60 / 0.5) and u.rate_unit == "µm/min"
    assert u.pair() == (0.65, 0.5)
    bare = Units()
    assert bare.time(1200) == 1200 and bare.length(10) == 10 and bare.rate_unit == "px/1000 frames"
    assert bare.pair() is None and Units(s_per_frame=0.5).rate_unit == "px/min"
    assert Units.from_setup({"duration_s": 3600, "um_per_px": 0.5}, n_frames=7200).s_per_frame == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------------- run folders
def test_a_movie_gets_its_own_run_folder_named_as_sparsetrack_names_it(tmp_path):
    f = folder_for(tmp_path / "Pollen tube movie 2 7-14-26.mp4", tmp_path / "runs")
    assert f.root == (tmp_path / "runs" / "Pollen_tube_movie_2_7-14-26").resolve()
    assert safe_name("test1lowdensjoshua-28c-hz.mp4 ") == "test1lowdensjoshua-28c-hz.mp4"
    run = make_run(tmp_path / "runs" / "tiny")
    assert folder_for(run / "analysis", tmp_path / "runs").root == run.resolve()  # the analysis folder inside one
    assert [r.name for r in find_runs(tmp_path / "runs")] == ["tiny"]


def test_setup_takes_a_duration_or_a_frame_interval(tmp_path):
    f = RunFolder(make_run(tmp_path / "run"))
    assert not f.is_setup()
    s = f.apply_setup({"hours": "3", "minutes": "20", "um_per_px": "0.65", "sample_id": " A1 ", "genotype": "WT",
                       "replicate": "2", "notes": ""})
    assert s["duration_s"] == 12000 and s["um_per_px"] == 0.65 and s["sample_id"] == "A1" and f.is_setup()
    assert Units.from_setup(f.load_setup(), f.n_frames()).s_per_frame == pytest.approx(12000 / (FPB * N_BINS))
    s = f.apply_setup({"hours": "", "minutes": "", "s_per_frame": "2", "um_per_px": ""})
    assert s["s_per_frame"] == 2 and s["duration_s"] == 2 * FPB * N_BINS and s["um_per_px"] is None
    s = f.apply_setup({"hours": "", "minutes": "", "s_per_frame": ""})
    assert s["duration_s"] is None and s["s_per_frame"] is None  # times stay in frames
    with pytest.raises(ValueError):
        f.apply_setup({"um_per_px": "-2"})
    listing = f.listing()
    assert listing["analysed"] and listing["prepared"] and listing["reviewed"] is None and listing["movie"] == "tiny.mp4"


def test_a_bare_analysis_folder_finds_the_cache_its_predictions_name(tmp_path):
    run = make_run(tmp_path / "run")
    bare = tmp_path / "bare"
    bare.mkdir()
    pred = json.loads((run / "analysis" / "predictions.json").read_text())
    pred["cache"] = str(run / "cache")
    (bare / "predictions.json").write_text(json.dumps(pred))
    f = RunFolder(bare)
    assert f.analysis == bare.resolve() and f.cache == run / "cache" and f.has_cache()


# ---------------------------------------------------------------------------------------------- the analysis shown
@pytest.fixture
def data(tmp_path):
    return RunData(RunFolder(make_run(tmp_path / "run", {"duration_s": FPB * N_BINS * 2.0, "um_per_px": 0.5})))


def test_series_are_by_bin_even_when_the_reference_starts_late(data):
    g1 = data.grain("g001")
    assert len(g1["L"]) == N_BINS and g1["L"][9] == 0.0 and g1["L"][10] == 2.0 and g1["L"][20] == 22.0
    assert g1["onset"] == 10 and g1["onset_by"] == frame(10)
    g3 = data.grain("g003")
    assert g3["drift"][REF_START] == [REF_START * 0.5, 0.0] and g3["drift"][0] == g3["drift"][REF_START]
    assert overlay.pos_at(g3, 20) == (60.0 + 10.0, 115.0)
    assert g3["lost"] == 26 and g3["lost_why"] == "gap"


def test_grain_states_over_time(data):
    g = {gid: data.grain(gid) for gid in data.order}
    assert overlay.state_at(g["g001"], 9) == "notyet" and overlay.state_at(g["g001"], 10) == "germinated"
    assert overlay.state_at(g["g002"], 30) == "never"
    assert overlay.state_at(g["g003"], 25) == "germinated" and overlay.state_at(g["g003"], 26) == "lost"
    assert overlay.state_at(g["g004"], 0) == "germinated"


def test_the_tube_is_drawn_along_its_route_to_its_length_where_the_grain_is(data):
    g1 = data.grain("g001")
    tube = overlay.tube_at(g1, 20)
    assert tube[0] == [68.0, 50.0] and overlay.path_length(tube) == pytest.approx(22.0)
    assert overlay.tube_at(g1, 5) is None
    g3 = data.grain("g003")
    t3 = overlay.tube_at(g3, 20)
    assert t3[0] == [68.0 + 10.0, 115.0]  # moved with the grain


def test_a_persons_route_is_drawn_near_their_trace_and_the_models_own_route_elsewhere():
    """Away from the time a person traced it, the tube follows the model's route as it lay then (it sways and turns as
    the tube grows), carried on along the person's where the tube is longer than the model's route reaches."""
    n = 80
    model_L = [0.0] * 4 + [min(2.0 * (b - 3), 60.0) for b in range(4, n)]  # the model's reading: 60 px by bin 33
    g = {"path": [[10.0 + k, 50.0] for k in range(0, 61, 2)], "pivot": [10.0, 50.0],
         "rot": [0.2 * (b - 30) for b in range(n)],  # the tube swings round its exit: straight at bin 30
         "human": [{"bin": 30, "state": "full", "L": 80.0, "pts": [[10.0, 50.0], [50.0, 51.0], [90.0, 52.0]]}],
         "L": [0.0] * 4 + [min(2.0 * (b - 3), 80.0) for b in range(4, n)], "Lm": model_L}
    person = g["human"][0]["pts"]
    assert overlay.route_at(g, 30) == (person, True) and overlay.route_at(g, 12) == (person, True)  # near it
    far, mine = overlay.route_at(g, 5)  # 25 bins before: the model's route as it lay then, turned 5 degrees
    assert not mine and far == overlay.model_route(g, 5) and far[-1][1] == pytest.approx(50.0 - 60 * 0.0872, abs=0.05)
    after, mine = overlay.route_at(g, 60)  # 30 bins after, the tube (80 px) reaches beyond the model's 60 px route:
    assert not mine and overlay.path_length(after) >= 79.5  # carried on along the person's route
    assert after[:31] == overlay.model_route(g, 60)
    swung = {**g, "rot": [2.0 * (b - 30) for b in range(n)]}  # its end far off the person's route: theirs instead
    assert overlay.route_at(swung, 60)[1] and overlay.route_at(swung, 5) == (overlay.model_route(swung, 5), False)
    g["Lm"] = [0.0] * 10 + model_L[10:]  # where the model read no tube yet, its route is not this tube's
    assert overlay.route_at(g, 5) == (person, True)
    assert overlay.route_at({**g, "human": []}, 5) == (overlay.model_route(g, 5), False)  # nothing traced


def test_events_mark_germinations_losses_stalls_focus_and_t50(data):
    grains = data.grains()
    pop = data.population(grains)
    ev = data.events(grains, pop)
    kinds = {(e["kind"], e.get("gid")) for e in ev}
    assert ("germination", "g001") in kinds and ("germination", "g003") in kinds
    assert ("lost", "g003") in kinds and ("focus", None) in kinds and ("t50", None) in kinds
    assert ("stall", "g004") in kinds  # 20 px from bin 8 on: stood still for over 15 bins
    assert {gid for kind, gid in kinds if kind == "check"} == {c["gid"] for c in data.checks(grains) if not c["done"]}
    assert [e["bin"] for e in ev] == sorted(e["bin"] for e in ev)
    assert stall_bin([0, 5, 10, 10, 10], 4) is None  # too short a stand-still


def test_population_counts_isolated_grains_and_censors_a_lost_one(data):
    grains = data.grains()
    pop = data.population(grains)
    assert pop["n"] == 3  # g004 is in a clump
    assert pop["counts"] == {"germinated": 2, "not_germinated": 1, "lost_before": 0}
    assert pop["t50_frame"] == frame(12) and pop["t50_bin"] == 12
    assert pop["certain"][-1] == pytest.approx(2 / 3, abs=1e-3)


def test_the_check_list_puts_the_least_sure_first_with_short_reasons(data):
    grains = data.grains()
    checks = data.checks(grains)
    by = {c["gid"]: c for c in checks}
    assert "touches g002" in by["g001"]["reasons"]
    assert by["g003"]["reasons"][0].startswith("lost at ") and "onset at focus change" in by["g003"]["reasons"]
    assert all(len(r) < 40 for c in checks for r in c["reasons"])  # short phrases
    detail = {r["code"]: r["detail"] for r in data.grain("g003")["check"]}
    assert detail["lost"] and detail["focus"]  # a sentence on demand
    assert checks == sorted(checks, key=lambda c: (c["done"], not c["isolated"], c["conf"] is None, c["conf"] or 1.0,
                                                   c["gid"]))


def test_times_and_lengths_read_in_the_movies_units(data):
    assert data.when(frame(10)) == f"{frame(10) * 2.0 / 60:.0f} min"
    assert data.when_range(frame(9), frame(10)) == f"{frame(9) * 2 / 60:.0f}-{frame(10) * 2 / 60:.0f} min"
    assert data.length_words(10) == "5.0 µm"
    data.set_units({})
    assert data.when(frame(10)) == f"frame {frame(10)}" and data.length_words(10) == "10.0 px"


def test_summary_and_warnings_are_short(data):
    grains = data.grains()
    s = data.summary(grains, data.population(grains))
    assert s["n"] == 3 and s["germinated"] == 2 and s["lost"] == 1 and "T50" in s["line"]
    w = data.warnings()
    assert w and w[0]["text"].startswith("focus change at")


def test_an_onset_within_one_minute_reads_as_that_minute_and_the_time_shown_reads_against_the_end(data):
    # 2 s per frame: frames 2850 and 2851 are both at 95 min
    assert data.when_range(2850, 2851) == "95 min"
    assert data.time_of(10) == f"{frame(10) * 2 / 60:.0f} of {frame(39) * 2 / 60:.0f} min"
    data.set_units({})
    assert data.time_of(10) == f"frame {frame(10):,} of {frame(39):,}"


def test_the_summary_says_which_grains_are_counted_and_a_grain_what_else_was_found(data):
    grains = data.grains()
    s = data.summary(grains, data.population(grains))
    assert "counted" in s["line"] and "1 in clumps or at the edge" in s["detail"]
    g3 = data.grain("g003")
    assert any(n.startswith("moved ") for n in data.notes(g3))  # it drifted 12.5 px and was followed
    unsure = [r for g in grains for r in g["check"] if r["code"] == "unsure"]
    assert all("the model reads" in r["detail"] for r in unsure)
