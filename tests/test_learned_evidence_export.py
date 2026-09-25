"""Exporting reviewed labels: checked and changed answers, and the germination curve's input."""

import csv
import json

import pytest

pytest.importorskip("torch")

from prototypes.learned_evidence import export_review as E  # noqa: E402


def _onset(fv, origin, fpb=300):
    return {"verdict": "emerged_within", "first_visible_bin": fv, "last_absent_bin": fv - 1,
            "first_visible_frame": fv * fpb + fpb // 2, "last_absent_frame": (fv - 1) * fpb + fpb // 2,
            "review_origin": origin}


def _trace(b, length, origin, state="full"):
    return {"bin": b, "source_frame": b * 300 + 150, "state": state, "length_px": length, "review_origin": origin}


def test_export_counts_what_was_checked_and_changed(tmp_path):
    model = {"frames_per_bin": 300, "n_bins": 100, "grains": {"g1": {}, "g2": {}, "g3": {"excluded": True}},
             "labels": {"g1": {"onset": _onset(30, "model"), "traces": {"36": _trace(36, 20.0, "model")}},
                        "g2": {"onset": _onset(50, "model"), "traces": {"69": _trace(69, 30.0, "model")}},
                        "g3": {"onset": _onset(10, "model")}}}
    reviewed = json.loads(json.dumps(model))
    reviewed["labels"]["g1"]["onset"] = _onset(30, "human")                 # confirmed
    reviewed["labels"]["g1"]["traces"]["36"] = _trace(36, 21.0, "human")   # confirmed, within tolerance
    reviewed["labels"]["g2"]["onset"] = _onset(55, "human")                 # moved
    reviewed["labels"]["g2"]["traces"]["69"] = _trace(69, 45.0, "human")   # longer
    path = tmp_path / "review_labels.json"
    path.write_text(json.dumps(reviewed))
    path.with_suffix(".model.json").write_text(json.dumps(model))
    E.main(["--labels", str(path), "--um-per-px", "0.5", "--s-per-frame", "10"])
    rows = list(csv.DictReader(open(tmp_path / "reviewed_grains.csv")))
    assert [r["grain"] for r in rows] == ["g1", "g2"]  # excluded grains are not results
    assert (rows[0]["onset_checked"], rows[0]["onset_changed"]) == ("yes", "no")
    assert (rows[1]["onset_checked"], rows[1]["onset_changed"]) == ("yes", "yes")
    assert rows[1]["last_length_um"] == "22.5" and rows[0]["onset_by_min"] == str(round(9150 * 10 / 60, 2))
    traces = list(csv.DictReader(open(tmp_path / "reviewed_traces.csv")))
    assert [(t["grain"], t["changed"]) for t in traces] == [("g1", "no"), ("g2", "yes")]
    assert (tmp_path / "population.csv").exists()


def test_population_input_uses_the_reviewed_onsets():
    doc = {"frames_per_bin": 300, "n_bins": 10, "grains": {"a": {}, "b": {}},
           "labels": {"a": {"onset": _onset(4, "human")}, "b": {"onset": {"verdict": "no_emergence_by_end"}}}}
    got = {g["id"]: g for g in E.population_input(doc)["grains"]}
    assert got["a"]["onset_interval"] == [3 * 300 + 150, 4 * 300 + 150] and got["a"]["status"] == "emerged_within"
    assert got["b"]["status"] == "no_emergence_by_end" and got["a"]["length"]["frames"] == [150, 9 * 300 + 150]


def _write(tmp_path, model, reviewed, with_model=True):
    path = tmp_path / "review_labels.json"
    path.write_text(json.dumps(reviewed))
    if with_model:
        path.with_suffix(".model.json").write_text(json.dumps(model))
    return path


def test_only_the_traces_the_tool_asks_for_now_are_results(tmp_path, capsys):
    # the model traced bins 36, 69 and 98 (onset at bin 30); the reviewer moves the onset to 40, which moves
    # the first trace to 46, traces it, and marks the tube burst at 69: the model's 36 and 98 are left out
    model = {"frames_per_bin": 300, "n_bins": 100, "grains": {"g1": {}},
             "labels": {"g1": {"onset": _onset(30, "model"),
                               "traces": {str(b): _trace(b, 10.0 + b / 10, "model") for b in (36, 69, 98)}}}}
    reviewed = json.loads(json.dumps(model))
    lab = reviewed["labels"]["g1"]
    lab["onset"] = _onset(40, "human")
    path = _write(tmp_path, model, reviewed)
    E.main(["--labels", str(path)])
    assert "1 traces still to answer" in capsys.readouterr().out  # bin 46: the model had nothing there
    lab["traces"]["46"] = _trace(46, 12.0, "human")
    lab["traces"]["69"] = _trace(69, 0.0, "human", state="burst")
    path = _write(tmp_path, model, reviewed)
    E.main(["--labels", str(path)])
    assert "2 of the model's traces left out" in capsys.readouterr().out
    rows = list(csv.DictReader(open(tmp_path / "reviewed_traces.csv")))
    assert [(r["bin"], r["state"], r["checked"], r["changed"]) for r in rows] == [
        ("46", "full", "yes", "yes"), ("69", "burst", "yes", "yes")]
    g = list(csv.DictReader(open(tmp_path / "reviewed_grains.csv")))[0]
    assert (g["last_traced_bin"], g["traces_checked"], g["onset_changed"]) == ("46", "2/2", "yes")


def test_no_tube_after_all_leaves_no_traces_and_a_lost_model_file_is_said(tmp_path, capsys):
    model = {"frames_per_bin": 300, "n_bins": 100, "grains": {"g1": {}},
             "labels": {"g1": {"onset": _onset(30, "model"), "traces": {"36": _trace(36, 8.0, "model")}}}}
    reviewed = json.loads(json.dumps(model))
    reviewed["labels"]["g1"]["onset"] = {"verdict": "no_emergence_by_end", "review_origin": "human"}
    E.main(["--labels", str(_write(tmp_path, model, reviewed, with_model=False))])
    assert "can't be told" in capsys.readouterr().out
    assert list(csv.DictReader(open(tmp_path / "reviewed_traces.csv"))) == []
    g = list(csv.DictReader(open(tmp_path / "reviewed_grains.csv")))[0]
    assert (g["onset_checked"], g["onset_changed"], g["last_traced_bin"]) == ("yes", "", "")


def test_a_partly_checked_review_also_gets_the_checked_onsets_alone(tmp_path, capsys):
    model = {"frames_per_bin": 300, "n_bins": 100, "grains": {"g1": {}, "g2": {}},
             "labels": {"g1": {"onset": _onset(30, "model")}, "g2": {"onset": _onset(50, "model")}}}
    reviewed = json.loads(json.dumps(model))
    reviewed["labels"]["g1"]["onset"] = _onset(32, "human")
    path = _write(tmp_path, model, reviewed)
    assert [g["id"] for g in E.population_input(reviewed, checked_only=True)["grains"]] == ["g1"]
    E.main(["--labels", str(path)])
    assert (tmp_path / "population.csv").exists() and (tmp_path / "checked_only" / "population.csv").exists()
    assert "checked onsets alone" in capsys.readouterr().out


def test_trace_once_anchors_are_the_checked_latest_traces(tmp_path):
    def traced(b, L, origin):
        return {**_trace(b, L, origin), "path_xy_ref": [[0.0, 0.0], [L, 0.0]]}

    plan = [36, 40, 69, 98]  # the tool's trace bins for an onset at bin 30 in a 100-bin movie
    doc = {"frames_per_bin": 300, "n_bins": 100, "grains": {"g1": {}, "g2": {}, "g3": {}, "g4": {"excluded": True}},
           "labels": {"g1": {"onset": _onset(30, "model"),
                             "traces": {str(b): traced(b, 10.0 + b, "human" if b == 98 else "model") for b in plan}},
                      "g2": {"onset": _onset(30, "model"),  # its latest trace is still the model's
                             "traces": {"36": traced(36, 9.0, "human"), "98": traced(98, 40.0, "model")}},
                      "g3": {"onset": _onset(30, "human"), "traces": {"98": _trace(98, 30.0, "human")}},  # no path
                      "g4": {"onset": _onset(30, "human"), "traces": {"98": traced(98, 30.0, "human")}}}}
    anchors = E.checked_anchors(doc)
    assert set(anchors) == {"g1"} and anchors["g1"]["bin"] == 98 and anchors["g1"]["length_px"] == 108.0
    path = _write(tmp_path, doc, doc)
    with pytest.raises(SystemExit, match="probability cache"):  # decoding needs the movie's probability cache
        E.main(["--labels", str(path), "--anchored", "--field", str(tmp_path / "cache")])
