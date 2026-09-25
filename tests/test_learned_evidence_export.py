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
             "labels": {"g1": {"onset": _onset(30, "model"), "traces": {"40": _trace(40, 20.0, "model")}},
                        "g2": {"onset": _onset(50, "model"), "traces": {"60": _trace(60, 30.0, "model")}},
                        "g3": {"onset": _onset(10, "model")}}}
    reviewed = json.loads(json.dumps(model))
    reviewed["labels"]["g1"]["onset"] = _onset(30, "human")                 # confirmed
    reviewed["labels"]["g1"]["traces"]["40"] = _trace(40, 21.0, "human")   # confirmed, within tolerance
    reviewed["labels"]["g2"]["onset"] = _onset(55, "human")                 # moved
    reviewed["labels"]["g2"]["traces"]["60"] = _trace(60, 45.0, "human")   # longer
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
