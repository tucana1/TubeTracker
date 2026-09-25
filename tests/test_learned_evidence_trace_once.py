"""Trace once: the test is every anchored grain's traces before its anchor, and a cache the model built is reused."""

import json

import pytest

pytest.importorskip("torch")

from prototypes.learned_evidence import trace_once  # noqa: E402
from prototypes.learned_evidence.prefix import anchors_from_labels  # noqa: E402


def _full(b, L):
    return {"bin": b, "state": "full", "length_px": L, "path_xy_ref": [[0.0, 0.0], [L, 0.0]]}


LABELS = {"grains": {"g1": {"x": 1}, "g2": {"x": 2}, "g3": {"x": 3}},
          "labels": {"g1": {"onset": {"verdict": "emerged_within"},
                            "traces": {"36": _full(36, 7.0), "70": _full(70, 16.0), "122": _full(122, 30.0),
                                       "174": {"bin": 174, "state": "burst"}}},
                     "g2": {"onset": {"verdict": "no_emergence_by_end"},
                            "traces": {"50": {"bin": 50, "state": "no_tube", "length_px": 0.0}}},
                     "g3": {"onset": {"verdict": "emerged_within"}, "traces": {"40": _full(40, 9.0)}}}}


def test_the_test_is_the_traces_before_each_anchor():
    anchors = anchors_from_labels(LABELS)
    assert {g: a["bin"] for g, a in anchors.items()} == {"g1": 122, "g3": 40}
    test = trace_once.leave_anchor_out(LABELS, anchors)
    assert set(test["grains"]) == {"g1", "g3"}
    assert sorted(test["labels"]["g1"]["traces"]) == ["36", "70"] and test["labels"]["g3"]["traces"] == {}
    assert test["labels"]["g1"]["onset"] == LABELS["labels"]["g1"]["onset"]
    assert sorted(LABELS["labels"]["g1"]["traces"]) == ["122", "174", "36", "70"]  # the labels are left as they were


def test_a_cache_the_model_built_is_found(tmp_path, monkeypatch):
    for name, sha in (("a", "other"), ("b", "mine")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "meta.json").write_text(json.dumps({"model_sha1": sha}))
    monkeypatch.setattr(trace_once.evaluate, "fingerprint", lambda net: "mine")
    assert trace_once.find_cache(None, [tmp_path / "missing", tmp_path / "a", tmp_path / "b"]) == tmp_path / "b"
    monkeypatch.setattr(trace_once.evaluate, "fingerprint", lambda net: "new")
    assert trace_once.find_cache(None, [tmp_path / "a", tmp_path / "b"]) is None


def test_movie_2_is_refused():
    with pytest.raises(SystemExit, match="held-out"):
        trace_once.main(["--labels", "benchmark/labels/m2_v1.json"])
