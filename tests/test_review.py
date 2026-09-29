"""The review loop on SparseTrack's own predictions: prefill -> the labelling tool -> a reviewer -> export."""

import csv
import json

import numpy as np
import pytest

from sparsetrack import stack
from sparsetrack.bench.server import Bench, trace_bins
from sparsetrack.review import export, prefill, to_length

SIZE, N_BINS, FPB = 200, 40, 300


def _cache(path, grains):
    path.mkdir()
    rng = np.random.default_rng(0)
    np.save(path / "bins.npy", (150 + 5 * rng.standard_normal((N_BINS, SIZE, SIZE))).astype(np.float16))
    (path / "meta.json").write_text(json.dumps({
        "schema": stack.SCHEMA, "frames_per_bin": FPB, "n_bins": N_BINS, "shifts": [[0.0, 0.0]] * N_BINS,
        "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": FPB * N_BINS, "width": SIZE, "height": SIZE}}))
    (path / "grains.json").write_text(json.dumps({"grains": grains}))


def _pred(onset_bin=10, rate=2.0):
    """SparseTrack-shaped predictions: g001 grows 2 px per bin straight down from bin 10; g002 never germinates."""
    frames = [b * FPB + FPB // 2 for b in range(N_BINS)]
    px = [max(0.0, rate * (b - onset_bin + 1)) if b >= onset_bin else 0.0 for b in range(N_BINS)]
    path = [[60.0, 60.0 + 10.0 + k] for k in range(0, 80, 5)]  # from the rim (r = 10) straight down
    g1 = {"id": "g001", "x": 60.0, "y": 60.0, "r": 10.0, "flags": ["reader:flood"], "status": "emerged_within",
          "onset_frame": frames[onset_bin], "onset_interval": [frames[onset_bin - 1], frames[onset_bin]],
          "length": {"frames": frames, "px": px}, "path": path, "exit_xy": path[0], "rotation_deg": [0.0] * N_BINS,
          "final_length_px": px[-1]}
    g2 = {"id": "g002", "x": 140.0, "y": 140.0, "r": 10.0, "flags": [], "status": "no_emergence_by_end",
          "onset_frame": None, "onset_interval": None, "length": {"frames": frames, "px": [0.0] * N_BINS}, "path": []}
    return {"method": "SparseTrack test", "params": {"rot_pivot": "exit"}, "frames_per_bin": FPB, "grains": [g1, g2]}


@pytest.fixture
def reviewed(tmp_path):
    grains = [{"id": "g001", "x": 60.0, "y": 60.0, "r": 10.0, "isolated": True},
              {"id": "g002", "x": 140.0, "y": 140.0, "r": 10.0, "isolated": True}]
    cache = tmp_path / "cache"
    _cache(cache, grains)
    pred = tmp_path / "predictions.json"
    pred.write_text(json.dumps(_pred()))
    return cache, pred, tmp_path / "review" / "review_labels.json"


def test_to_length_cuts_and_extends_at_most_a_little():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    cut = to_length(pts, 15.0)
    assert np.allclose(cut[-1], [10.0, 5.0]) and len(cut) == 3
    longer = to_length(pts, 40.0)  # 20 px of path: carried on by at most 5 px
    assert np.allclose(longer[-1], [10.0, 15.0])


def test_prefill_holds_the_models_answers_in_the_tools_format(reviewed):
    cache, pred, out = reviewed
    prefill(cache, pred, out, log=lambda *a: None)
    doc = json.loads(out.read_text())
    assert out.with_suffix(".model.json").exists() and doc["prefill"]["check_first"] == {"g001": ["reader:flood"]}
    on = doc["labels"]["g001"]["onset"]
    assert (on["verdict"], on["first_visible_bin"], on["review_origin"]) == ("emerged_within", 10, "model")
    assert doc["labels"]["g002"]["onset"]["verdict"] == "no_emergence_by_end"
    plan = trace_bins(10, N_BINS)
    traces = doc["labels"]["g001"]["traces"]
    assert sorted(map(int, traces)) == plan
    for b in plan:  # the model's path cut to the length it read there, from the rim
        t = traces[str(b)]
        want = 2.0 * (b - 10 + 1)
        assert t["state"] == "full" and abs(t["length_px"] - want) < 0.05 and t["review_origin"] == "model"
        assert t["path_xy_view"][0] == [60.0, 70.0]
    with pytest.raises(FileExistsError):  # never over a review that may hold a person's answers
        prefill(cache, pred, out, log=lambda *a: None)


def test_review_asks_until_checked_and_exports_what_was_checked(reviewed):
    cache, pred, out = reviewed
    prefill(cache, pred, out, log=lambda *a: None)
    bench = Bench(cache, out, annotator="reviewer")
    st = bench.state()
    assert st["review"]["model"].startswith("SparseTrack")
    assert st["progress"]["onset_checked"] == 0 and st["progress"]["traces_checked"] == 0
    plan = st["trace_plan"]["g001"]
    first, last = (st["labels"]["g001"]["traces"][str(b)] for b in (plan[0], plan[-1]))
    # the reviewer confirms the first trace as it stands (Enter) and moves the last one's apex 20 px on
    bench.set_trace("g001", {"bin": plan[0], "state": "full", "points": first["path_xy_view"]})
    pts = last["path_xy_view"]
    bench.set_trace("g001", {"bin": plan[-1], "state": "full", "points": pts[:-1] + [[pts[-1][0], pts[-1][1] + 20]]})
    bench.set_onset("g002", {"verdict": "no_emergence_by_end"})
    st = bench.state()
    assert st["progress"]["traces_checked"] == 2 and st["progress"]["onset_checked"] == 1
    export(out, um_per_px=0.5, log=lambda *a: None)
    rows = {int(r["bin"]): r for r in csv.DictReader(open(out.parent / "reviewed_traces.csv"))}
    assert (rows[plan[0]]["checked"], rows[plan[0]]["changed"]) == ("yes", "no")
    assert (rows[plan[-1]]["checked"], rows[plan[-1]]["changed"]) == ("yes", "yes")
    assert all(rows[b]["checked"] == "no" for b in plan[1:-1])
    grains = {r["grain"]: r for r in csv.DictReader(open(out.parent / "reviewed_grains.csv"))}
    assert grains["g001"]["traces_checked"] == f"2/{len(plan)}" and grains["g001"]["onset_checked"] == "no"
    assert grains["g002"]["onset_checked"] == "yes" and grains["g002"]["onset_changed"] == "no"
    assert float(grains["g001"]["last_length_um"]) == pytest.approx(0.5 * float(grains["g001"]["last_length_px"]))
    assert (out.parent / "population.png").exists()


def test_an_ordinary_labels_file_is_not_a_review(tmp_path):
    cache = tmp_path / "cache"
    _cache(cache, [{"id": "g001", "x": 60.0, "y": 60.0, "r": 10.0, "isolated": True}])
    bench = Bench(cache, tmp_path / "labels.json")
    bench.set_onset("g001", {"verdict": "no_emergence_by_end"})
    st = bench.state()
    assert st["review"] is None and st["progress"]["onset_checked"] == 1  # a person's own answers count as checked
