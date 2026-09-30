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


def test_reviewed_curve_goes_through_checked_lengths_shaped_like_the_model():
    from sparsetrack.review import reviewed_curve

    model = np.r_[np.zeros(5), np.linspace(2, 40, 20), np.full(5, 40.0)]  # grows from bin 5, stops at bin 24
    same = reviewed_curve(model, 5, [])
    assert np.allclose(same, np.maximum.accumulate(model))  # nothing checked: the model's curve
    L = reviewed_curve(model, 5, [(14, 30.0), (29, 60.0)])  # the person read longer tubes
    assert L[14] == pytest.approx(30.0) and L[29] == pytest.approx(60.0) and L[4] == 0.0
    assert np.all(np.diff(L) >= 0) and L[26] == pytest.approx(60.0)  # the model stopped at 24: so does the curve
    later = reviewed_curve(model, 12, [(20, 10.0)])  # onset moved later: nothing before it
    assert later[:12].max() == 0.0 and later[20] == pytest.approx(10.0)
    assert np.all(reviewed_curve(model, None, [(20, 10.0)]) == 0.0)  # no germination


def test_export_writes_the_reviewed_growth_curves(reviewed):
    cache, pred, out = reviewed
    prefill(cache, pred, out, log=lambda *a: None)
    bench = Bench(cache, out, annotator="reviewer")
    plan = bench.state()["trace_plan"]["g001"]
    pts = bench.state()["labels"]["g001"]["traces"][str(plan[-1])]["path_xy_view"]
    bench.set_trace("g001", {"bin": plan[-1], "state": "full", "points": pts[:-1] + [[pts[-1][0], pts[-1][1] + 20]]})
    fixed = bench.state()["labels"]["g001"]["traces"][str(plan[-1])]["length_px"]
    export(out, log=lambda *a: None)
    rows = [r for r in csv.DictReader(open(out.parent / "reviewed_growth.csv")) if r["grain"] == "g001"]
    L = {int(r["bin"]): float(r["length_px"]) for r in rows}
    assert L[plan[-1]] == pytest.approx(fixed, abs=0.01) and L[9] == 0.0  # through the fix; nothing before onset
    assert rows[0]["checked_lengths"] == "1" and (out.parent / "growth_curves.png").exists()


def test_trace_confidence_prefers_long_growing_readings_and_orders_the_review(reviewed):
    from sparsetrack.review import trace_confidence

    growing = np.r_[np.zeros(10), np.arange(1, 31) * 2.0]      # still growing at the end
    stalled = np.r_[np.zeros(10), np.arange(1, 6) * 2.0, np.full(25, 10.0)]  # stopped 25 bins ago
    assert trace_confidence(growing, 39) > trace_confidence(growing, 12)   # a longer reading, more often right
    assert trace_confidence(growing, 39) > trace_confidence(stalled, 39)   # a long stall is suspect
    cache, pred, out = reviewed
    prefill(cache, pred, out, log=lambda *a: None)
    doc = json.loads(out.read_text())
    assert set(doc["prefill"]["confidence"]) == {"g001"}
    assert all(0 < t["model_confidence"] < 1 for t in doc["labels"]["g001"]["traces"].values())
    assert Bench(cache, out, annotator="reviewer").order()[0] == "g001"  # grains with traces, least sure first


def test_the_gallery_opens_with_the_least_sure_grain(tmp_path):
    from sparsetrack.report import grain_confidence, write_gallery

    frames = [b * FPB + FPB // 2 for b in range(N_BINS)]
    growing = {"id": "g001", "status": "emerged_within", "onset_frame": frames[5], "onset_interval": [frames[4], frames[5]],
               "length": {"frames": frames, "px": [0.0] * 5 + [2.0 * k for k in range(1, N_BINS - 4)]}, "flags": []}
    stalled = {"id": "g002", "status": "emerged_within", "onset_frame": frames[5], "onset_interval": [frames[4], frames[5]],
               "length": {"frames": frames, "px": [0.0] * 5 + [3.0] * (N_BINS - 5)}, "flags": []}
    none = {"id": "g003", "status": "no_emergence_by_end", "length": {"frames": frames, "px": [0.0] * N_BINS}, "flags": []}
    assert grain_confidence(stalled, FPB) < grain_confidence(growing, FPB) and grain_confidence(none, FPB) is None
    page = write_gallery({"grains": [growing, none, stalled], "frames_per_bin": FPB}, tmp_path).read_text()
    assert page.index("id='g002'") < page.index("id='g001'") < page.index("id='g003'")
    assert "model confidence" in page


def test_growth_rate_is_the_slope_between_a_tenth_and_nine_tenths_of_the_final_length():
    from sparsetrack.report import growth_rate, summary_line

    frames = np.arange(40) * FPB + FPB // 2
    px = np.r_[np.zeros(10), np.arange(1, 21) * 1.5, np.full(10, 30.0)]  # 1.5 px per bin, then stops
    assert growth_rate(frames, px) == pytest.approx(1.5 / FPB, rel=0.05)
    assert growth_rate(frames, np.r_[np.zeros(30), np.full(10, 5.0)]) is None  # shorter than a tube
    g = {"id": "g1", "status": "emerged_within", "length": {"frames": frames.tolist(), "px": px.tolist()}}
    line = summary_line({"grains": [g], "frames_per_bin": FPB}, None, None, (0.5, 10.0))
    assert "1.50 px per bin" in line and "um/min" in line


def test_calibration_comes_from_the_command_line_or_calibration_json(tmp_path, monkeypatch):
    from sparsetrack import cli

    monkeypatch.chdir(tmp_path)
    assert cli.calibration() is None
    (tmp_path / "calibration.json").write_text('{"um_per_px": 0.65, "s_per_frame": 30}')
    assert cli.calibration() == (0.65, 30.0)
    assert cli.calibration(0.5, None) == (0.5, 30.0)  # what is given wins


def test_a_prefilled_trace_keeps_the_models_whole_route_to_slide_along(reviewed):
    cache, pred, out = reviewed
    prefill(cache, pred, out, log=lambda *a: None)
    tr = json.loads(out.read_text())["labels"]["g001"]["traces"]
    first = tr[min(tr, key=int)]
    route = np.asarray(first["model_path"])
    arc = float(np.sum(np.hypot(*np.diff(route, axis=0).T)))
    assert arc > first["length_px"] + 20  # the whole route (75 px + a little), not the trace cut at 2-40 px
    assert np.allclose(route[0], first["path_xy_view"][0])  # the same start as the trace


def test_the_gallery_warns_when_many_grains_could_not_be_followed(tmp_path):
    from sparsetrack.report import movie_warnings, write_gallery

    frames = [b * FPB + FPB // 2 for b in range(N_BINS)]
    grains = [{"id": f"g{k:03d}", "status": "no_emergence_by_end", "length": {"frames": frames, "px": [0.0] * N_BINS},
               "flags": ["reader:flood", "drift_rejected"] if k < 4 else []} for k in range(10)]
    assert movie_warnings({"grains": grains}) and "4 of 10 grains" in movie_warnings({"grains": grains})[0]
    assert not movie_warnings({"grains": grains[2:]})  # 2 of 8: no warning
    page = write_gallery({"grains": grains, "frames_per_bin": FPB}, tmp_path).read_text()
    assert "class='warn'" in page and "could not be followed" in page


def test_census_warns_when_the_grains_are_not_the_expected_size():
    from sparsetrack.cli import census_warnings
    assert census_warnings([]) and "no grains" in census_warnings([])[0]
    assert not census_warnings([{"r": 12.0 + 0.2 * k} for k in range(10)])
    assert census_warnings([{"r": 17.8} for _ in range(8)] + [{"r": 12.0}])  # most at the top of the range


def test_the_growth_view_shows_what_changed_over_the_last_bins():
    from sparsetrack.render import GROWTH_LAG, Renderer
    n = 30
    bins = np.full((n, 40, 40), 150.0, np.float32)
    bins[12:, 20, 20] = 100.0  # new dark material at bin 12, there from then on
    meta = {"shifts": [[0.0, 0.0]] * n, "n_bins": n, "frames_per_bin": 300}
    R = Renderer(bins, meta)
    at = lambda b: R.growth_crop(b, b, 20.0, 20.0, 10)[10, 10]  # crop pixel (10, 10) = reference pixel (20, 20)
    assert at(12) == -50.0 and at(12 + GROWTH_LAG - 1) == -50.0  # while the lag still reaches back before bin 12
    assert at(12 + GROWTH_LAG) == 0.0 and at(5) == 0.0  # still material cancels; nothing new before bin 12
    assert R.contrast("k", 20.0, 20.0, 10, "g") == (-30.0, 30.0)


def test_summary_puts_movies_side_by_side(tmp_path):
    from sparsetrack.summary import write_summary
    fpb, nb = 300, 40
    frames = [b * fpb + fpb // 2 for b in range(nb)]
    folders = []
    for name, first in (("early", 5), ("late", 20)):  # onsets from bin `first` on, one grain per bin; growth 0.5 px/bin
        f = tmp_path / name
        (f / "analysis").mkdir(parents=True)
        (f / "cache").mkdir()
        grains = []
        for k in range(8):
            fv = first + k if k < 6 else None  # 6 of 8 germinate
            px = [0.0 if fv is None or b < fv else 0.5 * (b - fv) for b in range(nb)]
            grains.append({"id": f"g{k}", "status": "emerged_within" if fv is not None else "no_emergence_by_end",
                           "onset_interval": [frames[fv - 1], frames[fv]] if fv is not None else None,
                           "length": {"frames": frames, "px": px}, "flags": []})
        (f / "analysis" / "predictions.json").write_text(json.dumps({"frames_per_bin": fpb, "grains": grains}))
        (f / "cache" / "grains.json").write_text(json.dumps({"grains": [{"id": g["id"], "isolated": True} for g in grains]}))
        folders.append(f)
    rows = write_summary(folders, tmp_path / "summary", units=(0.5, 2.0), log=lambda *a: None)
    early, late = rows
    assert early["grains"] == late["grains"] == 8 and abs(early["germinated"] - 0.75) < 1e-6
    assert late["t50_frame"] - early["t50_frame"] == 15 * fpb  # the late movie's onsets are 15 bins later
    assert abs(early["growth_px_per_bin"] - 0.5) < 0.05
    assert abs(early["growth_um_per_min"] - 0.5 * 0.5 / (fpb * 2.0 / 60.0)) < 0.01  # px/bin -> um/min
    assert (tmp_path / "summary" / "summary.png").exists()
    text = (tmp_path / "summary" / "summary.csv").read_text()
    assert text.splitlines()[0].startswith("movie,grains,germinated") and "late" in text


def test_a_refocus_is_found_and_warned_about():
    import cv2
    from sparsetrack.analyze import focus_changes
    from sparsetrack.report import movie_warnings
    rng = np.random.default_rng(0)
    sharp = cv2.GaussianBlur(rng.normal(150, 25, (120, 160)).astype(np.float32), (0, 0), 0.8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3.0)
    nb, fpb = 60, 300
    bins = np.stack([blurred if b < 30 else sharp for b in range(nb)]) + rng.normal(0, 0.5, (nb, 120, 160)).astype(np.float32)
    meta = {"n_bins": nb, "frames_per_bin": fpb}
    found = focus_changes(bins, meta)
    assert len(found) == 1 and abs(found[0]["bin"] - 30) <= 1 and found[0]["ratio"] > 2
    assert focus_changes(np.stack([sharp] * nb), meta) == []  # steady focus: nothing
    w = movie_warnings({"grains": [], "focus_changes": found})
    assert len(w) == 1 and "focus changed" in w[0] and "sharper" in w[0]


def test_an_excluded_sample_grain_is_replaced_from_a_fixed_reserve(tmp_path):
    grains = [{"id": f"g{k:03d}", "x": 20.0 + 15 * k, "y": 40.0, "r": 5.0, "isolated": True, "border": False}
              for k in range(8)]
    _cache(tmp_path / "c", grains)
    bench = Bench(tmp_path / "c", tmp_path / "labels.json", sample=4)
    sample = list(bench.doc["sample"]["grains"])
    in_sample = lambda: [g for g in bench.doc["sample"]["grains"] if not bench.doc["grains"][g].get("excluded")]
    out = bench.set_exclusion(sample[0], {"excluded": True, "reason": "not_a_grain"})
    first = out["replaced_by"]
    assert first and first not in sample and len(in_sample()) == 4
    assert bench.doc["sample"]["replacements"][0] == {**bench.doc["sample"]["replacements"][0], "out": sample[0],
                                                      "in": first, "reason": "not_a_grain"}
    # the next exclusion takes the next grain of the same fixed reserve
    second = bench.set_exclusion(sample[1], {"excluded": True, "reason": "clump"})["replaced_by"]
    reserve = bench.doc["sample"]["reserve"]
    assert [first, second] == reserve[:2]
    # a grain outside the sample brings nobody in; including a grain again and excluding it again does not overfill
    outside = next(g for g in bench.doc["grains"] if g not in bench.doc["sample"]["grains"])
    assert bench.set_exclusion(outside, {"excluded": True, "reason": "not_a_grain"})["replaced_by"] is None
    bench.set_exclusion(sample[0], {"excluded": False})
    assert len(in_sample()) == 5
    assert bench.set_exclusion(sample[0], {"excluded": True, "reason": "not_a_grain"})["replaced_by"] is None
    # kept in the labels file: reopening gives the same reserve and the recorded swaps
    again = Bench(tmp_path / "c", tmp_path / "labels.json")
    assert again.doc["sample"]["reserve"] == reserve and len(again.doc["sample"]["replacements"]) == 2
