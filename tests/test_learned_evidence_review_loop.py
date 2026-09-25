"""The review loop end to end against the labelling tool's own code: pipeline readings -> prefill.py ->
the tool (``sparsetrack.bench.server.Bench``) -> a reviewer's answer -> export_review.py. It fails if the
tool's API or label format changes under the pre-fill."""

import csv
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from sparsetrack import stack  # noqa: E402

SIZE, N_BINS, GX, GY, R = 240, 30, 100.5, 50.5, 9.0


def _cache(path, bins, grains):
    path.mkdir()
    np.save(path / "bins.npy", bins.astype(np.float16))
    (path / "meta.json").write_text(json.dumps({
        "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": N_BINS, "shifts": [[0.0, 0.0]] * N_BINS,
        "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": 300 * N_BINS, "width": SIZE, "height": SIZE}}))
    (path / "grains.json").write_text(json.dumps({"grains": grains}))


@pytest.fixture
def analysed(tmp_path):
    """A grain whose tube grows 3 px per bin from bin 5 (straight down), analysed as pipeline.py would."""
    from prototypes.learned_evidence import reach

    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    img = np.repeat(np.where(np.hypot(xx - 100.0, yy - 50.0) < R, 120.0, 180.0)[None], N_BINS, axis=0)
    prob = np.zeros((N_BINS, SIZE, SIZE))
    for b in range(5, N_BINS):
        end = min(60 + 3 * (b - 4), 200)
        img[b, 60:end, 99:102] = 150.0
        prob[b, 57:end, 99:102] = 16.0  # P = 1 (x16), from the grain's ring
    grains = [{"id": "g001", "x": GX, "y": GY, "r": R, "isolated": True}]
    field, work = tmp_path / "cache", tmp_path / "work"
    _cache(field, img, grains)
    work.mkdir()
    _cache(work / "prob_cache", prob, grains)
    dec = {"burst": True, "vmax": 4.0}
    pred = reach.analyze(work / "prob_cache", field, log=lambda *a: None, **dec)
    (work / "perbin").mkdir()
    (work / "perbin" / "predictions.json").write_text(json.dumps({**pred, "decoder": dec}))
    return field, work


def test_prefill_review_export(analysed):
    from sparsetrack.bench.server import Bench, trace_bins

    from prototypes.learned_evidence import export_review, prefill

    field, work = analysed
    out = prefill.main(["--field", str(field), "--work", str(work)])
    assert out.with_suffix(".model.json").exists() and not (work / "review_labels.building.json").exists()
    bench = Bench(field, out, annotator="reviewer")  # the tool opens it
    st = bench.state()
    lab = st["labels"]["g001"]
    assert lab["onset"]["verdict"] == "emerged_within" and lab["onset"]["review_origin"] == "model"
    plan = trace_bins(lab["onset"]["first_visible_bin"], N_BINS)
    assert st["progress"]["traces_done"] == st["progress"]["traces_needed"] == len(plan)
    last = lab["traces"][str(plan[-1])]
    path = np.asarray(last["path_xy_ref"])
    assert last["state"] == "full" and np.abs(path[:, 0] - GX).max() <= 1.0 and abs(path[0, 1] - (GY + R)) <= 1.5
    # the reviewer moves the apex of one trace 20 px on; the tool marks it theirs
    first = lab["traces"][str(plan[0])]
    pts = first["path_xy_view"]
    bench.set_trace("g001", {"bin": plan[0], "state": "full", "points": pts[:-1] + [[pts[-1][0], pts[-1][1] + 20]]})
    export_review.main(["--labels", str(out)])
    rows = list(csv.DictReader(open(work / "reviewed_traces.csv")))
    assert [(r["checked"], r["changed"]) for r in rows] == [("yes", "yes")] + [("no", "")] * (len(plan) - 1)
    grains = list(csv.DictReader(open(work / "reviewed_grains.csv")))
    assert grains[0]["onset_checked"] == "no" and grains[0]["traces_checked"] == f"1/{len(plan)}"


def test_prefill_reads_with_the_recorded_onset_length(analysed):
    from prototypes.learned_evidence import prefill

    field, work = analysed
    first = {}
    for onset_px in (None, 8.0):
        pred_path = work / "perbin" / "predictions.json"
        pred = json.loads(pred_path.read_text())
        pred["decoder"] = {"burst": True, "vmax": 4.0, **({"onset_px": onset_px} if onset_px else {})}
        pred_path.write_text(json.dumps(pred))
        out = prefill.main(["--field", str(field), "--work", str(work),
                            "--out", str(work / f"review_{onset_px}.json")])
        first[onset_px] = json.loads(out.read_text())["labels"]["g001"]["onset"]["first_visible_bin"]
    assert first[8.0] > first[None]  # a longer onset length calls the onset later, as calibration adopted it


def test_a_disc_judged_not_a_grain_is_excluded_for_review_and_can_come_back(tmp_path, capsys):
    """Six sharp grains and an out-of-focus ghost disc: the analysis leaves the ghost out, the pre-fill excludes it
    in the tool (model's call, listed to check first), the export says so, and the reviewer can include it again."""
    import cv2
    from sparsetrack.bench.server import Bench

    from prototypes.learned_evidence import export_review, prefill, reach

    size, T = 400, 30
    pos = [(80.5, 80.5), (200.5, 80.5), (320.5, 80.5), (80.5, 250.5), (200.5, 250.5), (320.5, 250.5), (200.5, 350.5)]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    im = np.full((size, size), 180.0)
    for k, (x, y) in enumerate(pos):
        d = np.hypot(xx - x, yy - y)
        disc = np.where(d < 12, -60.0, 0.0) + np.where((d >= 10) & (d < 13), -40.0, 0.0)
        im += cv2.GaussianBlur(disc.astype(np.float32), (0, 0), 5.0) * 0.5 if k == 6 else disc
    rng = np.random.default_rng(0)
    img = np.stack([im + rng.normal(0, 2.0, im.shape) for _ in range(T)])
    grains = [{"id": f"g{k + 1:03d}", "x": x, "y": y, "r": 12.0, "isolated": True} for k, (x, y) in enumerate(pos)]
    field, work = tmp_path / "cache", tmp_path / "work"
    for path, bins in ((field, img), (work / "prob_cache", np.zeros_like(img))):
        path.mkdir(parents=True)
        np.save(path / "bins.npy", bins.astype(np.float16))
        (path / "meta.json").write_text(json.dumps({
            "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": T, "shifts": [[0.0, 0.0]] * T,
            "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": 300 * T, "width": size, "height": size}}))
        (path / "grains.json").write_text(json.dumps({"grains": grains}))
    dec = {"burst": True, "vmax": 4.0}
    pred = reach.analyze(work / "prob_cache", field, log=lambda *a: None, **dec)
    assert set(pred["ghosts"]) == {"g007"} and len(pred["grains"]) == 6
    (work / "perbin").mkdir()
    (work / "perbin" / "predictions.json").write_text(json.dumps({**pred, "decoder": dec}))
    out = prefill.main(["--field", str(field), "--work", str(work)])
    doc = json.loads(out.read_text())
    g7 = doc["grains"]["g007"]
    assert g7["excluded"] and g7["exclude_reason"] == "not_a_grain" and g7["exclude_origin"] == "model"
    assert set(doc["prefill"]["check_first"]) == {"g007"}
    bench = Bench(field, out, annotator="reviewer")
    assert bench.state()["progress"]["grains"] == 6  # the tool does not ask for the ghost
    capsys.readouterr()
    export_review.main(["--labels", str(out)])
    assert "judged not grains are left out (g007)" in capsys.readouterr().out
    grains_csv = list(csv.DictReader(open(work / "reviewed_grains.csv")))
    assert "g007" not in {r["grain"] for r in grains_csv} and len(grains_csv) == 6
    bench.set_exclusion("g007", {"excluded": False})  # the reviewer disagrees: it is a grain
    assert bench.state()["progress"]["grains"] == 7
