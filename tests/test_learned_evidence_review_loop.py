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
