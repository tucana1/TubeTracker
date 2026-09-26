"""Label-free self-training (experiment): which readings become pseudo-traces, the check on labels, and the whole
step on a small movie."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from sparsetrack import stack  # noqa: E402

from prototypes.learned_evidence import selftrain as ST  # noqa: E402


def _reading(T=40, fpb=300, flags=(), status="emerged_within", unsteady=False):
    """A grain growing 1.5 px per bin from bin 10 to 30 px, read exactly except at bin 20 (10 px off the curve)."""
    fit = np.clip((np.arange(T) - 10) * 1.5, 0.0, 30.0)
    raw = fit.copy()
    raw[20] += 10.0
    frames = [b * fpb + fpb // 2 for b in range(T)]
    paths = {str(i): [[100.0, 60.0 + d] for d in np.linspace(0.0, max(fit[i], 1.0) + 1.0, 8)] for i in range(T)}
    return {"id": "g001", "status": status, "final_length_px": float(fit[-1]), "raw_reach_px": raw.tolist(),
            "length": {"frames": frames, "px": fit.tolist()}, "_paths": paths,
            "flags": list(flags) + (["unsteady:12/30"] if unsteady else [])}


def test_pseudo_traces_are_readings_that_agree_with_the_growth_curve():
    meta = {"n_bins": 40, "ref_start": 0}
    census = [{"id": "g001", "x": 100.0, "y": 50.0, "r": 10.0}]
    doc, n = ST.pseudo_labels([_reading(flags=["burst_after:10650", "gone:7650-8250"])], census, meta)
    bins = sorted(int(b) for b in doc["labels"]["g001"]["traces"])
    assert n == {"grains": 1, "traces": len(bins)} and bins
    assert 20 not in bins and not set(bins) & {25, 26, 27}  # off the curve; the grain away (frames 7650-8250)
    assert max(bins) <= 35 and all(b - a >= 6 for a, b in zip(bins, bins[1:]))  # before the burst, spaced
    for b in bins:  # each stops end_px + apex_margin short of the fitted length
        pts = np.asarray(doc["labels"]["g001"]["traces"][str(b)]["path_xy_ref"])
        fit = min(max((b - 10) * 1.5, 0.0), 30.0)
        assert abs(float(np.hypot(*np.diff(pts, axis=0).T).sum()) - max(2.0, fit - 2.0)) <= 0.05
    onset = doc["labels"]["g001"]["onset"]  # 8 bins before the decoded onset (first bin at 2 px: 12)
    assert onset == {"verdict": "emerged_within", "last_absent_bin": 12 - 1 - 8}
    for bad in (_reading(unsteady=True), _reading(status="no_emergence_by_end")):
        assert ST.pseudo_labels([bad], census, meta)[1]["grains"] == 0


def _labels_and_preds(n=6, fpb=300, T=20):
    grains, labels, start, tuned = {}, {}, [], []
    frames = [b * fpb + fpb // 2 for b in range(T)]
    for k in range(n):
        gid, x, y = f"g{k + 1:03d}", 60.0 + 80.0 * k, 60.0
        grains[gid] = {"id": gid, "x": x, "y": y, "r": 10.0}
        truth = np.clip((np.arange(T) - 5) * 3.0, 0.0, None)
        labels[gid] = {"onset": {"verdict": "emerged_within", "last_absent_frame": frames[5], "first_visible_frame": frames[6]},
                       "traces": {str(b): {"bin": b, "state": "full", "length_px": float(truth[b])} for b in (8, 12, 16)}}
        for out, scale in ((start, 1.3), (tuned, 1.0)):  # the start reads 30% long, the tuned model right
            out.append({"id": gid, "x": x, "y": y, "status": "emerged_within", "onset_frame": frames[6],
                        "length": {"frames": frames, "px": (truth * scale).tolist()}})
    doc = {"grains": grains, "labels": labels, "frames_per_bin": fpb, "n_bins": T}
    return doc, {"grains": start}, {"grains": tuned}


def test_the_check_on_labels_scores_both_readings_and_says_whether_it_would_be_adopted(tmp_path):
    doc, start, tuned = _labels_and_preds()
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(doc))
    lines = ST.check(start, tuned, path)
    assert "6 grains scored" in lines[0] and lines[-1].startswith("would be adopted")
    assert ST.check(tuned, start, path)[-1].startswith("not adopted")


def test_the_whole_step_on_a_small_movie(tmp_path):
    """A grain whose tube grows 3 px per bin from bin 5: first pass, pseudo-traces, image traces, a short tuning."""
    size, T = 240, 30
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.repeat(np.where(np.hypot(xx - 100.0, yy - 50.0) < 9.0, 120.0, 180.0)[None], T, axis=0)
    prob = np.zeros((T, size, size))
    for b in range(5, T):
        end = min(60 + 3 * (b - 4), 200)
        img[b, 60:end, 99:102] = 150.0
        prob[b, 57:end, 99:102] = 16.0
    grains = [{"id": "g001", "x": 100.5, "y": 50.5, "r": 9.0, "isolated": True}]
    for path, bins in ((tmp_path / "cache", img), (tmp_path / "prob", prob)):
        path.mkdir()
        np.save(path / "bins.npy", bins.astype(np.float16))
        (path / "meta.json").write_text(json.dumps({
            "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": T, "shifts": [[0.0, 0.0]] * T,
            "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": 300 * T, "width": size, "height": size}}))
        (path / "grains.json").write_text(json.dumps({"grains": grains}))
    model = ST.adapt(tmp_path / "cache", tmp_path / "prob", tmp_path / "work", dec={"burst": True, "vmax": 4.0},
                     steps=2, log=lambda *a: None)
    labels = json.loads((tmp_path / "work" / "pseudo_labels.json").read_text())
    traces = labels["labels"]["g001"]["traces"]
    assert model.exists() and len(traces) >= 3 and all(t["state"] == "full" for t in traces.values())
    last = traces[str(max(int(b) for b in traces))]
    assert np.abs(np.asarray(last["path_xy_ref"])[:, 0] - 100.5).max() <= 1.5  # along the drawn tube
