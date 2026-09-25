"""Fine-tuning on human traces: which pixels a trace supervises, and the grain folds."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from sparsetrack import stack  # noqa: E402

SIZE, N_BINS, LINE_COL, NEIGHBOUR_COL, THICK_COL = 240, 8, 100, 111, 167


@pytest.fixture
def movie(tmp_path):
    """A thin dark tube (frame column 100, rows 60-150) from bin 4 on, a second one 11 px beside it, and a
    thick one (15 px wide) further off."""
    bins = np.full((N_BINS, SIZE, SIZE), 180.0)
    bins[4:, 60:151, LINE_COL] = 120.0
    bins[4:, 60:151, NEIGHBOUR_COL] = 120.0
    bins[4:, 60:151, THICK_COL - 7:THICK_COL + 8] = 120.0
    np.save(tmp_path / "bins.npy", bins.astype(np.float16))
    (tmp_path / "meta.json").write_text(json.dumps({
        "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": N_BINS, "shifts": [[0.0, 0.0]] * N_BINS,
        "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": 300 * N_BINS, "width": SIZE, "height": SIZE}}))
    (tmp_path / "grains.json").write_text(json.dumps({"grains": []}))
    # the labelling tool's reference coordinates: frame pixel i covers [i, i + 1)
    trace = {"bin": 5, "state": "full", "path_xy_ref": [[LINE_COL + 0.5, 60.5], [LINE_COL + 0.5, 150.5]]}
    labels = {
        "grains": {"g1": {"id": "g1", "x": 100.5, "y": 50.5, "r": 9.0}, "g2": {"id": "g2", "x": 40.5, "y": 190.5,
                                                                              "r": 9.0},
                   "g3": {"id": "g3", "x": 111.5, "y": 200.5, "r": 9.0}, "g4": {"id": "g4", "x": 200.5, "y": 40.5,
                                                                              "r": 9.0, "excluded": True},
                   "g5": {"id": "g5", "x": THICK_COL + 0.5, "y": 50.5, "r": 9.0}},
        "labels": {"g1": {"traces": {"5": trace}},
                   "g2": {"traces": {"5": {"bin": 5, "state": "no_tube", "path_xy_ref": []}}},
                   "g3": {"traces": {"5": {"bin": 5, "state": "full",
                                           "path_xy_ref": [[NEIGHBOUR_COL + 0.5, 150.5], [NEIGHBOUR_COL + 0.5, 60.5]]}}},
                   "g4": {"traces": {"5": trace}},
                   "g5": {"traces": {"5": {"bin": 5, "state": "full",
                                           "path_xy_ref": [[THICK_COL + 0.5, 60.5], [THICK_COL + 0.5, 150.5]]}}}}}
    return tmp_path, labels


def _crop_of(real, gid):
    return [i for i, (g, *_rest) in enumerate(real["info"]) if g == gid]


def test_trace_supervises_its_tube_and_a_background_band(movie):
    from prototypes.learned_evidence.finetune import NEG_BAND, NEG_MIN, POS_PX, real_samples

    cache, labels = movie
    real = real_samples(cache, labels, grains={"g1", "g3"}, jitter=0.0)
    i = _crop_of(real, "g1")[1]
    _, b, cx, cy = real["info"][i]
    half = real["x"].shape[-1] // 2
    j = LINE_COL - int(cx - half)  # crop column of the drawn tube
    x = real["x"][i].astype(np.float32)
    change = np.abs(x[0] - x[1]).mean(axis=0)  # the image puts the tube there too
    assert change[j] > 0 and change[j - 1] == change[j + 1] == 0
    rows = slice(20, 70)
    body, w = real["body"][i].astype(bool)[rows], real["w"][i].astype(bool)[rows]
    assert body[:, j - int(POS_PX):j + int(POS_PX) + 1].all() and not body[:, j + int(POS_PX) + 1:].any()
    assert w[:, j].all() and not w[:, j - int(NEG_MIN) + 1:j - 2].any()  # tube wall or blur: not supervised
    out = int(NEG_MIN + NEG_BAND)  # a thin tube: background from NEG_MIN out
    assert w[:, j - out:j - int(NEG_MIN)].all() and not w[:, :j - out - 1].any()
    # g3's traced tube runs 11 px to the right: background stays away from it
    assert not w[:, NEIGHBOUR_COL - int(cx - half) - 3:].any()
    # untraced, it still does not become background: it widens the tube measured beside g1's trace
    alone = real_samples(cache, labels, grains={"g1"}, jitter=0.0)
    k = [n for n in _crop_of(alone, "g1") if alone["info"][n][2:] == (cx, cy)][0]
    assert real["neg_start"] == [NEG_MIN, NEG_MIN] and alone["neg_start"] == [14.0]
    assert not alone["w"][k].astype(bool)[rows][:, NEIGHBOUR_COL - int(cx - half)].any()


def test_background_starts_past_a_thick_tubes_walls(movie):
    from prototypes.learned_evidence.finetune import NEG_BAND, real_samples

    cache, labels = movie
    real = real_samples(cache, labels, grains={"g5"}, jitter=0.0)
    assert real["neg_start"] == [10.0]  # its change reaches 7 px out: background from 10
    i = _crop_of(real, "g5")[1]
    _, _, cx, cy = real["info"][i]
    j = THICK_COL - int(cx - real["x"].shape[-1] // 2)
    w = real["w"][i].astype(bool)[20:70]
    assert not w[:, j + 3:j + 10].any() and w[:, j + 10:j + 10 + int(NEG_BAND) + 1].all()


def test_supervision_spreads_over_time(movie):
    """Before onset: background along the path. In a traced bin: background along the path past the
    trace too. Between two traces: tube up to the earlier one's apex, nothing claimed past it."""
    import copy

    from prototypes.learned_evidence.finetune import real_samples

    cache, labels = movie
    labels = copy.deepcopy(labels)
    col = LINE_COL + 0.5
    labels["labels"]["g1"] = {"onset": {"verdict": "emerged_within", "last_absent_bin": 3}, "traces": {
        "4": {"bin": 4, "state": "full", "path_xy_ref": [[col, 60.5], [col, 90.5]]},  # 30 px
        "7": {"bin": 7, "state": "full", "path_xy_ref": [[col, 60.5], [col, 150.5]]}}}  # 90 px
    real = real_samples(cache, labels, grains={"g1", "g3"}, jitter=0.0, between=1)
    assert sorted({b for g, b, *_ in real["info"] if g == "g1"}) == [3, 4, 5, 6, 7]
    half = real["x"].shape[-1] // 2

    def column(b, cy=120):  # g1's crops centred at frame (100, cy); frame rows 72..167
        (i,) = [i for i, (g, b_, cx, cy_) in enumerate(real["info"]) if g == "g1" and b_ == b and cy_ == cy]
        j = LINE_COL - (100 - half)
        return real["body"][i][:, j].astype(bool), real["w"][i][:, j].astype(bool), cy - half

    body, w, oy = column(3)  # before onset
    assert not body.any() and w[80 - oy:150 - oy].all()
    body, w, oy = column(4)  # traced at 30 px (apex at row 90)
    assert body[72 - oy:90 - oy].all() and not body[91 - oy:].any()
    assert not w[91 - oy:97 - oy].any() and w[98 - oy:150 - oy].all() and not w[155 - oy:].any()
    body, w, oy = column(5)  # between 30 and 90 px: tube to row 90, beyond it could be either
    assert body[72 - oy:90 - oy].all() and not w[91 - oy:150 - oy].any()  # past the later apex: background
    body, w, oy = column(7)
    assert body[72 - oy:150 - oy].all()
    # a held-out bin gets no crops, but its trace still bounds the bins around it
    held = real_samples(cache, labels, grains={"g1", "g3"}, jitter=0.0, between=1, exclude_bins={6, 7})
    assert sorted({b for g, b, *_ in held["info"] if g == "g1"}) == [3, 4, 5]
    same = [(i, k) for i, a in enumerate(real["info"]) for k, c in enumerate(held["info"]) if a == c and a[:2] == ("g1", 5)]
    assert same and all((real["w"][i] == held["w"][k]).all() and (real["body"][i] == held["body"][k]).all()
                        for i, k in same)


def test_tip_target_sits_on_the_apex(movie):
    from prototypes.learned_evidence.finetune import real_samples

    cache, labels = movie
    real = real_samples(cache, labels, grains={"g1"}, jitter=0.0)
    i = _crop_of(real, "g1")[-1]
    _, _, cx, cy = real["info"][i]
    half = real["x"].shape[-1] // 2
    tip = real["tip"][i].astype(np.float32)
    apex = (150 - int(cy - half), LINE_COL - int(cx - half))
    assert np.unravel_index(np.argmax(tip), tip.shape) == apex
    assert real["wt"][i].astype(bool)[tip > 0.5].all()
    body, w = real["body"][i].astype(bool), real["w"][i].astype(bool)
    assert body[apex[0] - 1, apex[1]] and not body[apex[0] + 1:].any()  # no tube past the apex...
    assert not w[apex[0] + 1:apex[0] + 7, apex[1] - 3:apex[1] + 4].any()  # ...nor background: the front is blurred


def test_no_tube_answer_makes_a_ring_background(movie):
    from prototypes.learned_evidence.finetune import real_samples

    cache, labels = movie
    real = real_samples(cache, labels, grains={"g2"}, jitter=0.0)
    (i,) = _crop_of(real, "g2")
    _, _, cx, cy = real["info"][i]
    half = real["x"].shape[-1] // 2
    yy, xx = np.mgrid[0:2 * half, 0:2 * half]
    r = np.hypot(xx - (40.5 - (cx - half) - 0.5), yy - (190.5 - (cy - half) - 0.5))
    w = real["w"][i].astype(bool)
    assert not real["body"][i].any()
    assert w[(r > 12) & (r < 23)].all() and not w[r < 10].any() and not w[r > 25].any()


def test_excluded_grains_and_movie_2_are_left_out(movie, tmp_path):
    from prototypes.learned_evidence.finetune import main, real_samples

    cache, labels = movie
    assert {g for g, *_ in real_samples(cache, labels)["info"]} == {"g1", "g2", "g3", "g5"}
    m2 = tmp_path / "m2_v1.json"
    m2.write_text(json.dumps(labels))
    with pytest.raises(SystemExit, match="held-out"):
        main(["--field", str(cache), "--labels", str(m2), "--work", str(tmp_path / "w")])


def test_folds_split_grains_and_traced_bins():
    from prototypes.learned_evidence.finetune import assign_folds, fold_labels

    full = {"state": "full", "path_xy_ref": [[0, 0], [1, 1]]}
    labels = {"labels": {f"g{i:03d}": {"onset": {"verdict": "emerged_within"},
                                      "traces": {"70": full, "122": full, "174": full, str(20 + i): full}}
                         for i in range(31)}}
    grain, bins = assign_folds(labels, 3)
    assert sorted(np.bincount(list(grain.values()))) == [10, 10, 11]
    assert sorted(bins[b] for b in (70, 122, 174)) == [0, 1, 2]  # the three late bins in three folds
    per_fold = np.bincount([bins[int(k)] for lab in labels["labels"].values() for k in lab["traces"]])
    assert per_fold.max() - per_fold.min() <= 1  # the scattered early bins fill up the lighter folds
    assert (grain, bins) == assign_folds(labels, 3)
    scored = fold_labels(labels, grain, bins)
    for gid, lab in scored["labels"].items():
        assert lab["onset"] and all(bins[int(k)] == grain[gid] for k in lab["traces"])
    assert sum(len(lab["traces"]) for lab in scored["labels"].values()) >= 31  # each grain keeps its fold's late bin


def test_real_loss_holds_unsupervised_pixels_to_the_teacher():
    import torch

    from prototypes.learned_evidence.finetune import real_loss

    logits = torch.full((2, 2, 8, 8), -3.0)
    body, tip, none = torch.zeros(2, 8, 8), torch.zeros(2, 8, 8), torch.zeros(2, 8, 8)
    assert float(real_loss(logits, body, tip, none, none)) == 0.0  # nothing supervised, no teacher
    agree = real_loss(logits, body, tip, none, none, teacher=logits.clone())
    differ = real_loss(logits, body, tip, none, none, teacher=torch.full((2, 2, 8, 8), 6.0))
    assert float(differ) > float(agree)
