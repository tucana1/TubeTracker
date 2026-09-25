"""Calibrating the per-bin decoder's end offset: which offset the traces pick, and when that is adopted."""

import json

import pytest

pytest.importorskip("torch")

from prototypes.learned_evidence import calibrate as C  # noqa: E402


def _table(gain_for, n_grains=30, onset_worse=False):
    """end -> grain -> (onset hit, length hits, traces): the default reads 1 of 3 lengths right per grain;
    offset -3 reads one more for the grains ``gain_for`` says, and one fewer for the others it names."""
    table = {}
    for e in C.ENDS:
        rows = {}
        for i in range(n_grains):
            g = f"g{i:03d}"
            extra = gain_for(i) if e == -3.0 else 0
            onset = 0 if (onset_worse and e == -3.0 and i < 3) else 1
            rows[g] = (onset, 1 + extra, 3)
        table[e] = rows
    return table


def test_a_clear_gain_is_adopted():
    check = C.cross_check(_table(lambda i: 1))
    assert check["picks"] == [-3.0, -3.0, -3.0] and check["length_diff"] == 30
    assert check["length_ci"][0] > 0 and check["adopted"]
    assert C.pick(_table(lambda i: 1), [f"g{i:03d}" for i in range(30)]) == -3.0


def test_a_gain_within_noise_is_not_adopted():
    # one more for a few grains, one fewer for others: the best of ten on the same traces, not a real gain
    check = C.cross_check(_table(lambda i: {0: 1, 1: 1, 2: 1, 3: -1, 4: -1}.get(i % 10, 0)))
    assert not check["adopted"]


def test_worse_onsets_block_adoption():
    check = C.cross_check(_table(lambda i: 1, onset_worse=True))
    assert check["length_ci"][0] > 0 and check["onset_diff"] < 0 and not check["adopted"]


def test_ties_keep_the_default():
    assert C.pick(_table(lambda i: 0), [f"g{i:03d}" for i in range(30)]) == C.DEFAULT_END


def test_decoder_settings_belong_to_their_model(tmp_path):
    f = tmp_path / "decoder.json"
    f.write_text(json.dumps({"end_px": -4.0, "model": str(tmp_path / "a.pt")}))
    notes = []
    assert C.decoder_settings(f, tmp_path / "a.pt", log=notes.append) == {"end_px": -4.0} and not notes
    assert C.decoder_settings(f, tmp_path / "b.pt", log=notes.append) == {"end_px": -4.0} and notes
    assert C.decoder_settings(None) == {}


def test_movie_2_and_models_tuned_on_the_same_traces_are_refused(tmp_path):
    import torch

    labels = tmp_path / "ld_v1.json"
    labels.write_text(json.dumps({"grains": {}, "labels": {}}))
    m2 = tmp_path / "m2_v1.json"
    m2.write_text("{}")
    with pytest.raises(SystemExit, match="held-out"):
        C.main(["--field", str(tmp_path), "--labels", str(m2), "--work", str(tmp_path / "w")])
    tuned = tmp_path / "unet_ft.pt"
    torch.save({"state": {}, "args": {"labels": str(labels)}}, tuned)
    with pytest.raises(SystemExit, match="in-sample"):
        C.main(["--field", str(tmp_path), "--labels", str(labels), "--work", str(tmp_path / "w"),
                "--model", str(tuned)])


def test_onsets_read_at_another_length_as_the_decoder_calls_them():
    frames = [150, 450, 750, 1050, 1350]
    pred = {"grains": [
        {"id": "a", "status": "emerged_within", "onset_frame": 450, "onset_interval": [150, 450],
         "length": {"frames": frames, "px": [0.0, 2.5, 3.5, 9.0, 12.0]}},
        {"id": "b", "status": "no_emergence_by_end", "onset_frame": None, "onset_interval": None,
         "length": {"frames": frames, "px": [0.0] * 5}}]}
    later = C.with_onset(pred, 4.0)["grains"]
    assert later[0]["onset_frame"] == 1050 and later[0]["onset_interval"] == [750, 1050]
    assert C.with_onset(pred, 2.0)["grains"][0]["onset_frame"] == 450  # the default reproduces the decoder
    assert later[1]["status"] == "no_emergence_by_end"  # nothing grew: unchanged
    assert pred["grains"][0]["onset_frame"] == 450  # the input is left as it was


def test_an_onset_threshold_is_adopted_on_onsets_alone():
    # every grain's onset is right at 4 px and wrong at the default 2 px; lengths are the same
    table = {t: {f"g{i:03d}": (1 if t == 4.0 else 0, 2, 3) for i in range(30)} for t in C.ONSETS}
    check = C.cross_check(table, default=C.DEFAULT_ONSET, gain="onset")
    assert check["picks"] == [4.0, 4.0, 4.0] and check["onset_ci"][0] > 0 and check["adopted"]
    assert not C.cross_check(table, default=C.DEFAULT_ONSET, gain="length")["adopted"]  # no length gain


def test_decoder_settings_carry_an_adopted_onset(tmp_path):
    f = tmp_path / "decoder.json"
    f.write_text(json.dumps({"end_px": 1.0, "onset_px": 4.0, "model": "m.pt"}))
    assert C.decoder_settings(f) == {"end_px": 1.0, "onset_px": 4.0}


def test_no_note_for_a_model_fine_tuned_with_the_decoder(tmp_path):
    import torch

    f = tmp_path / "decoder.json"
    f.write_text(json.dumps({"end_px": -2.0, "model": str(tmp_path / "start.pt")}))
    tuned = tmp_path / "unet_ft.pt"
    torch.save({"state": {}, "args": {"decoder": str(f)}}, tuned)  # as finetune.py records the check's decoder
    notes = []
    assert C.decoder_settings(f, tuned, log=notes.append) == {"end_px": -2.0} and not notes
    torch.save({"state": {}, "args": {"decoder": None}}, tuned)  # tuned and checked without it
    assert C.decoder_settings(f, tuned, log=notes.append) == {"end_px": -2.0} and notes


def _annotated(n=12, fpb=300, nb=60, rate=0.5, start=10, at_px=4.0):
    """Labels whose annotator calls a tube visible when it is at_px long, and predictions whose fitted length
    grows rate px/bin from bin start: the decoder's 2 px comes 4 bins before the annotator's 4 px."""
    frames = [b * fpb + fpb // 2 for b in range(nb)]
    fit = [max(0.0, rate * (b - start)) for b in range(nb)]
    fv = next(b for b in range(nb) if fit[b] >= at_px)
    grains, labs, preds = {}, {}, []
    for i in range(n):
        gid = f"g{i:02d}"
        x, y = 40.0 + 60 * i, 50.0
        grains[gid] = {"id": gid, "x": x, "y": y, "r": 10.0}
        labs[gid] = {"onset": {"verdict": "emerged_within", "last_absent_frame": frames[fv - 1],
                               "first_visible_frame": frames[fv]}}
        on = next(b for b in range(nb) if fit[b] >= 2.0)
        preds.append({"id": f"p{i}", "x": x, "y": y, "status": "emerged_within", "onset_frame": frames[on],
                      "onset_interval": [frames[on - 1], frames[on]], "length": {"frames": frames, "px": fit}})
    return {"grains": grains, "labels": labs, "frames_per_bin": fpb, "n_bins": nb}, {"grains": preds}


def test_germination_length_is_measured_where_the_annotator_calls_tubes_visible():
    labels, pred = _annotated()
    reads = C.onset_readings(labels, pred)
    assert len(reads) == 12 and all(abs(v - 3.75) < 1e-9 for v in reads.values())  # midway between 3.5 and 4.0 px
    est, (lo, hi) = C.onset_estimate(reads)
    assert est == 3.8 and lo <= est <= hi
    check = C.onset_check(labels, pred, reads)
    assert check["onset_diff"] == 12 and check["onset_ci"][0] > 0  # every onset: 4 bins early at 2 px, right at 3.8 px
    assert C.with_onset(pred, est)["grains"][0]["onset_frame"] == labels["labels"]["g00"]["onset"]["first_visible_frame"]


def test_germination_length_ignores_grains_without_a_timed_onset():
    labels, pred = _annotated(n=3)
    labels["labels"]["g00"]["onset"] = {"verdict": "no_emergence_by_end"}
    labels["grains"]["g01"]["excluded"] = True
    assert set(C.onset_readings(labels, pred)) == {"g02"}
