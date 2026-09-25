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
