"""The global tip-trajectory reader (sparsetrack/tiptraj.py) on synthetic maps: a tube growing from its grain beside a
foreign tube whose tip the detector also sees; the tube map losing the tube for a few bins."""

import numpy as np

from sparsetrack import tiptraj
from sparsetrack.analyze import Params

N = 40          # bins
X0, Y0, R = 100.0, 100.0, 10.0   # the grain (reference coordinates of its centre)
RATE = 1.5      # px a bin from bin 10


def tip_at(b: int) -> float:
    """x of our tube's tip at bin b (it leaves the rim at x = X0 + R going +x)."""
    return X0 + R + RATE * (b - 9)


def frames(gap=()):
    """Per bin: (tube map uint8 P x 250, detector map uint8 P x 250), reference pixel (i, j) at (j + 0.5, i + 0.5)."""
    yy, xx = np.mgrid[0:200, 0:240].astype(np.float32) + 0.5
    out = []
    for b in range(N):
        P = np.zeros((200, 240), np.float32)
        D = np.zeros((200, 240), np.float32)
        # a foreign tube along y = 150 (there from the start), its tip growing towards -x
        ftip = 230.0 - 2.0 * b
        P[(np.abs(yy - 150) <= 2.5) & (xx >= ftip)] = 1.0
        D += 0.9 * np.exp(-((xx - ftip) ** 2 + (yy - 150) ** 2) / 8.0)
        if b >= 10 and b not in gap:
            t = tip_at(b)
            P[(np.abs(yy - Y0) <= 2.5) & (xx >= X0 + R - 1) & (xx <= t)] = 1.0
            D += 0.8 * np.exp(-((xx - t) ** 2 + (yy - Y0) ** 2) / 8.0)
        out.append((np.round(P * 250).astype(np.uint8), np.round(np.clip(D, 0, 1) * 250).astype(np.uint8)))
    return out


def run(gap=()):
    tr = tiptraj._Track(X0, Y0, R, np.zeros((N, 2)), np.zeros(72), half=80)
    for b, (P, D) in enumerate(frames(gap)):
        tr.step(b, P, D)
    w = dict(tiptraj.WEIGHTS, det_norm=0)
    return tr.bins, tiptraj.viterbi(tr.bins, 0, N, w)


def test_reads_the_growing_tube_not_the_foreign_one():
    bins, (choice, L) = run()
    germ = np.flatnonzero(choice != -1)
    assert len(germ) and 9 <= germ[0] <= 12          # germinates when the tube appears
    assert (choice[:9] == -1).all()                   # not on the foreign tip, there from the start
    for b in (15, 25, 39):
        assert abs(L[b] - (tip_at(b) - X0 - R)) <= 2.5, (b, L[b])
        k = choice[b]
        if k >= 0:
            tip = bins[b]["tips"][k]
            assert abs(tip[1] - Y0) <= 3 and abs(tip[0] - tip_at(b)) <= 3


def test_holds_through_a_gap_in_the_maps():
    _, (choice, L) = run(gap=range(20, 25))
    assert (choice[20:25] != -1).all()                # still germinated
    assert L[24] >= L[19] - 3.5                      # held, not reset
    assert abs(L[30] - (tip_at(30) - X0 - R)) <= 2.5  # and read again after it


def test_on_by_default_as_frozen():
    """0.9.0's defaults are the frozen candidate (tag sparsetrack-0.9.0-candidate): its detector, its weights file's
    settings, lengths along the middle of the band."""
    import json
    from pathlib import Path
    p = Params()
    assert p.tiptraj == "flood" and p.tiptraj_mid is True and p.tiptraj_guided is False
    assert p.tiptraj_model is None and tiptraj.MODEL.name == "tips_v3_all.pt" and tiptraj.MODEL.exists()
    frozen = Path(__file__).parents[1] / "prototypes" / "tip_trajectory" / "weights_ld_v3.json"
    best = json.loads(frozen.read_text())["best"]
    assert tiptraj.weights(p) == {k: best[k] for k in tiptraj.WEIGHTS}


def test_guided_second_pass_follows_the_first_reading():
    fr = frames()
    w = dict(tiptraj.WEIGHTS, det_norm=0)
    tr = tiptraj._Track(X0, Y0, R, np.zeros((N, 2)), np.zeros(72), half=80)
    for b, (P, D) in enumerate(fr):
        tr.step(b, P, D)
    choice, _ = tiptraj.viterbi(tr.bins, 0, N, w)
    guide = {b: tr.bins[b]["bodies"][k] for b, k in enumerate(choice) if k >= 0}
    tr2 = tiptraj._Track(X0, Y0, R, np.zeros((N, 2)), np.zeros(72), half=80, guide=guide)
    for b, (P, D) in enumerate(fr):
        tr2.step(b, P, D)
    choice2, L2 = tiptraj.viterbi(tr2.bins, 0, N, w)  # a clean tube: the second pass reads it as the first did
    for b in (25, 39):
        assert abs(L2[b] - (tip_at(b) - X0 - R)) <= 2.5, (b, L2[b])


def test_mid_correction_measures_along_the_middle_of_a_curve():
    yy, xx = np.mgrid[0:200, 0:200].astype(np.float32) + 0.5
    rad = np.hypot(xx - 100, yy - 100)
    P = ((rad >= 26) & (rad <= 34) & (yy <= 100)).astype(np.float32)  # a half ring, 8 px wide, middle at radius 30
    a = np.linspace(np.pi, 2 * np.pi, 200)
    inner = np.stack([100 + 27 * np.cos(a), 100 + 27 * np.sin(a)], 1)  # a body hugging the inside of the turn
    corr = tiptraj.mid_correction(inner, P)
    assert 6.0 <= corr <= 12.0, corr  # pi x (30 - 27) = 9.4 px longer along the middle
    straight = np.stack([np.linspace(40, 160, 100), np.full(100, 70.0)], 1)
    P2 = (np.abs(yy - 70) <= 4).astype(np.float32)
    assert abs(tiptraj.mid_correction(straight + np.array([0.0, 2.5]), P2)) <= 0.5  # off-centre but straight: ~0


def test_extension_follows_the_map_ahead():
    Pc = np.zeros((200, 200), np.float32)
    Pc[98:103, 50:160] = 1.0  # a band along +x; another one going back (behind the tip) must not be taken
    pts = tiptraj._extension(Pc, np.array([80.0, 100.0]), np.array([1.0, 0.0]))
    xs = np.array([q[0] for q in pts])
    assert len(pts) >= 5 and (xs > 80).all() and abs(xs.max() - 159) <= 2  # ahead only, out to the band's end
    assert all(abs(q[1] - 100) <= 2.5 for q in pts)
    assert abs(xs[0] - 86) <= 1.5  # the first sample 6 px of arc on


class _Frames:
    """Stand-ins for the renderer (a dark grain disc on a flat background), the tube maps and the detector maps."""

    def __init__(self):
        fr = frames()
        self.bins = np.stack([P for P, _ in fr])
        self.det = [D for _, D in fr]

    def crop(self, b, cx, cy, half):
        yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float32) + 0.5
        return np.where(np.hypot(xx - half, yy - half) <= R, 60.0, 120.0).astype(np.float32)

    def tip(self, b):
        return self.det[b]


def test_read_replaces_a_reading():
    f = _Frames()
    meta = {"ref_start": 0, "n_bins": N, "frames_per_bin": 300}
    res = {"id": "g1", "x": X0, "y": Y0, "r": R, "flags": ["reader:flood", "onset_lookback:3"], "status": "emerged_within",
           "onset_frame": 150, "length": {"frames": [b * 300 + 150 for b in range(N)], "px": [0.0] * N},
           "path_by_bin": {"routes": [], "index": []}}
    p = Params(tiptraj="flood", tiptraj_half=80, tiptraj_weights='{"det_norm": 0}')
    out = tiptraj.read(res, f, f, meta, {"x": X0, "y": Y0, "r": R}, p, f)
    assert out["status"] == "emerged_within" and "reader:tiptraj" in out["flags"] and "reader:flood" in out["flags"]
    assert not any(fl.startswith("onset_lookback") for fl in out["flags"]) and "path_by_bin" not in out
    assert 9 * 300 <= out["onset_frame"] <= 12 * 300 + 150
    L = np.asarray(out["length"]["px"])
    assert (np.diff(L) >= -1e-9).all() and abs(L[-1] - (tip_at(N - 1) - X0 - R)) <= 3.0
    tip = out["tip"]["xy"][-1]
    assert abs(tip[0] - tip_at(N - 1)) <= 3 and abs(tip[1] - Y0) <= 3
    assert len(out["path"]) >= 2 and abs(out["path"][0][1] - Y0) <= 3


# ----------------------------------------------------------------------------- detector checkpoints (v2 and v3)
def _cache(tmp_path, nb=34, rs=2, shape=(40, 56)):
    """A small synthetic cache: a smooth background and a bright line growing from bin 6 (no drift)."""
    import json
    from sparsetrack import stack
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    bins = np.empty((nb,) + shape, np.float32)
    for b in range(nb):
        img = 120.0 + 0.3 * xx + 0.2 * yy
        if b >= 6:
            img[(np.abs(yy - 20) <= 1.5) & (xx >= 8) & (xx <= 8 + 1.2 * (b - 6))] += 6.0
        bins[b] = img
    d = tmp_path / "cache"
    d.mkdir()
    np.save(d / "bins.npy", bins)
    (d / "meta.json").write_text(json.dumps({"schema": stack.SCHEMA, "shifts": [[0.0, 0.0]] * nb, "ref_start": rs,
                                             "n_bins": nb, "frames_per_bin": 10}))
    return d, bins


def _checkpoint(path, channels, in_ch):
    import torch
    from sparsetrack.learned import _unet
    torch.manual_seed(0)
    net = _unet((4, 8), "batch", in_ch=in_ch)
    torch.save({"channels": channels, "widths": (4, 8), "out_ch": 2, "state": net.state_dict()}, str(path))
    return net


def test_version3_detector_builds_maps_as_its_inputs_say(tmp_path, monkeypatch):
    import json
    import torch
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    d, bins = _cache(tmp_path)
    rs, nb = 2, len(bins)
    net = _checkpoint(tmp_path / "tip3_x.pt", list(tiptraj.DET_CHANNELS3), 5).eval()
    loaded = tiptraj.load_detector(tmp_path / "tip3_x.pt")
    assert loaded.version == 3 and loaded.channels == tiptraj.DET_CHANNELS3
    out = tiptraj.det_cache(d, tmp_path / "tip3_x.pt", log=lambda *a: None, out=tmp_path / "maps")
    names = sorted(f.name for f in out.glob("b*.npz"))
    assert names == [f"b{b:03d}.npz" for b in range(rs + 7, nb - 1)]
    assert json.loads((out / "meta.json").read_text())["version"] == 3
    R = lambda k: np.asarray(bins[k], np.float32)
    # a lag reaching before the reference bins is the change from them: at bin rs + 12, D24 x 8 = D12 x 8 = C x 20
    x = tiptraj._inputs3(R, rs, nb, rs + 12)
    assert np.allclose(x[3] * 8.0, x[4] * 20.0, atol=1e-4) and np.allclose(x[2] * 8.0, x[4] * 20.0, atol=1e-4)
    assert not np.allclose(x[1], x[3])  # the 6-bin change does not reach back that far
    # the stored map is the network's on those inputs (the frame is smaller than a tile's context)
    b = nb - 5
    x = tiptraj._inputs3(R, rs, nb, b)
    xp = np.pad(x, ((0, 0), (0, (-x.shape[1]) % 2), (0, (-x.shape[2]) % 2)), mode="reflect")
    with torch.no_grad():
        y = torch.sigmoid(net(torch.from_numpy(xp)[None]))[0, 0].numpy()[:x.shape[1], :x.shape[2]]
    q = np.clip(np.round(y * 250.0), 0, 250).astype(np.uint8)
    q[q < 3] = 0
    assert np.array_equal(np.load(out / f"b{b:03d}.npz")["tip"], q)


def test_version2_detectors_still_load_and_bad_inputs_are_refused(tmp_path, monkeypatch):
    import json
    import pytest
    import torch
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    d, bins = _cache(tmp_path)
    _checkpoint(tmp_path / "tip2_x.pt", "ADC", 3)
    net = tiptraj.load_detector(tmp_path / "tip2_x.pt")
    assert net.version == 2 and net.channels == "ADC"
    out = tiptraj.det_cache(d, tmp_path / "tip2_x.pt", log=lambda *a: None, out=tmp_path / "maps2")
    assert len(list(out.glob("b*.npz"))) == len(bins) - 1 - (2 + 7)
    assert "version" not in json.loads((out / "meta.json").read_text())
    _checkpoint(tmp_path / "bad.pt", ["A", "C", "D6"], 3)  # not in the version 3 order
    with pytest.raises(ValueError):
        tiptraj.load_detector(tmp_path / "bad.pt")
