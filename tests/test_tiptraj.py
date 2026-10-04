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


def test_off_by_default():
    assert Params().tiptraj == "off" and Params().tiptraj_guided is False


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
