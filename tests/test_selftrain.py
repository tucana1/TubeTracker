"""Label-free per-movie self-training (sparsetrack/selftrain.py) on a tiny synthetic cache: which readings become
pseudo-traces, the crops built from them, the fine-tuning, and the step end to end."""

import json
import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from sparsetrack import selftrain as st

FPB, NB, RS, W, H, R = 300, 40, 2, 240, 200, 8.0
CENTRES = {"g001": (60.0, 50.0), "g002": (60.0, 150.0), "g003": (130.0, 150.0), "g004": (180.0, 40.0),
           "g005": (130.0, 90.0), "g006": (200.0, 150.0)}


def frame(b):
    return b * FPB + FPB // 2


def lengths(gid):
    """Length per bin from bin 0 of each grain's tube (all grow along +x from their exit)."""
    out = []
    for b in range(NB):
        if gid in ("g001", "g003", "g005"):
            out.append(0.0 if b < 10 else 2.0 * (b - 9))
        elif gid == "g004":
            out.append(0.0 if b < 8 else 2.0 * (b - 7))
        elif gid == "g006":
            out.append(0.0 if b < 10 else 2.0 * (min(b, 17) - 9))  # grows to 16 px by bin 17, then stands still
        else:
            out.append(0.0)
    return out


def reading(gid, flags=()):
    x, y = CENTRES[gid]
    L = lengths(gid)
    grown = any(v > 0 for v in L)
    frames = [frame(b) for b in range(RS, NB)]
    res = {"id": gid, "x": x, "y": y, "r": R, "flags": list(flags), "length": {"frames": frames, "px": L[RS:]},
           "rotation_deg": [0.0] * len(frames), "final_length_px": L[-1]}
    if grown:
        on = next(b for b in range(NB) if L[b] > 0)
        res.update(status="emerged_within", onset_frame=frame(on), exit_xy=[x + R, y],
                   path=[[x + R + k, y] for k in range(0, 100, 2)])
    else:
        res.update(status="no_emergence_by_end", onset_frame=None, path=[])
    return res


def predictions():
    return {"schema": "sparsetrack.pred.v1", "frames_per_bin": FPB, "params": {"rot_pivot": "exit"},
            "grains": [reading("g001", ["reader:flood"]), reading("g002"), reading("g003", ["touches:g005"]),
                       reading("g004"), reading("g005"), reading("g006")]}


def tube_pixels(gid, b, width=2.5):
    x, y = CENTRES[gid]
    yy, xx = np.mgrid[0:H, 0:W]
    L = lengths(gid)[b]
    return (np.abs(yy + 0.5 - y) <= width) & (xx + 0.5 >= x + R) & (xx + 0.5 <= x + R + L)


def prob_movie():
    """P x 250: every grown tube marked as it is at each bin, except g005's (its reading has no map support)."""
    arr = np.zeros((NB, H, W), np.uint8)
    for b in range(NB):
        for gid in ("g001", "g003", "g004", "g006"):
            arr[b][tube_pixels(gid, b)] = 250
    return arr


def make_cache(tmp_path, start=None):
    """The cache (static texture, dark grains, bright growing tubes), its census, the predictions, and (for a start
    network) its probability movie where learned.prob_cache finds it."""
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    texture = gaussian_filter(rng.normal(0, 1, (H, W)), 4) * 30 + 150
    yy, xx = np.mgrid[0:H, 0:W]
    bins = np.repeat(texture[None], NB, axis=0)
    for (cx, cy) in CENTRES.values():
        bins[:, (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2] -= 60.0
    for b in range(NB):
        for gid in ("g001", "g003", "g004", "g005", "g006"):
            bins[b][tube_pixels(gid, b, 1.0)] += 40.0
    bins += rng.normal(0, 1.0, bins.shape)
    np.save(cache / "bins.npy", bins.astype(np.float16))
    meta = {"schema": "sparsetrack.cache.v1", "frames_per_bin": FPB, "n_bins": NB, "ref_start": RS,
            "shifts": [[0.0, 0.0]] * NB, "movie": {"name": "tiny.mp4", "width": W, "height": H}}
    (cache / "meta.json").write_text(json.dumps(meta))
    census = [{"id": gid, "x": x, "y": y, "r": R, "isolated": True, "border": False, "clump_size": 1}
              for gid, (x, y) in CENTRES.items()]
    (cache / "grains.json").write_text(json.dumps({"grains": census}))
    pred = tmp_path / "pred.json"
    pred.write_text(json.dumps(predictions()))
    if start is not None:
        prob = cache / f"prob_{start.stem}"
        prob.mkdir()
        np.save(prob / "bins.npy", prob_movie())
        (prob / "meta.json").write_text(json.dumps({**meta, "model_sha1": st.sha1(start)}))
    return cache, pred


def census():
    return {gid: {"id": gid, "x": x, "y": y, "r": R} for gid, (x, y) in CENTRES.items()}


def selected(s=None):
    maps = st.Maps(prob_movie(), [[0.0, 0.0]] * NB)
    return st.select(predictions(), census(), maps, NB, s or st.Settings(), W, H, log=lambda *a: None)


def test_confident_map_supported_readings_become_pseudo_traces():
    pseudo, stats = selected()
    g = pseudo["g001"]
    assert g["kind"] == "germinated"
    # confident from 12 px of steady growth (bin 15), every 3rd confident bin
    assert sorted(g["traces"]) == list(range(15, NB - 1, 3))
    for b, t in g["traces"].items():
        path = np.asarray(t["path_xy_ref"])
        assert t["state"] == "partial" and not t["contact"]
        assert np.allclose(path[:, 1], CENTRES["g001"][1], atol=0.6)          # on the middle of the marked band
        apex = np.asarray(t["apex_xy_ref"])
        assert abs(apex[0] - (CENTRES["g001"][0] + R + lengths("g001")[b])) < 1.5
        assert np.hypot(*(path[-1] - apex)) == pytest.approx(3.0, abs=0.6)    # cut 3 px short of the apex


def test_unsafe_unsupported_stalled_and_edge_readings_are_left_out():
    pseudo, stats = selected()
    grains = stats["grains"]
    assert "g003" not in pseudo and grains["g003"]["skipped"].startswith("unsafe flags touches:")
    assert "g005" not in pseudo and grains["g005"]["rejected_bins"]["share"] > 0   # no map under its reading
    assert max(pseudo["g006"]["traces"]) <= 17                                     # not once it stood still
    assert grains["g006"]["rejected_bins"]["conf"] > 0
    assert grains["g004"]["rejected_bins"]["edge"] > 0                             # it runs out of the frame
    for t in pseudo["g004"]["traces"].values():
        assert np.asarray(t["path_xy_ref"])[:, 0].max() <= W - st.Settings().edge_px
    # a grain read as never germinated, whose surroundings stay clean on the map: background
    assert pseudo["g002"]["kind"] == "clean" and pseudo["g002"]["clean_bins"][0] == RS + 3


def test_the_cap_spreads_pseudo_traces_over_the_confident_bins():
    pseudo, _ = selected(st.Settings(cap=3))
    assert sorted(pseudo["g001"]["traces"]) == [15, 27, 36]


def test_only_the_census_grains_are_used():
    maps = st.Maps(prob_movie(), [[0.0, 0.0]] * NB)
    c = {k: v for k, v in census().items() if k != "g001"}
    pseudo, stats = st.select(predictions(), c, maps, NB, st.Settings(), W, H, log=lambda *a: None)
    assert "g001" not in pseudo and "g001" not in stats["grains"]


def test_traced_crops_mark_the_route_and_leave_the_apex_unscored(tmp_path):
    cache, _ = make_cache(tmp_path)
    pseudo, _ = selected()
    s = st.Settings()
    crops = st.traced_crops(st.View(cache), {"g001": pseudo["g001"], "g002": pseudo["g002"]}, census(), s, 0,
                            log=lambda *a: None)
    n_tr = len(pseudo["g001"]["traces"])
    kinds = list(crops["kind"])
    assert kinds.count("trace") == s.along * n_tr and kinds.count("exit") == s.exits * n_tr
    assert kinds.count("neighbour") == 2 * n_tr            # the grain did not move: the route at b - 1 and b + 1
    assert kinds.count("clean") == len(pseudo["g002"]["clean_bins"])
    assert crops["x"].shape[1:] == (3, 2 * s.half, 2 * s.half) and crops["x"].dtype == np.float16
    for k in np.flatnonzero(crops["kind"] == "trace"):
        b, cx, cy = crops["info"][k]
        t = pseudo["g001"]["traces"][int(b)]
        jj, ii = np.meshgrid(np.arange(2 * s.half), np.arange(2 * s.half))
        px, py = jj + cx - s.half + 0.5, ii + cy - s.half + 0.5      # reference coordinates of the crop's pixels
        body = crops["body"][k] > 0
        assert np.all(np.abs(py[body] - CENTRES["g001"][1]) <= st.BODY + 0.6)
        ax, ay = t["apex_xy_ref"]
        assert not np.any(crops["w"][k][np.hypot(px - ax, py - ay) < s.tip_blind - 1])
    for k in np.flatnonzero(crops["kind"] == "clean"):
        assert crops["body"][k].sum() == 0 and crops["w"][k].sum() > 0


def test_propagated_crops_carry_the_tube_between_and_after_pseudo_traces(tmp_path):
    cache, _ = make_cache(tmp_path)
    s = st.Settings(spacing=6)
    pseudo, _ = selected(s)
    assert sorted(pseudo["g001"]["traces"]) == [15, 21, 27, 33]
    crops = st.propagated_crops(st.View(cache), {"g001": pseudo["g001"]}, census(), s, 0, log=lambda *a: None)
    kinds = set(crops["kind"])
    assert kinds == {"between", "after"}
    y0 = CENTRES["g001"][1]
    for k in range(len(crops["x"])):
        b, cx, cy = crops["info"][k]
        ii = np.arange(2 * s.half)[:, None] + cy - s.half + 0.5
        body = crops["body"][k] > 0
        assert body.any()
        assert np.all(np.abs(np.broadcast_to(ii, body.shape)[body] - y0) <= s.prop_body_px + 1.0)  # the static tube


def _tiny_net(tmp_path, torch, learned, name="start.pt", bg_px=16):
    torch.manual_seed(0)
    net = learned._unet((8, 16), norm="batch")
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.2, 0.2)
            m.running_var.uniform_(0.5, 1.5)
    path = tmp_path / name
    torch.save({"state": net.state_dict(), "widths": (8, 16), "norm": "batch", "bg_px": bg_px}, path)
    return path


def _shards(tmp_path, n=6, size=96):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (n, 3, size, size)).astype(np.float16)
    body = np.zeros((n, size, size), np.uint8)
    body[:, 40:44, 10:80] = 1
    real, synth = tmp_path / "real.npz", tmp_path / "synth.npz"
    np.savez(real, x=x, body=body, tip=np.zeros(body.shape, np.float16), w=np.ones(body.shape, np.uint8))
    np.savez(synth, x=x, body=body, tip=np.zeros(body.shape, np.float16))
    return [real, synth]


def test_fine_tuning_keeps_the_checkpoint_options_and_frozen_batchnorm_statistics(tmp_path):
    torch = pytest.importorskip("torch")
    from sparsetrack import learned
    start = _tiny_net(tmp_path, torch, learned)
    s = st.Settings()
    sets = [("a", st.load_shards([_shards(tmp_path)[0]]), 0.5), ("b", st.load_shards([_shards(tmp_path)[1]]), 0.5)]
    ck = st.fine_tune(start, sets, steps=3, seed=0, s=s, device="cpu", log=lambda *a: None)
    ck0 = torch.load(start, weights_only=False)
    assert (ck["widths"], ck["norm"], ck["bg_px"]) == ((8, 16), "batch", 16)
    stats = [k for k in ck0["state"] if k.endswith(("running_mean", "running_var"))]
    assert stats and all(torch.equal(ck["state"][k], ck0["state"][k]) for k in stats)
    assert not torch.equal(ck["state"]["head.weight"], ck0["state"]["head.weight"])


def test_replay_pools_trace_shards_and_splits_the_synthetic_share(tmp_path):
    real, synth = _shards(tmp_path)
    sets = st.replay_sets([real, synth, synth], st.Settings())
    assert [round(sh, 4) for _, _, sh in sets] == [0.25, 0.125, 0.125]


def test_the_step_end_to_end_writes_the_adapted_network_and_its_record(tmp_path):
    torch = pytest.importorskip("torch")
    from sparsetrack import learned
    start = _tiny_net(tmp_path, torch, learned)
    cache, pred = make_cache(tmp_path, start)
    out = tmp_path / "adapted.pt"
    rec = st.selftrain(cache, out, predictions=pred, start=start, replay=_shards(tmp_path), steps=2, seed=0,
                       device="cpu", log=lambda *a: None)
    assert out.exists() and not rec["unchanged"]
    net = learned.load_model(out, device="cpu")                      # loads as a tube network
    assert net.bg_px == 16
    doc = json.loads(out.with_suffix(".json").read_text())
    assert doc["start"]["sha1"] == st.sha1(start) and doc["seed"] == 0 and doc["steps"] == 2
    assert [r["sha1"] for r in doc["replay"]] == [st.sha1(p) for p in _shards(tmp_path)]
    assert doc["pseudo"]["germinated_used"] == 3 and doc["crops"]["traced"] > 0 and doc["crops"]["propagated"] > 0
    assert torch.load(out, weights_only=False)["selftrain"]["start"]["sha1"] == st.sha1(start)


def test_nothing_confident_keeps_the_starting_network(tmp_path):
    torch = pytest.importorskip("torch")
    from sparsetrack import learned
    start = _tiny_net(tmp_path, torch, learned)
    cache, pred = make_cache(tmp_path, start)
    doc = json.loads(pred.read_text())
    for g in doc["grains"]:
        g["flags"].append("drift_rejected")
    pred.write_text(json.dumps(doc))
    out = tmp_path / "adapted.pt"
    rec = st.selftrain(cache, out, predictions=pred, start=start, replay=_shards(tmp_path), steps=2, device="cpu",
                       log=lambda *a: None)
    assert rec["unchanged"] and st.sha1(out) == st.sha1(start)


def test_without_predictions_the_census_is_read_with_the_starting_network(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from sparsetrack import analyze as an
    from sparsetrack import learned
    start = _tiny_net(tmp_path, torch, learned)
    cache, _ = make_cache(tmp_path, start)
    calls = []

    def fake(cache_dir, out_dir, grains_path=None, params=None, **kw):
        calls.append((cache_dir, grains_path, params))
        out_dir.mkdir(parents=True)
        (out_dir / "predictions.json").write_text(json.dumps(predictions()))
        return predictions()
    monkeypatch.setattr(an, "analyze", fake)
    st.selftrain(cache, tmp_path / "a.pt", start=start, replay=_shards(tmp_path), steps=1, device="cpu",
                 work=tmp_path / "work", log=lambda *a: None)
    (cache_dir, grains_path, params), = calls
    assert grains_path is None                  # the cache's own census, no labels file
    assert params.model == str(start) and params.reader == an.Params().reader and params.tiptraj == "off"


def test_another_starting_network_needs_its_own_replay(tmp_path):
    torch = pytest.importorskip("torch")
    from sparsetrack import learned
    start = _tiny_net(tmp_path, torch, learned)
    cache, pred = make_cache(tmp_path, start)
    with pytest.raises(ValueError, match="replay"):
        st.selftrain(cache, tmp_path / "a.pt", predictions=pred, start=start, steps=1, log=lambda *a: None)


def test_command_line(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from sparsetrack import cli, learned
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    start = _tiny_net(tmp_path, torch, learned)
    cache, pred = make_cache(tmp_path, start)
    real, synth = _shards(tmp_path)
    out = tmp_path / "cli.pt"
    cli.main(["selftrain", str(cache), "--out", str(out), "--predictions", str(pred), "--start", str(start),
              "--replay", str(real), str(synth), "--steps", "2", "--seed", "3"])
    doc = json.loads(out.with_suffix(".json").read_text())
    assert doc["seed"] == 3 and doc["steps"] == 2 and math.isfinite(doc["timing"]["total_s"])


def test_several_runs_are_averaged_into_one_network(tmp_path):
    torch = pytest.importorskip("torch")
    from sparsetrack import learned
    start = _tiny_net(tmp_path, torch, learned)
    cache, pred = make_cache(tmp_path, start)
    replay = _shards(tmp_path)
    kw = dict(predictions=pred, start=start, replay=replay, steps=2, device="cpu", log=lambda *a: None)
    rec = st.selftrain(cache, tmp_path / "soup.pt", seed=5, runs=2, **kw)
    st.selftrain(cache, tmp_path / "a.pt", seed=5, **kw)
    st.selftrain(cache, tmp_path / "b.pt", seed=6, **kw)
    load = lambda p: torch.load(p, weights_only=False)
    want = st.soup([load(tmp_path / "a.pt"), load(tmp_path / "b.pt")])
    got = load(tmp_path / "soup.pt")
    assert rec["runs"] == 2 and [c["seed"] for c in rec["crops_by_run"]] == [5, 6]
    assert all(torch.allclose(got["state"][k].float(), want["state"][k].float(), atol=1e-6) for k in want["state"])
    a = load(tmp_path / "a.pt")["state"]["head.weight"]
    assert not torch.allclose(got["state"]["head.weight"], a)
