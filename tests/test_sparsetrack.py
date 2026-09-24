import json
import shutil
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from scipy.special import erfc

from sparsetrack import stack
from sparsetrack.bench.server import Bench, make_handler, trace_bins
from sparsetrack.grains import annotate_layout
from sparsetrack.render import Renderer
from sparsetrack.video import iter_keyframes, keyframe_indices, probe


def test_keyframe_indices_parse_ffprobe_csv():
    assert keyframe_indices(["0.000000,", "0.857143,", "N/A", "1.714286"], 14.0) == (0, 12, 24)
    with pytest.raises(ValueError):
        keyframe_indices(["1.0", "0.5"], 14.0)


def test_build_bins_averages_keyframes_per_bin(tmp_path):
    frames = [(f, np.full((4, 5), f, np.uint8)) for f in range(0, 30, 3)]  # keyframes every 3 frames
    counts = stack.build_bins(frames, n_frames=30, frames_per_bin=12, shape=(4, 5), out_path=tmp_path / "b.npy")
    bins = np.load(tmp_path / "b.npy")
    assert counts.tolist() == [4, 4, 2]
    assert np.allclose(bins[:, 0, 0], [np.mean([0, 3, 6, 9]), np.mean([12, 15, 18, 21]), np.mean([24, 27])])


def _field(dx=0.0, dy=0.0, size=128, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    img = np.full((size, size), 170.0)
    for cx, cy in rng.uniform(20, size - 20, (12, 2)):
        img -= 60 * np.exp(-((xx - cx - dx) ** 2 + (yy - cy - dy) ** 2) / 18.0)
    return img


def _texture(dx=0.0, dy=0.0, size=160):
    from scipy.ndimage import gaussian_filter, shift
    base = gaussian_filter(np.random.default_rng(3).normal(0, 20, (size, size)), 2.0) + 170
    return shift(base, (dy, dx), order=3, mode="reflect")


def test_estimate_shifts_recovers_translation_and_rejects_outlier():
    true = [(0, 0)] * 3 + [(0.4, -0.3), (2.5, 1.0), (2.6, 1.1), (2.4, 0.9), (2.5, 1.0), (2.5, 1.0)]
    bins = np.stack([_texture(dx, dy) for dx, dy in true])
    bins[5] = _texture(40.0, -30.0)  # one badly registered bin
    shifts, raw, outlier = stack.estimate_shifts(bins, ref_bins=3)
    assert outlier.tolist() == [False] * 5 + [True] + [False] * 3
    for b in (3, 4, 6, 7, 8):
        assert np.allclose(shifts[b], true[b], atol=0.2)  # real-frame precision is <0.06 px
    assert np.allclose(shifts[5], true[5], atol=0.5)


def test_renderer_crop_follows_registration_and_canvas_mapping():
    size = 96
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    ref_dot = (40.0, 50.0)
    shift = (3.0, -2.0)  # the dot appears at ref + shift in this bin
    img = 100 + 100 * np.exp(-((xx - ref_dot[0] - shift[0]) ** 2 + (yy - ref_dot[1] - shift[1]) ** 2) / 2.0)
    bins = img[None].astype(np.float16)
    r = Renderer(bins, {"shifts": [list(shift)], "n_bins": 1})
    half, cx, cy = 16, 44.0, 47.0
    crop = r.crop(0, cx, cy, half)
    j, i = np.unravel_index(np.argmax(crop), crop.shape)[::-1]
    # documented mapping: pixel j covers reference x in [cx - half + j, cx - half + j + 1)
    assert cx - half + j <= ref_dot[0] < cx - half + j + 1
    assert cy - half + i <= ref_dot[1] < cy - half + i + 1


def test_trace_bins_policy():
    assert trace_bins(20, 176) == [26, 70, 122, 174]
    assert trace_bins(68, 176) == [74, 122, 174]  # 70 is skipped: traces start 6 bins after onset
    assert trace_bins(173, 176) == []


def test_annotate_layout_flags_clumps_isolation_and_edges():
    grains = [{"x": 100, "y": 100, "r": 13}, {"x": 126, "y": 100, "r": 13}, {"x": 300, "y": 300, "r": 13},
              {"x": 8, "y": 300, "r": 13}]
    out = {(g["x"], g["y"]): g for g in annotate_layout(grains, (400, 400))}
    assert out[(100, 100)]["clump_size"] == 2 and not out[(100, 100)]["isolated"]
    assert out[(300, 300)]["isolated"]
    assert out[(8, 300)]["border"] and not out[(8, 300)]["isolated"]


@pytest.fixture
def tiny_cache(tmp_path):
    n_bins, h, w = 12, 80, 80
    bins = np.stack([_field(0, 0, size=80, seed=1) for _ in range(n_bins)]).astype(np.float16)
    np.save(tmp_path / "bins.npy", bins)
    meta = {"schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": n_bins, "shifts": [[0.0, 0.0]] * n_bins,
            "movie": {"name": "synthetic.mp4", "size_bytes": 1, "n_frames": n_bins * 300, "width": w, "height": h}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    grains = annotate_layout([{"x": 40.0, "y": 40.0, "r": 10.0}], (h, w))
    (tmp_path / "grains.json").write_text(json.dumps({"grains": grains}))
    return tmp_path


def test_bench_onset_trace_persistence_and_validation(tiny_cache):
    labels = tiny_cache / "labels" / "t.json"
    bench = Bench(tiny_cache, labels)
    rec = bench.set_onset("g001", {"verdict": "emerged_within", "first_visible_bin": 4, "consulted_bins": [3, 4]})
    assert (rec["last_absent_bin"], rec["first_visible_frame"], rec["last_absent_frame"]) == (3, 1350, 1050)
    with pytest.raises(ValueError):
        bench.set_onset("g001", {"verdict": "emerged_within", "first_visible_bin": 4, "last_absent_bin": 5})
    with pytest.raises(ValueError):
        bench.set_onset("g001", {"verdict": "unknown"})
    rec = bench.set_onset("g001", {"verdict": "no_emergence_by_end", "first_visible_bin": 4})
    assert rec["first_visible_bin"] is None and rec["first_visible_frame"] is None
    t = bench.set_trace("g001", {"bin": 9, "state": "partial", "points": [[40, 30], [43, 26]]})
    assert t["length_px"] == 5.0 and t["path_complete"] is False and t["direct_state"] == "direct_visible"
    with pytest.raises(ValueError):
        bench.set_trace("g001", {"bin": 9, "state": "full", "points": [[40, 30]]})
    reopened = Bench(tiny_cache, labels)  # persisted and reloadable
    assert reopened.doc["labels"]["g001"]["onset"]["verdict"] == "no_emergence_by_end"
    assert labels.with_suffix(".journal.jsonl").read_text().count("\n") == 4  # create, 2 onsets, 1 trace


def test_bench_refuses_labels_from_another_movie(tiny_cache):
    labels = tiny_cache / "t.json"
    Bench(tiny_cache, labels)
    doc = json.loads(labels.read_text())
    doc["movie"]["name"] = "other.mp4"
    labels.write_text(json.dumps(doc))
    with pytest.raises(SystemExit):
        Bench(tiny_cache, labels)


def test_bench_http_roundtrip(tiny_cache):
    bench = Bench(tiny_cache, tiny_cache / "t.json")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(bench))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        state = json.load(urllib.request.urlopen(base + "/api/state"))
        assert state["order"] == ["g001"] and state["n_bins"] == 12
        for path in ("/", "/static/app.js", "/api/img/coarse/g001", "/api/img/fine/g001?start=2",
                     "/api/img/frame/g001?bin=5&view=near", "/api/img/field?which=late"):
            assert urllib.request.urlopen(base + path).status == 200
        req = urllib.request.Request(base + "/api/onset/g001", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"verdict": "emerged_at_start"}).encode())
        assert json.load(urllib.request.urlopen(req))["first_visible_bin"] == 0
        assert json.load(urllib.request.urlopen(base + "/api/state"))["trace_plan"]["g001"] == [10]
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_probe_and_keyframe_decode_on_a_synthetic_movie(tmp_path):
    movie = tmp_path / "m.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=14", "-frames:v", "48",
                    "-c:v", "libx264", "-x264-params", "keyint=12:min-keyint=12:scenecut=0", "-pix_fmt", "yuv420p",
                    str(movie)], check=True)
    info = probe(movie)
    assert info.keyframes == (0, 12, 24, 36) and info.keyframe_interval == 12 and info.n_frames == 48
    frames = [f for f, img in iter_keyframes(info)]
    assert frames == [0, 12, 24, 36]


# ---------------------------------------------------------------- analysis and scoring
from sparsetrack.analyze import Params, analyze_grain, dp_front, rotation_track, sustained_onset  # noqa: E402
from sparsetrack.evaluate import interval_distance, score  # noqa: E402


def test_interval_distance_signs():
    assert interval_distance(500, 300, 600) == 0.0
    assert interval_distance(300, 300, 600) == -0.0 and interval_distance(250, 300, 600) == -50.0
    assert interval_distance(700, 300, 600) == 100.0
    assert interval_distance(10, None, 600) == 0.0


def test_dp_front_is_monotone_and_ignores_a_transient():
    n_bins, n = 30, 40
    ev = -np.ones((n_bins, n))
    for t in range(10, n_bins):  # tube grows 1 point per bin from bin 10
        ev[t, :t - 9] = 1.0
    ev[4, :25] = 1.0  # a one-bin blob over the path
    front = dp_front(ev, vmax=4)
    assert np.all(np.diff(front) >= 0)
    assert front[4] == 0 and front[9] == 0 and front[20] == 11 and front[-1] == 20


def test_rotation_track_follows_evidence_and_is_pinned_at_the_end():
    angles = np.arange(-30.0, 30.1, 3.0)
    score_ = np.zeros((20, len(angles)))
    for t in range(12):
        score_[t, np.argmin(np.abs(angles - 15.0))] = 5.0  # early: rotated by +15 degrees
    theta = rotation_track(score_, angles, penalty=0.1, max_turn=6.0, anchor_bins=3)
    assert theta[-1] == 0.0 and theta[0] == 15.0
    assert np.all(np.abs(np.diff(theta)) <= 6.0 + 1e-9)


def test_sustained_onset_rejects_transient_and_needs_persistence():
    p = Params()
    sig = np.zeros(40)
    sig[8:10] = 20.0            # transient bump
    sig[20:] = 20.0             # real, persistent rise
    t, _ = sustained_onset(sig, p)
    assert t == 20
    sig2 = np.zeros(40)
    sig2[20:30] = 20.0          # rises then falls back: not a tube
    assert sustained_onset(sig2, p)[0] is None
    sig3 = np.zeros(120)
    sig3[20:60] = 20.0          # a long rise, a mid-movie dip (the exit changes look), back to the end
    sig3[95:] = 20.0
    assert sustained_onset(sig3, Params(persist_bins=30))[0] == 20
    assert sustained_onset(sig3, Params(persist_bins=0))[0] != 20


def _synthetic_growth(n_bins=40, onset_bin=12, rate=1.5, size=320, gx=160.0, gy=160.0, r=13.0, angle=200.0,
                      seed=0, max_len=np.inf):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    d = np.hypot(xx - gx, yy - gy)
    grain = 175 - 90 * np.exp(-((d - r) ** 2) / 3.0)
    a = np.deg2rad(angle)
    u = np.array([np.cos(a), np.sin(a)])
    along = (xx - gx) * u[0] + (yy - gy) * u[1] - r
    across = -(xx - gx) * u[1] + (yy - gy) * u[0]
    bins, lengths = [], []
    for t in range(n_bins):
        L = min(max_len, max(0.0, (t - onset_bin + 1) * rate))
        lengths.append(L)
        # the tube's end is blurred by the optics, like a real tip: its signal fades out over ~3 px
        cap = np.where(along > 0, 0.5 * erfc((along - L) / (np.sqrt(2) * 1.2)), 0.0) if L > 0 else 0.0
        img = grain.copy()
        img += cap * (18 * np.exp(-across ** 2 / 0.8) - 22 * np.exp(-(np.abs(across) - 1.6) ** 2 / 0.5))
        bins.append(img + rng.normal(0, 0.4, img.shape))
    return np.stack(bins).astype(np.float16), np.array(lengths)


def test_analyze_grain_recovers_synthetic_onset_and_length():
    bins, true = _synthetic_growth()
    meta = {"shifts": [[0.0, 0.0]] * len(bins), "n_bins": len(bins), "frames_per_bin": 300}
    grain = {"id": "g001", "x": 160.0, "y": 160.0, "r": 13.0}
    res = analyze_grain(Renderer(bins, meta), meta, grain, [], Params(half=100))
    assert res["status"] == "emerged_within"
    onset_bin = res["onset_frame"] // 300
    assert abs(onset_bin - 12) <= 2
    est = np.array(res["length"]["px"])
    late = slice(20, len(bins))
    assert np.median(np.abs(est[late] - true[late])) < 2.0
    assert np.all(np.diff(est) >= -1e-9)


def test_analyze_grain_rejects_a_stub_that_never_grows():
    bins, _ = _synthetic_growth(max_len=3.0)  # a persistent 3 px bump at the rim that never elongates
    meta = {"shifts": [[0.0, 0.0]] * len(bins), "n_bins": len(bins), "frames_per_bin": 300}
    res = analyze_grain(Renderer(bins, meta), meta, {"id": "g001", "x": 160.0, "y": 160.0, "r": 13.0}, [],
                        Params(half=100))
    assert res["status"] == "no_emergence_by_end"
    assert max(res["length"]["px"]) == 0.0


def test_score_counts_onset_and_length_hits():
    labels = {"frames_per_bin": 300, "grains": {"g1": {"x": 10, "y": 10, "isolated": True}},
              "labels": {"g1": {"onset": {"verdict": "emerged_within", "last_absent_frame": 1000,
                                          "first_visible_frame": 2000},
                                "traces": {"20": {"state": "full", "length_px": 10.0, "source_frame": 6150},
                                           "2": {"state": "no_tube", "length_px": 0.0, "source_frame": 750}}}}}
    pred = {"method": "m", "grains": [{"id": "g1", "x": 10, "y": 10, "status": "emerged_within", "onset_frame": 2500,
                                        "length": {"frames": [750, 6150], "px": [0.0, 11.5]}}]}
    rep = score(labels, pred, onset_tol=600)
    assert rep["onset"]["hits"] == 1 and rep["onset"]["late"] == 0
    assert rep["length_full"]["within_tolerance"] == 1 and rep["absences"]["correct"] == 1


def test_settled_start_skips_initial_settling():
    rng = np.random.default_rng(5)
    base = rng.normal(170, 5, (64, 64))
    bins = []
    for t in range(40):
        drift = 6.0 * np.exp(-t / 3.0)  # large early motion that dies away
        bins.append(np.roll(base, int(round(drift * (t % 2 * 2 - 1))), axis=1) + rng.normal(0, 0.3, base.shape))
    assert 3 <= stack.settled_start(np.stack(bins), stride=1) <= 12


def test_flat_field_removes_vignetting_and_keeps_median():
    from sparsetrack.grains import flat_field
    yy, xx = np.mgrid[0:200, 0:200].astype(np.float64)
    vignette = 170 - 0.3 * xx  # dark right side
    out = flat_field(vignette)
    assert abs(np.median(out) - np.median(vignette)) < 0.01  # brightness level preserved
    assert np.std(out[50:150, 50:150]) < 0.1 * np.std(vignette[50:150, 50:150])


def test_turnbull_interval_censored_masses():
    from math import inf
    from sparsetrack.report import turnbull
    # two grains emerge in (0, 10], one in (10, 20], one is right-censored after 20
    masses = turnbull([(0, 10), (0, 10), (10, 20), (20, inf)])
    assert [(q, p) for q, p, _ in masses] == [(0, 10), (10, 20), (20, inf)]
    assert np.allclose([m for _, _, m in masses], [0.5, 0.25, 0.25])
    # overlapping brackets share mass on the innermost interval
    masses = turnbull([(0, 10), (5, 15)])
    assert [(q, p) for q, p, _ in masses] == [(5, 10)] and np.isclose(masses[0][2], 1.0)


def _grain_image(size=320, gx=160.0, gy=160.0, r=13.0):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    return 175 - 90 * np.exp(-((np.hypot(xx - gx, yy - gy) - r) ** 2) / 3.0)


def test_grain_settling_flags_debris_and_a_landing_grain():
    from sparsetrack.analyze import grain_settling
    rng = np.random.default_rng(0)
    p = Params(half=100)
    empty = np.stack([175 + rng.normal(0, 0.5, (200, 200)) for _ in range(30)]).astype(np.float32)
    assert grain_settling(empty, 99.5, 13.0, p)["no_grain"]
    still = np.stack([_grain_image(200, 99.5, 99.5) + rng.normal(0, 0.5, (200, 200)) for _ in range(30)])
    assert grain_settling(still.astype(np.float32), 99.5, 13.0, p) == {"no_grain": False, "b0": 0,
                                                                        "rim_median": pytest.approx(
                                                                            grain_settling(still.astype(np.float32),
                                                                                           99.5, 13.0, p)["rim_median"])}
    landing = still.copy()
    landing[0] = np.roll(_grain_image(200, 99.5, 99.5), 30, axis=1)  # still gliding in: elsewhere
    landing[1] = 0.5 * (landing[0] + still[1])                        # a blur on its way
    st = grain_settling(landing.astype(np.float32), 99.5, 13.0, p)
    assert not st["no_grain"] and st["b0"] == 2


def test_path_candidates_keep_the_grains_own_tube_over_a_late_foreign_one():
    bins, _ = _synthetic_growth(n_bins=40, onset_bin=6, rate=1.2)
    yy, xx = np.mgrid[0:320, 0:320].astype(np.float64)
    # a longer foreign tube touching the rim at 20 degrees appears only at bin 32
    a = np.deg2rad(20.0)
    along = (xx - 160) * np.cos(a) + (yy - 160) * np.sin(a) - 13.5
    across = -(xx - 160) * np.sin(a) + (yy - 160) * np.cos(a)
    foreign = np.where((along > 0) & (along < 90), -25 * np.exp(-across ** 2 / 3.0), 0.0)
    bins = bins.astype(np.float32)
    bins[32:] += foreign.astype(np.float32)
    meta = {"shifts": [[0.0, 0.0]] * len(bins), "n_bins": len(bins), "frames_per_bin": 300}
    res = analyze_grain(Renderer(bins, meta), meta, {"id": "g001", "x": 160.0, "y": 160.0, "r": 13.0}, [],
                        Params(half=100))
    ex, ey = res["exit_xy"]
    exit_deg = np.degrees(np.arctan2(ey - 160.0, ex - 160.0)) % 360
    assert abs(exit_deg - 200.0) < 20  # the own tube's exit, not the foreign one's at 20 degrees
    assert res["status"] == "emerged_within"


def test_cross_section_template_rows_are_zero_mean_unit_norm():
    from sparsetrack.analyze import cross_section_template
    yy, xx = np.mgrid[0:320, 0:320].astype(np.float64)
    line = -20 * np.exp(-((yy - 160) ** 2) / 3.4) * ((xx > 50) & (xx < 130))  # a dark tube along y = 160
    pts = np.stack([np.linspace(120, 60, 40), np.full(40, 160.0)], axis=1)
    normal = np.tile([0.0, 1.0], (40, 1))
    t = cross_section_template(line, pts, normal, np.arange(-3.5, 3.6, 0.5))
    assert t[:, 7].mean() < 0  # the centre of a dark tube is negative
    assert np.allclose(t.mean(axis=1), 0, atol=1e-5) and np.allclose(np.linalg.norm(t, axis=1), 1, atol=1e-4)


def test_synthetic_presets_render_every_phenomenon(tmp_path):
    from sparsetrack.synth import Scene, preset
    rng = np.random.default_rng(3)
    size, n_bins = 420, 4
    field = np.full((size, size), 180.0)
    centres = [(70 + 95 * (i % 4), 70 + 95 * (i // 4)) for i in range(12)]
    for cx, cy in centres:
        field = np.minimum(field, _grain_image(size, cx, cy, 11.0))
    bins = np.stack([field + rng.normal(0, 0.5, field.shape) for _ in range(n_bins)]).astype(np.float16)
    np.save(tmp_path / "bins.npy", bins)
    (tmp_path / "meta.json").write_text(json.dumps({
        "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": n_bins, "shifts": [[0.0, 0.0]] * n_bins,
        "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": 1200, "width": size, "height": size}}))
    census = annotate_layout([{"x": float(cx), "y": float(cy), "r": 11.0, "ring_contrast": 40.0} for cx, cy in centres],
                             (size, size))
    (tmp_path / "grains.json").write_text(json.dumps({"grains": census}))
    for name in ("v1", "v2", "v3", "v4"):
        cfg = preset(name, n_frames=80, frames_per_bin=5, onset_bins=(1.0, 4.0), rate_px_per_bin=(3.0, 6.0),
                     n_debris=1, seed=1, **({} if name == "v1" else dict(p_move=0.2, p_dock=0.5)),
                     **(dict(p_anchor=0.3, p_arrive=0.3, p_stub=0.3, p_evolve=1.0) if name in ("v3", "v4") else {}),
                     **(dict(p_sway=1.0) if name == "v4" else {}))
        scene = Scene(tmp_path, cfg)
        frames = [scene.render(k) for k in range(0, 80, 7)]
        assert all(f.shape == (size, size) and f.dtype == np.uint8 for f in frames)
        truth = scene.truth(name)
        assert truth["labels"] and all(g["id"] in truth["labels"] for g in truth["grains"].values())
        assert float(np.abs(frames[-1].astype(float) - frames[0]).max()) > 20  # tubes were drawn


def test_bench_views_follow_a_drifting_grain_and_store_reference_coordinates(tmp_path):
    rng = np.random.default_rng(5)
    n_bins, size = 24, 200
    frames = []
    for b in range(n_bins):
        drift = 0.5 * max(0, b - 6)  # the grain drifts right from bin 7 on, 8.5 px by the end
        frames.append(_grain_image(size, 100.0 + drift, 100.0, 12.0) + rng.normal(0, 0.5, (size, size)))
    np.save(tmp_path / "bins.npy", np.stack(frames).astype(np.float16))
    (tmp_path / "meta.json").write_text(json.dumps({
        "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": n_bins, "shifts": [[0.0, 0.0]] * n_bins,
        "movie": {"name": "drift.mp4", "size_bytes": 1, "n_frames": n_bins * 300, "width": size, "height": size}}))
    (tmp_path / "grains.json").write_text(json.dumps({"grains": annotate_layout(
        [{"x": 100.0, "y": 100.0, "r": 12.0}], (size, size))}))
    bench = Bench(tmp_path, tmp_path / "labels" / "t.json")
    off = bench.follow("g001")
    assert abs(off[-1][0] - 8.5) < 0.6 and abs(off[-1][1]) < 0.6 and abs(off[3][0]) < 0.3
    rec = bench.set_trace("g001", {"bin": n_bins - 1, "state": "full", "points": [[112, 100], [122, 100]]})
    assert rec["length_px"] == 10.0
    assert abs(rec["path_xy_ref"][0][0] - (112 + off[-1][0])) < 1e-6 and rec["path_xy_view"][0] == [112.0, 100.0]


def test_bench_sample_labels_a_fixed_random_subset(tmp_path):
    size, n_bins = 400, 6
    rng = np.random.default_rng(1)
    np.save(tmp_path / "bins.npy", np.stack([np.full((size, size), 180.0) + rng.normal(0, 0.5, (size, size))
                                             for _ in range(n_bins)]).astype(np.float16))
    (tmp_path / "meta.json").write_text(json.dumps({
        "schema": stack.SCHEMA, "frames_per_bin": 300, "n_bins": n_bins, "shifts": [[0.0, 0.0]] * n_bins,
        "movie": {"name": "s.mp4", "size_bytes": 1, "n_frames": 1800, "width": size, "height": size}}))
    grains = annotate_layout([{"x": 60.0 + 70 * (i % 5), "y": 60.0 + 70 * (i // 5), "r": 10.0} for i in range(20)],
                             (size, size))
    (tmp_path / "grains.json").write_text(json.dumps({"grains": grains}))
    a = Bench(tmp_path, tmp_path / "a.json", sample=5, seed=3)
    b = Bench(tmp_path, tmp_path / "b.json", sample=5, seed=3)
    kept = [g for g, v in a.doc["grains"].items() if not v.get("excluded")]
    assert len(kept) == 5 and kept == [g for g, v in b.doc["grains"].items() if not v.get("excluded")]
    assert all(a.doc["grains"][g]["isolated"] for g in kept)
    assert {v["exclude_reason"] for v in a.doc["grains"].values() if v.get("excluded")} == {"not_sampled"}
