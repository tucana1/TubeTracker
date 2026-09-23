import json
import shutil
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

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
