"""Synthetic benchmark movies with exact ground truth.

A synthetic movie reuses a real pre-germination field (the registered reference of a
prepared cache: real grains, real optics) and grows synthetic tubes from its grains:
a slow start then steady growth, a wandering heading, a real cross-section profile
(dark line or bright core with dark walls), contrast that matures after the front
passes, rigid rotation of some grain + tube pairs, drifting debris, slow illumination
change, random drift and a stage jump. Each rendered frame stands for one keyframe of
the real movies and is encoded as an intra frame with the real movies' x264 settings
(QP 30 floor, AQ on), then written with its truth in the benchmark-label format.

Time units in the synthetic movie are its own frames (= keyframes of the real movies);
``frames_per_bin`` 25 matches the real 300-frame bins.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import stack
from .cli import registered_mean
from .video import FFMPEG

X264 = ("keyint=1:min-keyint=1:scenecut=0:bframes=0:ref=3:cabac=1:deblock=0,0:me=hex:subme=7:psy-rd=1.00,0.00:"
        "mixed-refs=1:merange=16:trellis=1:8x8dct=1:deadzone-inter=21:deadzone-intra=11:fast-pskip=1:"
        "chroma-qp-offset=-2:slices=4:sliced-threads=1:nr=0:qpmin=30:qpmax=50:qpstep=4:ipratio=1.40:"
        "pbratio=1.30:aq-mode=1:aq-strength=1.0:mbtree=0:qcomp=1.0:ratetol=1.0")


@dataclass
class SynthConfig:
    n_frames: int = 4382                 # = keyframes of the real sparse movie
    frames_per_bin: int = 25
    seed: int = 0
    noise_sigma: float = 4.5             # calibrated: keyframe temporal std 0.46 vs real 0.47
    p_germinate: float = 0.85
    onset_bins: tuple = (6.0, 115.0)     # physical onset, in bins
    rate_px_per_bin: tuple = (0.15, 1.0)
    lag_bins: tuple = (2.0, 12.0)        # slow start: growth at slow_factor x rate for this long
    slow_factor: float = 0.3
    curvature: float = 0.035             # heading random walk (rad per sqrt px)
    max_length: float = 160.0
    p_bright_core: float = 0.3
    amplitude: tuple = (0.9, 2.0)        # x the post-codec real profile: encoded tubes then match real ones
    maturation_bins: tuple = (0.0, 10.0)  # contrast reaches 63% this long after the front passes
    p_rotate: float = 0.3
    rotate_deg: float = 40.0
    n_debris: int = 5
    jump_px: tuple = (-10.5, -1.0)
    jump_bin: tuple = (40.0, 80.0)
    drift_sigma: float = 0.01            # px per frame random walk
    gain_amp: float = 0.02
    encode: bool = True                  # False writes a lossless (FFV1) movie: the no-codec control


def tube_profile(d: np.ndarray, bright: bool) -> np.ndarray:
    """Intensity change across a tube at distance ``d`` from its centreline (real-tube fits).

    These shapes were measured on the *encoded* real movies. The QP-30 codec passes
    strong thin structures almost intact but attenuates faint ones (measured transfer
    for a dark line: 50% at 5 grey levels, 64% at 10, 83% at 20, ~100% from 45; bright
    cores: 5% at 5, 62% at 10, ~97% from 30), so the pre-codec amplitudes are drawn at
    0.9-2x these shapes to make encoded synthetic tubes match the real ones.
    """
    if bright:  # bright core with dark walls (g033/g036-like)
        return 11.0 * np.exp(-d ** 2 / 2.0) - 14.0 * np.exp(-(np.abs(d) - 3.5) ** 2 / 1.28)
    return -23.0 * np.exp(-d ** 2 / 3.4) + 1.5 * np.exp(-(np.abs(d) - 4.2) ** 2 / 0.5)  # dark line (median)


@dataclass
class Tube:
    grain: dict
    onset: float                  # frame at which growth starts (physical emergence)
    rate: float                   # px per frame after the slow start
    lag: float                    # frames of slow start
    slow: float
    bright: bool
    amp: float
    tau: float                    # maturation time constant (frames)
    path: np.ndarray = field(default=None)       # (n, 2) centreline from the exit, 0.25 px steps
    rot: np.ndarray = field(default=None)        # rotation (deg) per frame, or None

    def length(self, t: np.ndarray) -> np.ndarray:
        dt = np.clip(t - self.onset, 0.0, None)
        slow_part = np.minimum(dt, self.lag) * self.rate * self.slow
        fast_part = np.clip(dt - self.lag, 0.0, None) * self.rate
        return np.minimum(slow_part + fast_part, (len(self.path) - 1) * 0.25)

    def birth(self, s: np.ndarray) -> np.ndarray:
        """Frame at which the front reached arclength s (inverse of length)."""
        s_slow = self.lag * self.rate * self.slow
        return np.where(s <= s_slow, self.onset + s / (self.rate * self.slow),
                        self.onset + self.lag + (s - s_slow) / self.rate)


def _grow_path(rng, g: dict, obstacles: list[tuple[float, float, float]], lines: list[np.ndarray],
               max_len: float, shape: tuple[int, int], curvature: float, tries: int = 40) -> np.ndarray:
    """Centreline from a random exit on the grain rim, avoiding other grains, tubes and the border.

    Keeps the longest of up to ``tries`` attempts; stops early once one reaches max_len.
    """
    h, w = shape
    best = np.zeros((0, 2))
    for _ in range(tries):
        a = rng.uniform(0, 2 * np.pi)
        pts = [np.array([g["x"] + g["r"] * np.cos(a), g["y"] + g["r"] * np.sin(a)])]
        heading = a + rng.normal(0, 0.25)
        for _ in range(int(max_len / 0.25)):
            heading += rng.normal(0, curvature * 0.5)
            nxt = pts[-1] + 0.25 * np.array([np.cos(heading), np.sin(heading)])
            if np.hypot(nxt[0] - g["x"], nxt[1] - g["y"]) < g["r"] - 0.25:
                heading += 0.3  # steer away from its own grain
                continue
            if not (8 < nxt[0] < w - 8 and 8 < nxt[1] < h - 8):
                break
            if any(np.hypot(nxt[0] - ox, nxt[1] - oy) < orr + 7 for ox, oy, orr in obstacles):
                break
            if lines and len(pts) % 8 == 0 and any(np.min(np.hypot(*(ln - nxt).T)) < 9 for ln in lines):
                break
            pts.append(nxt)
        if len(pts) > len(best):
            best = np.array(pts)
        if (len(best) - 1) * 0.25 >= 0.95 * max_len:
            break
    return best


class Scene:
    def __init__(self, cache_dir: str | Path, cfg: SynthConfig):
        self.cfg = cfg
        bins, meta = stack.load(cache_dir)
        rs = int(meta.get("ref_start", 0))
        self.background = registered_mean(bins, meta["shifts"], [rs, rs + 1, rs + 2]).astype(np.float32)
        self.h, self.w = self.background.shape
        census = json.loads((Path(cache_dir) / "grains.json").read_text())["grains"]
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng
        grains = [g for g in census if g["isolated"] and g.get("ring_contrast", 0) >= 20]
        obstacles = [(g["x"], g["y"], g["r"]) for g in census]
        fpb = cfg.frames_per_bin
        self.tubes: list[Tube] = []
        self.controls: list[dict] = []
        lines: list[np.ndarray] = []
        for g in grains:
            if rng.random() > cfg.p_germinate:
                self.controls.append(g)
                continue
            onset = rng.uniform(*cfg.onset_bins) * fpb
            rate = rng.uniform(*cfg.rate_px_per_bin) / fpb
            remaining = cfg.n_frames - onset
            max_len = min(cfg.max_length, rate * remaining + 6)
            others = [o for o in obstacles if (o[0], o[1]) != (g["x"], g["y"])]
            path = _grow_path(rng, g, others, lines, max_len, (self.h, self.w), cfg.curvature)
            if len(path) < 8:
                self.controls.append(g)
                continue
            lines.append(path[::4])
            t = Tube(grain=g, onset=onset, rate=rate, lag=rng.uniform(*cfg.lag_bins) * fpb, slow=cfg.slow_factor,
                     bright=bool(rng.random() < cfg.p_bright_core), amp=float(rng.uniform(*cfg.amplitude)),
                     tau=float(rng.uniform(*cfg.maturation_bins) * fpb), path=path)
            if rng.random() < cfg.p_rotate:
                steps = rng.normal(0, 1, cfg.n_frames).cumsum()
                steps = cv2.GaussianBlur(steps.reshape(1, -1), (0, 0), 200).ravel()
                steps -= steps[-1]  # rotation is 0 at the end, like the traced end state
                t.rot = steps / (np.abs(steps).max() + 1e-9) * rng.uniform(10, cfg.rotate_deg)
            self.tubes.append(t)
        # per-tube lookup maps in the grain frame: nearest centreline arclength and distance
        self.maps = []
        for t in self.tubes:
            lo = np.floor(t.path.min(axis=0) - 8).astype(int)
            hi = np.ceil(t.path.max(axis=0) + 8).astype(int)
            gx, gy = t.grain["x"], t.grain["y"]
            if t.rot is not None:  # a rotating tube sweeps a disc around its grain
                reach = float(np.max(np.hypot(t.path[:, 0] - gx, t.path[:, 1] - gy))) + 8
                lo = np.floor([gx - reach, gy - reach]).astype(int)
                hi = np.ceil([gx + reach, gy + reach]).astype(int)
            lo = np.maximum(lo, 0)
            hi = np.minimum(hi, [self.w - 1, self.h - 1])
            yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1].astype(np.float32)
            tree = cKDTree(t.path)
            d, idx = tree.query(np.stack([xx.ravel(), yy.ravel()], 1))
            s = (idx * 0.25).reshape(xx.shape).astype(np.float32)
            d = d.reshape(xx.shape).astype(np.float32)
            self.maps.append((lo, hi, xx, yy, s, d))
        # debris blobs: straight drifting paths over a window of frames
        self.debris = []
        for _ in range(cfg.n_debris):
            start = rng.uniform(0, cfg.n_frames * 0.9)
            dur = rng.uniform(20, 60) * fpb / 25 * 25
            p0 = rng.uniform([50, 50], [self.w - 50, self.h - 50])
            v = rng.normal(0, 1, 2)
            v = v / np.linalg.norm(v) * rng.uniform(150, 300) / dur
            self.debris.append((start, dur, p0, v, rng.uniform(7, 14), rng.uniform(10, 25)))
        jb = rng.uniform(*cfg.jump_bin) * fpb
        drift = np.cumsum(rng.normal(0, cfg.drift_sigma, (cfg.n_frames, 2)), axis=0)
        drift[int(jb):] += cfg.jump_px
        self.drift = drift
        self.gain = 1.0 + cfg.gain_amp * np.sin(np.linspace(0, rng.uniform(1, 3) * np.pi, cfg.n_frames))

    def render(self, k: int) -> np.ndarray:
        cfg = self.cfg
        img = self.background * self.gain[k]
        for t, (lo, hi, xx, yy, s, d) in zip(self.tubes, self.maps):
            L = float(t.length(np.array([k]))[0])
            if L <= 0:
                continue
            ss, dd = s, d
            if t.rot is not None and abs(t.rot[k]) > 1e-3:
                a = math.radians(-t.rot[k])  # look up the unrotated tube at each pixel
                gx, gy = t.grain["x"], t.grain["y"]
                rx = gx + math.cos(a) * (xx - gx) - math.sin(a) * (yy - gy) - lo[0]
                ry = gy + math.sin(a) * (xx - gx) + math.cos(a) * (yy - gy) - lo[1]
                rx, ry = rx.astype(np.float32), ry.astype(np.float32)
                ss = cv2.remap(s, rx, ry, cv2.INTER_NEAREST, borderValue=1e6)
                dd = cv2.remap(d, rx, ry, cv2.INTER_NEAREST, borderValue=1e6)
            on = (ss <= L) & (dd < 6)
            if not on.any():
                continue
            age = k - t.birth(ss[on])
            mature = 1.0 - np.exp(-age / t.tau) if t.tau > 0 else np.ones_like(age)
            patch = img[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            patch[on] += t.amp * tube_profile(dd[on], t.bright) * mature
        for start, dur, p0, v, rad, con in self.debris:
            if start <= k < start + dur:
                c = p0 + v * (k - start)
                x0, x1 = int(max(0, c[0] - 3 * rad)), int(min(self.w, c[0] + 3 * rad))
                y0, y1 = int(max(0, c[1] - 3 * rad)), int(min(self.h, c[1] + 3 * rad))
                if x1 <= x0 or y1 <= y0:  # the blob has drifted out of the field
                    continue
                yy, xx = np.mgrid[y0:y1, x0:x1]
                img[y0:y1, x0:x1] -= con * np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * rad ** 2))
        dx, dy = self.drift[k]
        img = cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (self.w, self.h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        img = img + self.rng.standard_normal(img.shape, dtype=np.float32) * np.float32(cfg.noise_sigma)
        return np.clip(np.round(img), 0, 255).astype(np.uint8)

    def truth(self, name: str) -> dict:
        """Benchmark-format labels (reference coordinates, synthetic frame units)."""
        cfg, fpb = self.cfg, self.cfg.frames_per_bin
        n_bins = -(-cfg.n_frames // fpb)
        centres = np.array([b * fpb + fpb // 2 for b in range(n_bins)])
        grains, labels = {}, {}
        for i, t in enumerate(self.tubes):
            gid = f"s{i + 1:03d}"
            grains[gid] = {**{k: t.grain[k] for k in ("x", "y", "r")}, "id": gid, "isolated": True,
                           "source": "synthetic", "census_id": t.grain["id"]}
            L = t.length(centres.astype(float))
            visible = int(np.argmax(L >= 2.0)) if np.any(L >= 2.0) else None
            traces = {}
            for b in range(8, n_bins - 1, 12):
                traces[str(b)] = ({"state": "full", "length_px": round(float(L[b]), 3), "source_frame": int(centres[b])}
                                  if L[b] > 0 else {"state": "no_tube", "length_px": 0.0,
                                                    "source_frame": int(centres[b])})
            if visible is None:
                onset = {"verdict": "no_emergence_by_end"}
            else:
                onset = {"verdict": "emerged_within", "last_absent_frame": int(t.onset),
                         "first_visible_frame": int(centres[visible])}
            labels[gid] = {"onset": onset, "traces": traces,
                           "truth": {"onset_frame": float(t.onset), "rate_px_per_bin": t.rate * fpb,
                                     "lag_bins": t.lag / fpb, "tau_bins": t.tau / fpb, "bright_core": t.bright,
                                     "amplitude": t.amp, "rotates": t.rot is not None,
                                     "max_rotation_deg": float(np.abs(t.rot).max()) if t.rot is not None else 0.0,
                                     "final_length_px": float(L[-1])}}
        for j, g in enumerate(self.controls):
            gid = f"c{j + 1:03d}"
            grains[gid] = {**{k: g[k] for k in ("x", "y", "r")}, "id": gid, "isolated": True, "source": "synthetic",
                           "census_id": g["id"]}
            labels[gid] = {"onset": {"verdict": "no_emergence_by_end"},
                           "traces": {str(b): {"state": "no_tube", "length_px": 0.0, "source_frame": int(centres[b])}
                                      for b in range(8, n_bins - 1, 24)}}
        return {"schema": "sparsetrack.bench.v1", "origin": f"synthetic movie {name}", "config": asdict(cfg),
                "frames_per_bin": fpb, "n_bins": n_bins, "grains": grains, "labels": labels,
                "retest": {"grains": [], "labels": {}}}


def make_direct_cache(cache_dir: str | Path, out_cache: str | Path, cfg: SynthConfig, name: str = "synth",
                      log=print) -> Path:
    """No-codec control: bin the rendered frames straight into a cache (no encoding at all)."""
    out_cache = Path(out_cache)
    out_cache.mkdir(parents=True, exist_ok=True)
    started = time.time()
    scene = Scene(cache_dir, cfg)
    counts = stack.build_bins(((k, scene.render(k)) for k in range(cfg.n_frames)), cfg.n_frames, cfg.frames_per_bin,
                              (scene.h, scene.w), out_cache / "bins.npy")
    bins = np.load(out_cache / "bins.npy", mmap_mode="r")
    shifts, raw, outlier = stack.estimate_shifts(bins, ref_bins=3)
    meta = {"schema": stack.SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "movie": {"path": f"(direct render of {name})", "name": f"{name}_direct", "size_bytes": 0,
                      "width": scene.w, "height": scene.h, "n_frames": cfg.n_frames},
            "frames_per_bin": cfg.frames_per_bin, "n_bins": int(len(counts)), "keyframes_per_bin": counts.tolist(),
            "ref_bins": 3, "ref_start": 0, "shifts": np.round(shifts, 3).tolist(),
            "raw_shifts": np.round(raw, 3).tolist(), "shift_outlier_bins": [], "jump_bins": []}
    (out_cache / "meta.json").write_text(json.dumps(meta, indent=1))
    from .cli import write_census
    write_census(out_cache, 3, False)
    (out_cache.parent / f"{name}_truth.json").write_text(json.dumps(scene.truth(name)))
    log(f"{name}: direct (no-codec) cache in {time.time() - started:.0f} s -> {out_cache}")
    return out_cache


def make_movie(cache_dir: str | Path, out_dir: str | Path, cfg: SynthConfig, name: str = "synth", log=print) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    scene = Scene(cache_dir, cfg)
    log(f"{name}: {len(scene.tubes)} growing tubes, {len(scene.controls)} controls "
        f"({sum(t.rot is not None for t in scene.tubes)} rotating)")
    movie = out_dir / f"{name}.{'mp4' if cfg.encode else 'mkv'}"
    codec = (["-c:v", "libx264", "-preset", "medium", "-b:v", "12000k", "-x264-params", X264, "-pix_fmt", "yuv420p"]
             if cfg.encode else ["-c:v", "ffv1", "-pix_fmt", "gray"])
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{scene.w}x{scene.h}",
           "-r", "14/12", "-i", "-", *codec, str(movie)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for k in range(cfg.n_frames):
            proc.stdin.write(scene.render(k).tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    (out_dir / f"{name}_truth.json").write_text(json.dumps(scene.truth(name)))
    log(f"{name}: rendered and encoded {cfg.n_frames} frames in {time.time() - started:.0f} s -> {movie}")
    return movie
