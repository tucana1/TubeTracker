"""Kymograph training samples with exact truth from SparseTrack's codec-exact synthetic movies.

    python -m prototypes.kymo_reader.synth_data synthv5_s0 [synthv5_s1 ...]

For each movie: bin it into a temporary cache (``stack.prepare(..., frames_per_bin=25, ref_bins=3,
ref_start=0)`` + census, as ``scripts/synth_bench.py``), rebuild its ``Scene`` from the config in its
truth file, and for every synthetic tube read a kymograph along

- ``exact``: the tube's own end-state centreline (the grain frame at the end of the movie), with its
  end moved by up to +/-2 px, and
- ``perturbed``: the same centreline bent sideways by a smooth offset (up to 2.5 px), its end cut or
  extended by up to 8 px (the kind of route SparseTrack or a human draws),

each turned per bin by SparseTrack's own rotation track along that route (``strot.read_route``),
plus a random route from each non-germinating control grain (target: no tube). The target is the
true length L(t) at each bin centre; SparseTrack's own front along the route is kept as a
baseline. The temporary cache (0.4-0.9 GB) is deleted afterwards.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import kymo, store, strot

REPO = Path(__file__).resolve().parents[2]
SYN = REPO / "runs/sparsetrack/synth"
OUT = REPO / "runs/kymo_reader"


def field_of(name: str) -> Path:
    return REPO / ("runs/sparsetrack/m2" if name.startswith("synthv5m2") else "runs/sparsetrack/ld")


def scene_of(name: str):
    from sparsetrack.synth import Scene, SynthConfig
    doc = json.loads((SYN / f"{name}_truth.json").read_text())
    cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in doc["config"].items()}
    return Scene(field_of(name), SynthConfig(**cfg)), doc


def build_cache(name: str, cache: Path, log=print) -> None:
    from sparsetrack import stack
    from sparsetrack.cli import write_census
    if (cache / "grains.json").exists():
        return
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        stack.prepare(SYN / f"{name}.mp4", cache, frames_per_bin=25, ref_bins=3, ref_start=0, log=lambda *a: None)
        write_census(cache, 3, False)
    log(f"{name}: cache built in {time.time() - t0:.0f} s")


def smooth_noise(rng, n: int, sigma: float) -> np.ndarray:
    import cv2
    w = cv2.GaussianBlur(rng.normal(0, 1, (1, n + 6 * int(sigma))), (0, 0), sigma).ravel()[3 * int(sigma):3 * int(sigma) + n]
    return w / (np.abs(w).max() + 1e-9)


def perturb(path_xy: np.ndarray, rng, lateral_max: float = 2.5, end_max: float = 8.0) -> np.ndarray:
    """A route near the true centreline: bent sideways by a smooth offset (zero at the exit), end cut or
    extended."""
    p = kymo.resample(path_xy, 1.0, extend=end_max)
    n = p.n_path
    amp = rng.uniform(0.5, lateral_max)
    w = amp * smooth_noise(rng, len(p.pts), rng.uniform(6.0, 20.0)) * np.clip(p.s / 6.0, 0.0, 1.0)
    bent = p.pts + w[:, None] * p.normal
    d = rng.uniform(-end_max, end_max)
    stop = int(np.clip(n + round(d), 4, len(bent)))
    return bent[:stop]


def control_path(g: dict, grains: list[dict], rng, shape, length=(15.0, 60.0)) -> np.ndarray:
    """A random smooth route from a random point on the rim, clear of other grains."""
    h, w = shape
    for _ in range(50):
        a = rng.uniform(0, 2 * np.pi)
        pts = [np.array([g["x"] + g["r"] * math.cos(a), g["y"] + g["r"] * math.sin(a)])]
        heading = a + rng.normal(0, 0.3)
        L = rng.uniform(*length)
        ok = True
        for _ in range(int(L)):
            heading += rng.normal(0, 0.06)
            nxt = pts[-1] + np.array([math.cos(heading), math.sin(heading)])
            if not (12 < nxt[0] < w - 12 and 12 < nxt[1] < h - 12) or any(
                    math.hypot(nxt[0] - o["x"], nxt[1] - o["y"]) < o["r"] + 3 for o in grains if o["id"] != g["id"]):
                ok = False
                break
            pts.append(nxt)
        if ok and len(pts) > 8:
            return np.array(pts)
    return None


def extract_movie(name: str, variants=("exact", "perturbed"), max_tubes: int | None = None, seed: int = 0,
                  keep_cache: bool = False, log=print) -> Path:
    from sparsetrack import stack
    from sparsetrack.render import Renderer
    out = OUT / "data" / f"synth_{name}.npz"
    if out.exists():
        log(f"{out} exists")
        return out
    cache = OUT / "cache_tmp" / name
    t0 = time.time()
    scene, doc = scene_of(name)
    log(f"{name}: scene rebuilt in {time.time() - t0:.0f} s ({len(scene.tubes)} tubes, {len(scene.controls)} controls)")
    build_cache(name, cache, log)
    bins, meta = stack.load(cache)
    r = Renderer(bins, meta)
    fpb, nb = int(meta["frames_per_bin"]), int(meta["n_bins"])
    T = nb - r.ref_start
    frames = np.array([b * fpb + fpb // 2 for b in range(r.ref_start, nb)], float)
    census = json.loads((field_of(name) / "grains.json").read_text())["grains"]
    physical = [g for g in census if g.get("exclude_reason") != "not_a_grain"]
    rng = np.random.default_rng(seed + 7919 * sum(map(ord, name)))
    vmax = 4.5 if name.startswith("synthv5m2") else 4.0
    prm = strot.params(vmax_px=vmax, onset_source="front")
    order = list(range(len(scene.tubes)))
    if max_tubes is not None and len(order) > max_tubes:
        scored = [i for i in order if scene.tubes[i].scored]
        foreign = [i for i in order if not scene.tubes[i].scored]
        rng.shuffle(foreign)
        order = sorted(scored + foreign[:max(0, max_tubes - len(scored))])[:max_tubes]
    samples = []
    t_read = 0.0
    for n_done, i in enumerate(order):
        t = scene.tubes[i]
        g = t.grain
        others = [o for o in physical if o["id"] != g["id"]]
        L = t.length(frames)
        L_end = float(t.length(np.array([float((nb - 2) * fpb + fpb // 2)]))[0])
        n_pts = max(int(L_end / 0.25) + 1, 25)
        true_path = t.path[:n_pts] + 0.5          # frame-grid index -> reference coordinates
        follow = kymo.follow_offsets(r, g)
        info_t = {"movie": name, "tube": i, "scored": bool(t.scored), "grain": g["id"], "bright": bool(t.bright),
                  "rate_px_per_bin": float(t.rate * fpb), "amp": float(t.amp), "width": float(t.width),
                  "rotates": t.rot is not None, "sways": t.sway is not None,
                  "drifts": t.move is not None or t.anchor is not None, **{k: v for k, v in t.info.items()
                                                                         if isinstance(v, (bool, int, float, str))},
                  "final_len": float(L[-1])}
        vs = list(variants)
        if len(vs) > 1 and max_tubes is not None and name.startswith("synthv5m2"):
            vs = [vs[int(rng.integers(len(vs)))]]  # crowded long movies: one route per tube
        for v in vs:
            if v == "exact":
                d = rng.uniform(-2.0, 2.0)
                p0 = kymo.resample(true_path, 1.0, extend=2.0)
                route = p0.pts[:int(np.clip(p0.n_path + round(d), 4, len(p0.pts)))]
            else:
                route = perturb(true_path, rng)
            tr0 = time.time()
            try:
                rr = strot.read_route(r, meta, g, others, route, prm)
            except Exception as e:  # noqa: BLE001 - a failed SparseTrack read leaves the route unturned
                log(f"  tube {i} {v}: SparseTrack read failed ({e!r})")
                rr = None
            t_read += time.time() - tr0
            path = kymo.resample(route, back=kymo.BACK)
            ky = kymo.extract(r, g, path, follow, others, rot_deg=None if rr is None else rr["rotation_deg"],
                              pivot=None if rr is None else rr["pivot"])
            ky["target"] = L.astype(np.float32)
            ky["st_len"] = (np.zeros(T, np.float32) if rr is None else rr["length"].astype(np.float32))
            samples.append(store.pack(ky, {**info_t, "variant": v, "st_status": None if rr is None else rr["status"]}))
        if (n_done + 1) % 10 == 0:
            log(f"  {n_done + 1}/{len(order)} tubes, {time.time() - t0:.0f} s (SparseTrack reads {t_read:.0f} s)")
    for j, g in enumerate(scene.controls):
        route = control_path(g, census, rng, (r.height, r.width))
        if route is None:
            continue
        others = [o for o in physical if o["id"] != g["id"]]
        rr = strot.read_route(r, meta, g, others, route, prm)
        ky = kymo.extract(r, g, kymo.resample(route, back=kymo.BACK), kymo.follow_offsets(r, g), others,
                          rot_deg=None if rr is None else rr["rotation_deg"], pivot=None if rr is None else rr["pivot"])
        ky["target"] = np.zeros(T, np.float32)
        ky["st_len"] = (np.zeros(T, np.float32) if rr is None else rr["length"].astype(np.float32))
        samples.append(store.pack(ky, {"movie": name, "tube": -1 - j, "scored": True, "grain": g["id"],
                                       "variant": "control", "final_len": 0.0,
                                       "st_status": None if rr is None else rr["status"]}))
    store.save(out, samples)
    del bins, r
    if not keep_cache:
        shutil.rmtree(cache, ignore_errors=True)
    log(f"{name}: {len(samples)} samples in {time.time() - t0:.0f} s -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--max-tubes", type=int, default=None)
    ap.add_argument("--keep-cache", action="store_true")
    a = ap.parse_args()
    for nm in a.names:
        extract_movie(nm, max_tubes=a.max_tubes, keep_cache=a.keep_cache, log=lambda *x: print(*x, flush=True))
