"""Does learned evidence beat SparseTrack's change-based evidence? Two tests.

1. **End to end.** The network reads every registered bin of a movie; its tube probability
   (x ``SCALE_P`` so SparseTrack's grey-level thresholds sit near P = 0.3-0.6) is written as a
   SparseTrack cache and the *unchanged* pipeline runs on it: path candidates, rotation,
   monotone growth front, onset, contact censoring. Per-grain local registration is measured
   on the image cache and applied to the probability crops. Both runs are scored with
   ``sparsetrack.evaluate.score`` against the synthetic truth (onset tolerance 50 synthetic
   frames = 600 source frames, as in ``scripts/synth_bench.py``).
2. **Oracle path.** For every scored synthetic tube, evidence is sampled along the tube's true
   per-bin geometry and fed to SparseTrack's ``dp_front``: the growth front then depends only on
   the evidence (``abs``, ``matched``, ``union`` as in ``read_path``, or ``learned``).

    python -m prototypes.learned_evidence.evaluate --model unet.pt --field FIELD_CACHE \
        --movie v5s3_cache:v5:3 --movie v5s4_cache:v5:4 --out OUT
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import sparsetrack.analyze as A
from sparsetrack import stack
from sparsetrack.evaluate import load, score
from sparsetrack.render import Renderer
from sparsetrack.synth import Scene, preset

from .data import normalise
from .model import load as load_model, predict
from .truth import tube_point

SCALE_P = 16.0


# ----------------------------------------------------------------------------- probability cache
def registered_frame(bins: np.ndarray, shifts: np.ndarray, b: int) -> np.ndarray:
    dx, dy = shifts[b]
    img = np.asarray(bins[b], np.float32)
    return cv2.warpAffine(img, np.float32([[1, 0, -dx], [0, 1, -dy]]), (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def prob_cache(image_cache: str | Path, net, out_cache: str | Path, log=print) -> Path:
    """A SparseTrack cache whose "bins" are the network's tube probability x SCALE_P, in
    reference coordinates (shifts zero); census and meta copied from the image cache."""
    out_cache = Path(out_cache)
    if (out_cache / "meta.json").exists():
        return out_cache
    out_cache.mkdir(parents=True, exist_ok=True)
    bins, meta = stack.load(image_cache)
    shifts = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    early = np.mean([registered_frame(bins, shifts, b) for b in range(rs, rs + 3)], axis=0)
    late = np.mean([registered_frame(bins, shifts, b) for b in range(nb - 4, nb - 1)], axis=0)
    out = np.lib.format.open_memmap(out_cache / "bins.npy", mode="w+", dtype=np.float16, shape=bins.shape)
    started = time.time()
    for b in range(nb):
        x = normalise(registered_frame(bins, shifts, b), early, late)
        out[b] = (predict(net, x)[0] * SCALE_P).astype(np.float16)
    out.flush()
    del out
    m = {**meta, "shifts": [[0.0, 0.0]] * nb, "raw_shifts": [[0.0, 0.0]] * nb,
         "evidence": f"learned tube probability x {SCALE_P} (from {image_cache})"}
    (out_cache / "meta.json").write_text(json.dumps(m, indent=1))
    shutil.copy(Path(image_cache) / "grains.json", out_cache / "grains.json")
    log(f"probability cache {out_cache.name}: {nb} bins in {time.time() - started:.0f} s")
    return out_cache


def truth_prob_cache(image_cache: str | Path, field_cache: str | Path, preset_name: str, seed: int,
                     out_cache: str | Path, log=print) -> Path:
    """Upper bound: the exact built-tube mask of every bin (all tubes) as the "probability"."""
    from .truth import frame_truth
    out_cache = Path(out_cache)
    if (out_cache / "meta.json").exists():
        return out_cache
    out_cache.mkdir(parents=True, exist_ok=True)
    bins, meta = stack.load(image_cache)
    scene = Scene(field_cache, preset(preset_name, seed=seed))
    fpb, nb = int(meta["frames_per_bin"]), int(meta["n_bins"])
    out = np.lib.format.open_memmap(out_cache / "bins.npy", mode="w+", dtype=np.float16, shape=bins.shape)
    for b in range(nb):
        out[b] = (frame_truth(scene, b * fpb + fpb // 2)["body"] * SCALE_P).astype(np.float16)
    out.flush()
    del out
    m = {**meta, "shifts": [[0.0, 0.0]] * nb, "raw_shifts": [[0.0, 0.0]] * nb, "evidence": "exact truth masks"}
    (out_cache / "meta.json").write_text(json.dumps(m, indent=1))
    shutil.copy(Path(image_cache) / "grains.json", out_cache / "grains.json")
    log(f"truth cache {out_cache.name}: {nb} bins")
    return out_cache


@contextlib.contextmanager
def image_registration(image_cache: str | Path):
    """While active, SparseTrack's per-grain local registration is measured on the image
    cache (the probability maps have no grain rims to register on)."""
    bins, meta = stack.load(image_cache)
    img_r = Renderer(bins, meta)
    orig_grain, orig_shifts = A.analyze_grain, A.local_shifts
    current: dict = {}

    def grain_hook(renderer, meta_, grain, others, p, _settled=False):
        current.update(grain=grain, rs=int(meta_.get("ref_start", 0)), n=int(meta_["n_bins"]))
        return orig_grain(renderer, meta_, grain, others, p, _settled)

    def shifts_hook(crops, centre, radius, pad, ref_bins, *a, **k):
        g, rs = current["grain"], current["rs"]
        half = crops.shape[-1] // 2
        img = np.stack([img_r.crop(b, g["x"], g["y"], half) for b in range(rs, rs + len(crops))])
        if np.isnan(img).any():
            img = np.nan_to_num(img, nan=float(np.nanmedian(img)))
        return orig_shifts(img, centre, radius, pad, ref_bins, *a, **k)

    A.analyze_grain, A.local_shifts = grain_hook, shifts_hook
    try:
        yield
    finally:
        A.analyze_grain, A.local_shifts = orig_grain, orig_shifts


def _chunked(matched_kymograph):
    """``matched_kymograph`` remaps (angles x path points) rows in one call, and OpenCV's remap
    takes fewer than 32,767 rows: with 61 angles a path longer than ~268 px raises cv2.error.
    Angles are independent, so they are read in groups small enough for remap."""
    def run(signed, template, pts, normal, centre, across, angles_deg, lateral_offset=0.0):
        per = max(1, 32000 // max(len(pts), 1))
        return np.concatenate([matched_kymograph(signed, template, pts, normal, centre, across,
                                                 angles_deg[i:i + per], lateral_offset=lateral_offset)
                               for i in range(0, len(angles_deg), per)], axis=1)
    return run


@contextlib.contextmanager
def adaptive_crop(big: int = 300, gap_px: float = 3.0):
    """While active, a grain whose chosen path ends within ``gap_px`` of the edge of its square
    crop (``Params.half``, 150 px either way) is read again with ``half = big``. Otherwise a tube
    that leaves the crop stops there, with no flag: movie 2 runs twice as long as the dev movie,
    and its longest tubes pass 128 px. Label-free, and a grain whose path stays inside the crop is
    untouched. A grain read again is flagged ``crop_grown:<big>``. Long paths also need the
    matched kymograph read in angle groups (``_chunked``); results are otherwise identical."""
    orig, orig_mk = A.analyze_grain, A.matched_kymograph

    def hook(renderer, meta, grain, others, p, _settled=False):
        res = orig(renderer, meta, grain, others, p, _settled)
        diag = res.get("_diag")
        if not diag or diag[3] is None or len(diag[3]) == 0 or any(
                f.startswith("crop_grown") for f in res.get("flags", [])):
            return res
        side = diag[0].shape[0]  # the crop actually read
        x, y = diag[3][-1]
        if side < 2 * big and min(x, y, side - 1 - x, side - 1 - y) <= gap_px:
            res = orig(renderer, meta, grain, others, replace(p, half=big), _settled)
            res["flags"].append(f"crop_grown:{big}")
        return res

    A.analyze_grain, A.matched_kymograph = hook, _chunked(orig_mk)
    try:
        yield
    finally:
        A.analyze_grain, A.matched_kymograph = orig, orig_mk


def run_baseline(image_cache: str | Path, work: str | Path, grains_path: str | Path | None = None) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return A.analyze(image_cache, Path(work) / "baseline", grains_path=grains_path, params=A.Params(),
                         log=lambda *a: None)


def run_on_prob_cache(pcache: str | Path, image_cache: str | Path, work: str | Path, tag: str,
                      tip_offset: float = 0.0, grains_path: str | Path | None = None,
                      onset_source: str = "front") -> dict:
    """The unchanged SparseTrack pipeline on a probability cache (no settling: probability maps
    have no grain rims; per-grain registration is measured on the image cache).

    Onset comes from the growth front by default: SparseTrack's matched stub filter z-scores the
    exit against control angles, and on probability maps those controls are exactly zero, so the
    z-score explodes on noise-level values (grains called "emerged at start")."""
    lp = A.Params(settle=False, tip_offset_px=tip_offset, onset_source=onset_source)
    with image_registration(image_cache), contextlib.redirect_stdout(io.StringIO()):
        return A.analyze(pcache, Path(work) / tag, grains_path=grains_path, params=lp, log=lambda *a: None)


def per_grain_hits(rep: dict, onset_tol: float, len_abs: float = 2.0, len_rel: float = 0.10) -> dict[str, tuple]:
    """grain -> (onset hit 0/1 or None, length hits, length traces) from a ``score`` report."""
    out = {}
    for r in rep["rows"]:
        on = None
        if r.get("human") == "emerged_within":
            on = int("onset_error" in r and abs(r["onset_error"]) <= onset_tol)
        full = r.get("full", [])
        out[r["grain"]] = (on, sum(abs(f["error"]) <= max(len_abs, len_rel * f["human"]) for f in full), len(full))
    return out


def paired_bootstrap(rep_a: dict, rep_b: dict, onset_tol: float, n: int = 10000, seed: int = 0) -> dict:
    """Difference a - b in onset hits and length hits, resampling grains (95% intervals)."""
    a, b = per_grain_hits(rep_a, onset_tol), per_grain_hits(rep_b, onset_tol)
    grains = sorted(set(a) & set(b))
    on = np.array([[(a[g][0] or 0) - (b[g][0] or 0)] for g in grains], float)[:, 0]
    ln = np.array([a[g][1] - b[g][1] for g in grains], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(grains), size=(n, len(grains)))
    don, dln = on[idx].sum(axis=1), ln[idx].sum(axis=1)
    return {"grains": len(grains), "onset_diff": float(on.sum()), "onset_ci": np.percentile(don, [2.5, 97.5]).tolist(),
            "length_diff": float(ln.sum()), "length_ci": np.percentile(dln, [2.5, 97.5]).tolist(),
            "p_onset_not_better": float(np.mean(don <= 0)), "p_length_not_better": float(np.mean(dln <= 0))}


# ----------------------------------------------------------------------------- oracle path
def _sample(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear samples of ``img`` at (..., 2) coordinates, returned with shape ``xy.shape[:-1]``."""
    shape = xy.shape[:-1]
    mx = np.ascontiguousarray(xy[..., 0].reshape(1, -1), dtype=np.float32)
    my = np.ascontiguousarray(xy[..., 1].reshape(1, -1), dtype=np.float32)
    return cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=0).reshape(shape)


def oracle_tube(R: Renderer, RP: Renderer, t, fpb: int, nb: int, rs: int, p: A.Params) -> dict | None:
    """Growth fronts from each evidence type along the tube's true per-bin geometry."""
    lmax = (len(t.path) - 1) * 0.25
    s = np.arange(0.0, lmax, p.step)
    if len(s) < 8:
        return None
    frames = [b * fpb + fpb // 2 for b in range(nb)]
    geo = np.array([[tube_point(t, k, si) for si in s] for k in frames])  # (nb, n, 2) frame grid
    lo = np.floor(np.min(geo.reshape(-1, 2), axis=0) - 24)
    hi = np.ceil(np.max(geo.reshape(-1, 2), axis=0) + 24)
    cx, cy = (lo + hi) / 2.0
    cx, cy = float(round(cx)), float(round(cy))
    half = int(min(160, max(48, np.ceil(max(hi - lo) / 2 / 8) * 8)))
    reg = np.stack([R.crop(b, cx, cy, half) for b in range(nb)])
    if np.isnan(reg).any():
        reg = np.nan_to_num(reg, nan=float(np.nanmedian(reg)))
    early, late = reg[rs:rs + 3].mean(axis=0), reg[nb - 4:nb - 1].mean(axis=0)
    uv = geo - [cx - half, cy - half]  # crop pixel coordinates (OpenCV centres)
    tang = np.gradient(uv, axis=1)
    tang /= np.linalg.norm(tang, axis=2, keepdims=True) + 1e-9
    nrm = np.stack([-tang[..., 1], tang[..., 0]], axis=2)
    lat = lambda arr_b, b: np.max([_sample(arr_b, uv[b] + o * nrm[b]) for o in (-p.lateral, 0.0, p.lateral)], axis=0)
    # abs evidence (read_path, evidence="abs", with per-bin background subtraction)
    diffs = np.abs(reg - early[None])
    grid = np.stack(np.meshgrid(np.arange(2 * half), np.arange(2 * half)), axis=-1).astype(np.float32)
    near = np.zeros((2 * half, 2 * half), np.uint8)
    for b in range(nb):
        for q in uv[b][::4]:
            cv2.circle(near, (int(round(q[0])), int(round(q[1]))), 8, 1, -1)
    gxy = np.array([t.grain["x"] - (cx - half), t.grain["y"] - (cy - half)])
    bg_mask = (near == 0) & (np.hypot(*(grid - gxy).transpose(2, 0, 1)) > t.grain["r"] + 8)
    bg = np.array([float(np.median(d[bg_mask])) if bg_mask.any() else 0.0 for d in diffs])
    k_abs = np.clip(np.stack([lat(diffs[b], b) for b in range(nb)]) - bg[:, None], 0, None)
    base, noise = k_abs[rs:rs + 3].mean(axis=0), np.maximum(k_abs[rs:rs + 3].std(axis=0), 0.5)
    tau = np.maximum(p.evid_floor, base + p.evid_k * noise)
    ev = {"abs": np.clip((k_abs - tau) / tau, -1, 1)}
    # matched evidence: signed change projected on the end-state cross-section (per point)
    across = np.arange(-p.mk_half, p.mk_half + 1e-9, 0.5)
    tb = nb - 2
    q_end = uv[tb][:, None, :] + across[None, :, None] * nrm[tb][:, None, :]
    prof = _sample(late - early, q_end)
    prof = prof - prof.mean(axis=1, keepdims=True)
    template = prof / np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-3)
    signed = reg - early[None]
    def proj(b, off=0.0):
        q = uv[b][:, None, :] + (across[None, :, None] + off) * nrm[b][:, None, :]
        return np.einsum("pu,pu->p", _sample(signed[b], q), template)
    k_m = np.stack([proj(b) for b in range(nb)])
    ctrl = np.stack([np.concatenate([proj(b, -p.mk_control_px), proj(b, p.mk_control_px)]) for b in range(nb)])
    csig = np.maximum(1.4826 * np.median(np.abs(ctrl - np.median(ctrl, axis=1, keepdims=True)), axis=1), 0.3)
    tau_m = np.maximum(p.mk_floor, k_m[rs:rs + 3].mean(axis=0)[None] + p.mk_k * csig[:, None])
    ev["matched"] = np.clip((k_m - tau_m) / tau_m, -1, 1)
    ev["union"] = np.maximum(ev["abs"], ev["matched"])
    # learned evidence: log-odds of the tube probability along the path
    probs = np.stack([RP.crop(b, cx, cy, half) for b in range(nb)]) / SCALE_P
    pk = np.clip(np.stack([lat(probs[b], b) for b in range(nb)]), 1e-4, 1 - 1e-4)
    ev["learned"] = np.clip(np.log(pk / (1 - pk)) / 4.0, -1, 1)
    vmax = max(1, int(round(p.vmax_px / p.step)))
    truth_len = np.array([float(t.length(np.array([float(k)]))[0]) for k in frames])
    out = {"truth": truth_len}
    for name, e in ev.items():
        e = e.copy()
        e[:, s < p.skip_px] = 0.0
        e[:rs] = 0.0
        out[name] = A.dp_front(e, vmax) * p.step
    return out


def oracle(image_cache, pcache, field_cache, preset_name: str, seed: int, log=print) -> list[dict]:
    scene = Scene(field_cache, preset(preset_name, seed=seed))
    bins, meta = stack.load(image_cache)
    R = Renderer(bins, meta)
    RP = Renderer(*stack.load(pcache))
    fpb, nb, rs = int(meta["frames_per_bin"]), int(meta["n_bins"]), int(meta.get("ref_start", 0))
    p = A.Params()
    rows = []
    for i, t in enumerate(scene.tubes):
        if not t.scored:
            continue
        r = oracle_tube(R, RP, t, fpb, nb, rs, p)
        if r is not None:
            r.update(tube=i, info=dict(t.info), bright=t.bright, rotates=t.rot is not None,
                     drifts=t.move is not None or t.anchor is not None, tau_bins=t.tau / fpb, amp=t.amp)
            rows.append(r)
    log(f"oracle {Path(image_cache).name}: {len(rows)} tubes")
    return rows


def oracle_summary(rows: list[dict], offsets: dict[str, float]) -> str:
    """Trace-bin scoring as in the synthetic truth (every 12th bin from 8): lengths within
    max(2 px, 10%) where the truth is >= 2 px, absences where it is 0, onset within 2 bins."""
    lines = [f"{'evidence':10s} {'len in tol':>12s} {'med |err|':>10s} {'bias':>7s} {'absent ok':>10s} {'onset<=2 bins':>14s}"]
    for name in ("abs", "matched", "union", "learned"):
        hit = n = ab_ok = ab_n = on_hit = on_n = 0
        errs = []
        for r in rows:
            tr, pr = r["truth"], np.maximum(r[name] - offsets.get(name, 0.0), 0.0)
            for b in range(8, len(tr) - 1, 12):
                if tr[b] >= 2.0:
                    e = pr[b] - tr[b]
                    errs.append(e)
                    hit += abs(e) <= max(2.0, 0.1 * tr[b])
                    n += 1
                elif tr[b] == 0:
                    ab_ok += pr[b] < 2.0
                    ab_n += 1
            t_on = np.argmax(tr >= 2.0) if np.any(tr >= 2.0) else None
            p_on = np.argmax(pr >= 2.0) if np.any(pr >= 2.0) else None
            if t_on is not None:
                on_n += 1
                on_hit += p_on is not None and abs(int(p_on) - int(t_on)) <= 2
        e = np.array(errs)
        lines.append(f"{name:10s} {hit:>5d}/{n:<6d} {np.median(np.abs(e)):>10.2f} {np.mean(e):>+7.2f} "
                     f"{ab_ok:>4d}/{ab_n:<5d} {on_hit:>6d}/{on_n:<7d}")
    return "\n".join(lines)


def e2e_summary(name: str, rep: dict) -> str:
    o, L, a = rep["onset"], rep["length_full"], rep["absences"]
    conf = rep["germination_confusion"]
    ctrl = conf.get("no_emergence_by_end", {})
    fp = sum(v for k, v in ctrl.items() if k != "no_emergence_by_end")
    missed = sum(v for k, v in conf.get("emerged_within", {}).items() if k != "emerged_within")
    return (f"{name:9s} onset {o['hits']:3d}/{o['n_human_emerged_within']:<3d} (median |err| "
            f"{(o['median_abs_error'] or 0):5.0f} frames, missed {missed}) | lengths {L['within_tolerance']:3d}/{L['n']:<3d} "
            f"(median |err| {L['median_abs_error'] or 0:.2f} px, bias {L['bias'] or 0:+.2f}) | absences "
            f"{a['correct']}/{a['n']} | control false positives {fp}/{sum(ctrl.values())}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--field", required=True, help="the real cache the synthetic movies were built on")
    ap.add_argument("--movie", action="append", required=True, help="CACHE:PRESET:SEED (truth next to the movie)")
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--upper-bound", action="store_true", help="also run SparseTrack on the exact truth masks")
    args = ap.parse_args(argv)
    net = load_model(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    all_rows, reps = [], {}
    for spec in args.movie:
        cache, pr, seed = spec.rsplit(":", 2)
        seed = int(seed)
        truth_path = Path(args.truth_dir) / f"synth_{pr}_s{seed}_truth.json"
        truth = load(truth_path)
        work = out / f"{pr}s{seed}"
        pcache = prob_cache(cache, net, work / "prob_cache")
        if not args.skip_e2e:
            runs = {"baseline": run_baseline(cache, work),
                    "learned": run_on_prob_cache(pcache, cache, work, "learned")}
            if args.upper_bound:
                tcache = truth_prob_cache(cache, args.field, pr, seed, work / "truth_cache")
                runs["perfect"] = run_on_prob_cache(tcache, cache, work, "perfect")
            for k, pred in runs.items():
                rep = score(truth, pred, onset_tol=50)
                reps.setdefault(k, []).append(rep)
                report.append(f"[{pr} seed {seed}] {e2e_summary(k, rep)}")
                print(report[-1], flush=True)
        if not args.skip_oracle:
            all_rows += oracle(cache, pcache, args.field, pr, seed)
    if not args.skip_oracle:
        txt = oracle_summary(all_rows, {"abs": 2.5, "matched": 2.5, "union": 2.5, "learned": 0.0})
        report.append("oracle path (true geometry; SparseTrack's 2.5 px tip offset on its own evidence):\n" + txt)
        print(report[-1], flush=True)
    (out / "report.txt").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
