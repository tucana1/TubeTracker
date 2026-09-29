"""Self-training on one movie without labels: tip growth as the supervisor (experiment, off by default).

A pollen tube grows only at its apex, a built tube stays built until it bursts, and it is attached to its grain.
The per-bin decoder (``reach.py``) already reads every bin's tube attached to the grain and fits a monotone
growth curve. This module turns the readings that agree with that curve into pseudo-traces and tunes the network on
the movie's own frames with ``finetune.py``'s labeller, as if a person had traced them:

1. first pass: the start model's probability cache read by the per-bin decoder, with every bin's path;
2. pseudo-traces, per grain the decoder calls germinated (final length >= ``min_final``) and does not flag
   ``unsteady``: its path at bins whose raw reading is within max(2 px, 10%) of the fitted curve, before a burst
   and before the grain left its place, ``spacing`` bins apart. Each stops ``apex_margin`` px short of the fitted
   apex, and only a ``pos_px`` core along it is taught as tube: where a tube's walls and tip end is not known to a
   pixel, and teaching finetune's 2-px band made thin tubes read ~0.3 px longer (development seed 3: -15 lengths);
   the onset ``onset_guard`` bins before the decoded one;
3. image traces (``images=True``): per grain, the structure of the after-minus-before change that leaves the grain
   at one place and holds the start model's own reading; in each bin, its part attached to the rim whose pixels
   are already at their end-state level (built stays built, and the growth front stops the prefix). Used only
   where it reaches well past the start model's fitted length (>= 8 px and 25%), as tube only (no background past
   it), cut 4 px short of its end. Where the end state reaches well past the start model's final length (the
   network under-reads the grain), the grain's own pseudo-traces are made tube-only too, and its onset dropped,
   so that no under-read is taught as the tube's end. Grains the start model called tubeless get none: on a
   synthetic movie both structures found there were not tubes;
4. ``finetune.real_samples`` turns the traces into training crops (tube along a trace, background past its walls,
   between two traces the part both agree on, background along the path past a full trace's length and before
   onset, a ring round each grain holding only its own tube), and ``finetune.tune`` tunes the start model on them
   with ``distill`` holding every other pixel to the start model's output (and synthetic crops in every batch
   when ``--synthetic`` shards are given);
5. second pass: the tuned model's probability cache, read by the same decoder.

About 16 min per movie on one CPU thread (tuning 600 steps ~8 min, probability cache ~8 min); a few minutes on a
laptop's cores.

Results (development numbers, 26 Sep 2026; per-bin decoder on both sides, paired bootstrap over grains):
- synthetic development movies (v5 seeds 3-6, 108 grains): lengths -7 (95% CI -24 to +9) of 793 on the dense
  truth, +9 (0 to +19) of 282 on human-style traces; onsets +6 (+2 to +11) and +7 (+2 to +13) of 89;
- variant movies: faint tubes -5 (-12 to +1), thick +11 (-5 to +28), wide -2 (-12 to +9);
- the real sample movie, against an audit by eye: end lengths within 25% 14 -> 18 of 32, onsets within 6 bins
  27 -> 25 of 33. It learns the movie's thick-tube look (tubes the shipped network misses), and moves some onsets
  early.
So it is not a default step. It uses no labels, so a movie with labels tests it cleanly: ``--labels`` scores the
start and the self-trained readings on them and says whether it would be adopted (``finetune.adopted``: better
beyond noise on lengths or onsets, worse on neither). Give it the training shards the dev test made
(``--synthetic``); without them thin tubes drifted on a development movie.

    python -m prototypes.learned_evidence.selftrain --field runs/sparsetrack/<movie>/cache \\
        --pcache runs/learned_evidence/<movie>/prob_<cache name> --work runs/learned_evidence/<movie>_selftrain \\
        --synthetic 'runs/learned_evidence/ld/shards/train_*.npz' [--labels benchmark/labels/ld_v1.json]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize

from .finetune import SHIPPED

# the per-bin decoder as frozen before 25 Sep 2026 (the development numbers were measured with it)
FROZEN = dict(edge_mask="movie", gone="flag", fit="l1")


def census_of(field: Path, grains_path: Path | None = None) -> list[dict]:
    doc = json.loads(Path(grains_path or Path(field) / "grains.json").read_text())
    return list(doc["grains"].values()) if isinstance(doc["grains"], dict) else list(doc["grains"])


def first_pass(pcache: Path, field: Path, grains_path: Path | None = None, log=print, ghosts: bool = True,
               **dec) -> list[dict]:
    """Every grain read by the per-bin decoder (``reach.analyze``, so census ghosts are handled as the pipeline
    handles them), with its path at every bin."""
    from sparsetrack import stack

    from . import reach

    _, meta = stack.load(pcache)
    n = int(meta["n_bins"]) - int(meta.get("ref_start", 0))
    pred = reach.analyze(pcache, field, grains_path=grains_path, log=log, ghosts=ghosts, paths=tuple(range(n)), **dec)
    for res in pred["grains"]:
        res["_paths"] = {str(k): v for k, v in res.get("_paths", {}).items()}
    return pred["grains"]


def _frame_index(res: dict, flag: str) -> int | None:
    for f in res.get("flags", []):
        if f.startswith(flag + ":"):
            return int(np.searchsorted(res["length"]["frames"], int(f.split(":")[1])))
    return None


def pseudo_labels(reads: list[dict], census: list[dict], meta: dict, min_final: float = 8.0, agree_px: float = 2.0,
                  agree_rel: float = 0.10, spacing: int = 6, onset_guard: int = 8, apex_margin: float = 1.0,
                  end_px: float = 1.0) -> tuple[dict, dict]:
    """The decoder's readings that agree with its growth curve, as labels in the labelling tool's format."""
    from .prefill import simplify, to_length

    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    labels, n_tr = {}, 0
    for res in reads:
        if res["status"] not in ("emerged_within", "emerged_at_start") or res["final_length_px"] < min_final:
            continue
        if any(f.startswith("unsteady") for f in res.get("flags", [])):
            continue
        raw, fit = np.asarray(res["raw_reach_px"], float), np.asarray(res["length"]["px"], float)
        stop = len(fit)
        for flag, extra in (("burst_after", 1), ("no_grain_after", 0)):
            k = _frame_index(res, flag)
            if k is not None:
                stop = min(stop, k + extra)
        away = set()  # bins where the grain had left its place for a while (reach.py's gone="hold"): not its tube
        for f in res.get("flags", []):
            if f.startswith("gone:"):
                a, b = (int(v) for v in f.split(":")[1].split("-"))
                frames = res["length"]["frames"]
                away |= set(range(int(np.searchsorted(frames, a)), int(np.searchsorted(frames, b)) + 1))
        paths = {int(i): np.asarray(p, float) for i, p in (res.get("_paths") or {}).items()}
        ok = [i for i in range(stop) if fit[i] >= 2.0 and raw[i] > 0 and i in paths and len(paths[i]) >= 2
              and i not in away and abs(raw[i] - fit[i]) <= max(agree_px, agree_rel * fit[i])]
        chosen, last = [], None
        for i in reversed(ok):
            if last is None or last - i >= spacing:
                chosen.append(i)
                last = i
        if not chosen:
            continue
        traces = {}
        for i in sorted(chosen):
            pts = to_length(simplify(paths[i]), max(2.0, float(fit[i]) - end_px - apex_margin), max_extend=0.0)
            traces[str(rs + i)] = {"bin": rs + i, "state": "full", "path_xy_ref": np.round(pts, 2).tolist()}
        onset = {"verdict": "unknown"}
        if res["status"] == "emerged_within":
            absent = rs + int(np.argmax(fit >= 2.0)) - 1 - onset_guard
            if absent >= rs + 3:
                onset = {"verdict": "emerged_within", "last_absent_bin": absent}
        labels[res["id"]] = {"onset": onset, "traces": traces}
        n_tr += len(traces)
    doc = {"grains": {g["id"]: g for g in census}, "n_bins": nb, "labels": labels, "origin": "selftrain"}
    return doc, {"grains": len(labels), "traces": n_tr}


# ----------------------------------------------------------------------------- image (end-state) traces
def _registered(R_img, meta, grain, half):
    import sparsetrack.analyze as A
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    img = np.stack([R_img.crop(b, grain["x"], grain["y"], half) for b in range(rs, nb)]).astype(np.float32)
    if np.isnan(img).any():
        img = np.nan_to_num(img, nan=float(np.nanmedian(img)))
    ls = A.local_shifts(img, half - 0.5, float(grain["r"]), 12.0, 3)
    for i, (dx, dy) in enumerate(ls):
        img[i] = cv2.warpAffine(img[i], np.float32([[1, 0, -dx], [0, 1, -dy]]), img.shape[1:][::-1],
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return img, ls


def _axis(region, ring, rg, gr):
    sk = skeletonize(region)
    sk = sk if sk.sum() >= 2 else region
    near = sk & ring if (sk & ring).any() else sk
    rmin = float(rg[near].min())
    mcp = MCP_Geometric(np.where(sk, 1.0, np.inf))
    cum, _ = mcp.find_costs(list(zip(*np.nonzero(sk & (rg <= rmin + 0.5)))))
    good = sk & np.isfinite(cum)
    if not good.any():
        return None, 0.0
    far = np.unravel_index(int(np.argmax(np.where(good, cum, -1.0))), cum.shape)
    return np.asarray(mcp.traceback(far), float)[:, ::-1], float(cum[far]) + max(0.0, rmin - gr)


def endstate_traces(R_img, meta, grain, others, hint, half: int = 150, smooth: float = 1.0, k_end: float = 6.0,
                    k_rel: float = 8.0, disc: float = 4.0, ring_w: float = 6.0, max_exit_deg: float = 60.0,
                    min_len: float = 10.0, min_elong: float = 3.0, spacing: int = 6) -> dict[int, np.ndarray]:
    """{bin index: path (reference x, y)} from the images for one grain (see the module doc, step 3)."""
    from .reach import monotone_l1

    x, ls = _registered(R_img, meta, grain, half)
    T = len(x)
    x -= np.median(x.reshape(T, -1), axis=1)[:, None, None]
    if smooth > 0:
        x = np.stack([cv2.GaussianBlur(f, (0, 0), smooth) for f in x])
    B, A = x[:3].mean(0), x[T - 4:T - 1].mean(0)
    noise = 1.4826 * float(np.median(np.abs(np.diff(x, axis=0)))) / np.sqrt(2.0)
    D = A - B
    gx, gy, gr, c = float(grain["x"]), float(grain["y"]), float(grain["r"]), half - 0.5
    yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float64)
    rg = np.hypot(xx - c, yy - c)
    blocked = rg < gr + disc  # grains change and move a little: their discs and edges are not tube evidence
    for o in others:
        ox, oy = o["x"] - gx + c, o["y"] - gy + c
        if -o["r"] - 10 < ox < 2 * half + o["r"] + 10 and -o["r"] - 10 < oy < 2 * half + o["r"] + 10:
            blocked |= np.hypot(xx - ox, yy - oy) < o["r"] + disc
    k = 3 + int(np.ceil(np.abs(ls).max()))
    blocked[:k], blocked[-k:], blocked[:, :k], blocked[:, -k:] = True, True, True, True
    mad = 1.4826 * float(np.median(np.abs(D[~blocked] - np.median(D[~blocked]))))
    S = (np.abs(D) > max(k_end * noise * np.sqrt(2.0 / 3.0), k_rel * mad)) & ~blocked
    ring = (rg >= gr + disc) & (rg <= gr + disc + ring_w)
    hm = np.zeros(S.shape, np.uint8)
    q = np.round((np.asarray(hint, float) - [gx - half + 0.5, gy - half + 0.5]) * 4).astype(np.int32)
    cv2.polylines(hm, [q.reshape(-1, 1, 2)], False, 1, 3, shift=2)
    hm = (hm > 0) & ~blocked
    if hm.sum() < 3:
        return {}
    _, lab = cv2.connectedComponents(S.astype(np.uint8), connectivity=8)
    best, cover = None, 0.3
    for cid in np.unique(lab[S & ring]):
        if cid == 0:
            continue
        comp = lab == cid
        ey, ex = np.nonzero(comp & ring)
        ang = np.degrees(np.arctan2(ey - c, ex - c))
        a0 = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(ang)))))
        if np.max(np.abs((ang - a0 + 180) % 360 - 180)) * 2 > max_exit_deg:
            continue  # hugging the grain (it moved or swelled), or several tubes joined
        route, L = _axis(comp, ring, rg, gr)
        if route is None or L < min_len or L < min_elong * comp.sum() / max(L, 1.0):
            continue
        cv_ = float((comp & hm).sum()) / hm.sum()
        if cv_ >= cover:
            best, cover = comp, cv_
    if best is None:
        return {}
    traces, lens = {}, {}
    for i in list(range(3, T - 1, spacing)) + [T - 2]:
        like = best & (np.abs(x[i] - A) < np.abs(x[i] - B))  # already at its end-state level: built by bin i
        if not (like & ring).any():
            continue
        _, lb = cv2.connectedComponents(like.astype(np.uint8), connectivity=8)
        keep = np.unique(lb[like & ring])
        region = np.isin(lb, keep[keep > 0])
        route, L = _axis(region, ring, rg, gr)
        if route is None or L < 2.0:
            continue
        ey, ex = np.nonzero(region & ring)
        v = (np.array([ex.mean(), ey.mean()]) if len(ex) else route[0]) - c
        rim = c + v * gr / max(float(np.hypot(*v)), 1e-9)
        traces[i] = np.vstack([rim, route]) + [gx - half + ls[i][0] + 0.5, gy - half + ls[i][1] + 0.5]
        lens[i] = L + disc
    if not traces:
        return {}
    idx = sorted(lens)
    raw = np.array([lens[i] for i in idx])
    fit = monotone_l1(raw, vmax=max(4.0, float(raw.max())))
    return {i: traces[i] for i, r, f in zip(idx, raw, fit) if abs(r - f) <= max(2.0, 0.1 * f)}


def _hint(read: dict):
    """The start model's path at its latest reading that agrees with its growth curve."""
    raw, fit = np.asarray(read["raw_reach_px"], float), np.asarray(read["length"]["px"], float)
    paths = read.get("_paths") or {}
    ok = [i for i in range(len(fit)) if fit[i] >= 2.0 and raw[i] > 0 and len(paths.get(str(i), [])) >= 2
          and abs(raw[i] - fit[i]) <= max(2.0, 0.1 * fit[i])]
    return paths[str(max(ok))] if ok else None


def add_image_traces(labels: dict, reads: list[dict], field: Path, meta: dict, extend_px: float = 8.0,
                     extend_rel: float = 0.25, under_px: float = 10.0, under_rel: float = 0.25,
                     apex_cut: float = 4.0, log=print) -> dict:
    """Step 3 of the module doc, in place; returns counts."""
    from sparsetrack import stack
    from sparsetrack.render import Renderer

    from .prefill import to_length

    R = Renderer(*stack.load(field))
    census = census_of(field)
    by_id = {g["id"]: g for g in census}
    physical = [g for g in census if g.get("exclude_reason") != "not_a_grain"]
    rs = int(meta.get("ref_start", 0))
    stats = {"grains_extended": 0, "image_traces": 0, "grains_made_tube_only": 0}
    t0 = time.time()
    for rd in reads:
        g, hint = by_id.get(rd["id"]), None
        if g is not None and rd["status"] != "no_emergence_by_end":
            hint = _hint(rd)
        if hint is None:
            continue
        tr = endstate_traces(R, meta, g, [o for o in physical if o["id"] != g["id"]], hint)
        if not tr:
            continue
        fit = np.asarray(rd["length"]["px"], float)
        lens = {i: float(np.sum(np.hypot(*np.diff(p, axis=0).T))) for i, p in tr.items()}
        lab = labels["labels"].setdefault(g["id"], {"onset": {"verdict": "unknown"}, "traces": {}})
        if lens[max(lens)] > fit[-1] + max(under_px, under_rel * fit[-1]):
            for t in lab["traces"].values():
                t["state"] = "partial"
            lab["onset"] = {"verdict": "unknown"}
            stats["grains_made_tube_only"] += 1
        absent = lab["onset"].get("last_absent_bin")
        if absent is not None and any(rs + i <= absent and L >= 2.0 for i, L in lens.items()):
            lab["onset"] = {"verdict": "unknown"}
        added = 0
        for i, p in tr.items():
            if str(rs + i) in lab["traces"] or lens[i] < fit[i] + max(extend_px, extend_rel * fit[i]):
                continue
            cut = to_length(p, max(2.0, lens[i] - apex_cut), max_extend=0.0)
            lab["traces"][str(rs + i)] = {"bin": rs + i, "state": "partial", "path_xy_ref": np.round(cut, 2).tolist(),
                                          "source": "image"}
            added += 1
        stats["image_traces"] += added
        stats["grains_extended"] += added > 0
        if not lab["traces"]:
            del labels["labels"][g["id"]]
    log(f"image traces: {stats} ({time.time() - t0:.0f} s)")
    return stats


# ----------------------------------------------------------------------------- the whole step
def adapt(field: Path, pcache: Path, work: Path, start: Path = SHIPPED, dec: dict | None = None, steps: int = 600,
          lr: float = 3e-4, distill: float = 0.5, pos_px: float = 1.0, images: bool = True, synthetic=(),
          seed: int = 0, ghosts: bool = True, log=print, **plab) -> Path:
    """First pass, pseudo-labels, tuning; returns the tuned model (``work/unet_selftrain.pt``)."""
    from sparsetrack import stack

    from . import finetune as FT
    from .train import load_shards

    work.mkdir(parents=True, exist_ok=True)
    reads = first_pass(pcache, field, log=log, ghosts=ghosts, **(dec or {}))
    _, meta = stack.load(field)
    labels, summary = pseudo_labels(reads, census_of(field), meta, **plab)
    if images:
        summary.update(add_image_traces(labels, reads, field, meta, log=log))
    (work / "pseudo_labels.json").write_text(json.dumps(labels))
    if not labels["labels"]:
        raise SystemExit("no reading agrees with its growth curve: nothing to adapt on")
    keep, FT.POS_PX = FT.POS_PX, pos_px  # only the core of a tube is taught; its walls are held to the start model
    try:
        real = FT.real_samples(field, labels, seed=seed)
    finally:
        FT.POS_PX = keep
    log(f"pseudo-labels: {summary}; {len(real['x'])} training crops")
    syn = load_shards(list(synthetic)) if synthetic else None
    if syn is None:  # without them thin tubes drifted on a development movie (-6 lengths against with them)
        log("warning: no synthetic shards (--synthetic): tuning without replay of the training crops")
    net = FT.tune(start, real, syn=syn, steps=steps, lr=lr, distill=distill, seed=seed, log=log)
    return FT.save(net, work / "unet_selftrain.pt", field=str(field), started_from=str(start), steps=steps, lr=lr,
                   distill=distill, pos_px=pos_px, images=images, synthetic=list(synthetic), **summary)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", required=True, help="the movie's prepared cache")
    ap.add_argument("--pcache", required=True, help="the start model's probability cache of that movie")
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", default=str(SHIPPED), help="start model (the one that built --pcache)")
    ap.add_argument("--vmax", type=float, default=4.0, help="the per-bin decoder's speed cap, as the pipeline set it")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--no-images", action="store_true", help="pseudo-traces from the start model's readings only")
    ap.add_argument("--synthetic", nargs="*", default=[], help="synthetic shard globs mixed into every batch")
    ap.add_argument("--frozen-decoder", action="store_true",
                    help="the per-bin decoder as frozen before 25 Sep 2026 (as the development numbers were measured)")
    ap.add_argument("--labels", default=None,
                    help="the movie's labels: score the start and the self-trained readings on them (not used to adapt)")
    args = ap.parse_args(argv)
    field, work = Path(args.field), Path(args.work)
    if args.labels and "m2" in Path(args.labels).name:
        raise SystemExit("movie 2 is the held-out benchmark: test self-training on the dev movie's labels")
    dec = dict(big=300, burst=True, vmax=args.vmax, **(FROZEN if args.frozen_decoder else {}))
    import glob
    shards = sorted({f for p in args.synthetic for f in glob.glob(p)})
    ghosts = not args.frozen_decoder  # the frozen decoder analysed every census disc
    model = adapt(field, Path(args.pcache), work, start=Path(args.model), dec=dec, steps=args.steps,
                  images=not args.no_images, synthetic=shards, ghosts=ghosts)
    from . import evaluate, reach
    from .model import load as load_model
    grains_path = Path(args.labels) if args.labels else None  # scored as the dev test scores: the labels' census
    pc = evaluate.prob_cache(field, load_model(str(model)), work / "prob_cache")
    try:
        pred = reach.analyze(pc, field, grains_path=grains_path, log=print, ghosts=ghosts, **dec)
    finally:
        shutil.rmtree(pc, ignore_errors=True)
    (work / "predictions.json").write_text(json.dumps(pred))
    print(f"{work / 'predictions.json'}: the movie read with the self-trained model {model}")
    if args.labels:
        start = reach.analyze(Path(args.pcache), field, grains_path=grains_path, log=lambda *a: None, ghosts=ghosts,
                              **dec)
        lines = check(start, pred, Path(args.labels))
        print("\n".join(lines))
        (work / "check.txt").write_text("\n".join(lines) + "\n")


def check(start: dict, tuned: dict, labels_path: Path) -> list[str]:
    """Start and self-trained readings scored on the movie's labels, and whether self-training would be adopted."""
    from sparsetrack.evaluate import load, score

    from . import evaluate
    from .finetune import adopted

    labels = load(labels_path)
    rs, rt = score(labels, start), score(labels, tuned)
    pb = evaluate.paired_bootstrap(rt, rs, rs["onset"]["tolerance_frames"])
    verdict = ("would be adopted: better beyond noise, worse on neither" if adopted(pb) else
               "not adopted: not better beyond noise (95% interval above zero) without being worse on the other")
    return [f"{labels_path.name}: {rs['grains_scored']} grains scored (the labels were not used to adapt)",
            evaluate.e2e_summary("start", rs), evaluate.e2e_summary("selftrain", rt),
            f"selftrain - start over {pb['grains']} grains: onset {pb['onset_diff']:+.0f} [{pb['onset_ci'][0]:+.0f}, "
            f"{pb['onset_ci'][1]:+.0f}], lengths {pb['length_diff']:+.0f} [{pb['length_ci'][0]:+.0f}, "
            f"{pb['length_ci'][1]:+.0f}] (95% paired bootstrap over grains)", verdict]


if __name__ == "__main__":
    main()
