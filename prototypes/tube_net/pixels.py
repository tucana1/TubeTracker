"""Pixel check of a tube network on a labelled movie without writing its probability movie to disk.

    python -m prototypes.tube_net.pixels MODEL.pt MOVIE [MOVIE ...] [--json OUT]     # MOVIE = ld | m2 | m1

Only the bins the checks read are computed (full frames, exactly as ``learned.prob_cache`` computes them), held in
memory: the traced bins and every third bin before each grain's onset. Checks (``prototypes.synth_v6.recall``):
traced-tube marking from the grain's visible edge (all FULL traces, first trace per grain, young <= 8 px, long
>= 50 px), stubs and tips seen, marks 8-14 px beside tubes, and rim marks before onset (grain-bins with >= 6 px of
P >= 0.5 within r..r+7 px, where the flood could start a tube on the grain itself).

Movie 1 (30 Sep 2026; nothing is written to its cache): its grains move a median 33 px by the end, so its rim check
looks for each grain where it is at every bin (the labelling tool's own per-bin grain offsets, ``--follow``, the
default for movie 1) instead of at its first trace's place (ld and m2, as recorded).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from prototypes.synth_v6.recall import measure, rim_false_marks, summarise
from sparsetrack import learned, stack

REPO = Path(__file__).resolve().parents[2]
MOVIES = {"ld": ("runs/sparsetrack/ld", "benchmark/labels/ld_v1.json"),
          "m2": ("runs/sparsetrack/m2", "benchmark/labels/m2_v1.json"),
          "m1": ("runs/sparsetrack/m1", "benchmark/labels/m1_v1.json")}
FOLLOW_DEFAULT = {"m1"}  # movies whose rim check follows each grain bin by bin


def needed_bins(labels: dict, meta: dict, step: int = 3) -> list[int]:
    """The bins ``measure`` and ``rim_false_marks`` read."""
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    out = set()
    for gid, lab in labels["labels"].items():
        if labels["grains"][gid].get("excluded"):
            continue
        out |= {int(b) for b, t in lab.get("traces", {}).items() if t["state"] == "full"}
        on = lab.get("onset", {})
        end = (int(on["first_visible_bin"]) - 2 if on.get("verdict") == "emerged_within"
               else nb - 1 if on.get("verdict") == "no_emergence_by_end" else None)
        if end is not None:
            out |= set(range(rs + 3, end, step))
    return sorted(out)


def tiled_probability(net, img: np.ndarray, early: np.ndarray, late: np.ndarray, tile: int,
                      batch: int = 16) -> np.ndarray:
    """``learned.tube_probability`` on overlapping ``tile`` px tiles (half-tile stride, Hann-weighted blend), inputs
    normalised as there: with GroupNorm each tile is normalised over itself, not over the frame. (Measured, not
    adopted: see README.)"""
    import torch
    m = learned.local_background(early, net.bg_px) if getattr(net, "bg_px", 0) else float(np.nanmedian(early))
    x = np.nan_to_num((np.stack([img, early, late]).astype(np.float32) - m) / learned.IN_SCALE, nan=0.0)
    stride = tile // 2
    _, h, w = x.shape
    ny, nx = max(int(np.ceil((h - tile) / stride)), 0) + 1, max(int(np.ceil((w - tile) / stride)), 0) + 1
    H, W = max((ny - 1) * stride + tile, h), max((nx - 1) * stride + tile, w)
    x = np.pad(x, ((0, 0), (0, H - h), (0, W - w)), mode="reflect")
    win = np.outer(np.hanning(tile + 2)[1:-1], np.hanning(tile + 2)[1:-1]).astype(np.float32)
    out, wsum = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    origins = [(i * stride, j * stride) for i in range(ny) for j in range(nx)]
    dev = next(net.parameters()).device
    for k in range(0, len(origins), batch):
        chunk = origins[k:k + batch]
        xb = torch.from_numpy(np.stack([x[:, y:y + tile, xx:xx + tile] for y, xx in chunk])).to(dev)
        with torch.no_grad():
            pr = torch.sigmoid(net(xb))[:, 0].cpu().numpy()
        for (y, xx), pp in zip(chunk, pr):
            out[y:y + tile, xx:xx + tile] += pp * win
            wsum[y:y + tile, xx:xx + tile] += win
    return (out / np.maximum(wsum, 1e-6))[:h, :w]


def prob_movie(model: str | Path, movie: str, bins: list[int], net=None, log=print, tile: int = 0, bg: int = 0
               ) -> tuple[np.ndarray, dict]:
    """(uint8 P x P_SCALE for every bin - zero where not computed, meta in reference coordinates), as
    ``learned.prob_cache`` would store them (``bg`` > 0: override the checkpoint's ``bg_px``; ``tile`` > 0: on tiles
    of that size)."""
    cache = REPO / MOVIES[movie][0]
    src, meta = stack.load(cache)
    net = net or learned.load_model(model)
    if bg:
        net.bg_px = bg
    shifts = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    early = np.mean([learned._registered(src, shifts, b) for b in range(rs, rs + 3)], axis=0)
    late = np.mean([learned._registered(src, shifts, b) for b in range(nb - 4, nb - 1)], axis=0)
    arr = np.zeros(src.shape, np.uint8)  # untouched bins cost no memory
    t0 = time.time()
    for b in bins:
        img = learned._registered(src, shifts, b)
        p = tiled_probability(net, img, early, late, tile) if tile else learned.tube_probability(net, img, early, late)
        arr[b] = np.round(p * learned.P_SCALE).astype(np.uint8)
    log(f"  {movie}: {len(bins)} bins in {time.time() - t0:.0f} s")
    return arr, {**meta, "shifts": [[0.0, 0.0]] * nb, "raw_shifts": [[0.0, 0.0]] * nb}


def extra(rows: list[dict]) -> dict:
    long = [r for r in rows if r["length"] >= 50]
    ne, he = sum(r["n_edge"] for r in long), sum(r["hit_edge"] for r in long)
    return {"long>=50": {"traces": len(long), "point_recall": he / max(ne, 1)}}


def tip_extent(pm: tuple, labels: dict, step: float = 0.5, reach: float = 12.0) -> list[dict]:
    """Per FULL trace (>= 6 px, not touching anything): where the marks (P >= 0.5 within +/-1.5 px across) end
    along the tube, relative to the traced apex - walking out beyond it along the last 5 px's direction (> 0: marked
    past the apex; the flood reads that as length) or, if the apex is not marked, back along the trace (< 0)."""
    from prototypes.learned_flood.realpix import trace_points
    from sparsetrack.render import Renderer
    import cv2
    R = Renderer(*pm)
    out = []
    for gid, lab in labels["labels"].items():
        g = labels["grains"][gid]
        if g.get("excluded"):
            continue
        for b, t in lab.get("traces", {}).items():
            path = t.get("path_xy_ref") or []
            if t["state"] != "full" or t.get("contact") or len(path) < 2 or t["length_px"] < 6:
                continue
            pts, nrm = trace_points(path, step)
            d = pts[-1] - pts[max(len(pts) - 1 - int(5 / step), 0)]
            d = d / (np.linalg.norm(d) + 1e-9)
            n = np.array([-d[1], d[0]])
            ahead = pts[-1] + np.outer(np.arange(step, reach + 1e-9, step), d)
            half = int(np.ceil(np.max(np.abs(np.vstack([pts, ahead]) - [g["x"], g["y"]])))) + 8
            img = np.nan_to_num(R.crop(int(b), g["x"], g["y"], half) / learned.P_SCALE).astype(np.float32)

            def marked(q, nn):
                q = q - [g["x"] - half, g["y"] - half] - 0.5
                v = [cv2.remap(img, (q[:, 0] + o * nn[..., 0]).astype(np.float32)[None],
                               (q[:, 1] + o * nn[..., 1]).astype(np.float32)[None], cv2.INTER_LINEAR)[0]
                     for o in (-1.5, -0.75, 0, 0.75, 1.5)]
                return np.max(v, axis=0) >= 0.5
            if marked(pts[-1:], nrm[-1:])[0]:
                m = marked(ahead, np.repeat(n[None], len(ahead), 0))
                ext = float(step * (np.argmin(m) if not m.all() else len(m)))
            else:
                m = marked(pts[::-1], nrm[::-1])
                ext = -float(step * np.argmax(m)) if m.any() else -float(t["length_px"])
            out.append({"grain": gid, "bin": int(b), "length": float(t["length_px"]), "extent": ext})
    return out


def rim_followed(pm: tuple, labels: dict, offsets: dict, step: int = 3, min_px: int = 6) -> dict:
    """``recall.rim_false_marks`` with each grain looked for where it is at every bin (``offsets``: grain -> (n_bins,
    2) offsets from its census place) rather than at its first trace's place: grain-bins before the human's first
    visible bin - 2 (every bin of grains that never germinate) with >= ``min_px`` pixels of P >= 0.5 within r..r+7."""
    from sparsetrack.render import Renderer
    R = Renderer(*pm)
    rs, nb = int(pm[1].get("ref_start", 0)), int(pm[1]["n_bins"])
    marked = total = 0
    grains_marked = set()
    for gid, lab in labels["labels"].items():
        g = labels["grains"][gid]
        if g.get("excluded"):
            continue
        on = lab.get("onset", {})
        end = (int(on["first_visible_bin"]) - 2 if on.get("verdict") == "emerged_within"
               else nb - 1 if on.get("verdict") == "no_emergence_by_end" else None)
        if end is None:
            continue
        off = offsets.get(gid)
        half = int(g["r"] + 12)
        yy, xx = np.mgrid[0:2 * half, 0:2 * half]
        d = np.hypot(xx + 0.5 - half, yy + 0.5 - half)
        zone = (d >= g["r"]) & (d <= g["r"] + 7)
        for b in range(rs + 3, end, step):
            ox, oy = (off[b] if off is not None else (0.0, 0.0))
            p = np.nan_to_num(R.crop(b, g["x"] + ox, g["y"] + oy, half)) / learned.P_SCALE
            total += 1
            if int(((p >= 0.5) & zone).sum()) >= min_px:
                marked += 1
                grains_marked.add(gid)
    return {"grain_bins": total, "marked": marked, "share": marked / max(total, 1), "grains": len(grains_marked),
            "followed": True}


_OFFSETS: dict = {}


def check(model: str | Path, movie: str, log=print, net=None, tile: int = 0, bg: int = 0,
          follow: bool | None = None) -> dict:
    labels = json.loads((REPO / MOVIES[movie][1]).read_text())
    _, meta = stack.load(REPO / MOVIES[movie][0])
    pm = prob_movie(model, movie, needed_bins(labels, meta), net=net, log=log, tile=tile, bg=bg)
    rows = measure(pm, REPO / MOVIES[movie][1])
    s = summarise(rows) | extra(rows)
    if (movie in FOLLOW_DEFAULT) if follow is None else follow:
        if movie not in _OFFSETS:
            from prototypes.tube_net.realdata import grain_offsets
            _OFFSETS[movie] = grain_offsets(movie)
        s["rim_before_onset"] = rim_followed(pm, labels, _OFFSETS[movie])
    else:
        s["rim_before_onset"] = rim_false_marks(pm, REPO / MOVIES[movie][1])
    te = tip_extent(pm, labels)
    ext = np.array([r["extent"] for r in te])
    s["tip_extent"] = {"traces": len(te), "median": float(np.median(ext)), "p25": float(np.percentile(ext, 25)),
                       "p75": float(np.percentile(ext, 75)), "within2": int(np.sum(np.abs(ext) <= 2.0)),
                       "past4": int(np.sum(ext > 4.0)), "short4": int(np.sum(ext < -4.0))}
    return {"summary": s, "rows": rows, "tip_extent": te}


def line(s: dict) -> str:
    a, f, y, lg, rim = s["all"], s["first"], s["young<=8"], s["long>=50"], s["rim_before_onset"]
    te = s.get("tip_extent")
    return (f"all {100 * a['point_recall']:.0f}% (stub {a['stub_seen']}/{a['traces']}, tip {a['tip_seen']}, "
            f"beside {100 * a['beside']:.1f}%) | long {100 * lg['point_recall']:.0f}% ({lg['traces']}) | "
            f"first {100 * f['point_recall']:.0f}% stub {f['stub_seen']}/{f['traces']} tip {f['tip_seen']} | "
            f"young {100 * y['point_recall']:.0f}% stub {y['stub_seen']}/{y['traces']} tip {y['tip_seen']} | "
            f"rim before onset {100 * rim['share']:.1f}%"
            + (f" | marks end vs apex: median {te['median']:+.1f} px (IQR {te['p25']:+.1f}..{te['p75']:+.1f}), "
               f"within 2 px {te['within2']}/{te['traces']}, >4 past {te['past4']}, >4 short {te['short4']}"
               if te else ""))


def per_grain(rows: list[dict], grains: list[str]) -> str:
    out = []
    for r in rows:
        if r["grain"] in grains:
            out.append(f"{r['grain']}@{r['bin']} {r['length']:.0f}px {100 * r['hit_edge'] / max(r['n_edge'], 1):.0f}%")
    return ", ".join(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("movies", nargs="+", choices=tuple(MOVIES))
    ap.add_argument("--json", help="write {movie: {summary, rows}} here")
    ap.add_argument("--grains", nargs="*", default=["g069", "g082", "g043", "g054", "g038", "g005", "g092"])
    ap.add_argument("--tile", type=int, default=0, help="compute the maps on tiles of this size (0: full frames)")
    ap.add_argument("--bg", type=int, default=0, help="inputs relative to the local background over this many px")
    ap.add_argument("--follow", action="store_true", default=None,
                    help="rim check with each grain followed bin by bin (default: movie 1 only)")
    a = ap.parse_args()
    net = learned.load_model(a.model)
    res = {}
    for mv in a.movies:
        res[mv] = check(a.model, mv, net=net, tile=a.tile, bg=a.bg, follow=a.follow)
        tag = (f" tiles {a.tile}" if a.tile else "") + (f" bg {a.bg}" if a.bg else "")
        print(f"{Path(a.model).stem}{tag} on {mv}: {line(res[mv]['summary'])}", flush=True)
        if mv == "m2" and a.grains:
            print(f"  {per_grain(res[mv]['rows'], a.grains)}", flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res))
