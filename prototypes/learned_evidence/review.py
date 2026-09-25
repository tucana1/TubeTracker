"""Review pictures for the per-bin decoder, drawn on the movie itself, in SparseTrack's gallery.

For every grain: six registered image crops from its onset to the last bin, with the region the
decoder read (green) and its medial axis (yellow), then the length curve: the raw reading per bin
(dots), the monotone fit (line), onset and burst (if any) marked. ``index.html`` is SparseTrack's
own review gallery (``sparsetrack.report.write_gallery``) over these pictures.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sparsetrack import stack
from sparsetrack.render import Renderer
from sparsetrack.report import write_gallery

from .reach import reach_grain


def _tile(v, size: int, zoom_half: int, centre: float, gr: float) -> np.ndarray:
    im, region, axis = v
    c = int(round(centre))
    lo, hi = c - zoom_half, c + zoom_half
    crop = im[lo:hi, lo:hi]
    a, b = np.percentile(crop, [1, 99])
    g = np.clip((crop - a) / max(b - a, 1e-6) * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if region is not None:
        cnts, _ = cv2.findContours(region[lo:hi, lo:hi].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cnts, -1, (60, 170, 60), 1)
        rgb[axis[lo:hi, lo:hi]] = (0, 220, 255)
    cv2.circle(rgb, (zoom_half, zoom_half), int(round(gr)), (170, 170, 170), 1)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)


def _curve(res: dict, width: int, height: int, fpb: int) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = np.asarray(res["length"]["frames"], float)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.plot(frames, res["raw_reach_px"], ".", ms=2, color="#9a9890", label="per bin")
    ax.plot(frames, res["length"]["px"], "-", lw=1.5, color="#2a78d6", label="monotone fit")
    if res.get("onset_frame") is not None:
        ax.axvline(res["onset_frame"], color="#1baf7a", lw=1, label="onset")
    if res.get("burst_frame") is not None:
        ax.axvline(res["burst_frame"], color="#eb6834", lw=1, label="burst?")
    ax.set_xlabel("frame")
    ax.set_ylabel("length (px)")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1].copy()
    plt.close(fig)
    return img


def write_review(pcache: str | Path, image_cache: str | Path, pred: dict, out_dir: str | Path,
                 grains_path: str | Path | None = None, tiles: int = 6, tile_px: int = 150, zoom_half: int = 60,
                 **kw) -> Path:
    """Pictures in ``out_dir/diagnostics`` and SparseTrack's gallery in ``out_dir/index.html``.
    ``kw`` must match the settings that made ``pred`` (for instance ``burst``, ``big``, ``vmax``)."""
    out_dir = Path(out_dir)
    (out_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    bins_p, meta = stack.load(pcache)
    RP, R_img = Renderer(bins_p, meta), Renderer(*stack.load(image_cache))
    src = Path(grains_path) if grains_path else Path(pcache) / "grains.json"
    doc = json.loads(src.read_text())
    census = list(doc["grains"].values()) if isinstance(doc["grains"], dict) else doc["grains"]
    ghosts = set(pred.get("ghosts") or {})  # census discs the analysis judged not grains (reach.census_ghosts)
    physical = [g for g in census if g.get("exclude_reason") != "not_a_grain" and g["id"] not in ghosts]
    by_id = {g["id"]: g for g in census}
    fpb, rs, nb = int(meta["frames_per_bin"]), int(meta.get("ref_start", 0)), int(meta["n_bins"])
    n = nb - rs
    for res in pred["grains"]:
        g = by_id.get(res["id"])
        if g is None:
            continue
        first = rs
        if res.get("onset_frame") is not None:
            first = max(rs, int(res["onset_frame"]) // fpb)
        keep = tuple(sorted({int(round(v)) - rs for v in np.linspace(first, nb - 1, tiles)} & set(range(n))))
        others = [o for o in physical if o["id"] != g["id"]]
        full = reach_grain(RP, R_img, meta, g, others, keep=keep, **kw)
        views, centre = full["_views"], full["_centre"]
        zoom = int(min(centre, max(zoom_half, float(res.get("final_length_px") or 0.0) + g["r"] + 12)))
        row = np.hstack([np.pad(_tile(views[i], tile_px, zoom, centre, g["r"]),
                                ((0, 0), (0, 2), (0, 0)), constant_values=255) for i in sorted(views)])
        curve = _curve(res, row.shape[1], 170, fpb)
        curve = cv2.resize(curve, (row.shape[1], curve.shape[0]))
        cv2.imwrite(str(out_dir / "diagnostics" / f"{res['id']}.png"), np.vstack([row, curve]))
    shown = [{**r, "path_length_px": max(r.get("raw_reach_px") or [0.0])} for r in pred["grains"]]
    page = write_gallery({**pred, "grains": shown, "frames_per_bin": fpb,
                          "method": pred.get("method", "per-bin decoder"), "movie": meta.get("movie", {})}, out_dir)
    text = page.read_text()  # SparseTrack's gallery describes its own panels: describe these instead
    text = text.replace("<title>SparseTrack review</title>", "<title>Per-bin decoder review</title>")
    text = text.replace("<h1>SparseTrack review", "<h1>Per-bin decoder review")
    text = text.replace("panels: end state with the traced path, end-state change map, kymograph (time down, "
                        "arclength right) with the growth front.",
                        "panels: six registered bins from onset to the end, with the region the per-bin decoder "
                        "read (green) and its medial axis (yellow); below, the length read in each bin (dots) and "
                        "its monotone fit. \"path\" is the longest single-bin reading.")
    page.write_text(text)
    return page
