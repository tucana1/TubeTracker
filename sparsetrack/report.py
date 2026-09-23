"""Population and per-grain reports from SparseTrack predictions.

- ``turnbull``: nonparametric maximum-likelihood cumulative distribution of onset
  times from interval-censored observations (Turnbull's self-consistency EM). Onsets
  are only known to lie in an interval, some grains had emerged before observation
  started (left-censored) and some had not emerged by the end (right-censored).
- ``write_population`` / ``write_growth_curves``: CSV tables and their charts.
- ``write_video``: the registered field with every grain's tube drawn at its reported
  length, bin by bin (H.264 through ffmpeg).

Everything here reports model output; nothing is human-verified.
"""

from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path

import numpy as np

from .video import FFMPEG

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES, SERIES_LIGHT = "#2a78d6", "#cde2fb"


def turnbull(intervals: list[tuple[float, float]], tol: float = 1e-9, max_iter: int = 5000
             ) -> list[tuple[float, float, float]]:
    """Innermost intervals (q, p] with their probability masses for half-open (L, R] data.

    ``L`` may be -inf (left-censored) and ``R`` +inf (right-censored).
    """
    ends = [(lo, 1) for lo, _ in intervals] + [(hi, 0) for _, hi in intervals]
    ends.sort(key=lambda e: (e[0], e[1]))  # at ties a right end (0) precedes a left end (1)
    inner = [(a[0], b[0]) for a, b in zip(ends, ends[1:]) if a[1] == 1 and b[1] == 0]
    if not inner:
        return []
    alpha = np.array([[1.0 if lo <= q and p <= hi else 0.0 for q, p in inner] for lo, hi in intervals])
    mass = np.full(len(inner), 1.0 / len(inner))
    for _ in range(max_iter):
        denom = alpha @ mass
        new = (alpha * mass / denom[:, None]).mean(axis=0)
        if np.max(np.abs(new - mass)) < tol:
            mass = new
            break
        mass = new
    return [(q, p, float(m)) for (q, p), m in zip(inner, mass)]


def onset_intervals(pred: dict, ids: set[str] | None = None) -> list[tuple[float, float]]:
    """(L, R] per grain from predictions; unobservable grains are left out."""
    out = []
    for g in pred["grains"]:
        if ids is not None and g["id"] not in ids:
            continue
        frames = (g.get("length") or {}).get("frames") or [0]
        status = g.get("status")
        if status == "emerged_within" and g.get("onset_interval"):
            out.append((float(g["onset_interval"][0]), float(g["onset_interval"][1])))
        elif status == "emerged_at_start":
            out.append((-math.inf, float(frames[0])))
        elif status == "no_emergence_by_end":
            out.append((float(frames[-1]), math.inf))
    return out


def _axes_style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def write_population(pred: dict, out_dir: str | Path, ids: set[str] | None = None, label: str = "isolated grains"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    iv = onset_intervals(pred, ids)
    masses = turnbull(iv)
    counts = {s: sum(1 for g in pred["grains"] if (ids is None or g["id"] in ids) and g.get("status") == s)
              for s in ("emerged_within", "emerged_at_start", "no_emergence_by_end", "unobservable")}
    with open(out_dir / "population.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["interval_after_frame", "interval_by_frame", "mass", "cumulative_fraction_germinated"])
        cum = 0.0
        for q, p, m in masses:
            cum += m
            w.writerow([q, p, round(m, 6), round(cum, 6)])
    if not masses:
        return None
    frames = [f for g in pred["grains"] for f in ((g.get("length") or {}).get("frames") or [])]
    start, end = (float(min(frames)), float(max(frames))) if frames else (0.0, 1.0)
    # Two envelopes of the (interval-undetermined) NPMLE: mass counted once its interval
    # has closed ("germinated by t for certain") and once it has opened ("possibly by t").
    knots = sorted({start, end, *[v for q, p, _ in masses for v in (q, p) if math.isfinite(v) and start <= v <= end]})
    certain = np.array([sum(m for q, p, m in masses if p <= x) for x in knots])
    possible = np.array([sum(m for q, p, m in masses if q < x) for x in knots])
    t50 = next(((q, p) for (q, p, _), c in zip(masses, np.cumsum([m for _, _, m in masses]))
                if c >= 0.5 and math.isfinite(p)), None)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _axes_style(ax)
    ax.fill_between(knots, certain, possible, step="post", color=SERIES_LIGHT, linewidth=0)
    ax.step(knots, certain, where="post", color=SERIES, linewidth=2)
    if t50:
        ax.plot([t50[1]], [0.5], "o", color=SERIES, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"T50 by frame {t50[1]:.0f}", (t50[1], 0.5), xytext=(8, -14), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, end)
    ax.set_xlabel("Source frame", fontsize=9, color=INK2)
    ax.set_ylabel("Fraction germinated", fontsize=9, color=INK2)
    n = sum(counts.values()) - counts["unobservable"]
    ax.set_title(f"Cumulative germination, {label} (model output, SparseTrack)", fontsize=10, color=INK, loc="left",
                 pad=20)
    ax.text(0, 1.025, f"n = {n}: {counts['emerged_within']} emerged during the movie, {counts['emerged_at_start']} "
            f"already at start, {counts['no_emergence_by_end']} none by the end; band = onset-interval uncertainty",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_dir / "population.png", facecolor=SURFACE)
    plt.close(fig)
    return {"n": n, "counts": counts, "t50_interval": t50}


def write_growth_curves(pred: dict, out_dir: str | Path, ids: list[str], cols: int = 7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grains = {g["id"]: g for g in pred["grains"]}
    ids = [i for i in ids if i in grains]
    if not ids:
        return
    rows = -(-len(ids) // cols)
    peaks = np.array([max(grains[i]["length"]["px"] or [0]) for i in ids])
    ymax = float(min(peaks.max(), 1.1 * np.percentile(peaks, 90))) or 1.0  # a few runaway tubes must not flatten the rest
    xmax = max(max(grains[i]["length"]["frames"] or [0]) for i in ids)
    fig, axes = plt.subplots(rows, cols, figsize=(1.55 * cols, 1.25 * rows + 0.6), dpi=150, sharex=True, sharey=True,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for k, ax in enumerate(axes.flat):
        if k >= len(ids):
            ax.axis("off")
            continue
        g = grains[ids[k]]
        _axes_style(ax)
        ax.plot(g["length"]["frames"], g["length"]["px"], color=SERIES, linewidth=1.6)
        if g.get("onset_frame") is not None and g.get("status") == "emerged_within":
            ax.axvline(g["onset_frame"], color=INK2, linewidth=0.7)
        note = " · contact" if "contact_censored" in g.get("flags", []) else ""
        if max(g["length"]["px"] or [0]) > ymax * 1.05:
            note += f" · off scale ({max(g['length']['px']):.0f})"
        ax.set_title(f"{g['id']}{note}", fontsize=7.5, color=INK2, loc="left", pad=2)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax * 1.05)
        ax.tick_params(labelsize=6)
    fig.suptitle("Tube length (exit to apex, px) against source frame, per isolated grain; line = onset. "
                 "Model output, not human-verified.", fontsize=8.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(Path(out_dir) / "growth_curves.png", facecolor=SURFACE)
    plt.close(fig)


def write_video(renderer, meta: dict, pred: dict, out_path: str | Path, fps: int = 12, ids: set[str] | None = None):
    """Registered field per bin with each emerged grain's tube drawn at its reported length."""
    import cv2

    h, w = renderer.height, renderer.width
    rs, fpb = int(meta.get("ref_start", 0)), int(meta["frames_per_bin"])
    series_bgr = (214, 120, 42)
    grains = [g for g in pred["grains"] if ids is None or g["id"] in ids]
    geo = []
    for g in grains:
        path = np.array(g.get("path") or [], dtype=np.float64)
        if len(path) < 2:
            geo.append(None)
            continue
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
        geo.append((path, s))
    lo = hi = None
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for b in range(rs, renderer.n_bins):
            dx, dy = renderer.shifts[b]
            img = cv2.warpAffine(np.asarray(renderer.bins[b], np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]),
                                 (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if lo is None:
                lo, hi = np.percentile(img, [0.5, 99.8])
            frame = cv2.cvtColor(np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            t = b - rs
            n_em = 0
            for g, gg in zip(grains, geo):
                lengths = g["length"]["px"]
                L = lengths[t] if t < len(lengths) else 0.0
                if gg is None or L <= 0:
                    continue
                n_em += 1
                path, s = gg
                keep = path[s <= L]
                theta = math.radians((g.get("rotation_deg") or [0.0] * len(lengths))[t])
                c = np.array([g["x"], g["y"]])
                rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
                pts = (keep - c) @ rot.T + c
                if len(pts) >= 2:
                    cv2.polylines(frame, [np.round(pts * 4).astype(np.int32).reshape(-1, 1, 2)], False, series_bgr, 2,
                                  cv2.LINE_AA, shift=2)
                    tip = tuple(int(v) for v in np.round(pts[-1]))
                    cv2.putText(frame, f"{L:.0f}", (tip[0] + 4, tip[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(frame, f"{L:.0f}", (tip[0] + 4, tip[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (11, 11, 11), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (w, 22), (251, 252, 252), -1)
            cv2.putText(frame, f"frames {b * fpb}-{(b + 1) * fpb - 1}   tubes shown: {n_em}/{len(grains)}   "
                        f"(model output, lengths in px)", (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (11, 11, 11), 1,
                        cv2.LINE_AA)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
