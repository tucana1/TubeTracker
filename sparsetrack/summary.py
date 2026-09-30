"""Several movies side by side (conditions, genotypes, repeats): one row per movie and one figure.

    python -m sparsetrack summary MOVIE_OR_RUN_FOLDER [...] [--out runs/sparsetrack/summary]

Per movie, from its analysis (``runs/sparsetrack/<movie>/analysis/predictions.json``, isolated grains as in its
gallery): grains read, germinated share, T50 (the time half the grains had germinated, from the germination curve
that allows for each onset's interval), median growth rate and final length, grains lost partway; and, where the
movie has been reviewed (``review/review_labels.json``), the germinated share and T50 from the checked onsets. In
minutes and um with ``calibration.json``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from . import report

PALETTE = ("#2a78d6", "#d6632a", "#2aa876", "#8a4fd6", "#d62a6e", "#6e6e6e", "#b8860b", "#1f9aa6")


def t50_of(intervals: list[tuple[float, float]]) -> float | None:
    """The earliest time by which half the grains had surely germinated (the germination curve's lower envelope)."""
    masses = report.turnbull(intervals)
    cum = 0.0
    for q, p, m in masses:
        cum += m
        if cum >= 0.5 and math.isfinite(p):
            return float(p)
    return None


def germinated_share(intervals: list[tuple[float, float]]) -> float | None:
    """Share of grains germinated by the end of the movie (mass of every closed interval)."""
    masses = report.turnbull(intervals)
    return float(sum(m for q, p, m in masses if math.isfinite(p))) if masses else None


def envelope(intervals: list[tuple[float, float]], end: float) -> tuple[list[float], list[float]]:
    """(times, fraction surely germinated by then), a step curve for plotting."""
    masses = report.turnbull(intervals)
    knots = sorted({0.0, end, *[p for _, p, _ in masses if math.isfinite(p) and p <= end]})
    return knots, [sum(m for _, p, m in masses if p <= x) for x in knots]


def movie_row(folder: Path, units: tuple[float, float] | None = None) -> dict:
    pred = json.loads((folder / "analysis" / "predictions.json").read_text())
    census = folder / "cache" / "grains.json"
    iso = ({g["id"] for g in json.loads(census.read_text())["grains"] if g.get("isolated", True)}
           if census.exists() else {g["id"] for g in pred["grains"]})
    fpb = int(pred.get("frames_per_bin", 300))
    gs = [g for g in pred["grains"] if g["id"] in iso]
    observed = [g for g in gs if g.get("status") != "unobservable"]
    grown = [g for g in observed if g.get("status") in ("emerged_within", "emerged_at_start")]
    rates = [r * fpb for r in (report.growth_rate(g["length"]["frames"], g["length"]["px"]) for g in grown) if r]
    finals = [float(g.get("final_length_px") or (g["length"]["px"] or [0.0])[-1]) for g in grown]
    iv = report.onset_intervals(pred, iso)
    frames = [f for g in gs for f in ((g.get("length") or {}).get("frames") or [])]
    row = {"movie": folder.name, "grains": len(observed), "germinated": germinated_share(iv), "t50_frame": t50_of(iv),
           "growth_px_per_bin": float(np.median(rates)) if rates else None,
           "final_length_px": float(np.median(finals)) if finals else None,
           "lost_partway": sum(any(f.startswith("grain_lost_after") for f in g.get("flags", [])) for g in gs),
           "last_frame": float(max(frames)) if frames else None, "frames_per_bin": fpb,
           "_intervals": iv, "_rates": rates}
    rev = folder / "review" / "review_labels.json"
    if rev.exists():
        from .review import population_input
        doc = json.loads(rev.read_text())
        riv = report.onset_intervals(population_input(doc), iso)  # the same grains as the model's curve
        checked = sum(1 for gid, lab in doc.get("labels", {}).items()
                      if gid in iso and not doc["grains"].get(gid, {}).get("excluded")
                      and (lab.get("onset") or {}).get("review_origin") == "human")
        row.update(reviewed_germinated=germinated_share(riv), reviewed_t50_frame=t50_of(riv), onsets_checked=checked,
                   _reviewed_intervals=riv)
    if units:
        um, spf = units
        row["t50_min"] = None if row["t50_frame"] is None else row["t50_frame"] * spf / 60.0
        row["growth_um_per_min"] = None if row["growth_px_per_bin"] is None else row["growth_px_per_bin"] * um / (fpb * spf / 60.0)
        row["final_length_um"] = None if row["final_length_px"] is None else row["final_length_px"] * um
        if row.get("reviewed_t50_frame") is not None:
            row["reviewed_t50_min"] = row["reviewed_t50_frame"] * spf / 60.0
    return row


COLUMNS = ("movie", "grains", "germinated", "t50_frame", "t50_min", "growth_px_per_bin", "growth_um_per_min",
           "final_length_px", "final_length_um", "lost_partway", "reviewed_germinated", "reviewed_t50_frame",
           "reviewed_t50_min", "onsets_checked")


def fmt(v) -> str:
    if v is None:
        return ""
    return f"{v:.3f}" if isinstance(v, float) and abs(v) < 10 else (f"{v:.1f}" if isinstance(v, float) else str(v))


def write_summary(folders: list[Path], out: Path, units: tuple[float, float] | list | None = None, log=print,
                  extra: list[dict] | None = None) -> list[dict]:
    """``units``: (um per px, s per frame) for every movie, or a list with one (or None) per movie (the TubeTracker
    app keeps each movie's own); the figure is in minutes and um/min only when every movie has them. ``extra``: per
    movie, columns written before the results (e.g. sample id, genotype, replicate)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    per_movie = list(units) if isinstance(units, list) else [units] * len(folders)
    rows = [movie_row(f, u) for f, u in zip(folders, per_movie)]
    extra = extra or [{} for _ in rows]
    extra_cols = list(dict.fromkeys(k for e in extra for k in e))
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS[:1] + tuple(extra_cols) + COLUMNS[1:])
        for r, e in zip(rows, extra):
            w.writerow([fmt(r.get(COLUMNS[0]))] + [e.get(c, "") for c in extra_cols] + [fmt(r.get(c)) for c in COLUMNS[1:]])
    physical = bool(per_movie) and all(per_movie)  # minutes and um/min only when every movie has its units
    scales = [(lambda f, s=u[1]: f * s / 60.0) if physical else (lambda f: f) for u in per_movie]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150, gridspec_kw={"width_ratios": [1.6, 1.0]})
    fig.patch.set_facecolor(report.SURFACE)
    for ax in (a1, a2):
        report._axes_style(ax)
    for i, (r, u, scale) in enumerate(zip(rows, per_movie, scales)):
        c = PALETTE[i % len(PALETTE)]
        if r["_intervals"] and r["last_frame"]:
            t, y = envelope(r["_intervals"], r["last_frame"])
            a1.step([scale(x) for x in t], y, where="post", color=c, linewidth=2, label=r["movie"])
            if r["t50_frame"] is not None:
                a1.plot([scale(r["t50_frame"])], [0.5], "o", color=c, markersize=5, markeredgecolor=report.SURFACE)
        if r.get("_reviewed_intervals") and r["last_frame"]:
            t, y = envelope(r["_reviewed_intervals"], r["last_frame"])
            a1.step([scale(x) for x in t], y, where="post", color=c, linewidth=1.2, linestyle="--")
        rates = np.asarray(r["_rates"], float)
        if physical:
            rates = rates * u[0] / (r["frames_per_bin"] * u[1] / 60.0)
        if len(rates):
            jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(rates))
            a2.plot(i + jitter, rates, "o", color=c, markersize=3, alpha=0.6)
            a2.plot([i - 0.3, i + 0.3], [np.median(rates)] * 2, color=report.INK, linewidth=2)
    a1.set_ylim(0, 1.02)
    a1.set_xlabel("Minutes" if physical else "Source frame", fontsize=9, color=report.INK2)
    a1.set_ylabel("Fraction germinated", fontsize=9, color=report.INK2)
    a1.set_title("Germination (model; dashed: after review), dots = T50", fontsize=10, color=report.INK, loc="left")
    a1.legend(fontsize=7.5, frameon=False, loc="lower right")
    a2.set_xticks(range(len(rows)))
    a2.set_xticklabels([r["movie"] for r in rows], fontsize=7.5, rotation=20, ha="right")
    a2.set_ylabel("Growth rate (um/min)" if physical else "Growth rate (px per bin)", fontsize=9, color=report.INK2)
    a2.set_title("Growth rate per tube (bar = median)", fontsize=10, color=report.INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "summary.png", facecolor=report.SURFACE)
    plt.close(fig)
    for r, scale in zip(rows, scales):
        t50 = (f"{r['t50_min']:.0f} min" if r.get("t50_min") is not None else
               f"frame {r['t50_frame']:.0f}" if r["t50_frame"] is not None else "not reached")
        g = "" if r["germinated"] is None else f"{100 * r['germinated']:.0f}% germinated"
        rate = (f"{r['growth_um_per_min']:.2f} um/min" if r.get("growth_um_per_min") is not None else
                f"{r['growth_px_per_bin']:.2f} px/bin" if r["growth_px_per_bin"] is not None else "-")
        log(f"{r['movie']}: {r['grains']} grains, {g}, T50 {t50}, median growth {rate}"
            + (f", {r['lost_partway']} lost partway" if r["lost_partway"] else "")
            + (f"; reviewed ({r['onsets_checked']} onsets checked): T50 "
               f"{'-' if r.get('reviewed_t50_frame') is None else round(scale(r['reviewed_t50_frame']))}"
               if "reviewed_t50_frame" in r else ""))
    log(f"wrote {out / 'summary.csv'} and {out / 'summary.png'}")
    return rows
