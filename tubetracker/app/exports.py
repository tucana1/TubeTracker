"""Results for the lab: one movie's tables and figures, and several movies side by side.

``export_movie`` writes ``<run>/results/`` from what the app shows (the review where a person answered, the
model elsewhere), in the movie's units:

- ``grains.csv``: per grain, with the sample's metadata: germinated or not, onset (as an interval: after / by),
  final length, growth rate, lost partway, the model's confidence, and whether a person checked or corrected it;
- ``growth.csv``: every grain's tube length at every time (bin centre), the lengths a person checked marked;
- ``germination.png``: the germination curve (isolated grains, interval-censored onsets) with T50;
- ``growth_curves.png``: every tube's length against time;
- ``summary.txt``: the movie's result in words.

It also refreshes SparseTrack's reviewed exports in ``<run>/review/`` (``sparsetrack.review.export``). The model's
own tables stay in ``<run>/analysis/``.

``export_summary`` puts several movies side by side (``sparsetrack.summary.write_summary``, each movie in its own
units, with its sample metadata).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from sparsetrack import report

from .model import EMERGED, RunData
from .runfolder import RunFolder
from .units import Units, format_duration

STATUS_WORDS = {"emerged_within": "germinated", "emerged_at_start": "germinated before the start",
                "no_emergence_by_end": "not germinated", "unobservable": "could not be read"}


def _num(v, nd=2):
    return "" if v is None else round(float(v), nd)


def grain_rows(data: RunData, grains: list[dict]) -> tuple[list[str], list[list]]:
    u, s = data.units, data.setup
    head = ["sample_id", "genotype", "replicate", "grain", "x_px", "y_px", "isolated", "status",
            "onset_after_frame", "onset_by_frame"]
    if u.timed:
        head += ["onset_after_min", "onset_by_min"]
    head += ["final_length_px"] + (["final_length_um"] if u.scaled else [])
    head += ["growth_px_per_frame"] + ([f"growth_{u.rate_unit.replace('µ', 'u').replace('/', '_per_')}"]
                                        if u.timed else [])
    head += ["lost_after_frame"] + (["lost_after_min"] if u.timed else [])
    head += ["model_confidence", "source", "onset_checked", "lengths_checked", "flags"]
    rows = []
    for g in grains:
        status = "excluded: " + g["excluded"].replace("_", " ") if g["excluded"] else STATUS_WORDS.get(g["status"],
                                                                                                          g["status"])
        if g["status"] == "no_emergence_by_end" and g["lost"] is not None and not g["excluded"]:
            status = "not germinated before it was lost"
        lost_frame = data.frame(g["lost"] - 1) if g["lost"] is not None else None
        row = [s.get("sample_id", ""), s.get("genotype", ""), s.get("replicate", ""), g["id"], g["x"], g["y"],
               "yes" if g["isolated"] else "no", status, g["onset_after"] if g["onset_after"] is not None else "",
               g["onset_by"] if g["onset_by"] is not None else ""]
        if u.timed:
            row += [_num(u.minutes(g["onset_after"]), 1), _num(u.minutes(g["onset_by"]), 1)]
        grown = g["status"] in EMERGED and not g["excluded"]
        row += [_num(g["final"]) if grown else ""] + ([_num(u.um(g["final"])) if grown else ""] if u.scaled else [])
        row += [_num(g["rate"], 6) if grown else ""] + ([_num(u.rate(g["rate"]), 4) if grown else ""] if u.timed else [])
        row += ["" if lost_frame is None else lost_frame] + ([_num(u.minutes(lost_frame), 1)] if u.timed else [])
        rv = g["review"]
        source = {"model": "model", "partly checked": "partly reviewed", "checked": "reviewed",
                  "corrected": "reviewed (corrected)", "excluded": "reviewed (excluded)"}[rv["state"]]
        row += [_num(g["conf"], 3), source, {"model": "no", "checked": "yes", "changed": "yes (changed)"}[rv["onset"]],
                f"{rv['traces_checked']}" + (f" ({rv['traces_changed']} changed)" if rv["traces_changed"] else ""),
                ";".join(g["flags"])]
        rows.append(row)
    return head, rows


def growth_rows(data: RunData, grains: list[dict]) -> tuple[list[str], list[list]]:
    u = data.units
    head = ["grain", "bin", "frame"] + (["minutes"] if u.timed else []) + ["length_px"] + (
        ["length_um"] if u.scaled else []) + ["checked_here"]
    rows = []
    for g in grains:
        if g["excluded"] or g["status"] not in EMERGED:
            continue
        checked = {t["bin"] for t in g["human"] if t["state"] in ("full", "partial", "no_tube")}
        end = g["lost"] if g["lost"] is not None else data.n_bins
        for b in range(end):
            f = data.frame(b)
            rows.append([g["id"], b, f] + ([_num(u.minutes(f), 2)] if u.timed else []) + [g["L"][b]]
                        + ([_num(u.um(g["L"][b]))] if u.scaled else []) + ["yes" if b in checked else ""])
    return head, rows


def _axes(ax):
    report._axes_style(ax)


def germination_figure(data: RunData, pop: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u = data.units
    t = [u.time(data.frame(b)) for b in range(data.n_bins)]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    fig.patch.set_facecolor(report.SURFACE)
    _axes(ax)
    ax.fill_between(t, pop["certain"], pop["possible"], step="mid", color=report.SERIES_LIGHT, linewidth=0)
    ax.step(t, pop["certain"], where="mid", color=report.SERIES, linewidth=2)
    if pop["t50_frame"] is not None:
        x = u.time(pop["t50_frame"])
        ax.axvline(x, color=report.INK2, linewidth=0.8, linestyle="--")
        ax.plot([x], [0.5], "o", color=report.SERIES, markersize=6, markeredgecolor=report.SURFACE, markeredgewidth=2)
        ax.annotate(f"T50 {x:.0f} {u.time_unit}", (x, 0.5), xytext=(8, -14), textcoords="offset points", fontsize=8,
                    color=report.INK2)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Time (min)" if u.timed else "Source frame", fontsize=9, color=report.INK2)
    ax.set_ylabel("Fraction germinated", fontsize=9, color=report.INK2)
    c = pop["counts"]
    ax.set_title(f"Germination, {data.setup.get('sample_id') or data.folder.name}", fontsize=10, color=report.INK,
                 loc="left", pad=20)
    ax.text(0, 1.025, f"n = {pop['n']} isolated grains: {c['germinated']} germinated, {c['not_germinated']} not, "
            f"{c['lost_before']} lost before germinating; band = onset-interval uncertainty", transform=ax.transAxes,
            fontsize=7.5, color=report.MUTED)
    fig.tight_layout()
    fig.savefig(path, facecolor=report.SURFACE)
    plt.close(fig)


def growth_figure(data: RunData, grains: list[dict], path: Path) -> None:
    u = data.units
    curves = []
    for g in grains:
        if g["excluded"] or g["status"] not in EMERGED or not g["isolated"]:
            continue
        end = g["lost"] if g["lost"] is not None else data.n_bins
        frames = [data.frame(b) for b in range(end)]
        onset = data.frame(g["onset"]) if g["status"] == "emerged_within" and g["onset"] is not None else None
        curves.append({"id": g["id"] + (" (checked)" if g["review"]["state"] in ("checked", "corrected") else ""),
                       "status": g["status"], "onset_frame": u.time(onset), "flags": [],
                       "length": {"frames": [u.time(f) for f in frames], "px": [u.length(v) for v in g["L"][:end]]},
                       "anchors": [(u.time(data.frame(t["bin"])), u.length(t["L"])) for t in g["human"]
                                   if t["state"] in ("full", "partial")]})
    if not curves:
        return
    tmp = path.parent / ".growth_tmp"
    tmp.mkdir(exist_ok=True)
    report.write_growth_curves({"grains": curves}, tmp, [c["id"] for c in curves],
                               title=f"Tube length ({u.length_unit}) against time ({u.time_unit}), per isolated grain; "
                                     f"line = onset, dots = lengths you checked.")
    (tmp / "growth_curves.png").replace(path)
    tmp.rmdir()


def write_csv(path: Path, head: list, rows: list) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)


def export_movie(data: RunData, log=lambda *a: None) -> dict:
    """Write ``results/`` (and refresh ``review/``); returns {"folder", "files": [paths]}."""
    out = data.folder.results
    out.mkdir(parents=True, exist_ok=True)
    grains = data.grains()
    pop = data.population(grains)
    summ = data.summary(grains, pop)
    files = []
    head, rows = grain_rows(data, grains)
    write_csv(out / "grains.csv", head, rows)
    head, rows = growth_rows(data, grains)
    write_csv(out / "growth.csv", head, rows)
    files += [out / "grains.csv", out / "growth.csv"]
    germination_figure(data, pop, out / "germination.png")
    files.append(out / "germination.png")
    growth_figure(data, grains, out / "growth_curves.png")
    if (out / "growth_curves.png").exists():
        files.append(out / "growth_curves.png")
    s, u = data.setup, data.units
    lines = [f"Movie: {s.get('movie_name') or (data.meta.get('movie') or {}).get('name', data.folder.name)}",
             f"Sample: {s.get('sample_id', '')}  genotype: {s.get('genotype', '')}  replicate: {s.get('replicate', '')}",
             f"Duration: {format_duration(s.get('duration_s'))} ({u.s_per_frame:.3g} s per frame)" if u.timed
             else "Duration: not given (times in frames)",
             f"Pixel size: {u.um_per_px} um" if u.scaled else "Pixel size: not given (lengths in px)",
             f"Analysis: {data.pred.get('method', '')}", "", summ["line"],
             f"{summ['reviewed']} of {summ['grains']} grains checked or corrected by a person; the rest are the "
             f"model's readings.", "", "Files: grains.csv (per grain), growth.csv (length at every time), "
             "germination.png, growth_curves.png; the model's own tables are in ../analysis/."]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    files.append(out / "summary.txt")
    if data.folder.review_labels.exists():
        from sparsetrack.review import export
        try:
            export(data.folder.review_labels, u.um_per_px, u.s_per_frame, log=log)
            review = data.folder.review_labels.parent
            files += [p for p in (review / "reviewed_grains.csv", review / "reviewed_traces.csv",
                                  review / "reviewed_growth.csv") if p.exists()]
        except Exception as exc:  # noqa: BLE001 - the app's own tables are written; say what failed
            log(f"the reviewed exports in review/ could not be refreshed: {exc}")
    return {"folder": str(out), "files": [str(f) for f in files], "summary": summ["line"]}


def movie_label(folder: RunFolder) -> str:
    s = folder.load_setup()
    bits = [s.get("sample_id") or folder.name]
    bits += [b for b in (s.get("genotype"), f"rep {s['replicate']}" if s.get("replicate") else "") if b]
    return " · ".join(bits)


def compare_rows(folders: list[RunFolder]) -> list[dict]:
    """Several movies side by side, as ``sparsetrack summary`` reads them (the model's results, and the reviewed
    onsets where a person has checked), each in its own units, with its germination curve for plotting."""
    from sparsetrack.summary import envelope, movie_row

    rows = []
    for f in folders:
        u = Units.from_setup(f.load_setup(), f.n_frames())
        r = movie_row(f.root, None)  # frames and px; converted below with the movie's own units
        spf = u.s_per_frame
        to_t = (lambda fr: fr * spf / 60.0) if spf else (lambda fr: fr)
        curve = envelope(r["_intervals"], r["last_frame"]) if r["_intervals"] and r["last_frame"] else ([], [])
        checked = bool(r.get("onsets_checked"))  # a review with nothing checked yet is still the model's
        rcurve = (envelope(r["_reviewed_intervals"], r["last_frame"]) if checked and r.get("_reviewed_intervals")
                  and r["last_frame"] else ([], []))
        if not checked:
            r["reviewed_t50_frame"] = r["reviewed_germinated"] = None
        fpb = r["frames_per_bin"]
        rates = [x / fpb for x in r["_rates"]]  # px per frame
        s = f.load_setup()
        rows.append({
            "folder": str(f.root), "name": f.name, "label": movie_label(f), "sample_id": s.get("sample_id", ""),
            "genotype": s.get("genotype", ""), "replicate": s.get("replicate", ""), "timed": u.timed,
            "scaled": u.scaled, "grains": r["grains"], "germinated": r["germinated"],
            "t50": None if r["t50_frame"] is None else to_t(r["t50_frame"]),
            "reviewed_t50": None if r.get("reviewed_t50_frame") is None else to_t(r["reviewed_t50_frame"]),
            "reviewed_germinated": r.get("reviewed_germinated"), "onsets_checked": r.get("onsets_checked"),
            "median_rate": u.rate(float(np.median(rates))) if rates else None, "rate_unit": u.rate_unit,
            "median_final": u.length(r["final_length_px"]) if r["final_length_px"] is not None else None,
            "length_unit": u.length_unit, "time_unit": u.time_unit, "lost": r["lost_partway"],
            "rates": [u.rate(x) for x in rates],
            "curve": {"t": [to_t(x) for x in curve[0]], "y": list(curve[1])},
            "reviewed_curve": {"t": [to_t(x) for x in rcurve[0]], "y": list(rcurve[1])},
        })
    return rows


def export_summary(folders: list[RunFolder], out: Path, log=lambda *a: None) -> dict:
    from sparsetrack.summary import write_summary

    units = []
    for f in folders:
        u = Units.from_setup(f.load_setup(), f.n_frames())
        units.append(u.pair())
    extra = [{"sample_id": f.load_setup().get("sample_id", ""), "genotype": f.load_setup().get("genotype", ""),
              "replicate": f.load_setup().get("replicate", "")} for f in folders]
    out.mkdir(parents=True, exist_ok=True)
    write_summary([f.root for f in folders], out, units=units, log=log, extra=extra)
    return {"folder": str(out), "files": [str(out / "summary.csv"), str(out / "summary.png")]}
