"""Score a method's predictions against human benchmark labels.

Labels: a ``sparsetrack.bench.v1`` document (from the labelling tool or converted
legacy answers). Predictions: a ``sparsetrack.pred.v1`` document::

    {"schema": "sparsetrack.pred.v1", "method": "...", "grains": [
        {"id": "g023", "x": 925.6, "y": 493.1,            # reference coordinates
         "status": "emerged_within",                      # GerminationEvent verdicts
         "onset_frame": 6150,                             # first frame judged visible
         "length": {"frames": [...], "px": [...]}}]}      # exit-to-apex length series

Onset timing error is the signed distance from the predicted onset frame to the human
bracket (0 inside ``(last_absent_frame, first_visible_frame]``, negative = early).
Lengths are compared at every human trace: FULL traces by error, PARTIAL traces as
lower bounds, "no tube" traces as absences.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PRED_SCHEMA = "sparsetrack.pred.v1"
EMERGED = ("emerged_within", "emerged_at_start")


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def match_grains(bench: dict, pred_grains: list[dict], radius: float = 12.0) -> dict[str, dict]:
    """Map benchmark grain id -> predicted grain (same id first, else nearest within ``radius``)."""
    by_id = {g.get("id"): g for g in pred_grains if g.get("id")}
    out: dict[str, dict] = {}
    used: set[int] = set()
    for gid, g in bench["grains"].items():
        if gid in by_id:
            out[gid] = by_id[gid]
            used.add(id(by_id[gid]))
    for gid, g in bench["grains"].items():
        if gid in out:
            continue
        best, bd = None, radius
        for p in pred_grains:
            if id(p) in used:
                continue
            d = float(np.hypot(p["x"] - g["x"], p["y"] - g["y"]))
            if d <= bd:
                best, bd = p, d
        if best is not None:
            out[gid] = best
            used.add(id(best))
    return out


def interval_distance(t: float, lo: float | None, hi: float | None) -> float:
    """Signed distance of ``t`` from the interval (lo, hi]; 0 inside, negative before."""
    if lo is not None and t <= lo:
        return float(t - lo)
    if hi is not None and t > hi:
        return float(t - hi)
    return 0.0


def length_at(pred: dict, frame: int) -> float | None:
    series = pred.get("length") or {}
    frames, px = series.get("frames") or [], series.get("px") or []
    if not frames:
        return 0.0 if pred.get("status") == "no_emergence_by_end" else None
    i = int(np.argmin(np.abs(np.asarray(frames) - frame)))
    return float(px[i])


def score(labels: dict, pred: dict, onset_tol: float = 600.0, len_abs: float = 2.0, len_rel: float = 0.10,
          absent_px: float = 2.0, subset: str = "isolated") -> dict:
    """Return a report dict; ``subset`` = "isolated" (sparse benchmark) or "all" included grains."""
    grains = {gid: g for gid, g in labels["grains"].items()
              if not g.get("excluded") and (subset == "all" or g.get("isolated", True))}
    matched = match_grains({"grains": grains}, pred.get("grains", []), radius=float(pred.get("match_radius_px", 12.0)))
    rows, onset_err, full_err, absences, partial_ok, contact_n = [], [], [], [], [], []
    confusion: dict[str, dict[str, int]] = {}
    for gid in sorted(grains):
        lab = labels["labels"].get(gid, {})
        on = lab.get("onset")
        p = matched.get(gid)
        row = {"grain": gid, "matched": p is not None}
        if on:
            hv = on["verdict"]
            pv = p.get("status") if p else "missing"
            confusion.setdefault(hv, {}).setdefault(pv, 0)
            confusion[hv][pv] += 1
            row.update(human=hv, pred=pv, human_bracket=[on.get("last_absent_frame"), on.get("first_visible_frame")])
            if hv == "emerged_within" and p and p.get("status") == "emerged_within" and p.get("onset_frame") is not None:
                d = interval_distance(p["onset_frame"], on.get("last_absent_frame"), on.get("first_visible_frame"))
                onset_err.append(d)
                row.update(pred_onset=p["onset_frame"], onset_error=d)
        for key, tr in sorted((lab.get("traces") or {}).items(), key=lambda kv: int(kv[0])):
            if tr["state"] == "unsure" or p is None:
                continue
            if tr.get("contact"):  # touching another tube/grain: scored apart from the sparse metric
                contact_n.append(1)
                continue
            frame = tr.get("source_frame") or int(key) * labels["frames_per_bin"] + labels["frames_per_bin"] // 2
            model = length_at(p, frame)
            if model is None:
                continue
            if tr["state"] == "no_tube":
                absences.append(model < absent_px)
                row.setdefault("absences", []).append({"frame": frame, "pred": round(model, 2)})
            elif tr["state"] == "full":
                h = tr["length_px"]
                full_err.append((model - h, h))
                row.setdefault("full", []).append({"frame": frame, "human": h, "pred": round(model, 2),
                                                   "error": round(model - h, 2)})
            else:
                partial_ok.append(model >= tr["length_px"] - len_abs)
        rows.append(row)
    e = np.array([x for x, _ in full_err]) if full_err else np.zeros(0)
    h = np.array([y for _, y in full_err]) if full_err else np.zeros(0)
    tol = np.maximum(len_abs, len_rel * h)
    oe = np.array(onset_err) if onset_err else np.zeros(0)
    human_emerged = sum(v for hv, c in confusion.items() if hv == "emerged_within" for v in c.values())
    return {
        "method": pred.get("method"), "subset": subset, "grains_scored": len(grains),
        "grains_matched": sum(r["matched"] for r in rows),
        "germination_confusion": confusion,
        "onset": {"n_timed": int(len(oe)), "n_human_emerged_within": int(human_emerged),
                  "tolerance_frames": onset_tol,
                  "hits": int(np.sum(np.abs(oe) <= onset_tol)),
                  "median_abs_error": float(np.median(np.abs(oe))) if len(oe) else None,
                  "mean_error": float(np.mean(oe)) if len(oe) else None,
                  "early": int(np.sum(oe < -onset_tol)), "late": int(np.sum(oe > onset_tol))},
        "length_full": {"n": int(len(e)), "within_tolerance": int(np.sum(np.abs(e) <= tol)),
                        "median_abs_error": float(np.median(np.abs(e))) if len(e) else None,
                        "mean_abs_error": float(np.mean(np.abs(e))) if len(e) else None,
                        "bias": float(np.mean(e)) if len(e) else None},
        "length_partial": {"n": len(partial_ok), "consistent": int(sum(partial_ok))},
        "absences": {"n": len(absences), "correct": int(sum(absences))},
        "traces_in_contact_skipped": len(contact_n),
        "rows": rows,
    }


def markdown(report: dict) -> str:
    o, L, a = report["onset"], report["length_full"], report["absences"]
    fmt = lambda v, n=1: "—" if v is None else f"{v:.{n}f}"
    lines = [f"## {report['method']} — {report['subset']} grains ({report['grains_matched']}/{report['grains_scored']} matched)",
             "",
             f"- Onset: {o['hits']}/{o['n_timed']} within ±{o['tolerance_frames']:.0f} frames of the human bracket "
             f"(of {o['n_human_emerged_within']} human 'emerged during movie'); median |error| {fmt(o['median_abs_error'], 0)}, "
             f"mean {fmt(o['mean_error'], 0)}; early {o['early']}, late {o['late']}",
             f"- Length (FULL traces): {L['within_tolerance']}/{L['n']} within max(2 px, 10%); median |error| "
             f"{fmt(L['median_abs_error'], 2)} px, bias {fmt(L['bias'], 2)} px",
             f"- PARTIAL traces consistent: {report['length_partial']['consistent']}/{report['length_partial']['n']}; "
             f"absences correct: {a['correct']}/{a['n']}",
             f"- Germination calls (human → predicted): {json.dumps(report['germination_confusion'])}",
             "", "| grain | human | predicted | bracket | pred onset | onset err | FULL errors (px) |", "|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        full = ", ".join(f"{x['error']:+.1f}" for x in r.get("full", []))
        br = r.get("human_bracket")
        lines.append(f"| {r['grain']} | {r.get('human', '')} | {r.get('pred', '' if r['matched'] else 'unmatched')} | "
                     f"{'' if not br else f'({br[0]}, {br[1]}]'} | {r.get('pred_onset', '')} | "
                     f"{'' if 'onset_error' not in r else int(r['onset_error'])} | {full} |")
    return "\n".join(lines) + "\n"
