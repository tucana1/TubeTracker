"""Review an analysis in the labelling tool: the model proposes, a person confirms or fixes.

``prefill`` writes a labels file holding SparseTrack's answers in the tool's own format (through the
tool's ``Bench``): an onset bracket per grain, and a traced tube at each bin the tool asks for (the
model's path turned to that bin's rotation and cut to the length it read there). Every answer is
marked ``review_origin: model``. The tool marks what a person answers ``human`` and, on a pre-filled
file, keeps asking for the model's answers until each is checked (Enter confirms one as it stands).

``export`` turns the reviewed file into results:
- ``reviewed_grains.csv``: per grain, the onset interval and the last traced length, and whether the
  onset and traces were checked and changed from the model's proposal;
- ``reviewed_traces.csv``: every traced length, checked or not;
- ``population.png`` / ``population.csv``: the germination curve (Turnbull, with T50).

The pre-fill and export were first written for the learned-evidence prototype (cloud branch, 26 Sep
2026: ``prototypes/learned_evidence/prefill.py``, ``export_review.py``); this is their decoder-free
part, fed by SparseTrack's own predictions.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

REVIEW_HINTS = ("touches:", "shared_change_split", "reader:flood", "drift_rejected", "rotates:")


# ---------------------------------------------------------------------------- pre-fill
def to_length(pts: np.ndarray, length: float, max_extend: float = 5.0) -> np.ndarray:
    """The path cut to ``length`` px of arc, or carried on along its last direction by at most
    ``max_extend`` px: a trace left short is easy to fix in review, an invented one misleads."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if length <= s[-1]:
        i = max(1, int(np.searchsorted(s, length, side="left")))
        a = (length - s[i - 1]) / max(s[i] - s[i - 1], 1e-9)
        return np.vstack([pts[:i], pts[i - 1] + a * (pts[i] - pts[i - 1])])
    back = next((q for q in pts[-2::-1] if np.hypot(*(pts[-1] - q)) >= 3.0), pts[0])
    d = pts[-1] - back
    d = d / max(float(np.hypot(*d)), 1e-9)
    return np.vstack([pts, pts[-1] + d * min(length - s[-1], max_extend)])


def onset_body(res: dict, fpb: int) -> dict:
    """The model's germination call as the tool's onset answer."""
    status = res.get("status")
    if status == "emerged_within" and res.get("onset_frame") is not None:
        fv = int(res["onset_frame"]) // fpb
        return {"verdict": "emerged_within", "first_visible_bin": fv, "last_absent_bin": fv - 1 if fv > 0 else None}
    if status == "emerged_at_start":
        return {"verdict": "emerged_at_start"}
    return {"verdict": "no_emergence_by_end"}


def trace_body(res: dict, b: int, pred: dict, min_px: float = 2.0) -> dict:
    """The model's reading at bin ``b`` as the tool's trace answer. SparseTrack's paths are in the grain's
    own frame (its drift removed), which is the tool's grain-following view."""
    from .report import turned_path

    px = res.get("length", {}).get("px") or []
    length = float(px[b]) if b < len(px) else 0.0
    path = turned_path(res, b, pred)
    if length < min_px or len(path) < 2:
        return {"bin": b, "state": "no_tube", "points": [], "view": "model"}
    return {"bin": b, "state": "full", "points": to_length(path, length).round(2).tolist(), "view": "model"}


def prefill(cache: str | Path, pred: dict | str | Path, out: str | Path, log=print) -> Path:
    """Write the review labels file ``out`` (and the proposals as made, ``*.model.json``)."""
    from . import __version__
    from .bench.server import Bench, trace_bins

    cache, out = Path(cache), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists and may hold a person's answers: move it aside to pre-fill again")
    source = str(pred) if not isinstance(pred, dict) else None
    pred = json.loads(Path(pred).read_text()) if not isinstance(pred, dict) else pred
    out.parent.mkdir(parents=True, exist_ok=True)
    building = out.with_name(out.stem + ".building.json")
    for f in (building, building.with_suffix(".journal.jsonl")):
        f.unlink(missing_ok=True)
    model_name = f"SparseTrack {pred.get('params', {}).get('version') or __version__}"
    bench = Bench(cache, building, annotator=model_name)
    bench.save = lambda *a, **k: None  # the answers go through the tool's own code; the file is written once, below
    census, fpb, nb = bench.doc["grains"], bench.fpb, bench.n_bins
    n_traces, check_first = 0, {}
    for res in pred.get("grains", []):
        g = census.get(res["id"])
        if g is None or g.get("excluded"):
            continue
        hints = [f for f in res.get("flags", []) if f.startswith(REVIEW_HINTS)]
        if hints:
            check_first[res["id"]] = hints
        body = onset_body(res, fpb)
        bench.set_onset(res["id"], body)
        if body["verdict"] not in ("emerged_within", "emerged_at_start"):
            continue
        for b in trace_bins(body.get("first_visible_bin") or 0, nb):
            bench.set_trace(res["id"], trace_body(res, b, pred))
            n_traces += 1
    doc = bench.doc
    for lab in doc["labels"].values():
        for rec in [lab.get("onset") or {}, *(lab.get("traces") or {}).values()]:
            if rec:
                rec["review_origin"] = "model"
        lab.pop("time_spent_s", None)
    info = {"source": source, "field": str(cache), "method": pred.get("method"), "model": model_name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "grains": len(doc["labels"]), "traces": n_traces,
            "check_first": check_first}
    doc["prefill"] = info
    doc["updated"] = info["created"]
    out.write_text(json.dumps(doc, indent=1))
    out.with_suffix(".model.json").write_text(json.dumps(doc, indent=1))
    out.with_suffix(".journal.jsonl").write_text(json.dumps({"t": info["created"], "event": "prefill",
                                                             "payload": info}) + "\n")
    for f in (building, building.with_suffix(".journal.jsonl")):
        f.unlink(missing_ok=True)
    log(f"pre-filled {info['grains']} grains and {n_traces} traces with {model_name}'s answers -> {out}")
    return out


# ---------------------------------------------------------------------------- export
def changed_onset(a: dict | None, b: dict | None) -> bool | None:
    """Whether answer ``a`` differs from the model's ``b``; an answer where the model gave none is a change."""
    if not a:
        return None
    if not b:
        return True
    return (a.get("verdict"), a.get("first_visible_bin")) != (b.get("verdict"), b.get("first_visible_bin"))


def changed_trace(a: dict | None, b: dict | None) -> bool | None:
    if not a:
        return None
    if not b:
        return True  # a bin the model had no proposal at (the onset was moved)
    if a.get("state") != b.get("state"):
        return True
    la, lb = float(a.get("length_px") or 0.0), float(b.get("length_px") or 0.0)
    return abs(la - lb) > max(2.0, 0.1 * max(la, lb))


def _yn(checked: bool, changed: bool | None) -> str:
    return "" if not checked or changed is None else ("yes" if changed else "no")


def asked_bins(onset: dict, saved: dict, n_bins: int) -> list[int]:
    """The bins the tool asks this grain's traces at now: its own plan for the current onset (a moved onset
    moves the first one), ending at a burst. The tool keeps listing answers given after a burst; the
    model's there were proposed before the burst was found, so only a person's are kept."""
    from .bench.server import grain_trace_plan

    if onset.get("verdict") not in ("emerged_within", "emerged_at_start"):
        return []
    plan = grain_trace_plan(onset.get("first_visible_bin") or 0, n_bins, saved)
    burst = next((b for b in plan if (saved.get(str(b)) or {}).get("state") == "burst"), None)
    return [b for b in plan
            if burst is None or b <= burst or (saved.get(str(b)) or {}).get("review_origin") == "human"]


def population_input(doc: dict, checked_only: bool = False) -> dict:
    """The reviewed onsets in the form ``report.write_population`` reads (with ``checked_only``, only the
    onsets answered in the tool, not the model's still unchecked)."""
    fpb, nb = int(doc["frames_per_bin"]), int(doc["n_bins"])
    frames = [fpb // 2, (nb - 1) * fpb + fpb // 2]
    grains = []
    for gid, lab in doc.get("labels", {}).items():
        on = lab.get("onset") or {}
        if doc["grains"].get(gid, {}).get("excluded") or not on or (checked_only and on.get("review_origin") != "human"):
            continue
        la, fv = on.get("last_absent_frame"), on.get("first_visible_frame")
        grains.append({"id": gid, "status": on.get("verdict"), "length": {"frames": frames},
                       "onset_interval": [la if la is not None else fv - fpb, fv] if fv is not None else None})
    return {"grains": grains}


def reviewed_curve(model_px, fv: int | None, anchors: list[tuple[int, float]]) -> np.ndarray:
    """A grain's length at every bin after review: zero before the first visible bin ``fv`` (None: never
    germinated), through the lengths a person checked (``anchors``: (bin, px)), and between them shaped
    like the model's own curve (its growth rescaled to reach each checked length; straight where the model
    did not grow). After the last checked length it grows as the model's did. Never shrinks."""
    m = np.maximum.accumulate(np.nan_to_num(np.asarray(model_px, float)))
    n = len(m)
    out = np.zeros(n)
    if fv is None or fv >= n:
        return out
    pts = sorted((b, float(L)) for b, L in anchors if fv <= b < n)
    if fv > 0:
        pts = [(fv - 1, 0.0)] + pts
    elif pts:  # there from the start: the model's shape up to the first checked length
        b1, L1 = pts[0]
        pts = [(0, m[0] * L1 / m[b1] if m[b1] > 0 else L1)] + pts
    else:
        pts = [(0, m[0])]
    for (a, La), (b, Lb) in zip(pts, pts[1:]):
        seg = np.arange(a, b + 1)
        dm = m[b] - m[a]
        frac = (m[seg] - m[a]) / dm if dm > 1e-6 else (seg - a) / max(b - a, 1)
        out[seg] = La + frac * (Lb - La)
    b_last, L_last = pts[-1]
    out[b_last:] = L_last + (m[b_last:] - m[b_last])
    out[:fv] = 0.0
    return np.maximum.accumulate(np.maximum(out, 0.0))


def write_reviewed_growth(doc: dict, pred: dict, out: Path, um: float | None, spf: float | None) -> int:
    """``reviewed_growth.csv`` and ``growth_curves.png``: every germinated grain's reviewed curve."""
    from .report import write_growth_curves

    fpb, nb = int(doc["frames_per_bin"]), int(doc["n_bins"])
    model = {g["id"]: g for g in pred.get("grains", [])}
    curves, rows = [], []
    for gid, lab in sorted(doc.get("labels", {}).items()):
        on = lab.get("onset") or {}
        if doc["grains"].get(gid, {}).get("excluded") or on.get("verdict") not in ("emerged_within", "emerged_at_start"):
            continue
        saved = lab.get("traces") or {}
        plan = asked_bins(on, saved, nb)
        checked = [saved[str(b)] for b in plan if str(b) in saved and saved[str(b)].get("review_origin") == "human"]
        anchors = [(t["bin"], float(t.get("length_px") or 0.0) if t["state"] in ("full", "partial") else 0.0)
                   for t in checked if t["state"] in ("full", "partial", "no_tube")]
        burst = next((t["bin"] for t in checked if t["state"] == "burst"), None)
        px = (model.get(gid) or {}).get("length", {}).get("px") or [0.0] * nb
        L = reviewed_curve(px, on.get("first_visible_bin") or 0, anchors)[:burst]  # nothing is measured after a burst
        frames = [b * fpb + fpb // 2 for b in range(len(L))]
        curves.append({"id": gid, "status": on["verdict"], "onset_frame": on.get("first_visible_frame"),
                       "length": {"frames": frames, "px": L.round(2).tolist()}, "flags": [],
                       "anchors": [(b * fpb + fpb // 2, v) for b, v in anchors]})
        rows += [[gid, b, f, round(f * spf / 60.0, 2) if spf else "", round(float(v), 2),
                  round(float(v) * um, 2) if um else "", len(anchors)] for b, (f, v) in enumerate(zip(frames, L))]
    with open(out / "reviewed_growth.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grain", "bin", "frame", "minutes", "length_px", "length_um", "checked_lengths"])
        w.writerows(rows)
    write_growth_curves({"grains": curves}, out, [c["id"] for c in curves],
                        title="Tube length (px) against source frame after review: the model's curve through the "
                              "lengths you checked (dots); line = onset.")
    return len(curves)


def export(labels_path: str | Path, um_per_px: float | None = None, s_per_frame: float | None = None,
           log=print) -> Path:
    """Results from a reviewed labels file, written next to it; returns that folder."""
    from .report import write_population

    path = Path(labels_path)
    doc = json.loads(path.read_text())
    model_path = path.with_suffix(".model.json")
    have_model = model_path.exists()
    model = json.loads(model_path.read_text()) if have_model else {"labels": {}}
    um, spf = um_per_px, s_per_frame
    nb = int(doc["n_bins"])
    hints = (doc.get("prefill") or model.get("prefill") or {}).get("check_first") or {}
    out = path.parent
    grains_rows, trace_rows = [], []
    n_on = n_on_checked = n_on_changed = n_tr = n_tr_checked = n_tr_changed = n_open = n_stale = 0
    for gid, lab in sorted(doc.get("labels", {}).items()):
        g = doc["grains"].get(gid, {})
        if g.get("excluded"):
            continue
        on = lab.get("onset") or {}
        mlab = model["labels"].get(gid) or {}
        on_checked = on.get("review_origin") == "human"
        on_changed = changed_onset(on, mlab.get("onset")) if (on_checked and have_model) else None
        n_on += 1
        n_on_checked += on_checked
        n_on_changed += bool(on_changed)
        saved = lab.get("traces") or {}
        plan = asked_bins(on, saved, nb)
        traces = [saved[str(b)] for b in plan if str(b) in saved]
        n_open += len(plan) - len(traces)
        n_stale += sum(1 for k, t in saved.items() if int(k) not in plan and t.get("review_origin") == "model")
        checked = 0
        for tr in traces:
            t_checked = tr.get("review_origin") == "human"
            t_changed = (changed_trace(tr, (mlab.get("traces") or {}).get(str(tr["bin"])))
                         if (t_checked and have_model) else None)
            checked += t_checked
            n_tr += 1
            n_tr_checked += t_checked
            n_tr_changed += bool(t_changed)
            L = float(tr.get("length_px") or 0.0) if tr.get("state") in ("full", "partial") else 0.0
            sf = tr.get("source_frame")
            trace_rows.append([gid, tr["bin"], sf, round(sf * spf / 60.0, 2) if (spf and sf is not None) else "",
                               tr.get("state"), round(L, 2), round(L * um, 2) if um else "",
                               "yes" if t_checked else "no", _yn(t_checked, t_changed)])
        last = next((t for t in reversed(traces) if t.get("state") in ("full", "partial")), None)
        last_len = float(last["length_px"]) if last else ""
        la, fv = on.get("last_absent_frame"), on.get("first_visible_frame")
        grains_rows.append([
            gid, g.get("x"), g.get("y"), on.get("verdict", ""), la if la is not None else "", fv if fv is not None else "",
            round(la * spf / 60.0, 2) if spf and la is not None else "",
            round(fv * spf / 60.0, 2) if spf and fv is not None else "",
            "yes" if on_checked else "no", _yn(on_checked, on_changed),
            last["bin"] if last else "", round(last_len, 2) if last else "",
            round(last_len * um, 2) if (um and last) else "", f"{checked}/{len(plan)}", " ".join(hints.get(gid, []))])
    with open(out / "reviewed_grains.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grain", "x", "y", "onset_verdict", "onset_after_frame", "onset_by_frame", "onset_after_min",
                    "onset_by_min", "onset_checked", "onset_changed", "last_traced_bin", "last_length_px",
                    "last_length_um", "traces_checked", "model_flags"])
        w.writerows(grains_rows)
    with open(out / "reviewed_traces.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grain", "bin", "frame", "minutes", "state", "length_px", "length_um", "checked", "changed"])
        w.writerows(trace_rows)
    # the curve says how much of it was checked; a partly checked review also gets the checked onsets alone
    write_population(population_input(doc), out, label=f"review, {n_on_checked} of {n_on} onsets checked")
    if 0 < n_on_checked < n_on:
        (out / "checked_only").mkdir(exist_ok=True)
        write_population(population_input(doc, checked_only=True), out / "checked_only",
                         label=f"the {n_on_checked} checked onsets only")
    lines = [f"{path}: {n_on} grains",
             f"onsets checked {n_on_checked}/{n_on} ({n_on_changed} changed from the model's)",
             f"traces checked {n_tr_checked}/{n_tr} ({n_tr_changed} changed by more than max(2 px, 10%))"]
    if n_open:
        lines.append(f"{n_open} traces still to answer: the tool asks for them after an onset was moved")
    if n_stale:
        lines.append(f"{n_stale} of the model's traces left out: the tool no longer asks for them (onset moved, "
                     "tube burst, or no tube)")
    if not have_model:
        lines.append(f"no {model_path.name}: what changed from the model's proposals can't be told")
    ghosts = sorted(gid for gid, g in doc["grains"].items() if g.get("excluded") and g.get("exclude_origin") == "model")
    if ghosts:
        lines.append(f"{len(ghosts)} census discs the model judged not grains are left out ({', '.join(ghosts)}): "
                     "include one again in the tool if it is a grain, answer it, then export again")
    if n_on_checked < n_on or n_tr_checked < n_tr or n_open:
        lines.append("answers not yet checked are the model's: carry on reviewing, then export again")
    if 0 < n_on_checked < n_on:
        lines.append(f"the germination curve of the checked onsets alone: {out / 'checked_only' / 'population.png'}")
    written = [out / "reviewed_grains.csv", out / "reviewed_traces.csv", out / "population.png"]
    source = (doc.get("prefill") or {}).get("source")
    if source and Path(source).exists():
        write_reviewed_growth(doc, json.loads(Path(source).read_text()), out, um, spf)
        written += [out / "reviewed_growth.csv", out / "growth_curves.png"]
    lines.append("wrote " + ", ".join(str(f) for f in written))
    log("\n".join(lines))
    return out
