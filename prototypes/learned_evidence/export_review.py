"""Results from reviewed labels: what the lab reports after checking the model's answers.

Reads a ``review_labels.json`` pre-filled by ``prefill.py`` and reviewed in the labelling tool, and
writes next to it:
- ``reviewed_grains.csv``: per grain, the onset interval, the last traced length, and whether its onset
  and traces were checked (answered in the tool) and changed from the model's proposal;
- ``reviewed_traces.csv``: every traced length, checked or not;
- ``population.png`` and ``population.csv``: the germination curve (Turnbull, with T50) from the onsets.
It says how much has been checked: an answer not yet looked at is still the model's. Traces are the ones
the tool asks for now: after you move an onset or mark a burst, the model's proposals at bins it no longer
asks about are left out (and a bin it newly asks about stays open until you trace it).

With ``--anchored`` ("trace once", ``trace_once.py``), every grain whose latest traced tube you checked is decoded
again by the prefix decoder anchored on that trace: its length at every bin and its onset, in ``trace_once/``
(``growth.csv``, ``growth_curves.png``, ``predictions.json``). It needs the movie's probability cache, which the
analysis writes and a re-run rebuilds.

    python -m prototypes.learned_evidence.export_review --labels runs/learned_evidence/<movie>/review_labels.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _changed_onset(a: dict | None, b: dict | None) -> bool | None:
    """Whether answer ``a`` differs from the model's ``b``; an answer where the model gave none is a change."""
    if not a:
        return None
    if not b:
        return True
    return (a.get("verdict"), a.get("first_visible_bin")) != (b.get("verdict"), b.get("first_visible_bin"))


def _changed_trace(a: dict | None, b: dict | None) -> bool | None:
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
    model's there were proposed before the burst was found, so only yours are kept."""
    from sparsetrack.bench.server import grain_trace_plan

    if onset.get("verdict") not in ("emerged_within", "emerged_at_start"):
        return []
    plan = grain_trace_plan(onset.get("first_visible_bin") or 0, n_bins, saved)
    burst = next((b for b in plan if (saved.get(str(b)) or {}).get("state") == "burst"), None)
    return [b for b in plan
            if burst is None or b <= burst or (saved.get(str(b)) or {}).get("review_origin") == "human"]


def population_input(doc: dict, checked_only: bool = False) -> dict:
    """The reviewed onsets in the form ``sparsetrack.report.write_population`` reads (with ``checked_only``,
    only the onsets answered in the tool, not the model's still unchecked)."""
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


def checked_anchors(doc: dict) -> dict[str, dict]:
    """Per grain whose latest asked FULL trace you checked (answered in the tool), that trace as the prefix
    decoder's anchor: {grain id: {"bin", "path_xy_ref", "length_px"}}."""
    nb, out = int(doc["n_bins"]), {}
    for gid, lab in doc.get("labels", {}).items():
        if doc["grains"].get(gid, {}).get("excluded"):
            continue
        saved = lab.get("traces") or {}
        full = [b for b in asked_bins(lab.get("onset") or {}, saved, nb) if (saved.get(str(b)) or {}).get("state") == "full"]
        tr = saved[str(full[-1])] if full else {}
        if tr.get("review_origin") == "human" and len(tr.get("path_xy_ref") or []) >= 2:
            out[gid] = {"bin": full[-1], "path_xy_ref": tr["path_xy_ref"], "length_px": float(tr["length_px"])}
    return out


def write_anchored(path: Path, doc: dict, field: Path, um: float | None, log=print) -> Path | None:
    """The prefix decoder anchored on each checked latest trace, for those grains, in ``trace_once/``."""
    import csv

    from sparsetrack.report import write_growth_curves

    from . import prefix

    anchors = checked_anchors(doc)
    pcache = path.parent / f"prob_{field.name}"
    out = path.parent / "trace_once"
    if not anchors:
        if out.exists():  # results of an earlier review must not pass for this one's
            import shutil
            shutil.rmtree(out.with_name("trace_once_old"), ignore_errors=True)
            out.rename(out.with_name("trace_once_old"))
        log("--anchored: no grain has its latest traced tube checked yet" + (" (an earlier trace_once/ is now "
            "trace_once_old/)" if out.with_name("trace_once_old").exists() else ""))
        return None
    if not (pcache / "meta.json").exists():
        raise SystemExit(f"--anchored needs the probability cache {pcache}: analyse the movie again to rebuild it")
    dec = (doc.get("prefill") or {}).get("decoder") or {}
    pp = prefix.Params(vmax_px=float(dec.get("vmax", prefix.Params.vmax_px)),
                       onset_px=float(dec.get("onset_px", prefix.Params.onset_px)))
    pred = prefix.analyze(pcache, field, grains_path=path, log=lambda *a: None, params=pp, anchors=anchors,
                          only=sorted(anchors))
    out.mkdir(exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(pred))
    with open(out / "growth.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grain", "frame", "length_px", "length_um", "anchor_bin", "status", "onset_frame"])
        for g in pred["grains"]:
            for f, L in zip(g["length"]["frames"], g["length"]["px"]):
                w.writerow([g["id"], f, L, round(L * um, 2) if um else "", g["anchor"]["bin"], g["status"],
                            g.get("onset_frame") if g.get("onset_frame") is not None else ""])
    write_growth_curves(pred, out, [g["id"] for g in pred["grains"]])
    log(f"trace once: {len(pred['grains'])} grains decoded along your checked latest trace -> {out / 'growth.csv'}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", required=True, help="review_labels.json, reviewed in the labelling tool")
    ap.add_argument("--um-per-px", type=float, default=None)
    ap.add_argument("--s-per-frame", type=float, default=None)
    ap.add_argument("--anchored", action="store_true",
                    help="also decode each grain whose latest traced tube you checked along that trace (trace_once/)")
    ap.add_argument("--field", default=None, help="the movie's prepared cache (default: the one pre-filled from)")
    args = ap.parse_args(argv)
    from sparsetrack.report import write_population

    path = Path(args.labels)
    doc = json.loads(path.read_text())
    model_path = path.with_suffix(".model.json")
    have_model = model_path.exists()
    model = json.loads(model_path.read_text()) if have_model else {"labels": {}}
    um, spf = args.um_per_px, args.s_per_frame
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
        on_changed = _changed_onset(on, mlab.get("onset")) if (on_checked and have_model) else None
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
            t_changed = (_changed_trace(tr, (mlab.get("traces") or {}).get(str(tr["bin"])))
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
    lines.append(f"wrote {out / 'reviewed_grains.csv'}, {out / 'reviewed_traces.csv'}, {out / 'population.png'}")
    print("\n".join(lines))
    if args.anchored:
        field = args.field or (doc.get("prefill") or model.get("prefill") or {}).get("field")
        if not field:
            raise SystemExit("--anchored: give the movie's prepared cache with --field (this review predates the note)")
        write_anchored(path, doc, Path(field), um)
    return out


if __name__ == "__main__":
    main()
