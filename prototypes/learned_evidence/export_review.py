"""Results from reviewed labels: what the lab reports after checking the model's answers.

Reads a ``review_labels.json`` pre-filled by ``prefill.py`` and reviewed in the labelling tool, and
writes next to it:
- ``reviewed_grains.csv``: per grain, the onset interval, the last traced length, and whether its onset
  and traces were checked (answered in the tool) and changed from the model's proposal;
- ``reviewed_traces.csv``: every traced length, checked or not;
- ``population.png`` and ``population.csv``: the germination curve (Turnbull, with T50) from the onsets.
It says how much has been checked: an answer not yet looked at is still the model's.

    python -m prototypes.learned_evidence.export_review --labels runs/learned_evidence/<movie>/review_labels.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _changed_onset(a: dict | None, b: dict | None) -> bool | None:
    if not a or not b:
        return None
    return (a.get("verdict"), a.get("first_visible_bin")) != (b.get("verdict"), b.get("first_visible_bin"))


def _changed_trace(a: dict | None, b: dict | None) -> bool | None:
    if not a or not b:
        return None
    if a.get("state") != b.get("state"):
        return True
    la, lb = float(a.get("length_px") or 0.0), float(b.get("length_px") or 0.0)
    return abs(la - lb) > max(2.0, 0.1 * max(la, lb))


def population_input(doc: dict) -> dict:
    """The reviewed onsets in the form ``sparsetrack.report.write_population`` reads."""
    fpb, nb = int(doc["frames_per_bin"]), int(doc["n_bins"])
    frames = [fpb // 2, (nb - 1) * fpb + fpb // 2]
    grains = []
    for gid, lab in doc.get("labels", {}).items():
        on = lab.get("onset") or {}
        if doc["grains"].get(gid, {}).get("excluded") or not on:
            continue
        la, fv = on.get("last_absent_frame"), on.get("first_visible_frame")
        grains.append({"id": gid, "status": on.get("verdict"), "length": {"frames": frames},
                       "onset_interval": [la if la is not None else fv - fpb, fv] if fv is not None else None})
    return {"grains": grains}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", required=True, help="review_labels.json, reviewed in the labelling tool")
    ap.add_argument("--um-per-px", type=float, default=None)
    ap.add_argument("--s-per-frame", type=float, default=None)
    args = ap.parse_args(argv)
    from sparsetrack.report import write_population

    path = Path(args.labels)
    doc = json.loads(path.read_text())
    model_path = path.with_suffix(".model.json")
    model = json.loads(model_path.read_text()) if model_path.exists() else {"labels": {}}
    um, spf = args.um_per_px, args.s_per_frame
    out = path.parent
    grains_rows, trace_rows = [], []
    n_on = n_on_checked = n_on_changed = n_tr = n_tr_checked = n_tr_changed = 0
    for gid, lab in sorted(doc.get("labels", {}).items()):
        g = doc["grains"].get(gid, {})
        if g.get("excluded"):
            continue
        on = lab.get("onset") or {}
        mon = (model["labels"].get(gid) or {}).get("onset")
        on_checked = on.get("review_origin") == "human"
        on_changed = _changed_onset(on, mon) if on_checked else False
        n_on += 1
        n_on_checked += on_checked
        n_on_changed += bool(on_changed)
        traces = sorted((lab.get("traces") or {}).values(), key=lambda t: int(t["bin"]))
        checked = 0
        for tr in traces:
            mtr = ((model["labels"].get(gid) or {}).get("traces") or {}).get(str(tr["bin"]))
            t_checked = tr.get("review_origin") == "human"
            t_changed = _changed_trace(tr, mtr) if t_checked else False
            checked += t_checked
            n_tr += 1
            n_tr_checked += t_checked
            n_tr_changed += bool(t_changed)
            L = float(tr.get("length_px") or 0.0) if tr.get("state") in ("full", "partial") else 0.0
            trace_rows.append([gid, tr["bin"], tr.get("source_frame"),
                               round(tr["source_frame"] * spf / 60.0, 2) if spf and tr.get("source_frame") else "",
                               tr.get("state"), round(L, 2), round(L * um, 2) if um else "",
                               "yes" if t_checked else "no", "yes" if t_changed else ("no" if t_checked else "")])
        last = next((t for t in reversed(traces) if t.get("state") in ("full", "partial")), None)
        last_len = float(last["length_px"]) if last else ""
        la, fv = on.get("last_absent_frame"), on.get("first_visible_frame")
        grains_rows.append([
            gid, g.get("x"), g.get("y"), on.get("verdict", ""), la if la is not None else "", fv if fv is not None else "",
            round(la * spf / 60.0, 2) if spf and la is not None else "",
            round(fv * spf / 60.0, 2) if spf and fv is not None else "",
            "yes" if on_checked else "no", "yes" if on_changed else ("no" if on_checked else ""),
            last["bin"] if last else "", round(last_len, 2) if last else "",
            round(last_len * um, 2) if (um and last) else "", f"{checked}/{len(traces)}"])
    with open(out / "reviewed_grains.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grain", "x", "y", "onset_verdict", "onset_after_frame", "onset_by_frame", "onset_after_min",
                    "onset_by_min", "onset_checked", "onset_changed", "last_traced_bin", "last_length_px",
                    "last_length_um", "traces_checked"])
        w.writerows(grains_rows)
    with open(out / "reviewed_traces.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grain", "bin", "frame", "minutes", "state", "length_px", "length_um", "checked", "changed"])
        w.writerows(trace_rows)
    write_population(population_input(doc), out, label="reviewed grains")
    lines = [f"{path}: {n_on} grains",
             f"onsets checked {n_on_checked}/{n_on} ({n_on_changed} changed from the model's)",
             f"traces checked {n_tr_checked}/{n_tr} ({n_tr_changed} changed by more than max(2 px, 10%))",
             f"wrote {out / 'reviewed_grains.csv'}, {out / 'reviewed_traces.csv'}, {out / 'population.png'}"]
    if n_on_checked < n_on or n_tr_checked < n_tr:
        lines.insert(3, "answers not yet checked are the model's: carry on reviewing, then export again")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    main()
