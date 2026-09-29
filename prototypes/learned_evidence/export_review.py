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
import json
from pathlib import Path

# the decoder-free part lives in SparseTrack now (sparsetrack/review.py); re-exported here
from sparsetrack.review import _yn, asked_bins, changed_onset as _changed_onset  # noqa: F401
from sparsetrack.review import changed_trace as _changed_trace, export, population_input  # noqa: F401


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
    dec = (doc.get("prefill") or {}).get("decoder") or {}
    fused = bool(dec.get("thick_model"))  # the analysis read fused evidence (pipeline --thick-model): so does this
    pcache = path.parent / (f"prob_fused_{field.name}" if fused else f"prob_{field.name}")
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
    path = Path(args.labels)
    out = export(path, args.um_per_px, args.s_per_frame)
    if args.anchored:
        doc = json.loads(path.read_text())
        model_path = path.with_suffix(".model.json")
        model = json.loads(model_path.read_text()) if model_path.exists() else {}
        field = args.field or (doc.get("prefill") or model.get("prefill") or {}).get("field")
        if not field:
            raise SystemExit("--anchored: give the movie's prepared cache with --field (this review predates the note)")
        write_anchored(path, doc, Path(field), args.um_per_px)
    return out


if __name__ == "__main__":
    main()
