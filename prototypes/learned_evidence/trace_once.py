"""Trace once: how well does one traced tube per grain give the rest of your traces?

The prefix decoder (``prefix.py``) decodes a grain's whole movie as prefixes of its tube at one state (a pollen tube
grows at its tip). Anchored on a human trace it needs nothing else from you: which tube is the grain's, and where
its end is, come from the trace; how long it was at every other bin, and when it came out, from the evidence.
This checks that on a labelled movie. Per grain, the latest FULL trace with a drawn path is the anchor; the
grain's earlier traces and its onset bracket are the test (traces at or after the anchor's bin are left out). Three
runs on the same learned evidence are scored on the same test: the per-bin decoder, the prefix decoder on its own,
and the prefix decoder anchored on your trace.

    python -m prototypes.learned_evidence.trace_once --field runs/sparsetrack/ld --labels benchmark/labels/ld_v1.json

A development-movie check, run by ``adapt.py`` as its last step: never on movie 2, whose labels are the held-out
test and cannot also be the anchors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sparsetrack.evaluate import load, score

from . import calibrate, evaluate, prefix, reach


def leave_anchor_out(labels: dict, anchors: dict) -> dict:
    """The labels of the anchored grains only, without their traces at or after the anchor's bin."""
    doc = json.loads(json.dumps(labels))
    doc["grains"] = {gid: g for gid, g in doc["grains"].items() if gid in anchors}
    for gid in doc["grains"]:
        lab = doc["labels"].get(gid) or {}
        lab["traces"] = {k: t for k, t in (lab.get("traces") or {}).items() if int(k) < int(anchors[gid]["bin"])}
    return doc


def find_cache(net, candidates, field: Path | None = None) -> Path | None:
    """A probability cache this network built already for this movie (the dev test's, calibration's or
    fine-tuning's): the same model, and the same prepared movie as far as the field's own metadata tells."""
    fp = evaluate.fingerprint(net)
    want = json.loads((field / "meta.json").read_text()) if field is not None and (field / "meta.json").exists() else {}
    for c in candidates:
        meta = Path(c) / "meta.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        if m.get("model_sha1") == fp and all(m.get(k) == want[k] for k in ("n_bins", "frames_per_bin", "created", "movie")
                                             if k in want):
            return Path(c)
    return None


def out_of_sample(model: Path, labels_path: Path) -> tuple[Path, str | None]:
    """A model fine-tuned on these very labels reads their traces in-sample: the model it started from instead."""
    try:
        import torch
        info = torch.load(str(model), map_location="cpu", weights_only=False).get("args") or {}
    except Exception:
        return model, None
    lab, start = info.get("labels"), info.get("started_from")
    if lab and start and Path(lab).resolve() == labels_path.resolve():
        return Path(start), (f"{model} was fine-tuned on these labels, so their traces are read with the model it "
                             f"started from, {start}")
    return model, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", default="runs/sparsetrack/ld", help="the dev movie's prepared cache")
    ap.add_argument("--labels", default="benchmark/labels/ld_v1.json", help="the dev movie's labels")
    ap.add_argument("--work", default="runs/learned_evidence/ld_once")
    ap.add_argument("--model", default=None, help="default: the model the launcher uses (see adapt.py)")
    ap.add_argument("--decoder", default=None, help="per-bin decoder settings (default: the launcher's)")
    args = ap.parse_args(argv)
    field, labels_path, work = Path(args.field), Path(args.labels), Path(args.work)
    if "m2" in labels_path.name:
        raise SystemExit("movie 2 is the held-out benchmark: its traces are the test, never anchors")
    from .adapt import CAL, DEV, FT, in_use
    from .finetune import speed_cap
    from .model import load as load_model

    use_model, use_decoder = in_use()
    model = Path(args.model) if args.model else use_model
    decoder = Path(args.decoder) if args.decoder else use_decoder
    model, note = out_of_sample(model, labels_path)
    labels = load(labels_path)
    anchors = prefix.anchors_from_labels(labels)
    if not anchors:
        raise SystemExit(f"{labels_path} has no FULL trace with a drawn path to anchor on")
    work.mkdir(parents=True, exist_ok=True)
    # what this check was asked about, so adapt.py runs it again when the model or decoder in use changes
    (work / "used.json").write_text(json.dumps({"in_use_model": str(use_model), "in_use_decoder": str(use_decoder),
                                                "model": str(model), "decoder": str(decoder)}))
    started = time.time()
    net = load_model(str(model))
    pcache = (find_cache(net, [DEV / f"prob_{field.name}", CAL / "prob", *sorted(FT.glob("prob_*"))], field)
              or evaluate.prob_cache(field, net, work / f"prob_{field.name}"))
    vmax = speed_cap(pcache, field, labels_path)
    kw = dict(big=300, burst=True, vmax=vmax) | calibrate.decoder_settings(decoder, model)
    pp = prefix.Params(vmax_px=vmax, onset_px=kw.get("onset_px", prefix.Params.onset_px))
    quiet = dict(grains_path=labels_path, log=lambda *a: None)
    runs = {"per-bin": reach.analyze(pcache, field, **quiet, **kw),
            "prefix": prefix.analyze(pcache, field, **quiet, params=pp),
            "anchored": prefix.analyze(pcache, field, **quiet, params=pp, anchors=anchors)}
    for name, pred in runs.items():
        (work / f"{name}.json").write_text(json.dumps(pred))
    test = leave_anchor_out(labels, anchors)
    reps = {name: score(test, pred) for name, pred in runs.items()}
    tol = reps["per-bin"]["onset"]["tolerance_frames"]
    n_scored = reps["per-bin"]["grains_scored"]
    if n_scored == 0:
        lines = [f"{labels_path.name}: {len(anchors)} grains have a traced tube, but none is among the scored grains "
                 "(isolated, not excluded): nothing to score"] + ([note] if note else [])
        print("\n".join(lines))
        (work / "report.txt").write_text("\n".join(lines) + "\n")
        return work

    def diff(a, b):
        p = evaluate.paired_bootstrap(reps[a], reps[b], tol)
        return (f"{a} - {b} over {p['grains']} grains: onset {p['onset_diff']:+.0f} "
                f"[{p['onset_ci'][0]:+.0f}, {p['onset_ci'][1]:+.0f}], lengths {p['length_diff']:+.0f} "
                f"[{p['length_ci'][0]:+.0f}, {p['length_ci'][1]:+.0f}] (95% paired bootstrap over grains)"), p

    bins = [a["bin"] for a in anchors.values()]
    n_test = reps["per-bin"]["length_full"]["n"]
    pairs = {k: diff(*k) for k in (("anchored", "per-bin"), ("anchored", "prefix"), ("prefix", "per-bin"))}
    lines = [f"{labels_path.name}: {n_scored} scored grains with a traced tube; anchored on each grain's latest FULL "
             f"trace (bins {min(bins)}-{max(bins)}, median {int(np.median(bins))}), scored on the {n_test} FULL traces "
             f"before it and the onset brackets ({time.time() - started:.0f} s; model {model}"
             + (f", decoder {decoder}" if decoder else "") + ")",
             *([note] if note else []),
             *[evaluate.e2e_summary(name, rep) for name, rep in reps.items()],
             *[text for text, _ in pairs.values()],
             "The anchored run was given one trace per grain and decoded the rest. The per-bin decoder's settings "
             "were fitted on these same traces, so the comparison leans its way."]
    print("\n".join(lines))
    (work / "report.txt").write_text("\n".join(lines) + "\n")
    (work / "scores.json").write_text(json.dumps({**reps, **{f"{a}_vs_{b}": p for (a, b), (_, p) in pairs.items()},
                                                  "anchors": {g: a["bin"] for g, a in anchors.items()}},
                                                 default=str, indent=1))
    return work


if __name__ == "__main__":
    main()
