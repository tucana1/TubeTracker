"""Pre-fill review labels from the per-bin decoder: the model proposes, you correct in the labelling tool.

The labelling tool (``sparsetrack bench``) opens any labels file. This writes one for a movie the pipeline
has analysed, with the decoder's answers in the tool's own format (it goes through the tool's ``Bench``):
- an onset bracket per grain;
- a traced tube at each bin the tool asks for (6 bins after onset, 40% and 70% through the movie, the
  last bin): the decoder's path from the grain's rim along the tube, as long as the length it reports.
Every answer is marked ``review_origin: model``; the tool marks what you answer ``human``, so a grain you
confirm or fix is told apart from one not yet looked at. The file sits with the pipeline's results
(``review_labels.json``, and the proposals as made in ``review_labels.model.json``), never with the
benchmark labels, and an existing one is not overwritten. ``export_review.py`` turns it into results.

    python -m prototypes.learned_evidence.prefill --field runs/sparsetrack/<movie>/cache \\
        --work runs/learned_evidence/<movie>
    python -m sparsetrack bench runs/sparsetrack/<movie>/cache \\
        --labels runs/learned_evidence/<movie>/review_labels.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "per-bin decoder (learned evidence)"
REVIEW_HINTS = ("no_grain_after", "unsteady", "burst_after")  # reach_grain's flags that ask for a look first


def simplify(pts: np.ndarray, eps: float = 1.0, max_gap: float = 25.0) -> np.ndarray:
    """Douglas-Peucker to ``eps`` px, and no segment longer than ``max_gap`` px (like a careful click trace)."""
    def dp(a, b):
        if b <= a + 1:
            return [a]
        t = pts[b] - pts[a]
        dev = np.abs((pts[a + 1:b, 0] - pts[a, 0]) * t[1] - (pts[a + 1:b, 1] - pts[a, 1]) * t[0]) / (np.hypot(*t) + 1e-9)
        k = int(np.argmax(dev))
        return dp(a, a + 1 + k) + dp(a + 1 + k, b) if dev[k] > eps else [a]
    keep = dp(0, len(pts) - 1) + [len(pts) - 1]
    out = [keep[0]]
    for j in keep[1:]:
        n = int(np.ceil(np.hypot(*(pts[j] - pts[out[-1]])) / max_gap))
        out += [int(round(out[-1] + (j - out[-1]) * f / n)) for f in range(1, n)] + [j]
    return pts[sorted(set(out))]


def arc(pts) -> float:
    return float(np.sum(np.hypot(*np.diff(np.asarray(pts, float), axis=0).T))) if len(pts) > 1 else 0.0


def to_length(pts: np.ndarray, length: float, max_extend: float = 5.0) -> np.ndarray:
    """The path cut to ``length`` px of arc, or carried on along its last direction to reach it, by at
    most ``max_extend`` px: a trace left short is easy to fix in review, an invented one misleads."""
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
    """The decoder's onset as the tool's onset answer."""
    status = res.get("status")
    if status == "emerged_within":
        fv = (int(res["onset_frame"]) - fpb // 2) // fpb
        return {"verdict": "emerged_within", "first_visible_bin": fv, "last_absent_bin": fv - 1}
    if status == "emerged_at_start":
        return {"verdict": "emerged_at_start"}
    return {"verdict": "no_emergence_by_end"}


def trace_body(res: dict, b: int, rs: int, follow: np.ndarray, unsure: bool = False) -> dict:
    """The decoder's reading at bin ``b`` as the tool's trace answer (points in the tool's grain-following
    view). The monotone fit can hold a length the bin's own reading fell short of (fading evidence, a
    burst), so the path is the latest one up to this bin that was about that long, cut to the length.
    ``unsure`` answers "unsure" with the same points (the tool keeps them): the grain had left its place
    by then, so what the decoder read there may be anything passing."""
    i = b - rs
    length = float(res["length"]["px"][i])
    # smoothed first, then cut: the tool measures the clicked polyline, so the trace is the length reported
    paths = {j: simplify(np.asarray(p, float)) for j, p in res.get("_paths", {}).items() if j <= i and len(p) >= 2}
    if length < 2.0 or not paths:
        return {"bin": b, "state": "no_tube", "points": [], "view": "model"}
    long_enough = [j for j in paths if arc(paths[j]) >= length - max(2.0, 0.1 * length)]
    src = paths[max(long_enough)] if long_enough else max(paths.values(), key=arc)
    ref = to_length(src, length)
    view = ref - np.asarray(follow[b], float)
    return {"bin": b, "state": "unsure" if unsure else "full", "points": view.round(2).tolist(), "view": "model"}


def left_at(flags) -> int | None:
    """The frame from which the grain had left its place (reach_grain's ``no_grain_after`` flag)."""
    return next((int(f.split(":")[1]) for f in flags if f.startswith("no_grain_after:")), None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", required=True, help="the movie's prepared cache, as given to pipeline.py")
    ap.add_argument("--work", required=True, help="the pipeline's work folder for that movie")
    ap.add_argument("--out", default=None, help="default: WORK/review_labels.json")
    args = ap.parse_args(argv)
    from sparsetrack import stack
    from sparsetrack.bench.server import Bench, trace_bins
    from sparsetrack.render import Renderer

    from .reach import reach_grain

    field, work = Path(args.field), Path(args.work)
    out = Path(args.out) if args.out else work / "review_labels.json"
    if Path("benchmark").resolve() in out.resolve().parents:
        raise SystemExit("review labels are the model's proposals: keep them out of benchmark/")
    if out.exists():
        raise SystemExit(f"{out} exists and may hold your corrections: move it aside to pre-fill again")
    pred_path = work / "perbin" / "predictions.json"
    pcache = work / f"prob_{field.name}"
    if not pred_path.exists() or not (pcache / "meta.json").exists():
        raise SystemExit(f"run pipeline.py on {field} with --work {work} first (needs {pred_path} and {pcache})")
    pred = json.loads(pred_path.read_text())
    dec = {k: v for k, v in (pred.get("decoder") or {}).items()
           if k in ("big", "burst", "vmax", "end_px", "onset_px")}
    if not dec:  # predictions from before the pipeline recorded its settings: its defaults then
        learned = work / "learned" / "predictions.json"
        vmax = json.loads(learned.read_text()).get("params", {}).get("vmax_px", 4.0) if learned.exists() else 4.0
        dec = {"big": 300, "burst": True, "vmax": float(vmax)}
    bins_p, meta = stack.load(pcache)
    RP, R_img = Renderer(bins_p, meta), Renderer(*stack.load(field))
    fpb, rs, nb = int(meta["frames_per_bin"]), int(meta.get("ref_start", 0)), int(meta["n_bins"])
    started = time.time()
    building = out.with_name(out.stem + ".building.json")
    for f in (building, building.with_suffix(".journal.jsonl")):
        f.unlink(missing_ok=True)
    bench = Bench(field, building, annotator=MODEL_NAME)  # the tool's own records, built in a scratch file
    census = bench.doc["grains"]
    physical = [g for g in census.values() if g.get("exclude_reason") != "not_a_grain"]
    n_traces, differ, check_first = 0, [], {}
    for res in pred["grains"]:
        g = census.get(res["id"])
        if g is None or g.get("excluded"):
            continue
        others = [o for o in physical if o["id"] != g["id"]]
        # read again, with its paths; onset and lengths both come from this reading, so they agree
        read = reach_grain(RP, R_img, meta, g, others, paths=tuple(range(nb - rs)), **dec)
        if (read["status"], read["final_length_px"]) != (res.get("status"), res.get("final_length_px")):
            differ.append(g["id"])
        hints = [f for f in read["flags"] if f.startswith(REVIEW_HINTS)]
        if hints:
            check_first[g["id"]] = hints
        body = onset_body(read, fpb)
        bench.set_onset(g["id"], body)
        if body["verdict"] not in ("emerged_within", "emerged_at_start"):
            continue
        plan = trace_bins(body.get("first_visible_bin", 0), nb)
        follow = bench.follow(g["id"])
        left = left_at(read["flags"])
        for b in plan:
            bench.set_trace(g["id"], trace_body(read, b, rs, follow,
                                                unsure=left is not None and bench.bin_centre(b) >= left))
            n_traces += 1
    doc = bench.doc
    for lab in doc["labels"].values():
        for rec in [lab.get("onset") or {}, *(lab.get("traces") or {}).values()]:
            if rec:
                rec["review_origin"] = "model"
    info = {"source": str(pred_path), "field": str(field), "decoder": pred.get("decoder"), "method": pred.get("method"),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "grains": len(doc["labels"]), "traces": n_traces,
            "check_first": check_first}
    doc["prefill"] = info
    doc["updated"] = info["created"]
    out.write_text(json.dumps(doc, indent=1))
    out.with_suffix(".model.json").write_text(json.dumps(doc, indent=1))  # the proposals as made, for export_review
    out.with_suffix(".journal.jsonl").write_text(json.dumps({"t": info["created"], "event": "prefill",
                                                             "payload": info}) + "\n")
    for f in (building, building.with_suffix(".journal.jsonl")):
        f.unlink(missing_ok=True)
    if check_first:
        print("check these first (the grain left its place, the reading keeps jumping, or the tube may have burst):\n  "
              + "\n  ".join(f"{gid}: {', '.join(v)}" for gid, v in sorted(check_first.items())))
    if differ:
        print(f"note: {len(differ)} grains read differently from {pred_path} (settings changed since?): "
              f"{', '.join(differ[:8])}{' ...' if len(differ) > 8 else ''}; the review holds this reading")
    print(f"{out}: {info['grains']} grains, {n_traces} traces pre-filled ({time.time() - started:.0f} s). Review with:\n"
          f"  python -m sparsetrack bench {field} --labels {out}")
    return out


if __name__ == "__main__":
    main()
