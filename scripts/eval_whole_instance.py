"""Whole-instance check (rev8 step 3): does the query decide the tube?

Holds ONE crop fixed and swaps only the owner query, then reads the
body head's response against each hand-painted tube mask in that crop.

A model that has learned whole instances shows a diagonal-dominant
response matrix: querying grain i lights up tube i, and nothing else.
Query-invariant output (the failure the review measured — a 9px
receptive field cannot carry a distant tube from a root prompt) shows
up as identical rows.

Crops and clips are built exactly as training builds them
(QUERY_OFFSETS, load_clip_pixels, native/255 grayscale), so the numbers
predict what the trained model does, not a re-implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GRAIN_R_NOMINAL = 16.0


def summarize_swap(iou: dict, owners: list) -> dict:
    """Read the response matrix: diag vs off-diag, row peaks, spread.

    Diagonal-dominant = the query selects the tube. Identical rows
    (row spread ~0) = query-invariant output, which is the failure the
    review measured: a distal answer that cannot depend on the root
    prompt.
    """
    import numpy as np

    diag = [float(iou[o][o]) for o in owners]
    off = [float(iou[q][t]) for q in owners for t in owners if q != t]
    # rev8: a row peaks on its own tube only STRICTLY, and only when the
    # row says anything at all. An all-zero matrix used to score 3/3
    # because `0 >= 0` holds for every column -- a collapsed model
    # (predict nothing) read as a passing swap test, which flatters
    # exactly the failure the test exists to catch (measured on the
    # weight-10 run: held-out IoU 0.000 with "self-peaked=3/3").
    row_peaks = 0
    for q in owners:
        rq = [float(iou[q][t]) for t in owners]
        if max(rq) <= 0.0:
            continue                     # no prediction: no evidence
        best_other = max((float(iou[q][t]) for t in owners if t != q),
                         default=float("-inf"))
        if float(iou[q][q]) > best_other:
            row_peaks += 1
    spread = 0.0
    for q in owners:
        rq = np.asarray([float(iou[q][t]) for t in owners])
        for p in owners:
            if p == q:
                continue
            rp = np.asarray([float(iou[p][t]) for t in owners])
            spread = max(spread, float(np.abs(rq - rp).max()))
    all_iou = diag + off
    degenerate = (not all_iou) or float(np.max(all_iou)) <= 0.0
    return {
        "diag_mean_iou": float(np.mean(diag)) if diag else 0.0,
        "offdiag_mean_iou": float(np.mean(off)) if off else 0.0,
        "rows_peaking_on_own_tube": int(row_peaks),
        "n_owners": len(owners),
        "max_row_spread_iou": spread,
        "degenerate": bool(degenerate),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--frame", required=True, help="movie:frame")
    ap.add_argument("--movie", action="append", default=[])
    ap.add_argument("--crop-size", type=int, default=288)
    ap.add_argument("--owner-prefix", default="ld|rev8p",
                    help="only masks whose owner_key starts with this")
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--multiscale", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--allow-legacy-semantics",
                    action="store_true",
                    help="replay a checkpoint that declares no activation/schema")
    ap.add_argument("--query-at", default="",
                    help="native x,y: place a SINGLE query disc there "
                    "(e.g. on background) and score every mask against "
                    "that one response. The control for query-invariance: "
                    "if a background query returns the same row as a "
                    "grain query, the query is being ignored.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from prototypes.v30_video_apex.dataset import QUERY_OFFSETS
    from prototypes.v30_video_apex.batch_builder import (
        linked_mask_target, QUERY_INDEX)
    from prototypes.v30_video_apex.model import (
        build_owner_prompt, build_model)
    from prototypes.v30_video_apex.train import load_checkpoint
    from prototypes.v30_video_apex.targets import (
        load_clip_pixels, mask_iou_split)
    from tubetracker.annotation_frames import FrameReader
    import torch

    movie, _, fr_s = a.frame.partition(":")
    frame = int(fr_s)
    movies: dict[str, str] = {}
    for spec in a.movie:
        k, _, p = spec.partition("=")
        movies[k] = p

    snap = Path(a.snapshot)
    masks = json.loads((snap / "body_masks.json").read_text())
    rows = [m for m in masks
            if str(m.get("owner_key", "")).startswith(a.owner_prefix)
            and int(m.get("source_frame", -1)) == frame
            and m.get("mask_raster")]
    if len(rows) < 1:
        print(f"need >=1 mask at {a.frame}; found {len(rows)}")
        return 1
    single = len(rows) == 1

    # crop that holds every mask of this frame (+ margin), then clamp
    cs = int(a.crop_size)
    rr = FrameReader(movies[movie])
    Hh, Ww = rr.native_size[1], rr.native_size[0]
    boxes = []
    for m in rows:
        r = m["mask_raster"]
        boxes.append((int(r["x0"]), int(r["y0"]),
                      int(r["x0"]) + int(r["w"]), int(r["y0"]) + int(r["h"])))
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ox = int(min(max(cx - cs / 2, 0), max(0, Ww - cs)))
    oy = int(min(max(cy - cs / 2, 0), max(0, Hh - cs)))
    crop = (ox, oy, cs, cs)
    print(f"crop {crop} holds masks x[{x0},{x1}] y[{y0},{y1}]")

    frames = tuple(frame + o for o in QUERY_OFFSETS)
    missing = tuple(1 if f < 0 else 0 for f in frames)
    frames = tuple(max(0, f) for f in frames)
    clip_np = load_clip_pixels(rr, frames, crop, missing)
    rr.close()

    # per-mask truth in crop coords + the query disc at each grain
    truths: dict[str, np.ndarray] = {}
    valids: dict[str, np.ndarray] = {}
    anchors: dict[str, tuple] = {}
    queries: dict[str, tuple] = {}
    for m in rows:
        owner = str(m["owner_key"]).split("|")[-1]
        # rev9 WP-A.3: the SAME builder the trainer runs, so a target
        # cannot mean one thing in training and another here.
        _b = linked_mask_target(cs, cs, (ox, oy), m,
                                link=str(m.get("link", "")))
        tgt_mask, val_mask = _b.target, _b.valid
        if not _b.extent_used and m.get("review_region"):
            print(f"{owner}: extent quarantined ({_b.quarantine_reason})")
        truths[owner] = tgt_mask
        valids[owner] = val_mask
        tgt = m.get("target_xy") or m.get("focus_xy") or []
        if not (len(tgt) == 2 and 0 <= tgt[0] - ox < cs
                and 0 <= tgt[1] - oy < cs):
            # fall back to the paint's centroid (the grain sits at one
            # end of the tube); never invent a point outside the crop
            ys_p, xs_p = np.nonzero(truths[owner])
            if len(xs_p) == 0:
                print(f"skip {owner}: empty mask")
                truths.pop(owner)
                valids.pop(owner)
                continue
            tgt = [float(xs_p.mean()) + ox, float(ys_p.mean()) + oy]
            print(f"{owner}: no target_xy, using paint centroid {tgt}")
        anchors[owner] = (float(tgt[0]) - ox, float(tgt[1]) - oy)
        queries[owner] = (float(tgt[0]) - ox, float(tgt[1]) - oy)
    owners = sorted(truths)
    if not owners:
        print("no usable owners")
        return 1
    control = None
    if a.query_at:
        if str(a.query_at).strip().lower() == "auto":
            # the EMPTIEST point of the crop: farthest from any painted
            # pixel. The mask-union centre is NOT background for a
            # clump (it landed inside g1's tube and returned a 0.29
            # response, which would have read as a false positive).
            _paint = np.zeros((cs, cs), bool)
            for _t in owners:
                _paint |= (truths[_t] > 0)
            _ys, _xs = np.nonzero(_paint)
            if len(_xs):
                _gx, _gy = np.meshgrid(np.arange(cs), np.arange(cs))
                _d2 = np.full((cs, cs), 1e9, np.float32)
                for _px, _py in zip(_xs, _ys):
                    _d2 = np.minimum(_d2, (_gx - _px) ** 2 + (_gy - _py) ** 2)
                _iy, _ix = np.unravel_index(int(np.argmax(_d2)), _d2.shape)
                _qx, _qy = float(_ix) + ox, float(_iy) + oy
            else:
                _qx = (x0 + x1) / 2.0
                _qy = (y0 + y1) / 2.0
        else:
            _q = [float(v) for v in a.query_at.split(",")]
            _qx = _q[0] if len(_q) == 2 else None
            _qy = _q[1] if len(_q) == 2 else None
        if _qx is not None:
            control = (_qx - ox, _qy - oy)
            queries = {"<control>": control}
            print(f"control query at native ({_qx:.0f},{_qy:.0f}) "
                  f"-> crop ({control[0]:.0f},{control[1]:.0f})")

    # rev9 WP-A.4: build from the checkpoint's OWN declared architecture
    # (no --base/--multiscale guessing), and refuse a partial load.
    from prototypes.v30_video_apex.model_factory import (
        CheckpointContractError, build_model_from_checkpoint,
        optimizer_groups_summary)
    try:
        model, _minfo = build_model_from_checkpoint(
            a.checkpoint,
            allow_legacy_semantics=getattr(a, "allow_legacy_semantics",
                                           False))
        print(f"model from checkpoint: {_minfo['model_kwargs']} "
              f"| groups: {optimizer_groups_summary(_minfo['config'])}")
    except CheckpointContractError as e:
        # rev9 WP-A.4: a checkpoint that predates the contract cannot
        # declare its architecture. The legacy path is DECLARED here
        # (--base/--multiscale on the command line) and loud, never a
        # silent guess.
        print(f"note: {e}")
        print(f"legacy fallback: base={a.base}, multiscale="
              f"{bool(a.multiscale)} (declared on the command line)")
        model = build_model("temporal", base=a.base,
                            multiscale=bool(a.multiscale))
        from prototypes.v30_video_apex.train import load_checkpoint
        _meta = load_checkpoint(a.checkpoint, model, strict=False)
        if _meta.get("missing_keys"):
            print(f"note: checkpoint gaps {_meta['missing_keys']}")
    model.eval()

    # ONE inference path, shared with fit_criteria.py (they disagreed
    # at 0.224 vs 0.007 while each carried its own copy)
    from prototypes.v30_video_apex.inference import predict_queries
    _pres = predict_queries(model, clip_np, dict(queries), a.threshold)
    mat_iou: dict[str, dict[str, float]] = {}
    mat_mass: dict[str, dict[str, float]] = {}
    fit_extra: dict[str, dict] = {}
    for q_owner in sorted(queries):
        prob, pred = _pres[q_owner]
        _row_iou = {}
        mat_iou[q_owner] = {}
        mat_mass[q_owner] = {}
        mat_dist: dict[str, dict] = {}
        for t_owner in owners:
            # masked-on-both-sides IoU over the reviewed validity, with
            # the distal split: comparable to the trainer's dev probe
            _split = mask_iou_split(pred, truths[t_owner],
                                    valids[t_owner],
                                    anchor_xy=anchors.get(q_owner),
                                    split_px="auto")
            mat_iou[q_owner][t_owner] = float(_split["mask_iou"] or 0.0)
            mat_dist[q_owner] = mat_dist.get(q_owner, {})
            mat_dist[q_owner][t_owner] = {
                "proximal": _split["proximal_iou"],
                "distal": _split["distal_iou"]}
            t = truths[t_owner] > 0
            mat_mass[q_owner][t_owner] = (float(prob[t].mean())
                                          if t.sum() else float("nan"))
        print(f"query {q_owner}: " + "  ".join(
            f"{t}: IoU={mat_iou[q_owner][t]:.2f} "
            f"prox={mat_dist[q_owner][t]['proximal'] if mat_dist[q_owner][t]['proximal'] is None else round(mat_dist[q_owner][t]['proximal'], 2)} "
            f"dist={mat_dist[q_owner][t]['distal'] if mat_dist[q_owner][t]['distal'] is None else round(mat_dist[q_owner][t]['distal'], 2)}"
            for t in owners))
        _pq = np.asarray(queries[q_owner], float)
        _ys, _xs = np.nonzero(pred)
        _att = (float(np.min(np.hypot(_xs - _pq[0], _ys - _pq[1])))
                if len(_xs) else None)
        _spill = 0.0
        for _o2 in owners:
            if _o2 != q_owner:
                _spill += float((pred & (truths[_o2] > 0)).sum())
        fit_extra[q_owner] = {
            "attachment_err_px": _att,
            "spill_into_foreign_frac": round(
                _spill / max(1.0, float(pred.sum())), 4),
            "pred_frac_crop": round(float(pred.mean()), 4),
            "all_zero": bool(pred.sum() == 0),
            "all_one": bool(pred.mean() > 0.99),
            "iou_own": float(mat_iou[q_owner].get(q_owner, 0.0) or 0.0),
        }

    # structure readouts: does the row peak on its own tube, and do the
    # rows differ from each other at all?
    summary = {}
    if control is not None:
        # one row only: the response to this query, scored on every mask
        print("control row: " + "  ".join(
            f"{t}: IoU={mat_iou['<control>'][t]:.3f}" for t in owners))
    elif not single:
        summary = summarize_swap(mat_iou, owners)
    if single:
        # one masked tube: a distal-fit readout, not a swap matrix
        o = owners[0]
        summary = {"diag_mean_iou": mat_iou[o][o],
                   "offdiag_mean_iou": 0.0,
                   "rows_peaking_on_own_tube": 1,
                   "n_owners": 1, "max_row_spread_iou": 0.0}
    spread = summary.get("max_row_spread_iou", 0.0)
    row_peaks = summary.get("rows_peaking_on_own_tube", 0)
    # distance profile from each query point: is the far end merely
    # below threshold (0.4x) or genuinely absent (0.0x)? A thresholded
    # IoU cannot tell those apart, and the difference decides whether
    # the failure is under-training or a design limit.
    prof = {}
    if len(queries) == 1:
        o = list(queries)[0]
        qx, qy = queries[o]
        d = np.hypot(xx - qx, yy - qy)
        # profile over the union of all masks' validity, so a control
        # query (which has no mask of its own) still profiles
        _v = np.zeros((cs, cs), bool)
        for _t in owners:
            _v |= (valids[_t] > 0)
        _paint = np.zeros((cs, cs), bool)
        for _t in owners:
            _paint |= (truths[_t] > 0)
        for lo in (0, 10, 20, 30, 40, 60):
            hi = lo + 10 if lo < 40 else lo + 20
            sel = (d >= lo) & (d < hi) & _v
            if not sel.any():
                continue
            prof[f"{lo}-{hi}px"] = {
                "mean_p": float(prob[sel].mean()),
                "paint_frac": float(_paint[sel].mean()),
                "n": int(sel.sum())}
    out = {
        "prob_profile": prof,
        "checkpoint": str(a.checkpoint),
        "snapshot": str(a.snapshot),
        "frame": a.frame,
        "crop_xywh": list(crop),
        "owners": owners,
        "iou": mat_iou,
        "prob_in_mask": mat_mass,
        **summary,
        "note": "diagonal-dominant = the query selects the tube; "
                "identical rows = query-invariant output",
    }
    _own_iou = [fit_extra[q]["iou_own"] for q in fit_extra]
    out["fit_criteria"] = {
        "per_query": fit_extra,
        "iou_ge_0.90_each": bool(_own_iou and all(
            v >= 0.90 for v in _own_iou)),
        "rows_peaking_on_own_tube": row_peaks,
        "no_all_zero_or_all_one": bool(all(
            not v["all_zero"] and not v["all_one"]
            for v in fit_extra.values())),
        # the masks occupy ~1% of a crop; a prediction covering a third
        # of it is the gross-over-inclusion version of the same
        # shortcut, which a >0.99 test does not catch
        "pred_frac_plausible": bool(all(
            v["pred_frac_crop"] < 0.10 for v in fit_extra.values())),
        "owned_absence_case": "NONE IN THIS SNAPSHOT - zero false "
                             "emissions here is not evidence",
        "note": "training-plumbing gates, not generalization "
                "claims",
    }
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"-> {a.out}")
    if control is None:
        print(f"diag mean IoU {out.get('diag_mean_iou', 0.0):.3f} vs off-diag "
          f"{out['offdiag_mean_iou']:.3f}; "
          f"{row_peaks}/{len(owners)} rows peak on their own tube; "
          f"max row spread {spread:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
