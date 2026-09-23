"""WP-C: one microscopy-pretrained model against the same ownership task.

The review: "Use one microscopy checkpoint, the repaired masks and the
same owner queries, train/dev events and scoring contract as the
multiscale control. First measure prompt-conditioned masks WITHOUT
adaptation."

This harness scores a pretrained model on the pinned paired-clump crop
with the SAME contract `eval_whole_instance.py` uses for our model: for
each owner query, IoU against each painted tube, the 3x3 matrix, the
diagonal/off-diagonal means, and how many rows peak on their own tube.

Backends:
  cellpose  (cpsam_v2 / cpdino): microscopy-pretrained, no prompts of
            its own — the owner query is applied by picking the
            instance that covers the queried grain (declared rule).
  micro_sam (vit_b_lm):         prompt-conditioned by the grain point.

Nothing here is adapted or fine-tuned: this is the review's first
measurement, not the fine-tuning step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_CROP = (475, 518, 288, 288)      # the pinned clump crop


def _load_crop(snapshot: Path, movie_key: str, frame: int,
               crop) -> tuple[np.ndarray, dict[str, np.ndarray],
                              dict[str, list]]:
    from prototypes.v30_video_apex.targets import (
        decode_mask_raster, load_clip_pixels, samples_from_snapshot)
    from tubetracker.annotation_frames import FrameReader

    man = json.loads((snapshot / "snapshot_manifest.json").read_text())
    movies = {k: v.get("path") for k, v in (man.get("movies") or {}).items()}
    reader = FrameReader(movies[movie_key])
    try:
        gray = load_clip_pixels(reader, (frame,), crop, (0,))[0]
    finally:
        reader.close()
    masks: dict[str, np.ndarray] = {}
    grains: dict[str, list] = {}
    for s in samples_from_snapshot(str(snapshot)):
        if (s.kind == "body_mask" and s.movie == movie_key
                and int(s.source_frame) == frame and s.mask_raster):
            owner = str(s.owner_key).split("|")[-1]
            m = decode_mask_raster(crop[3], crop[2],
                                   (crop[0], crop[1]), s.mask_raster) > 0
            if not m.any():
                continue          # this owner has no tube in this crop
            masks[owner] = m
            tgt = list(s.target_xy or []) or list(s.focus_xy or [])
            g = None
            if len(tgt) == 2:
                g = [float(tgt[0]) - crop[0], float(tgt[1]) - crop[1]]
                if not (0 <= g[0] < crop[2] and 0 <= g[1] < crop[3]):
                    g = None
            if g is None:
                # declared fallback: the queried grain's recorded point
                # lies outside this crop, so the paint's own root end
                # (lowest painted pixel column) is the query instead
                ys, xs = np.nonzero(m)
                if len(xs):
                    g = [float(xs.mean()), float(ys.mean())]
            if g is not None:
                grains[owner] = g
    return gray, masks, grains


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def _cellpose_masks(gray: np.ndarray, grains: dict[str, list],
                    model_type: str, diameter: float | None):
    from cellpose import models
    model = models.CellposeModel(gpu=False, pretrained_model=model_type)
    out = model.eval(gray, channels=[0, 0], diameter=diameter)
    labels = out[0] if isinstance(out, tuple) else out
    labels = np.asarray(labels)
    ids = [int(v) for v in np.unique(labels) if v > 0]
    masks: dict[str, np.ndarray] = {}
    info: dict[str, dict] = {}
    for owner, (gx, gy) in grains.items():
        j = int(round(gx))
        i = int(round(gy))
        lab = 0
        if 0 <= i < labels.shape[0] and 0 <= j < labels.shape[1]:
            lab = int(labels[i, j])
        if lab == 0 and ids:      # nearest instance centre (declared rule)
            best, best_lab = None, 0
            for _id in ids:
                ys, xs = np.nonzero(labels == _id)
                d = float(np.hypot(xs.mean() - gx, ys.mean() - gy))
                if best is None or d < best:
                    best, best_lab = d, _id
            lab = best_lab
        masks[owner] = (labels == lab) if lab else np.zeros_like(labels,
                                                                 dtype=bool)
        info[owner] = {"picked_instance": lab,
                       "instance_cover": (float(masks[owner].sum())
                                          / max(1.0, float(labels.size)))}
    info["_instances"] = {"n_instances": len(ids),
                          "labelled_frac": float((labels > 0).mean())}
    return masks, info


def _micro_sam_masks(gray: np.ndarray, grains: dict[str, list],
                     ckpt: str | None):
    """Prompt-conditioned masks from the light-microscopy SAM (vit_b_lm).

    NO adaptation: the official checkpoint, prompted with one positive
    point at the queried grain. Embeddings are computed once for the
    crop and reused for all queries (the review asks for the
    owner-independent encoder with an owner-conditioned decoder).
    """
    import torch
    from micro_sam import util as ms_util
    from micro_sam.prompt_based_segmentation import segment_from_points

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # micro_sam 1.8 returns a SamPredictor directly (verified above):
    # wrapping it again raised "'SamPredictor' object has no attribute
    # 'image_encoder'".
    predictor = ms_util.get_sam_model(model_type="vit_b_lm",
                                      device=device,
                                      checkpoint_path=(ckpt or None))
    img = np.clip(np.asarray(gray) * 255.0, 0, 255).astype(np.uint8)
    emb = ms_util.precompute_image_embeddings(predictor, img, ndim=2)
    masks: dict[str, np.ndarray] = {}
    info: dict[str, dict] = {}
    for owner, (gx, gy) in grains.items():
        pts = np.array([[float(gy), float(gx)]], dtype=float)   # (y, x)
        lbl = np.array([1], dtype=int)
        res = segment_from_points(predictor, pts, lbl,
                                  image_embeddings=emb,
                                  multimask_output=True)
        mask = np.asarray(res[0]) > 0
        masks[owner] = mask
        info[owner] = {"n_points": 1,
                       "cover": round(float(mask.mean()), 5),
                       "score": (round(float(res[1]), 4)
                                 if len(res) > 1 else None)}
    info["_instances"] = {"device": device,
                          "checkpoint": ckpt or "default vit_b_lm"}
    return masks, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cellpose",
                    choices=["cellpose", "micro_sam"])
    ap.add_argument("--model-type", default="cpsam_v2",
                    help="cellpose pretrained id (cpsam_v2 | cpdino | "
                         "cpsam)")
    ap.add_argument("--diameter", type=float, default=None)
    ap.add_argument("--prompt", default="grain",
                    choices=["grain", "proximal"],
                    help="grain = the queried grain's point (what the app "
                         "shows); proximal = a verified point INSIDE the "
                         "painted tube, nearest its root end (the review's "
                         "declared alternative for a tube-only mask)")
    ap.add_argument("--micro-sam-checkpoint", default="")
    ap.add_argument("--snapshot", default=str(
        REPO / "runs/prototypes/v30/snapshots/snap23"))
    ap.add_argument("--movie", default="ld")
    ap.add_argument("--frame", type=int, default=42000)
    ap.add_argument("--crop", default=",".join(str(v) for v in DEFAULT_CROP),
                    help="pinned crop x,y,w,h (default: the clump crop)")
    ap.add_argument("--control", default=str(
        REPO / "runs/prototypes/v30/threequery_baseline_learn3.json"),
        help="multiscale control JSON to compare against")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    crop = tuple(int(v) for v in a.crop.split(","))

    gray, truth, grains = _load_crop(Path(a.snapshot), a.movie, a.frame,
                                     crop)
    if a.prompt == "proximal":
        # replace each query with the painted pixel nearest the owner's
        # recorded grain point: inside the tube, at its proximal end
        for owner in list(grains):
            m = truth.get(owner)
            if m is None or not m.any():
                continue
            gx, gy = grains[owner]
            ys, xs = np.nonzero(m)
            d = np.hypot(xs - gx, ys - gy)
            i = int(np.argmin(d))
            grains[owner] = [float(xs[i]), float(ys[i])]
        print("prompt=proximal (verified in-tube points)")
    print(f"crop {crop} | owners {sorted(truth)} | grains "
          f"{ {k: [round(q, 1) for q in v] for k, v in grains.items()} }")
    owners = sorted(truth)
    if a.backend == "cellpose":
        preds, info = _cellpose_masks(gray, grains, a.model_type, a.diameter)
    else:
        preds, info = _micro_sam_masks(
            gray, grains, a.micro_sam_checkpoint or None)

    matrix = {q: {o: round(_iou(preds[q], truth[o]), 4) for o in owners}
              for q in owners}
    diag = [matrix[q][q] for q in owners]
    off = [matrix[q][o] for q in owners for o in owners if q != o]
    peaks = sum(1 for q in owners
                if matrix[q][q] == max(matrix[q].values()))
    out = {"backend": a.backend,
           "model": a.model_type if a.backend == "cellpose" else "vit_b_lm",
           "adapted": False,
           "prompt_mode": a.prompt,
           "crop_xywh": list(crop), "movie": a.movie, "frame": a.frame,
           "owners": owners, "iou": matrix,
           "diag_mean_iou": round(float(np.mean(diag)), 4),
           "offdiag_mean_iou": round(float(np.mean(off)), 4),
           "rows_peaking_on_own_tube": int(peaks),
           "n_owners": len(owners),
           "instance_info": info,
           "pred_cover": {q: round(float(preds[q].mean()), 4)
                          for q in owners}}
    ctrl = Path(a.control)
    if ctrl.exists():
        c = json.loads(ctrl.read_text())
        out["multiscale_control"] = {
            "diag_mean_iou": c.get("diag_mean_iou"),
            "offdiag_mean_iou": c.get("offdiag_mean_iou"),
            "rows_peaking_on_own_tube": c.get("rows_peaking_on_own_tube")}
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("instance_info",)}, indent=1)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
