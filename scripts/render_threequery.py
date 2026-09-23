"""Native evidence for the three-query state (milestone requirement).

Raw crop + human paints + the three owner queries + the model's three
prediction panels, with IoU and coverage annotated per query. Uses the
shared inference path so the figure cannot drift from the evaluator's
numbers.

usage: python scripts/render_threequery.py --checkpoint <pt> --out <png>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CROP = (475, 518, 288, 288)


def main() -> int:
    from prototypes.v30_video_apex.inference import (load_query_clip,
                                                     predict_queries)
    from prototypes.v30_video_apex.model import build_model
    from prototypes.v30_video_apex.model_factory import (
        CheckpointContractError, build_model_from_checkpoint)
    from prototypes.v30_video_apex.targets import (
        decode_mask_raster, samples_from_snapshot)
    from prototypes.v30_video_apex.train import load_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base", type=int, default=4)
    ap.add_argument("--frame", type=int, default=42000)
    ap.add_argument("--movie", default="ld")
    ap.add_argument("--snapshot", default=str(
        REPO / "runs/prototypes/v30/snapshots/snap23"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--evaluator-json", default="",
                    help="eval_whole_instance.py output: its IoU is masked "
                         "to the REVIEWED region, the render's is the whole "
                         "crop — showing both is the over-inclusion read")
    a = ap.parse_args()

    snap = Path(a.snapshot)
    truth, queries = {}, {}
    for s in samples_from_snapshot(str(snap)):
        if (s.kind == "body_mask" and s.movie == a.movie
                and int(s.source_frame) == a.frame and s.mask_raster):
            owner = str(s.owner_key).split("|")[-1]
            m = decode_mask_raster(CROP[3], CROP[2],
                                   (CROP[0], CROP[1]), s.mask_raster) > 0
            if not m.any():
                continue
            truth[owner] = m
            tgt = list(s.target_xy or [])
            if len(tgt) == 2 and 0 <= tgt[0] - CROP[0] < CROP[2] \
                    and 0 <= tgt[1] - CROP[1] < CROP[3]:
                queries[owner] = [tgt[0] - CROP[0], tgt[1] - CROP[1]]
            else:
                ys, xs = np.nonzero(m)
                queries[owner] = [float(xs.mean()), float(ys.mean())]
    owners = sorted(truth)

    gray = load_query_clip(snap, a.movie, a.frame, CROP)[4]  # centre frame
    try:
        model, _ = build_model_from_checkpoint(a.checkpoint)
        load_note = "factory (declared architecture)"
    except CheckpointContractError:
        model = build_model("temporal", base=a.base, multiscale=True)
        load_checkpoint(a.checkpoint, model, strict=False)
        load_note = f"legacy fallback base={a.base} multiscale=True"
    res = predict_queries(model, load_query_clip(snap, a.movie, a.frame, CROP),
                          {o: queries[o] for o in owners}, 0.5)

    ev = {}
    if a.evaluator_json:
        try:
            ev = json.loads(Path(a.evaluator_json).read_text()).get(
                "iou", {})
        except Exception as e:  # noqa: BLE001
            print("evaluator json ignored:", e)

    n = len(owners)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.4 * (n + 1), 4.2))
    axes[0].imshow(gray, cmap="gray")
    colours = ["#00e5ff", "#ff2ec4", "#ffd000", "#39ff14"]
    for c, o in zip(colours, owners):
        axes[0].contour(truth[o].astype(float), levels=[0.5], linewidths=1.6,
                        colors=[c])
        axes[0].plot(queries[o][0], queries[o][1], "+", ms=11, mew=2,
                     color=c)
    axes[0].set_title("human paints + owner queries\n(raw centre frame)",
                      fontsize=9)
    for k, o in enumerate(owners):
        prob, pred = res[o]
        t = truth[o]
        inter = float((pred & t).sum())
        union = float((pred | t).sum())
        iou = inter / union if union else 0.0
        show = np.stack([gray] * 3, -1)
        show[pred] = 0.45 * show[pred] + 0.55 * np.array([1.0, 0.25, 0.25])
        show[t & ~pred] = 0.45 * show[t & ~pred] + 0.55 * np.array(
            [0.2, 1.0, 0.3])
        show[pred & t] = 0.45 * show[pred & t] + 0.55 * np.array(
            [1.0, 1.0, 1.0])
        axes[k + 1].imshow(np.clip(show, 0, 1))
        _rev = (ev.get(o) or {}).get(o)
        axes[k + 1].set_title(
            f"query {o}\nIoU(crop) {iou:.3f}"
            + (f" | IoU(reviewed) {_rev:.3f}" if _rev is not None else "")
            + f"\npred {pred.mean()*100:.1f}% of crop | paint "
              f"{t.mean()*100:.1f}%", fontsize=9)
        axes[k + 1].plot(queries[o][0], queries[o][1], "+", ms=10, mew=2,
                         color=colours[k])
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{a.checkpoint} — {load_note}\nred = predicted only, "
                 f"green = painted only, white = agreement", fontsize=10)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)
    meta = {"checkpoint": a.checkpoint, "crop_xywh": list(CROP),
            "load": load_note,
            "evaluator_json": a.evaluator_json,
            "per_query": {o: {"iou_crop": float(((res[o][1] & truth[o]).sum())
                                           / max(1, (res[o][1] | truth[o]).sum())),
                              "iou_reviewed": (ev.get(o) or {}).get(o),
                              "pred_frac": float(res[o][1].mean()),
                              "paint_frac": float(truth[o].mean())}
                          for o in owners}}
    Path(str(a.out) + ".json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
