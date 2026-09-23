"""Audit actual native frames, scoped body errors and reserved cap proposals."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tubetracker.annotation_frames import FrameReader
from prototypes.v30_video_apex.native_body import load_body_checkpoint, body_input, predict_owned_body
from prototypes.v30_video_apex.native_caps import (load_native_checkpoint, detect_caps,
    extract_tile, file_hash, OFFSETS)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", default="runs/prototypes/v30/snap27_rev14")
    p.add_argument("--body-panel", default="runs/prototypes/v30/rev14_native_body_panel")
    p.add_argument("--cap-panel", default="runs/prototypes/v30/rev14_native_panel")
    p.add_argument("--body-checkpoint", default="runs/prototypes/v30/rev14_body_low_lr/step-1600.pt")
    p.add_argument("--cap-checkpoint", default="runs/prototypes/v30/rev14_native_single_refine/step-0200.pt")
    p.add_argument("--temporal-checkpoint", default="runs/prototypes/v30/rev14_native_temporal_refine/step-0300.pt")
    p.add_argument('--cap-label', default='single')
    p.add_argument('--comparison-label', default='temporal')
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    a = p.parse_args()
    torch.set_num_threads(2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    snapshot = Path(a.snapshot)
    sm = json.loads((snapshot / "snapshot_manifest.json").read_text())
    observations = json.loads((snapshot / "observations.json").read_text())
    bdoc = json.loads((Path(a.body_panel) / "panel.json").read_text())
    cdoc = json.loads((Path(a.cap_panel) / "panel.json").read_text())
    cases = []
    for uid in ("obs-rev13w4-004", "obs-rev13w4-002", "obs-r3-005", "obs-r3-007"):
        obs = next(o for o in observations if o["obs_uuid"] == uid)
        roi = [800, 200, 1160, 510] if obs["movie"] == "ld" else [512, 704, 896, 1024]
        cases.append({"id": uid, "movie": obs["movie"], "frame": obs["source_frame"],
                      "tip_xy": obs["direct_xy"], "revision": obs["obs_revision"], "roi_xyxy": roi})
    fit_frames = {(c["movie"], c["frame"]) for c in cdoc["cases"]}
    assert not any((c["movie"], c["frame"]) in fit_frames for c in cases)
    phases = [[0, 0], [17, 29], [47, 53]]
    manifest = {"scope": "same-movie reserved interval plus movie-2 cap transfer; no whole-field precision claim",
        "limits": "Low-density interval was already inspected during development. Movie-2 has two point reviews, not exhaustive cap or owner truth.",
        "frozen_before_this_evaluation": True, "cases": cases, "tile_phases_xy": phases,
        "cap_threshold": .5, "proposal_recall_radius_px": 5,
        "snapshot": str(snapshot.resolve()), "snapshot_sha256": file_hash(snapshot / "snapshot_manifest.json"),
        "body_panel_sha256": file_hash(Path(a.body_panel) / "panel.json"),
        "cap_panel_sha256": file_hash(Path(a.cap_panel) / "panel.json"),
        "checkpoint_sha256": {k: file_hash(v) for k, v in {
            "body": a.body_checkpoint, a.cap_label: a.cap_checkpoint, a.comparison_label: a.temporal_checkpoint}.items()},
        "source_sha256": file_hash(__file__), "excluded_from_cap_fit_by_movie_frame": True,
        'source_files': {name: file_hash(ROOT/name) for name in (
            'prototypes/v30_video_apex/native_caps.py', 'prototypes/v30_video_apex/native_body.py',
            'prototypes/v30_video_apex/cap_evidence.py', 'tubetracker/annotation_frames.py')}}
    dump(out / "frozen_panel.json", manifest)
    readers, images = {}, {}
    def frame(movie, fid):
        key = movie, fid
        if key not in images:
            if movie not in readers:
                readers[movie] = FrameReader(sm["movies"][movie]["path"])
            r = readers[movie].read(fid)
            if not r.exact:
                raise ValueError("inexact source frame in native audit")
            images[key] = cv2.cvtColor(r.frame, cv2.COLOR_BGR2GRAY)
        return images[key]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        arrays = dict(np.load(Path(a.body_panel) / "panel.npz"))
        assert file_hash(Path(a.body_panel) / "panel.npz") == bdoc["dataset_sha256"]
        body, _ = load_body_checkpoint(a.body_checkpoint, a.device)
        brows, visuals = [], []
        for i, c in enumerate(bdoc["cases"]):
            gray = frame(c["movie"], c["frame"])
            origin = (np.asarray(c["origin"]) + 16).tolist()
            native = extract_tile([gray], origin, 288)[0]
            stored = arrays["pixels"][i, 16:304, 16:304]
            if not np.array_equal(native, stored):
                raise ValueError("stored training panel differs from current native frame")
            owner = {"grain_native": c["grain"], "grain_radius_px": c["radius"]}
            prob, _ = predict_owned_body(body, gray, owner, origin)
            with torch.no_grad():
                direct = torch.sigmoid(body(torch.from_numpy(body_input(stored, origin,
                    c["grain"], c["radius"], body.image_gain))[None].to(a.device)))[0, 0].cpu().numpy()
            assert np.array_equal(prob, direct)
            pos, neg, foreign = [arrays[k][i, 16:304, 16:304] for k in ("positive", "ordinary", "foreign")]
            valid, pred = pos | neg | foreign, prob >= .5
            fp, fn = pred & neg, pos & ~pred
            distance = distance_transform_edt(~pos)
            _, components = label(fp & (distance > 2))
            row = {"id": c["id"], "kind": c["kind"], "frame": c["frame"],
                "native_pixels_identical": True, "runtime_probability_max_error": float(np.abs(prob-direct).max()),
                "iou": float((pred & pos).sum() / max(1, ((pred | pos) & valid).sum())) if pos.any() else None,
                "positive_pixels": int(pos.sum()), "false_negative_pixels": int(fn.sum()),
                "licensed_background_false_positive_pixels": int(fp.sum()),
                "false_positive_pixels_within_2px_of_paint": int((fp & (distance <= 2)).sum()) if pos.any() else 0,
                "far_false_positive_components": components,
                "foreign_positive_pixels": int((pred & foreign).sum()),
                "unreviewed_positive_pixels_unscored": int((pred & ~valid).sum())}
            brows.append(row)
            if pos.any():
                visuals.append((c, native, pos, fp, fn, pred & ~valid, row))
        dump(out / "body_error_audit.json", {"scope": bdoc["scope"], "checkpoint": a.body_checkpoint,
            "boundary_analysis_is_diagnostic_only": True, "gate_unchanged": "min IoU >= .90 and min recall >= .95; no threshold relaxation",
            "rows": brows})
        for c, native, pos, fp, fn, unknown, row in sorted(visuals, key=lambda v: v[-1]["iou"])[:4]:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for ax in axes:
                ax.imshow(native, cmap="gray", vmin=0, vmax=255); ax.axis("off")
            axes[0].contour(pos, levels=[.5], colors=["lime"], linewidths=.8)
            overlay = np.zeros((*native.shape, 4))
            overlay[fp] = [1, .1, .1, .9]; overlay[fn] = [0, .8, 1, .9]
            overlay[unknown] = [.8, .2, .9, .5]
            axes[1].imshow(overlay)
            axes[0].set_title("Preserved human paint: green")
            axes[1].set_title("Red: scoped extra; cyan: missed\nPurple: unreviewed, unscored")
            fig.suptitle(c["id"] + f"  IoU {row['iou']:.3f}")
            fig.tight_layout(); fig.savefig(out / (c["id"] + "-errors.png"), dpi=140); plt.close(fig)
        del body
        cap_rows = []
        for model_name, checkpoint in ((a.cap_label, a.cap_checkpoint), (a.comparison_label, a.temporal_checkpoint)):
            model, _ = load_native_checkpoint(checkpoint, a.device)
            for c in cases:
                clip = [frame(c["movie"], c["frame"] + delta) for delta in OFFSETS]
                for phase in phases:
                    detected = detect_caps(model, clip, movie=c["movie"], source_frame=c["frame"],
                        roi=c["roi_xyxy"], threshold=.5, tile_phase_xy=phase)
                    nearest = min(detected["caps"], key=lambda cap: np.linalg.norm(
                        np.asarray(cap["tip_xy"]) - c["tip_xy"]), default=None)
                    distance = float(np.linalg.norm(np.asarray(nearest["tip_xy"]) - c["tip_xy"])) if nearest else None
                    row = {"model": model_name, "case": c["id"], "movie": c["movie"], "frame": c["frame"],
                        "tile_phase_xy": phase, "n_cap_proposals": len(detected["caps"]),
                        "nearest_proposal_error_px": distance, "proposal_within_5px": distance is not None and distance <= 5,
                        "nearest_probability": nearest["probability"] if nearest else None,
                        "licensed_false_cap_count": None, "owner_selection_accuracy": None,
                        "caps": detected["caps"], "tile_input_hashes": detected["tile_hashes"],
                        'tiling': detected.get('tiling')}
                    cap_rows.append(row)
                    if phase == [0, 0]:
                        x0, y0, x1, y1 = c["roi_xyxy"]
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ax.imshow(clip[1][y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255,
                                  extent=(x0, x1, y1, y0))
                        for cap in detected["caps"]:
                            ax.plot(*cap["tip_xy"], "o", markerfacecolor="none", markeredgecolor="orange", markersize=7)
                        ax.plot(*c["tip_xy"], "+", color="lime", markersize=12)
                        ax.set_title(f"{model_name} / {c['id']}\nGreen: reviewed cap; orange: all proposals (unscored precision)")
                        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); fig.tight_layout()
                        fig.savefig(out / f"{model_name}-{c['id']}.png", dpi=140); plt.close(fig)
                    print(json.dumps({k: v for k, v in row.items() if k not in ("caps", "tile_input_hashes", 'tiling')}), flush=True)
            del model
        dump(out / "cap_roi_audit.json", {"manifest": manifest, "rows": cap_rows})
        print(json.dumps({"body_cases": len(brows), "cap_cases_and_phases": len(cap_rows), "out": str(out)}))
    finally:
        for reader in readers.values():
            reader.close()


if __name__ == "__main__":
    main()
