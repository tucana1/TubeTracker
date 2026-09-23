"""Every reviewed extent in the snapshot, shown natively.

The review requires exact annotation extents as evidence at each
milestone. For each body-mask sample: the raw crop, the human paint
(outline), the stored review rectangle, and the *supervision actually
licensed* (reviewed background = valid & ~paint). If those three do not
agree, the picture shows it.

usage: python scripts/render_extents_sheet.py [--snapshot ...] --out png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    from prototypes.v30_video_apex.inference import load_query_clip
    from prototypes.v30_video_apex.targets import (
        body_mask_channels_from_raster, decode_mask_raster,
        samples_from_snapshot)

    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(
        REPO / "runs/prototypes/v30/snapshots/snap24"))
    ap.add_argument("--out", default=str(
        REPO / "runs/prototypes/v30/framing_checks/extents_snap24.png"))
    a = ap.parse_args()
    snap = Path(a.snapshot)

    rows = []
    for s in samples_from_snapshot(str(snap)):
        if s.kind == "body_mask" and s.mask_raster:
            rows.append(s)
    rows.sort(key=lambda s: (str(s.movie), int(s.source_frame or 0),
                             str(s.owner_key)))

    cols = 4
    nrow = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(nrow, cols, figsize=(4.1 * cols, 4.3 * nrow))
    axes = np.atleast_2d(axes)
    for k, s in enumerate(rows):
        ax = axes[k // cols][k % cols]
        r = s.mask_raster
        x0, y0 = int(r["x0"]), int(r["y0"])
        w, h = int(r["w"]), int(r["h"])
        pad = 60
        crop = (max(0, x0 - pad), max(0, y0 - pad),
                min(288, w + 2 * pad), min(288, h + 2 * pad))
        cnr = crop[2], crop[3]
        try:
            gray = load_query_clip(snap, s.movie, int(s.source_frame),
                                   crop)[4]
        except Exception as e:  # noqa: BLE001
            ax.set_title(f"{s.obs_uuid}\nframe load failed: {e}",
                         fontsize=8)
            continue
        ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
        paint = decode_mask_raster(cnr[0], cnr[1], (crop[0], crop[1]),
                                   r) > 0
        ax.contour(paint.astype(float), levels=[0.5], linewidths=1.6,
                   colors=["#39ff14"])
        rr = list(s.review_region or [])
        ew = eh = 0
        if len(rr) == 2:
            (ax1, ay1), (ax2, ay2) = rr
            ew, eh = int(ax2 - ax1), int(ay2 - ay1)
            ax.add_patch(mpatches.Rectangle(
                (ax1 - crop[0], ay1 - crop[1]), ax2 - ax1, ay2 - ay1,
                fill=False, ec="#00e5ff", lw=1.6))
        # what supervision is actually licensed
        try:
            ch = body_mask_channels_from_raster(
                cnr[0], cnr[1], (crop[0], crop[1]), r,
                complete=bool(s.complete), review_region=rr)
            bg = (ch["valid"] > 0) & (ch["target"] == 0)
            if bg.any():
                show = np.stack([gray] * 3, -1)
                show[bg] = 0.55 * show[bg] + 0.45 * np.array(
                    [0.1, 0.5, 1.0])
                ax.imshow(np.clip(show, 0, 1))
        except Exception:  # noqa: BLE001
            pass
        ax.set_title(
            f"{s.obs_uuid}\n{str(s.owner_key).split('|')[-1]} | paint "
            f"{int(paint.sum())} px | extent "
            f"{ew}x{eh} | "
            f"{s.quarantine_reason or 'ok'}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(len(rows), nrow * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Reviewed extents, snap24 — green outline = human paint, "
                 "cyan = stored review rectangle,\nblue shading = reviewed "
                 "background actually licensed for supervision", fontsize=11)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=140)
    meta = [{"uuid": s.obs_uuid, "owner": str(s.owner_key).split("|")[-1],
             "complete": bool(s.complete),
             "review_region": list(s.review_region or []),
             "reason": s.quarantine_reason or "ok"} for s in rows]
    Path(str(a.out) + ".json").write_text(json.dumps(meta, indent=1))
    n_ok = sum(1 for s in rows if not s.quarantine_reason)
    print(f"wrote {a.out} | masks {len(rows)} | usable {n_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
