"""Render the rev9 WP-A.2 round-trip evidence (native pixels).

For each mask re-saved in the round trip: the raw frame with the painted
outline and the reviewed extent, the target channels the loader builds
(fg / band / reviewed background / unknown), and a zoom on the tube.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/Users/joshjiang/Documents/TubeTracker")
MOVIE = Path("/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4")
OUT = REPO / "runs/prototypes/v30/framing_checks/rev9_roundtrip_evidence.png"

import sys
sys.path.insert(0, str(REPO))
from prototypes.v30_video_apex.batch_builder import linked_mask_target
from prototypes.v30_video_apex.targets import (
    decode_mask_raster, _region_in_crop)
from tubetracker.annotation_frames import FrameReader

DB = Path.home() / "Documents/TubeTracker-annotator-projects/rev8masks2/annotations.db"
c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

reader = FrameReader(str(MOVIE))
try:
    probe = reader.read(0).frame
    H, W = probe.shape[:2]
    print(f"native frame: {W}x{H}")
    rows = []
    for uuid in ("mask-rev8m2-000", "mask-rev8m2-001"):
        rev, data = c.execute(
            "select revision, data from revisions where uuid=? "
            "order by revision desc limit 1", (uuid,)).fetchone()
        d = json.loads(data)
        # crop centred on the raster paint (clamped to the frame)
        r = d["mask_raster"]
        cx = float(r["x0"]) + float(r["w"]) / 2.0
        cy = float(r["y0"]) + float(r["h"]) / 2.0
        ox = int(max(0, min(cx - 144, W - 288)))
        oy = int(max(0, min(cy - 144, H - 288)))
        b = linked_mask_target(288, 288, (ox, oy), d, link="round-trip")
        frame = reader.read(int(d["source_frame"])).frame
        gray = frame if frame.ndim == 2 else frame[..., :3].mean(axis=2)
        crop = gray[oy:oy + 288, ox:ox + 288]
        extent = _region_in_crop(288, 288, (ox, oy), d.get("review_region"))
        rows.append({"uuid": uuid, "rev": rev, "crop": crop,
                     "extent": extent, "b": b,
                     "note": d.get("review_region_note"),
                     "frame": int(d["source_frame"]),
                     "origin": (ox, oy)})
finally:
    reader.close()

fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2))
for row, rec in enumerate(rows):
    crop, ext, b = rec["crop"], rec["extent"], rec["b"]
    # panel 1: raw + paint outline + reviewed extent
    ax = axes[row, 0]
    ax.imshow(crop, cmap="gray", vmin=np.percentile(crop, 1),
              vmax=np.percentile(crop, 99))
    ax.contour(b.fg.astype(float), levels=[0.5], colors="#00e5ff",
               linewidths=1.6)
    ax.contour(ext.astype(float), levels=[0.5], colors="#39ff14",
               linewidths=1.2, linestyles="--")
    ax.set_title(f"{rec['uuid']} rev{rec['rev']}  frame {rec['frame']}\n"
                 f"cyan = paint ({int(b.fg.sum())} px), green = reviewed "
                 f"extent", fontsize=9)
    # panel 2: channels
    ch = np.zeros(crop.shape + (3,), np.float32)
    ch[b.unknown] = (0.22, 0.10, 0.28)
    ch[b.bg_reviewed] = (0.16, 0.32, 0.85)
    ch[b.band] = (0.95, 0.55, 0.10)
    ch[b.fg] = (0.98, 0.92, 0.20)
    ax = axes[row, 1]
    ax.imshow(ch)
    band_px = int(b.band.sum())
    ax.set_title("target channels: yellow fg, blue reviewed bg, purple "
                 "unknown" + (", orange band" if band_px else
                              " (no band: a usable extent replaces it)")
                 + f"\nnote={rec['note']} "
                   f"quarantine={b.quarantine_reason}", fontsize=9)
    # panel 3: zoom on the tube
    ys, xs = np.nonzero(b.fg)
    ax = axes[row, 2]
    if len(xs):
        x0, x1 = max(0, xs.min() - 30), min(288, xs.max() + 31)
        y0, y1 = max(0, ys.min() - 30), min(288, ys.max() + 31)
        ax.imshow(crop[y0:y1, x0:x1], cmap="gray",
                  vmin=np.percentile(crop, 1), vmax=np.percentile(crop, 99))
        ax.contour(b.fg[y0:y1, x0:x1].astype(float), levels=[0.5],
                   colors="#00e5ff", linewidths=1.4)
    ax.set_title("zoom: exactly what was painted (native pixels)",
                 fontsize=9)
    for a_ in axes[row]:
        a_.set_xticks([])
        a_.set_yticks([])
fig.suptitle("rev9 WP-A.2 round trip: extents re-saved by the annotator, "
             "targets built from snap21", fontsize=11)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
for rec in rows:
    b = rec["b"]
    print(f"{rec['uuid']}: note={rec['note']} used={b.extent_used} "
          f"reason={b.quarantine_reason} fg={int(b.fg.sum())} "
          f"bg_reviewed={int(b.bg_reviewed.sum())} "
          f"band={int(b.band.sum())} unknown={int(b.unknown.sum())}")
