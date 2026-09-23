"""Render one v30 event for eye verification (rev5 #4/#5).

Video = the 9 clip frames around the query (temporal context), with the
automatic overlay drawn ONLY on the query frame: competing proposals
(thin yellow), winner route truncated at the selected front (green),
exported tip (green dot), human tip (white X). Other frames carry just
the frame number. libx264/yuv420p.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="run_v30_movie.py output dir")
    ap.add_argument("--movie", required=True, help="path to the movie")
    ap.add_argument("--event", required=True,
                    help="event obs_uuid substring")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--half-size", type=int, default=256)
    ap.add_argument("--snapshot", default="",
                    help="v30 snapshot (draws human tips as white X)")
    a = ap.parse_args()

    import cv2
    import numpy as np

    from prototypes.v30_video_apex.adapter_v29 import export_tip_path_length
    from prototypes.v30_video_apex.dataset import QUERY_OFFSETS
    from tubetracker.annotation_frames import FrameReader

    run = Path(a.run_dir)
    measurements = json.loads((run / "measurements.json").read_text())
    cands = json.loads((run / "candidates.json").read_text())
    m = next((d for d in measurements if a.event in d["event"]), None)
    if m is None:
        print(f"event {a.event!r} not in run")
        return 1
    frame = int(m["source_frame"])
    gold_tips = []
    if a.snapshot:
        for o in json.loads(
                (Path(a.snapshot) / "observations.json").read_text()):
            if (int(o.get("source_frame", -1)) == frame and o.get("direct_xy")
                    and str(o.get("direct_state", "")) in (
                        "direct_visible", "visible_imprecise")):
                gold_tips.append([float(o["direct_xy"][0]),
                                  float(o["direct_xy"][1])])
    reader = FrameReader(a.movie)
    H, W = reader.native_size[1], reader.native_size[0]

    ev_cands = [c for c in cands
                if c["owner_id"] == m["event"] and c["x_native"] is not None]
    # Zoom box around the winner tip (or gold): auto zoom-to-subject.
    snap_gold = None
    if m["tip"] is not None:
        cx, cy = float(m["tip"][0]), float(m["tip"][1])
    else:
        cx, cy = W / 2, H / 2
    hs = int(a.half_size)
    ox = int(min(max(cx - hs, 0), max(0, W - 2 * hs)))
    oy = int(min(max(cy - hs, 0), max(0, H - 2 * hs)))

    def draw_overlay(img, with_overlay: bool):
        if not with_overlay:
            return img
        # Competing proposals first (thin yellow), winner on top (green).
        for c in ev_cands:
            route_id = c.get("route_id", "")
            is_winner = (m.get("winner") or {}).get("route_id") == route_id
            poly = np.asarray(
                c.get("polyline_native") or
                [[m["root_xy"][0], m["root_xy"][1]],
                 [c["x_native"], c["y_native"]]], float)
            col = (60, 220, 60) if is_winner else (60, 220, 220)
            th = 2 if is_winner else 1
            pts = np.stack([poly[:, 0] - ox, poly[:, 1] - oy],
                           axis=1).astype(int)
            cv2.polylines(img, [pts], False, col, th, cv2.LINE_AA)
        if m["tip"] is not None:
            cv2.circle(img, (int(m["tip"][0] - ox), int(m["tip"][1] - oy)),
                       4, (60, 220, 60), -1, cv2.LINE_AA)
        for gx, gy in gold_tips:
            px, py = int(gx - ox), int(gy - oy)
            cv2.drawMarker(img, (px, py), (255, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
        return img

    frames = []
    for k, off in enumerate(QUERY_OFFSETS):
        fid = max(0, frame + off)
        res = reader.read(fid)
        g = res.frame[:, :, 0] if res.frame.ndim == 3 else res.frame
        crop = g[oy:oy + 2 * hs, ox:ox + 2 * hs]
        if crop.shape[:2] != (2 * hs, 2 * hs):
            pad = np.zeros((2 * hs, 2 * hs), np.uint8)
            pad[:crop.shape[0], :crop.shape[1]] = crop
            crop = pad
        img = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        img = draw_overlay(img, with_overlay=(k == 4))
        tag = f"frame {fid}" + ("  QUERY" if k == 4 else "")
        cv2.putText(img, tag, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        state = ("assisted" if m.get("assisted")
                 else (m.get("status") or ""))
        cv2.putText(img, f"{m['event'][:18]} {state}", (10, 2 * hs - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        frames.append(img)
    reader.close()

    # Hold the query frame longer: context, QUERY x4, context.
    seq = frames[:4] + [frames[4]] * 4 + frames[5:]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"),
                         6, (2 * hs, 2 * hs))
    if not vw.isOpened():
        print("avc1 unavailable, trying mp4v")
        vw = cv2.VideoWriter(str(out),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             6, (2 * hs, 2 * hs))
    for img in seq:
        vw.write(img)
    vw.release()
    print(f"rendered {out} ({len(seq)} frames, "
          f"{'assisted' if m.get('assisted') else m.get('status')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
