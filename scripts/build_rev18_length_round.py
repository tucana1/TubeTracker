"""rev18 human growth-curve round: one centerline trace task per (grain, frame).

Replaces the noisy automatic length proxy (per-frame reach max: spiky) with
direct human measurements, exactly as proposed: "mark more tips and trace tube
lengths at various points". At each ladder frame the investigator traces the
tube from where it leaves the grain rim to its far tip — the trace IS the tube
length at that frame and the LAST dot IS the tip mark. 6 frames per grain give
a human length-vs-frame curve per tube: the direct test of "is the tube
lengthening from that point of the pollen", the spike adjudication (human curve
should be smooth where the automatic one spiked), and the S11 dispute.

Every task is load-bearing: one (length, tip, rim-exit) measurement per time
point = one point on one grain's growth curve. Grain centring is per-task AT
ITS OWN FRAME (drift-tracked circle fit, incremental re-seed from the rev17
anchor), so each task opens centred on the grain at the measured frame — no
per-frame camera travel needed.

Roles keep rev17's split: S3/S4/S6 calibration, S9/S11 heldout (S11 also
adjudicates "emerged_at_start" vs my short/jumpy call). The traces bank
reviewed route geometry at multiple ages — route evidence the prototype's
"supported current paths" needs. Appended to the LIVE project; launches scoped
to just these tasks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "runs" / "prototypes" / "v30"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from rev17_precise_centers import centre  # noqa: E402
from tubetracker.annotation_frames import FrameReader  # noqa: E402
from tubetracker.annotation_store import AnnotationStore  # noqa: E402
from tubetracker.analysis_project import QUEUE  # noqa: E402

MOVIE = "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"
MOVIE_HASH = "238ebd68f4d97ceca32889cc68191d6d11e71be5fd5c2352bbf22438498b13e1"
PROJECT = Path("/Users/joshjiang/Documents/TubeTracker-annotator-projects/rev14analysis")
SCRATCH = Path("/Users/joshjiang/.hermes/cache/scratch")

# sid, anchor_centre (rev17 circle-fit), body radius, anchor_frame, ladder (asc)
GRAINS = [
    ("S3",  (490.66, 252.57),  11.86, 18500,
     [300, 1500, 4000, 10000, 17000, 24000]),
    ("S4",  (1085.39, 870.02), 11.87, 14500,
     [1500, 5000, 8500, 12500, 16000, 19500]),
    ("S6",  (202.11, 744.91),  13.03, 22500,
     [2500, 7000, 12000, 17000, 22000, 27500]),
    ("S9",  (883.91, 187.22),  12.98, 10500,
     [1000, 4000, 7000, 10000, 13000, 15500]),
    ("S11", (631.81, 642.40),  11.89, 20000,
     [3000, 7500, 12000, 16500, 21000, 25500]),
]
ROLES = {"S3": "calibration", "S4": "calibration", "S6": "calibration",
         "S9": "heldout", "S11": "heldout"}

WHY = (
    "TUBE LENGTH + TIP at ONE frame — {sid} grain (green ring), frame {f} "
    "(point {i} of {n} on this grain's growth curve).\n\n"
    "TRACE THIS GRAIN'S TUBE (ignore every other ball in view):\n"
    "  • 1st click: where the tube leaves/crosses this grain's rim — the exit "
    "point on the ball's edge.\n"
    "  • then keep clicking along the middle of the tube, outward.\n"
    "  • LAST click: the tube's far TIP (this last dot IS the tip mark).\n"
    "Then press 'Save FULL current path' if you traced all the way to the tip, "
    "or 'Save PARTIAL current path' if any stretch is hidden or the tip is "
    "unclear — never guess the tip.\n\n"
    "No tube or bump at all on THIS grain at this frame -> press 'No tube at "
    "this frame — save and next'. A tiny bump is still a trace: just 2 dots "
    "(rim exit, then bump tip) and Save FULL. Mark only what you actually "
    "see. You may scrub around first to orient (return to THIS frame to draw)."
)


def hop_fit(reader, cur, f_from, f_to, direction):
    """Incremental circle-fit track: hops of <=500 frames, fit at every hop."""
    frames = list(range(f_from + 500 * direction, f_to, 500 * direction)) + [f_to]
    for h in frames:
        g = cv2.cvtColor(reader.read(int(h)).frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        res = centre(g, cur[0], cur[1])
        if res is not None:
            cur = (res[0], res[1])
    return cur


def track_centres(reader, anchor_xy, anchor_f, targets):
    want = sorted(set(int(f) for f in targets) | {int(anchor_f)})
    anchor_f = int(anchor_f)
    centres = {anchor_f: (float(anchor_xy[0]), float(anchor_xy[1]))}
    for direction in (1, -1):
        cur = centres[anchor_f]
        prev = anchor_f
        span = [f for f in want if (f - anchor_f) * direction > 0]
        span = span if direction > 0 else span[::-1]
        for f in span:
            cur = hop_fit(reader, cur, prev, f, direction)
            centres[f] = cur
            prev = f
    return centres


def main() -> int:
    reader = FrameReader(MOVIE)
    fitted: dict[str, dict[int, tuple[float, float]]] = {}
    tiles = []
    for sid, anchor_xy, rad, anchor_f, ladder in GRAINS:
        centres = track_centres(reader, anchor_xy, anchor_f, ladder)
        fitted[sid] = {f: (round(c[0], 2), round(c[1], 2)) for f, c in centres.items()}
        for f in ladder:
            cx, cy = centres[f]
            g = cv2.cvtColor(reader.read(f).frame, cv2.COLOR_BGR2GRAY)
            half = 34
            h, w = g.shape
            x0, y0 = max(0, int(cx) - half), max(0, int(cy) - half)
            crop = g[y0:min(h, int(cy) + half), x0:min(w, int(cx) + half)]
            crop = cv2.copyMakeBorder(crop, 0, max(0, 2 * half - crop.shape[0]),
                                      0, max(0, 2 * half - crop.shape[1]),
                                      cv2.BORDER_REFLECT)
            S = 5
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            rgb = cv2.resize(rgb, (rgb.shape[1] * S, rgb.shape[0] * S),
                             interpolation=cv2.INTER_LANCZOS4)
            ccx, ccy = int((cx - x0) * S), int((cy - y0) * S)
            cv2.circle(rgb, (ccx, ccy), int(rad * S), (0, 255, 0), 2)
            cv2.drawMarker(rgb, (ccx, ccy), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(rgb, f"{sid} f{f}", (4, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)
            tiles.append(rgb)
    reader.close()
    # verification sheet: 5 grain rows x 6 frame cols, ascending frames
    h, w = tiles[0].shape[:2]
    cols = 6
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * h + 8 * (rows + 1), cols * w + 8 * (cols + 1), 3), 15, np.uint8)
    for k, im in enumerate(tiles):
        yy = 8 + (k // cols) * (h + 8)
        xx = 8 + (k % cols) * (w + 8)
        sheet[yy:yy + h, xx:xx + w] = im
    cv2.imwrite(str(SCRATCH / "rev18_views.png"), sheet)

    store = AnnotationStore(PROJECT / "annotations.db")
    order = 0
    try:
        before = sum(1 for t in store.unfinished_tasks(limit=500)
                     if t.get("review_queue") == QUEUE)
        for sid, _xy, rad, _af, ladder in GRAINS:
            for i, f in enumerate(ladder):
                cx, cy = fitted[sid][f]
                task = {
                    "uuid": f"rev18-len-{sid}-f{f:05d}",
                    "task_type": "centerline",
                    "movie": "ld",
                    "movie_uuid": "ld",
                    "movie_content_hash": MOVIE_HASH,
                    "owner_uuid": f"rev17-{sid}",   # same grain identity as rev17
                    "grain_id": f"rev17-{sid}",
                    "query_frames": [int(f)],
                    "focus_xy": [cx, cy],
                    "target_xy": [cx, cy],
                    "target_r": rad,
                    "view_zoom": 5.5,
                    "review_queue": QUEUE,
                    "queue_order": 200 + order,
                    "queue_batch": "rev18-len",
                    "queue_position": order + 1,
                    "stratum": "sparse_tube_length_growth",
                    "role": ROLES[sid],
                    "geometry_type": "path",
                    "class_scope": "tube",
                    "instance_scope": f"rev17-{sid}",
                    "completeness": "unreviewed",
                    "completed": False,
                    "why": WHY.format(sid=sid, f=f, i=i + 1, n=len(ladder)),
                }
                store.save("task", task["uuid"], task, actor="rev18len")
                order += 1
        after = sum(1 for t in store.unfinished_tasks(limit=500)
                    if t.get("review_queue") == QUEUE)
        print(f"QUEUE pending: {before} -> {after} (wrote {order} tasks)")
    finally:
        store.close()
    (SCRATCH / "rev18_centres.json").write_text(json.dumps(fitted, indent=1))
    print(f"wrote rev18_views.png + rev18_centres.json; {order} tasks published")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
