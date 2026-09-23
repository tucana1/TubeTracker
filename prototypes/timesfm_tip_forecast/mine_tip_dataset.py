"""Mine weak tip labels for the CNN point detector (H153).

Label filters compose every TimesFM/v29 instrument as a purity gate:
  1. accepted == 1 (pipeline measurement, not diagnostic);
  2. closed-loop replay verdict == keep (no flicker-out/back/burst);
  3. frame-normalized root support >= 0.15 (no detached-root/drift);
  4. owner allowlist: zero veto faults AND zero root-survey flags AND
     rollout end err < 25 (no instrument ever objected to this owner);
  5. mid-life samples only (exclude first 15 / last 8: emergence noise,
     tail defocus).
Grain channel is left unreviewed (masked out of the loss); only the tip
channel trains. Provenance is recorded per label — these are weak labels,
and the ledger records the protocol version with the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototypes.timesfm_tip_forecast.switch_cut import transverse_wall_energy
from tubetracker.cnn_prototype import (
    FrameReview,
    PointLabel,
    write_dataset_manifest,
    write_frame_reviews,
    write_point_labels,
)

NROOT_MIN = 0.15
ROLLOUT_MAX = 25.0
FRAMES_PER_MOVIE = 70
CONFIDENCE = 0.7
PROVENANCE = "v29-timesfm-weak-v1"


def grad_p95(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.percentile(np.hypot(gx, gy), 95))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--centerlines", required=True)
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--rollout-summary", default="")
    ap.add_argument("--movie", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = pd.read_csv(a.measurements)
    cl = pd.read_csv(a.centerlines)
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    roll_end: dict[int, float] = {}
    if a.rollout_summary:
        summ = json.loads(Path(a.rollout_summary).read_text())
        for k, v in summ.items():
            try:
                roll_end[int(k)] = float(v.get("rollout_err_end") or 1e9)
            except (TypeError, ValueError):
                continue

    rep_verdict: dict[tuple[int, int], str] = {}
    fault_owners: set[int] = set()
    for rf in Path(a.replay_dir).glob("replay_P*.csv"):
        pid = int(rf.stem.split("_")[1][1:])
        r = pd.read_csv(rf)
        for _, row in r.iterrows():
            rep_verdict[(pid, int(row.sample_index))] = row.verdict
        if (r.verdict == "fault-suspect").any() or (r.verdict == "burst").any():
            fault_owners.add(pid)

    root_flag_owners: set[int] = set()
    if Path("/tmp/rootsurvey_norm.csv").is_file():
        surv = pd.read_csv("/tmp/rootsurvey_norm.csv")
        tag = "dense" if "dense" in a.measurements else "lowdens"
        bad = surv[(surv.field == tag) & (surv.nrootsup < 0.10) & (surv.L >= 20)]
        root_flag_owners = set(int(p) for p in bad.pid.unique())

    owners = sorted(m[m.accepted == 1].pollen_id.unique())
    allow = [
        p for p in owners
        if p not in fault_owners and p not in root_flag_owners
        and roll_end.get(p, 0.0) < ROLLOUT_MAX
    ]
    print(f"allowlisted {len(allow)}/{len(owners)} owners", flush=True)

    cap = cv2.VideoCapture(a.movie)
    frame_cache: dict[int, np.ndarray] = {}
    p95_cache: dict[int, float] = {}

    def gray(src: int) -> np.ndarray:
        if src not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, src)
            ok, fr = cap.read()
            assert ok, src
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            frame_cache[src] = g
            p95_cache[src] = grad_p95(g)
        return frame_cache[src]

    # Candidate (owner, sample) -> source_frame, tip.
    cands: list[tuple[int, int, int, float, float]] = []
    for pid in allow:
        d = m[(m.pollen_id == pid) & (m.accepted == 1)].sort_values("sample_index")
        if len(d) < 30:
            continue
        lo, hi = d.sample_index.iloc[15], d.sample_index.iloc[-8]
        for _, row in d.iterrows():
            s = int(row.sample_index)
            if not (lo <= s <= hi):
                continue
            if rep_verdict.get((pid, s), "keep") != "keep":
                continue
            r = cl[(cl.pollen_id == pid) & (cl.sample_index == s)].sort_values(
                "point_index"
            )
            if len(r) < 8:
                continue
            pts = r[["source_x_px", "source_y_px"]].to_numpy(float)
            arc = r.arc_length_px.to_numpy()
            g = gray(int(row.source_frame))
            w = transverse_wall_energy(g, pts)
            if float(np.median(w[arc < 10])) / p95_cache[int(row.source_frame)] < NROOT_MIN:
                continue
            cands.append((pid, s, int(row.source_frame),
                          float(row.tip_x_px), float(row.tip_y_px)))
    print(f"clean tip candidates: {len(cands)}", flush=True)

    # Spread frames across the recording; keep all clean tips per frame.
    by_src: dict[int, list[tuple[int, int, float, float]]] = {}
    for pid, s, src, x, y in cands:
        by_src.setdefault(src, []).append((pid, s, x, y))
    srcs = sorted(by_src)
    pick = srcs[:: max(1, len(srcs) // FRAMES_PER_MOVIE)][:FRAMES_PER_MOVIE]
    print(f"exporting {len(pick)} frames", flush=True)

    labels: list[PointLabel] = []
    reviews: list[FrameReview] = []
    records = []
    for src in pick:
        img_name = f"{a.tag}_{src:06d}.png"
        cap.set(cv2.CAP_PROP_POS_FRAMES, src)
        ok, fr = cap.read()
        assert ok, src
        cv2.imwrite(str(out / "frames" / img_name), fr)
        for pid, s, x, y in by_src[src]:
            labels.append(PointLabel(img_name, src, "tip", x, y,
                                     PROVENANCE, CONFIDENCE))
        reviews.append(FrameReview(img_name, src, False, True))
        records.append({"image_name": img_name, "source_frame": src})
    cap.release()
    write_point_labels(out / "labels.csv", labels)
    write_frame_reviews(out / "frame_reviews.csv", reviews)
    write_dataset_manifest(out / "manifest.json", {"frames": records})
    print(f"wrote {len(labels)} tip labels on {len(records)} frames -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
