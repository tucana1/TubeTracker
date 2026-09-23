"""Corrected tip series (H150): truncation + recovery filter, no extension.

EXPERIMENTAL — DO NOT USE FOR MEASUREMENT. The H150 eye-audit
(/tmp/corr_check.png) convicted this output: truncation at support
valleys on the REAL tube (P22 s202-222) moves tips OFF-apex into
background while accepted was correct (corrected path 434px vs accepted
250px, jumps 6 vs 3). Kept as scaffolding for a future geometry-gated
rule; the recovery-pass HOLD filter (guided_tip_filter.py) is the
shipped acceptance logic.

Combines the two working repairs into one tip series per owner:
  1. switch-cut truncation where it fires (tip -> last supported point,
     eye-verified on-shaft; degenerate cuts with arc < 15px are SKIPPED);
  2. recovery-pass HOLD filter on fault-suspect samples (freeze departures,
     pass returns), applied AFTER truncation (holds chain from corrected
     tips, so s244 freezes at s243's truncated on-shaft tip);
  3. accepted tips everywhere else.
Known-open (documented, not hidden): burst-labeled chimeric tips
(s249/250 class) pass through; pre-onset junction-switches without a
background gap (s228 class, ratio 0.63) are rule-blind.

Render: green corrected tip vs red accepted tip + CUT/HOLD banners.
"""

from __future__ import annotations

import argparse
import subprocess

import cv2
import numpy as np
import pandas as pd

from .guided_tip_filter import RECOVERY_PX, apply_tip_filter
from .switch_cut import cut_at_support_dip, transverse_wall_energy

MIN_CUT_ARC = 15.0  # degenerate root-fallback cuts are skipped


def build_corrected(
    measurements: pd.DataFrame,
    centerlines: pd.DataFrame,
    replay: pd.DataFrame,
    gray_by_src: dict,
    owner: int,
    s0: int = 0,
    s1: int = 10**9,
) -> pd.DataFrame:
    m = measurements.query("pollen_id == @owner and accepted == 1")
    m = m[(m.sample_index >= s0) & (m.sample_index <= s1)].sort_values(
        "sample_index"
    )
    rep = replay.set_index("sample_index")
    # Stage 1: truncation map.
    trunc: dict[int, tuple[float, float, float]] = {}
    for _, row in m.iterrows():
        s = int(row.sample_index)
        r = centerlines[
            (centerlines.pollen_id == owner) & (centerlines.sample_index == s)
        ].sort_values("point_index")
        if len(r) < 8 or int(row.source_frame) not in gray_by_src:
            continue
        pts = r[["source_x_px", "source_y_px"]].to_numpy(float)
        arc = r.arc_length_px.to_numpy()
        w = transverse_wall_energy(gray_by_src[int(row.source_frame)], pts)
        cut = cut_at_support_dip(w, arc)
        if cut is not None and float(arc[cut]) >= MIN_CUT_ARC:
            trunc[s] = (float(pts[cut][0]), float(pts[cut][1]), float(arc[cut]))
    # Stage 2: recovery-pass holds chained on corrected tips.
    cx, cy, held, tag = [], [], [], []
    last = None
    keep_hist: list[tuple[int, tuple[float, float]]] = []
    for _, row in m.iterrows():
        s = int(row.sample_index)
        ax, ay = float(row.tip_x_px), float(row.tip_y_px)
        v = rep.loc[s].verdict if s in rep.index else "keep"
        base = trunc[s][:2] if s in trunc else (ax, ay)
        how = "CUT" if s in trunc else ""
        if v != "fault-suspect":
            keep_hist.append((s, base))
            keep_hist = [(ss, p) for (ss, p) in keep_hist if s - ss <= 10]
        if v == "fault-suspect" and last is not None:
            recent = [np.array(p) for (ss, p) in keep_hist if s - ss <= 10]
            if len(recent) and min(
                float(np.linalg.norm(np.array(base) - p)) for p in recent
            ) <= RECOVERY_PX:
                cx.append(base[0])
                cy.append(base[1])
                last = base
                held.append(False)
                keep_hist.append((s, base))
                tag.append(how if how else "RECOVER")
            else:
                cx.append(last[0])
                cy.append(last[1])
                held.append(True)
                tag.append("HOLD")
        else:
            cx.append(base[0])
            cy.append(base[1])
            last = base
            held.append(False)
            tag.append(how)
    out = pd.DataFrame(
        {
            "sample_index": m.sample_index.to_numpy(),
            "source_frame": m.source_frame.to_numpy(),
            "accepted_x": m.tip_x_px.to_numpy(dtype=float),
            "accepted_y": m.tip_y_px.to_numpy(dtype=float),
            "corrected_x": np.array(cx),
            "corrected_y": np.array(cy),
            "held": np.array(held),
            "tag": np.array(tag),
        }
    )
    out["corrected_step"] = np.linalg.norm(
        np.diff(
            np.column_stack([out.corrected_x, out.corrected_y]),
            axis=0,
            prepend=[[out.corrected_x.iloc[0], out.corrected_y.iloc[0]]],
        ),
        axis=1,
    )
    return out


def render_corrected(
    measurements: pd.DataFrame,
    corrected: pd.DataFrame,
    movie: str,
    owner: int,
    out: str,
    fps: float = 8.0,
) -> None:
    allp = np.vstack(
        [
            corrected[["accepted_x", "accepted_y"]].to_numpy(),
            corrected[["corrected_x", "corrected_y"]].to_numpy(),
        ]
    )
    x0, y0 = max(0, int(allp[:, 0].min() - 60)), max(0, int(allp[:, 1].min() - 60))
    x1, y1 = int(allp[:, 0].max() + 60), int(allp[:, 1].max() + 60)
    if (x1 - x0) % 2:
        x1 += 1
    if (y1 - y0) % 2:
        y1 += 1
    cap = cv2.VideoCapture(movie)
    tmp = "/tmp/corrected_tmp.mp4"
    vw = cv2.VideoWriter(
        tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (x1 - x0, y1 - y0)
    )
    for _, row in corrected.iterrows():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.source_frame))
        ok, fr = cap.read()
        if not ok:
            continue
        crop = fr[y0:y1, x0:x1].copy()
        H, W = y1 - y0, x1 - x0
        if crop.shape[0] != H or crop.shape[1] != W:
            crop = cv2.copyMakeBorder(
                crop, 0, H - crop.shape[0], 0, W - crop.shape[1],
                cv2.BORDER_REPLICATE,
            )
        ax, ay = int(row.accepted_x - x0), int(row.accepted_y - y0)
        gx, gy = int(row.corrected_x - x0), int(row.corrected_y - y0)
        cv2.circle(crop, (ax, ay), 5, (255, 255, 255), -1)
        cv2.circle(crop, (ax, ay), 4, (0, 0, 255), -1)
        cv2.circle(crop, (gx, gy), 5, (255, 255, 255), -1)
        cv2.circle(crop, (gx, gy), 4, (0, 255, 0), -1)
        label = "P%d s%d" % (owner, int(row.sample_index))
        cv2.rectangle(crop, (4, 4), (300, 26), (0, 0, 0), -1)
        cv2.putText(crop, label, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if row.tag:
            cv2.rectangle(crop, (4, 30), (300, 52), (0, 0, 0), -1)
            col = (0, 255, 0) if row.tag != "HOLD" else (0, 0, 255)
            cv2.putText(crop, row.tag, (9, 47), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 1, cv2.LINE_AA)
        vw.write(crop)
    vw.release()
    subprocess.run(
        ["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out],
        check=True,
    )
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--centerlines", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--movie", required=True)
    ap.add_argument("--owner", type=int, required=True)
    ap.add_argument("--s0", type=int, default=0)
    ap.add_argument("--s1", type=int, default=10**9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="")
    ap.add_argument("--fps", type=float, default=8.0)
    a = ap.parse_args()

    mall = pd.read_csv(a.measurements)
    cl = pd.read_csv(a.centerlines)
    rep = pd.read_csv(a.replay)
    use = mall[
        (mall.pollen_id == a.owner)
        & (mall.accepted == 1)
        & (mall.sample_index >= a.s0)
        & (mall.sample_index <= a.s1)
    ]
    cap = cv2.VideoCapture(a.movie)
    gray_by_src: dict[int, np.ndarray] = {}
    for src in sorted(use.source_frame.unique()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(src))
        ok, fr = cap.read()
        if ok:
            gray_by_src[int(src)] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    cap.release()
    corrected = build_corrected(mall, cl, rep, gray_by_src, a.owner, a.s0, a.s1)
    if a.csv:
        corrected.to_csv(a.csv, index=False)
    print(corrected[corrected.tag != ""].to_string(index=False))
    render_corrected(mall, corrected, a.movie, a.owner, a.out, a.fps)


if __name__ == "__main__":
    main()
