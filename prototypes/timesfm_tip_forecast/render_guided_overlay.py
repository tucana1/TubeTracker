"""Render guided-replay overlay: accepted tip (red) vs rollout (cyan) + veto flags."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--centerlines", default="")
    ap.add_argument("--replay", required=True)
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--guided", default="",
                    help="H152: guided tip series CSV (sample_index,guided_x,"
                    "guided_y,held) from guided_tip_filter — draws the green"
                    " shipped-logic tip; HOLD banner on freezes, RECOVER tag"
                    " where the veto fired but the filter passed a return.")
    ap.add_argument("--movie", required=True)
    ap.add_argument("--owner", type=int, required=True)
    ap.add_argument("--s0", type=int, default=0)
    ap.add_argument("--s1", type=int, default=10**9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--anchor-xy", default="",
                    help="H175: fixed crop center 'x,y' in source px (e.g. the "
                    "grain center) — keeps the pollen centered instead of the "
                    "joint trajectory bbox. Requires --half-size.")
    ap.add_argument("--half-size", type=float, default=0.0,
                    help="H175: crop half-width in source px around --anchor-xy.")
    ap.add_argument("--auto-anchor", action="store_true",
                    help="H178: detect the owner grain on the first frame with "
                    "the FRST grain detector and anchor there (nearest "
                    "detection to the trajectory start). Overrides "
                    "--anchor-xy; still needs --half-size.")
    ap.add_argument("--corrected", default="",
                    help="H190: corrected-tip CSV (sample_index,corrected_x,"
                    "corrected_y,moved_px) from run_corrected_series — draws "
                    "the green corrected tip with a FIX banner where it moved.")
    a = ap.parse_args()

    mall = pd.read_csv(a.measurements)
    m = mall.query("pollen_id == @a.owner and accepted == 1").sort_values("sample_index")
    rep = pd.read_csv(a.replay).query("sample_index >= @a.s0 and sample_index <= @a.s1")
    roll = pd.read_csv(a.rollout).query(
        "sample_index >= @a.s0 and sample_index <= @a.s1"
    )
    rmap = {int(r.sample_index): r for _, r in rep.iterrows()}
    omap = {int(r.sample_index): r for _, r in roll.iterrows()}
    gmap = {}
    if a.guided:
        gg = pd.read_csv(a.guided)
        gg = gg.query("sample_index >= @a.s0 and sample_index <= @a.s1")
        gmap = {int(r.sample_index): r for _, r in gg.iterrows()}
    cmap = {}
    if a.corrected:
        cc = pd.read_csv(a.corrected)
        cc = cc.query("sample_index >= @a.s0 and sample_index <= @a.s1")
        cmap = {int(r.sample_index): r for _, r in cc.iterrows()}
    use = m[(m.sample_index >= a.s0) & (m.sample_index <= a.s1)].reset_index(drop=True)
    if len(use) == 0:
        raise SystemExit("no samples in range")

    pts = np.column_stack([use.tip_x_px.to_numpy(), use.tip_y_px.to_numpy()])
    extra = []
    for _, r in roll.iterrows():
        extra.append([r.rollout_x, r.rollout_y])
    if a.centerlines:
        cl = pd.read_csv(a.centerlines)
        cl = cl.query("pollen_id == @a.owner and sample_index >= @a.s0"
                      " and sample_index <= @a.s1").sort_values(
                          ["sample_index", "point_index"])
        roots = cl.groupby("sample_index").first().reset_index()
        for _, r in roots.iterrows():
            extra.append([r.source_x_px, r.source_y_px])
    for _, r in gmap.items():
        extra.append([r.guided_x, r.guided_y])
    for _, r in cmap.items():
        extra.append([r.corrected_x, r.corrected_y])
    extra = np.array(extra) if extra else np.zeros((0, 2))
    allp = np.vstack([pts, extra]) if len(extra) else pts
    if a.auto_anchor:
        from grain_detect import detect_grains as _detect_grains

        # Median-of-3 over the first samples: single-frame FRST jitter
        # (a few px) otherwise offsets the whole fixed crop (H197).
        cap0 = cv2.VideoCapture(a.movie)
        found = []
        for src in use.source_frame.iloc[:3].tolist():
            cap0.set(cv2.CAP_PROP_POS_FRAMES, int(src))
            ok0, fr0 = cap0.read()
            if not ok0:
                continue
            dets = _detect_grains(
                cv2.cvtColor(fr0, cv2.COLOR_BGR2GRAY), threshold_frac=0.20
            )
            if dets:
                seed = allp[0]
                found.append(min(
                    dets,
                    key=lambda d: (d[0] - seed[0]) ** 2 + (d[1] - seed[1]) ** 2,
                ))
        cap0.release()
        if not found:
            raise SystemExit("auto-anchor: no grains detected")
        bx = float(np.median([d[0] for d in found]))
        by = float(np.median([d[1] for d in found]))
        bs = max(d[2] for d in found)
        print(f"auto-anchor: grain at ({bx:.1f},{by:.1f}) score {bs:.2f} "
              f"from {len(found)} frames", flush=True)
        a.anchor_xy, a.half_size = f"{bx:.1f},{by:.1f}", a.half_size or 127.0
    if a.anchor_xy and a.half_size > 0:
        cx, cy = (float(v) for v in a.anchor_xy.split(","))
        h = float(a.half_size)
        x0, y0, x1, y1 = int(cx - h), int(cy - h), int(cx + h), int(cy + h)
    else:
        x0 = max(0, int(allp[:, 0].min() - 60))
        y0 = max(0, int(allp[:, 1].min() - 60))
        x1 = int(allp[:, 0].max() + 60)
        y1 = int(allp[:, 1].max() + 60)
    if (x1 - x0) % 2:  # libx264 requires even dims
        x1 += 1
    if (y1 - y0) % 2:
        y1 += 1

    cap = cv2.VideoCapture(a.movie)
    tmp = "/tmp/guided_tmp.mp4"
    vw = cv2.VideoWriter(
        tmp, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (x1 - x0, y1 - y0)
    )
    for _, row in use.iterrows():
        s = int(row.sample_index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.source_frame))
        ok, fr = cap.read()
        if not ok:
            continue
        crop = fr[y0:y1, x0:x1].copy()
        H, W = y1 - y0, x1 - x0
        if crop.shape[0] != H or crop.shape[1] != W:  # frame-edge clamp
            crop = cv2.copyMakeBorder(
                crop, 0, H - crop.shape[0], 0, W - crop.shape[1],
                cv2.BORDER_REPLICATE)
        ax, ay = int(row.tip_x_px - x0), int(row.tip_y_px - y0)
        if s in omap:
            cx, cy = int(omap[s].rollout_x - x0), int(omap[s].rollout_y - y0)
            cv2.circle(crop, (cx, cy), 6, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.line(crop, (ax, ay), (cx, cy), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(crop, (ax, ay), 5, (255, 255, 255), -1)
        cv2.circle(crop, (ax, ay), 4, (0, 0, 255), -1)
        v = rmap.get(s)
        label = f"P{a.owner} s{s}"
        banner = None
        if v is not None:
            label += f" r={v.resid:.0f}px g={v.length_gain:.0f}"
            if v.verdict == "fault-suspect":
                banner = (f"VETO {v.verdict}", (0, 0, 255))
            elif v.verdict == "burst":
                banner = (f"BURST +{v.length_gain:.0f}px", (0, 200, 255))
        gr = gmap.get(s)  # H152 shipped-logic overlay: green guided tip.
        if gr is not None:
            gx, gy = int(gr.guided_x - x0), int(gr.guided_y - y0)
            cv2.circle(crop, (gx, gy), 5, (255, 255, 255), -1)
            cv2.circle(crop, (gx, gy), 4, (0, 255, 0), -1)
            if bool(gr.held):
                banner = ("HOLD last-guided", (0, 0, 255))
            elif v is not None and v.verdict == "fault-suspect":
                banner = ("VETO pass: RECOVER", (0, 255, 0))
        cr = cmap.get(s)  # H190 corrected tip: green ring + FIX banner.
        if cr is not None and float(cr.moved_px) > 0:
            qx, qy = int(cr.corrected_x - x0), int(cr.corrected_y - y0)
            cv2.circle(crop, (qx, qy), 7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(crop, (ax, ay), (qx, qy), (0, 255, 0), 1, cv2.LINE_AA)
            banner = (f"FIX +{float(cr.moved_px):.0f}px", (0, 255, 0))
        cv2.rectangle(crop, (4, 4), (300, 26), (0, 0, 0), -1)
        cv2.putText(crop, label, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if banner is not None:
            cv2.rectangle(crop, (4, 30), (300, 52), (0, 0, 0), -1)
            cv2.putText(crop, banner[0], (9, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        banner[1], 1, cv2.LINE_AA)
        vw.write(crop)
    vw.release()
    import subprocess

    subprocess.run(
        ["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.out],
        check=True,
    )
    print("wrote", a.out)


if __name__ == "__main__":
    main()
