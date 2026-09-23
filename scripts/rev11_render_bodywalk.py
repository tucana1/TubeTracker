"""rev11: native overlay of the body-walk demo (p00).

Draws, on the native frame, the winning body-walk route (bw2), its
rival walks (bw0/bw1), the best historic-family route, the detected
grain center and the walk's measured tip — the visual evidence the
review asks for ('native overlays and failed cases'). Reads the saved
demo artifacts; writes overlay.png beside them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="rev11_bodywalk_* output directory")
    ap.add_argument("--movie", default="/Users/joshjiang/Downloads/"
                    "test1lowdensjoshua-28c-hz.mp4 .mp4")
    ap.add_argument("--out", default="overlay.png")
    a = ap.parse_args()
    d = Path(a.dir)
    cands = json.loads((d / "candidates.json").read_text())
    if not cands:
        print("no candidates")
        return 1
    frame = int(cands[0]["source_frame"])
    from tubetracker.annotation_frames import FrameReader
    reader = FrameReader(a.movie)
    try:
        gray = reader.read(frame).frame
    finally:
        reader.close()
    gray = gray[:, :, 0] if gray.ndim == 3 else gray

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    ax.imshow(gray, cmap="gray")
    def _load(v):
        return json.loads(v) if isinstance(v, str) else v

    walks = [c for c in cands if str(c.get("route_id", "")).startswith("bw")]
    winner = next((c for c in cands if c.get("selected")), None)
    hist = [c for c in cands
            if not str(c.get("route_id", "")).startswith("bw")]
    hist.sort(key=lambda c: -(c.get("score_body") or 0))
    for c in hist[:3]:
        p = np.asarray(_load(c["polyline_native"]), float)
        ax.plot(p[:, 0], p[:, 1], "-", color="0.55", lw=1.0, alpha=0.8,
                label=f"{c['route_id']} (historic, sb="
                      f"{(c.get('score_body') or 0):.4f})")
    for c in walks:
        p = np.asarray(_load(c["polyline_native"]), float)
        win = bool(c.get("selected"))
        ax.plot(p[:, 0], p[:, 1], "-",
                color=("crimson" if win else "tab:orange"),
                lw=(2.6 if win else 1.6),
                label=f"{c['route_id']} (body-walk, sb="
                      f"{(c.get('score_body') or 0):.4f}"
                      f"{', WINNER' if win else ''})")
        ax.plot(p[-1, 0], p[-1, 1], "o", color=("crimson" if win else
                                                 "tab:orange"), ms=4)
    if winner is not None:
        ct = _load(winner["current_tip"])
        ax.plot(ct[0], ct[1], "*", color="lime", ms=14,
                label="winner measured tip")
        po = winner.get("physical_origin") or {}
        po = _load(po)
        gc = po.get("grain_center_native")
        if gc:
            ax.plot(gc[0], gc[1], "s", color="cyan", ms=7,
                    label=f"detected grain center ({po.get('grain_center_source')})")
    st = cands[0].get("root_source")
    ax.set_title(f"{d.name}: frame {frame} | root {st} | "
                 f"rank=score_body")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.set_xlim(max(0, gray.shape[1] * 0.0), gray.shape[1])
    out = d / a.out
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
