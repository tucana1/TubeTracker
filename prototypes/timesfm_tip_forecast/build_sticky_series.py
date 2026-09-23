"""Build a sticky coherent series from a raw corrected series (H210).

Pipeline: raw corrected series + guided-replay faults
  -> consensus_smooth (veto flicker)
  -> sticky_bias with resets (fault samples + consensus vetoes)
  -> sticky CSV (sample_index, corrected_x/y, moved_px, keep)

Pure post-processing: no images, no model. v29.38 consumes the output.
"""

import argparse

import numpy as np
import pandas as pd

from .sticky_bias import sticky_bias
from .tip_correct import consensus_smooth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True,
                    help="raw corrected series CSV (sample_index, "
                    "accepted_x/y, corrected_x/y, moved_px)")
    ap.add_argument("--replay", default="",
                    help="guided-replay CSV; fault-suspect samples reset")
    ap.add_argument("--keep-col", default="",
                    help="H210: input already carries consensus decisions "
                    "(e.g. eye-validated v2smooth 'keep' column) — use as "
                    "the fix mask instead of re-running consensus (which "
                    "would double-veto era transitions).")
    ap.add_argument("--vetoed-col", default="",
                    help="H210: input already carries veto flags "
                    "(e.g. 'vetoed' column) — these reset the bias.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    c = pd.read_csv(a.series).sort_values("sample_index").reset_index(drop=True)
    acc = c[["accepted_x", "accepted_y"]].to_numpy(float)
    cor = c[["corrected_x", "corrected_y"]].to_numpy(float)
    if "tier" in c.columns:
        tier = c["tier"].to_numpy(int)
    else:
        tier = np.zeros(len(c), dtype=int)  # legacy series: all standard
        tier[(c.moved_px <= 0).to_numpy(bool)] = -1
    # H212: tier-1 far candidates never enter the standard fix mask —
    # they join sticky agreement windows only (era adopts, isolated dies).
    fixed = (c.moved_px > 0).to_numpy(bool) & (tier == 0)
    cand = (tier == 1)

    if a.keep_col and a.keep_col in c.columns:
        keep = c[a.keep_col].to_numpy(bool)
        print("using upstream keep col "
              f"{a.keep_col}: {int(keep.sum())} fixed", flush=True)
    else:
        keep = consensus_smooth(cor, fixed)
    if a.vetoed_col and a.vetoed_col in c.columns:
        c["vetoed"] = c[a.vetoed_col].astype(int)
    else:
        c["vetoed"] = (fixed & ~keep).astype(int)

    reset = (c.vetoed == 1).to_numpy(bool)
    if a.replay:
        rep = pd.read_csv(a.replay)
        faults = set(rep[rep.verdict == "fault-suspect"].sample_index.astype(int))
        reset = reset | c.sample_index.isin(faults).to_numpy()
        print(f"resets: {int(reset.sum())} (incl. {len(faults)} faults)",
              flush=True)

    out = sticky_bias(acc, cor, keep, reset_mask=reset,
                      cand_xy=cor, cand_mask=cand)
    c["corrected_x"], c["corrected_y"] = out[:, 0], out[:, 1]
    c["moved_px"] = np.hypot(out[:, 0] - c.accepted_x,
                             out[:, 1] - c.accepted_y)
    c["keep"] = (c.moved_px > 0.5).astype(int)
    c.to_csv(a.out, index=False)
    st = np.hypot(np.diff(out[:, 0]), np.diff(out[:, 1]))
    print(f"wrote {a.out}: fixed={int(c.keep.sum())} "
          f"step med={np.median(st):.1f} max={st.max():.1f} "
          f"n>20px={int((st > 20).sum())}", flush=True)


if __name__ == "__main__":
    main()
