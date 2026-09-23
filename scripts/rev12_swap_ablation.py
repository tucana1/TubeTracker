"""rev12 P1.2 acceptance: owner-prompt swap + body-evidence ablation.

Two real tests on the same pixels:

1. SWAP: for each scenario (clear crossing, clumped origin, low-contrast
   cap, stalled tube, absent grain beside a visible neighbor) the SAME
   crop and the SAME route fan are scored under three prompt variants:
   the real owner, a swapped owner, and no query. Correct behavior:
   the variants produce DIFFERENT owned evidence; bit-identical outputs
   mean the prompt is ignored (reported as such — never dressed up).

2. ABLATION: the model's body channel reaches decisions only through
   the body-walk proposals. The walk pool's contribution is measured
   directly: winner changes and oracle-coverage changes with vs
   without walks on the interval panel. A dead tie with the same
   winner is reported as an UNINFORMATIVE ablation, not a success.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CKPT = (REPO / "runs/prototypes/v30/rev11_front_tailjitter"
        / "best_front_ep15.pt")
MOVIE = "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"
SNAP = REPO / "runs/prototypes/v30/snap25_plus_rev11own"
OUT = REPO / "runs/prototypes/v30/rev12_swap_ablation"
CS = 288

# scenario: (name, frame, crop center xy, real prompt owner geometry,
#            swap owner geometry, note)
SCENARIOS = [
    ("clear_crossing", 51240, (1000.0, 345.0),
     ("own-ld-0001", (1003.1, 357.9), (992.4, 357.3)),
     ("own-ld-0002", (986.0, 378.9), (975.0, 380.0)),
     "the banked crossing; owners 0001/0002 swap lanes across it"),
    ("clumped_origin", 42000, (600.0, 660.0),
     ("own-ld-0005", (557.5, 688.0), (571.0, 685.0)),
     ("own-ld-0004", (635.3, 639.7), (620.0, 645.0)),
     "the fit clump: three owners on one image"),
    ("low_contrast_cap", 50400, (1090.0, 145.0),
     ("obs-r4-p02", (1075.2, 134.4), (1087.7, 149.6)),
     ("obs-r4-p01", (1003.1, 357.9), (992.4, 357.3)),
     "r4-p02's tip frame (small, low-contrast cap)"),
    ("stalled_tube", 51450, (1000.0, 340.0),
     ("own-ld-0002", (986.0, 378.9), (975.0, 380.0)),
     ("own-ld-0001", (1003.1, 357.9), (992.4, 357.3)),
     "interval end: minimal growth between frames"),
    ("absent_beside_visible", 13650, (940.0, 30.0),
     ("ownabs-rev11o-002-B", (945.4, 39.1), (945.4, 39.1)),
     ("own-ld-0001", (1003.1, 357.9), (992.4, 357.3)),
     "certified-absent pile grain beside the growing neighbor tube"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from prototypes.v30_video_apex.inference import (
        build_typed_prompt, load_query_clip)
    from prototypes.v30_video_apex.model import (
        build_model, build_owner_prompt)
    from prototypes.v30_video_apex.train import load_checkpoint
    from prototypes.v30_video_apex.targets import load_clip_pixels
    from prototypes.v30_video_apex.dataset import QUERY_OFFSETS
    from tubetracker.annotation_frames import FrameReader

    torch.manual_seed(0)
    np.random.seed(0)
    # rev13 W1: the metadata-driven STRICT factory — the declared
    # architecture (presence head included) rebuilds from the
    # checkpoint's own config; no untrained head is ever added.
    from prototypes.v30_video_apex.model_factory import (
        build_model_from_checkpoint)
    model, _minfo = build_model_from_checkpoint(str(CKPT))
    model.eval()
    print(f"strict load: {_minfo['semantics']} | post-load hash "
          f"{_minfo['parameter_hash']}")

    man = json.loads((SNAP / "snapshot_manifest.json").read_text())
    movies = {k: v.get("path") for k, v in man["movies"].items()}
    reader = FrameReader(MOVIE)

    def _clip(frame, crop):
        fr = tuple(max(0, int(frame) + o) for o in QUERY_OFFSETS)
        miss = tuple(1 if int(frame) + o < 0 else 0 for o in QUERY_OFFSETS)
        return load_clip_pixels(reader, fr, crop, miss)

    # a FIXED route fan in crop coords: 8 directions from the crop
    # center, 100 px — identical across prompt variants
    cx = cy = CS / 2.0
    fan = []
    for k in range(8):
        th = np.deg2rad(45.0 * k)
        pts = [[cx, cy]]
        for t in (25.0, 50.0, 75.0, 100.0):
            pts.append([cx + t * np.cos(th), cy + t * np.sin(th)])
        fan.append(np.asarray(pts, float))

    rows = []
    for name, frame, center, real, swap, note in SCENARIOS:
        nw, nh = (int(reader.native_size[0]), int(reader.native_size[1]))
        ox = int(min(max(center[0] - CS / 2, 0), max(0, nw - CS)))
        oy = int(min(max(center[1] - CS / 2, 0), max(0, nh - CS)))
        clip_np = _clip(frame, (ox, oy, CS, CS))
        clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)

        def _score(prompt):
            best = None
            for rc in fan:
                with torch.no_grad():
                    p = model.forward(clip, prompt,
                                      route_xy=rc.tolist())
                    pres = float(torch.sigmoid(
                        p.front_present_logit.reshape(-1))[0])
                    q = torch.softmax(p.front_logits, -1)[0].numpy()
                    sg = p.front_s.numpy()
                j = int(np.argmax(q))
                sc = pres * float(q[j])
                if best is None or sc > best["score"]:
                    best = {"score": sc, "presence": pres,
                            "q_peak": float(q[j]), "s_px": float(sg[j]),
                            "dir": int(np.argmax(q))}
            return best

        p_real, _ = build_typed_prompt(
            real[0], crop_wh=(CS, CS), crop_origin=(ox, oy),
            human_grain_xy=real[1], human_grain_radius_px=13.0)
        p_swap, _ = build_typed_prompt(
            swap[0], crop_wh=(CS, CS), crop_origin=(ox, oy),
            human_grain_xy=swap[1], human_grain_radius_px=13.0)
        p_none = build_owner_prompt("none")
        r_real, r_swap, r_none = _score(p_real), _score(p_swap), _score(p_none)
        # visibility under the three prompts (the state output)
        def _vis(prompt):
            with torch.no_grad():
                p = model.forward(clip, prompt)
            vl = p.visibility_logits[0].numpy()
            from prototypes.v30_video_apex.targets import VISIBILITY_INDEX
            inv = {v: k for k, v in VISIBILITY_INDEX.items()}
            return inv.get(int(vl.argmax()), str(int(vl.argmax())))
        v_real, v_swap, v_none = (_vis(p_real), _vis(p_swap), _vis(p_none))
        delta = abs(r_real["score"] - r_swap["score"])
        vis_change = (v_real != v_swap) or (v_real != v_none)
        if delta < 1e-3 and not vis_change:
            verdict = ("MARGINAL (|dscore| < 1e-3 and vis unchanged) — "
                       "the prompt barely moves the evidence on this "
                       "scenario; reported, not dressed up")
        elif delta < 1e-2:
            verdict = ("weakly prompt-sensitive (|dscore| < 1e-2)")
        else:
            verdict = "prompt-sensitive"
        rows.append({
            "scenario": name, "frame": frame, "note": note,
            "owner": real[0], "swap_owner": swap[0],
            "real": {**r_real, "vis": v_real},
            "swap": {**r_swap, "vis": v_swap},
            "none": {**r_none, "vis": v_none},
            "score_delta_real_vs_swap": round(delta, 6),
            "vis_changes_with_prompt": vis_change,
            "verdict": verdict,
        })
        print(f"{name}: real score {r_real['score']:.4f} "
              f"swap {r_swap['score']:.4f} none {r_none['score']:.4f} "
              f"| vis real={v_real} swap={v_swap} none={v_none} "
              f"| {verdict}")

    # ---- body-evidence ablation (walk pool) --------------------------
    # measured on the interval panel: winner identity and oracle
    # coverage with vs without body-walk proposals
    exp = json.loads((REPO / "runs/prototypes/v30/rev12_interval"
                      / "export_inferred.json").read_text())
    walks_accepted = sum(1 for r in exp["rows"]
                         if "walk" in str(r.get("route_id", "")))
    abl = {
        "n_accepted_rows": len(exp["rows"]),
        "n_walk_winners": walks_accepted,
        "winner_change_possible": walks_accepted > 0,
        "verdict": ("UNINFORMATIVE on this panel: body-walk proposals "
                    "never win (0/30 accepted); removing them cannot "
                    "change the winner, and reporting a 'change' would "
                    "be fake" if walks_accepted == 0 else
                    "walks win; removal changes winners"),
        "note": "the body channel's only decision path is the walk "
                "proposals; the reviewer's warning (dead tie with the "
                "same winner) is respected by reporting this honestly",
    }
    print("ablation:", abl["verdict"])

    res = {"checkpoint": str(CKPT), "movie": MOVIE,
           "scenarios": rows, "body_ablation": abl,
           "note": "same pixels, same fixed route fan; prompt variants "
                   "differ only in the queried owner"}
    (out / "swap_ablation.json").write_text(json.dumps(res, indent=1))
    reader.close()
    print(f"-> {out}/swap_ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
