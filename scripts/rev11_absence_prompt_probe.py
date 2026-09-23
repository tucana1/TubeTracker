"""rev12 P0.2: the absence/present prompt probe, parameterized.

Exercises ALL nine explicit certified absence regions plus nearby
PRESENT controls (known neighbouring tubes in the same frames) under
four prompt constructions, on a chosen checkpoint:

- auto    : the detector's own grain at the sample anchor;
- human   : the human-identified grain disc (the certified record);
- none    : the resolved no-query prompt (owner-specific loss zero);
- swap    : ANOTHER case's human grain (the physical-owner swap).

A known present neighbouring grain must not be relabelled absent to
satisfy an absence gate; the probe shows what each construction
actually produces. Legacy observation records are NOT exercised here —
the probe's cases are the certified ones (plus present controls).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_CKPT = (REPO / "runs/prototypes/v30/rev11_front_absdiet_ms"
                / "best_front_ep15.pt")
DEFAULT_SNAP = REPO / "runs/prototypes/v30/snap25_plus_rev11own"
OUT = REPO / "runs/prototypes/v30/rev12_absence_prompt_probe.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--snapshot", default=str(DEFAULT_SNAP))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    import torch
    from prototypes.v30_video_apex.inference import (
        build_typed_prompt, load_query_clip)
    from prototypes.v30_video_apex.model_factory import (
        build_model_from_checkpoint)
    from prototypes.v30_video_apex.model import build_owner_prompt
    from prototypes.v30_video_apex.targets import (
        VISIBILITY_INDEX, samples_from_snapshot)
    from scripts.train_v30_front import sample_prompt
    from tubetracker.annotation_frames import FrameReader

    inv = {v: k for k, v in VISIBILITY_INDEX.items()}
    snap = Path(a.snapshot)
    # rev13 W1: strict metadata load (no legacy semantics under an
    # evaluation path); the operative state is recorded in the output.
    model, info = build_model_from_checkpoint(a.checkpoint)
    print(f"checkpoint {a.checkpoint}: {info.get('model_kwargs')}")
    model.eval()
    samples = samples_from_snapshot(str(snap))
    abs_cases = [s for s in samples if s.kind == "no_tube"
                 and getattr(s, "certified", False)]
    # nearby PRESENT controls: direct-visible samples in the same
    # (movie, frame) within 150 px of an absence grain; when the banked
    # data has none at those frames, fall back to the REGISTRY's
    # present grains (human-identified owners with paths) — known
    # present neighbors that must NOT be relabelled absent.
    present: list = []
    for ab in abs_cases:
        ax, ay = float(ab.focus_xy[0]), float(ab.focus_xy[1])
        for s in samples:
            if s is ab or s.movie != ab.movie \
                    or s.source_frame != ab.source_frame:
                continue
            if str(getattr(s, "direct_state", "")) not in (
                    "direct_visible", "visible_imprecise"):
                continue
            if not s.focus_xy:
                continue
            d = float(np.hypot(s.focus_xy[0] - ax, s.focus_xy[1] - ay))
            if d <= 150.0:
                present.append((ab, s, d))
    if not present:
        # registry fallback: present grains (with human paths) built as
        # REAL SampleDefs (same fields the trainer prompt reads)
        from prototypes.v30_video_apex.targets import SampleDef
        man = json.loads((REPO / "runs/prototypes/v30/"
                          "snap25_plus_rev11own/snapshot_manifest.json")
                         .read_text())
        movies = {k: v.get("path") for k, v in man["movies"].items()}
        reg = json.loads((REPO / "runs/prototypes/v30/grain_ids.json")
                         .read_text())
        owners = json.loads((REPO / "runs/prototypes/v30/"
                             "rev11own_verify/owners.json")
                            .read_text())["owners"]
        for o in owners:
            if o.get("no_tube"):
                continue
            gid = reg["aliases"].get(
                f"task|rev11own|{o.get('source_task')}|"
                f"{o.get('source_label')}", "")
            gx, gy = (float(o["grain_native"][0]),
                      float(o["grain_native"][1]))
            ax, ay = (o.get("attachment_native") or o["grain_native"])
            s = SampleDef(
                entry_id=f"present|{o['id']}",
                movie=o["movie"], movie_path=movies.get(o["movie"], ""),
                source_frame=int(o.get("identified_at_frame") or 0),
                tube_ref=f"present|{o['id']}",
                kind="path_tip",
                tip_xy=(gx, gy), path_xy=[[ax, ay], [gx, gy]],
                attachment_xy=(float(ax), float(ay)),
                focus_xy=(gx, gy), direct_state="direct_visible",
                owner_uuid=o["id"],
                owner_key=f"{o['movie']}|{o['id']}",
                grain_id=gid, query_kind="tube", certified=True,
                sample_key=f"present|{o['id']}")
            s.validate()
            present.append((None, s, float("nan")))
    print(f"absence cases: {len(abs_cases)} | present controls: "
          f"{len(present)}")

    man = json.loads((snap / "snapshot_manifest.json").read_text())
    movies = {k: v.get("path") for k, v in man["movies"].items()}
    sizes: dict[str, tuple[int, int]] = {}
    CS = 288

    def _clip_for(s):
        if s.movie not in sizes:
            r = FrameReader(movies[s.movie])
            sizes[s.movie] = (int(r.native_size[0]),
                              int(r.native_size[1]))
            r.close()
        nw, nh = sizes[s.movie]
        cx, cy = float(s.focus_xy[0]), float(s.focus_xy[1])
        crop = (int(min(max(cx - CS / 2, 0), max(0, nw - CS))),
                int(min(max(cy - CS / 2, 0), max(0, nh - CS))), CS, CS)
        clip_np = load_query_clip(snap, s.movie, int(s.source_frame),
                                  crop)
        return clip_np, crop

    def _full(clip, prompt):
        """rev13 W3.5: FULL distributions + emissions, not just argmax.
        Visibility class probabilities (all classes), the owned-cap
        presence probability, the route probability, the heat maximum
        and any cap emissions the checkpoint's declared heads produce.
        """
        with torch.no_grad():
            pr = model.forward(clip, prompt)
        vl = pr.visibility_logits[0].numpy()
        ex = np.exp(vl - vl.max())
        probs = (ex / ex.sum()).tolist()
        arg = int(vl.argmax())
        out = {
            "visibility_probs": {inv.get(i, str(i)): round(float(p), 5)
                                 for i, p in enumerate(probs)},
            "visibility_argmax": inv.get(arg, str(arg)),
            "visibility_conf": round(float(probs[arg]), 5),
            "heat_max": round(float(pr.heat[0, 0].max()), 4),
        }
        fpl = getattr(pr, "front_present_logit", None)
        out["front_present_prob"] = (
            round(float(torch.sigmoid(fpl.reshape(-1))[0]), 5)
            if fpl is not None else None)
        rl = getattr(pr, "route_logit", None)
        out["route_prob"] = (round(float(torch.sigmoid(
            rl.reshape(-1))[0]), 5) if rl is not None else None)
        cl = getattr(pr, "cap_logits", None)
        out["cap_logits"] = ([round(float(v), 4) for v in
                              cl.reshape(-1).numpy()]
                             if cl is not None else None)
        # declared decision rule: owned presence if the visible-class
        # mass is at least half OR the presence head says so when the
        # checkpoint declares one.
        vis_mass = float(sum(p for i, p in enumerate(probs)
                             if inv.get(i) in ("direct_visible",
                                               "visible_imprecise")))
        out["visible_mass"] = round(vis_mass, 5)
        out["decision"] = (
            "present" if (out["front_present_prob"] is not None
                          and out["front_present_prob"] >= 0.5)
            or vis_mass >= 0.5 else "absent")
        return out

    swap_src = abs_cases[1] if len(abs_cases) > 1 else abs_cases[0]
    rows = []
    for s in abs_cases + [p[1] for p in present]:
        clip_np, crop = _clip_for(s)
        ox, oy = crop[0], crop[1]
        cx, cy = float(s.focus_xy[0]), float(s.focus_xy[1])
        clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)
        p_auto, prov_auto, pk_auto = sample_prompt(
            s, clip_np, ox, oy, CS, CS, "auto-grain", False)
        p_human, prov_h = build_typed_prompt(
            str(s.owner_uuid or s.entry_id), crop_wh=(CS, CS),
            crop_origin=(ox, oy), human_grain_xy=(cx, cy),
            human_grain_radius_px=13.0)
        p_none = build_owner_prompt("none")
        p_swap, _ = build_typed_prompt(
            str(swap_src.owner_uuid or swap_src.entry_id),
            crop_wh=(CS, CS), crop_origin=(ox, oy),
            human_grain_xy=(float(swap_src.focus_xy[0]),
                            float(swap_src.focus_xy[1])),
            human_grain_radius_px=13.0)
        row = {
            "case": str(s.entry_id),
            "kind": "absent" if s.kind == "no_tube" else "present",
            "truth": str(getattr(s, "direct_state", "")),
            "grain_id": str(getattr(s, "grain_id", "") or ""),
            "query_kind": str(getattr(s, "query_kind", "") or ""),
            "auto": _full(clip, p_auto),
            "human": _full(clip, p_human),
            "none": _full(clip, p_none),
            "swap": _full(clip, p_swap),
            "auto_center": prov_auto.get("center_native"),
            "human_center": prov_h.get("center_native"),
        }
        rows.append(row)
        print(f"{row['case']} [{row['kind']}/{row['truth']}]: "
              f"auto={row['auto']['visibility_argmax']}"
              f"({row['auto']['visibility_conf']:.2f},"
              f"{row['auto']['decision']}) "
              f"human={row['human']['visibility_argmax']}"
              f"({row['human']['visibility_conf']:.2f},"
              f"{row['human']['decision']}) "
              f"none={row['none']['visibility_argmax']}"
              f"({row['none']['visibility_conf']:.2f}) "
              f"swap={row['swap']['visibility_argmax']}"
              f"({row['swap']['visibility_conf']:.2f})")
    # rev13 W3.5: paired present/absent confusion under the DECLARED
    # decision rule, with the trivial baselines reported side by side —
    # a uniform (always-present / always-absent) classifier must not
    # pass, and the confusion is over the paired cases only.
    conf = {}
    for construction in ("auto", "human", "none", "swap"):
        tp = fp = tn = fn = 0
        for r in rows:
            dec = r[construction]["decision"]
            truth = r["kind"]
            if truth == "present":
                tp += int(dec == "present")
                fn += int(dec == "absent")
            else:
                fp += int(dec == "present")
                tn += int(dec == "absent")
        n = tp + fp + tn + fn
        conf[construction] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": round((tp + tn) / max(1, n), 4),
            "n": n}
    always_present = sum(1 for r in rows if r["kind"] == "present") / max(
        1, len(rows))
    res = {"checkpoint": a.checkpoint,
           "model_kwargs": info.get("model_kwargs"),
           "operative": {"parameter_hash": info.get("parameter_hash"),
                         "semantics": info.get("semantics"),
                         "model_schema": info.get("model_schema"),
                         "activation": info.get("activation"),
                         "preprocessing": info.get("preprocessing")},
           "snapshot": str(snap), "n_absent": len(abs_cases),
           "n_present_controls": len(present),
           "decision_rule": "present iff presence-head prob >= 0.5 (when "
                            "declared) or visible-class mass >= 0.5",
           "confusion": conf,
           "baselines": {
               "always_present_accuracy": round(always_present, 4),
               "always_absent_accuracy": round(1.0 - always_present, 4),
               "note": "a construction at or below the trivial baselines "
                       "has no demonstrated discrimination"},
           "rows": rows}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
