"""rev11 oracle-support PERTURBATION gate (review section 8B, pre-experiment).

The rev11 review: the one-pixel front result was conditional on a support
constructed as human-route + a FIXED 24-px extension — `support_length-24`
(an image-free shortcut) scores 0.065 px on p05, and changing only the
post-tip extension to 12/48 px raised the model's error to 11.06/6.92 px.
Before the proposal experiment, the front head must be tested where the
IMAGE, not the support construction, determines the cap.

This harness rebuilds the trainer's EXACT saved inputs (clip hashes and
route hashes verified against runs/prototypes/v30/rev10_front_snap25/
samples.json), then measures the held-out front error under:

  * post-tip extension variants 12 / 24 / 48 px   (cap fixed in the image)
  * a 12-px proximal trim                          (start boundary moved)
  * mean-image ablation at 24 px                   (appearance dependence)

and reports the image-free baseline error per configuration (returning
`support_length - 24`). Pass criteria, per event: front error <= 5 px for
ALL extension variants; a configuration that only tracks the support
length is reported as such.

Usage:
  .venv/bin/python scripts/rev11_support_perturbation.py \
      --checkpoint runs/prototypes/v30/rev10_front_snap25/best_front_ep19.pt \
      --samples runs/prototypes/v30/rev10_front_snap25/samples.json \
      --snapshot runs/prototypes/v30/snapshots/snap25 \
      --out runs/prototypes/v30/rev11_support_perturbation.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.dataset import QUERY_OFFSETS  # noqa: E402
from prototypes.v30_video_apex.inference import (  # noqa: E402
    build_typed_prompt)
from prototypes.v30_video_apex.model_factory import (  # noqa: E402
    build_model_from_checkpoint)
from prototypes.v30_video_apex.targets import (  # noqa: E402
    blind_oracle_route, load_clip_pixels, resample_polyline,
    samples_from_snapshot)
from prototypes.v30_video_apex import train as _T  # noqa: E402

TOL_PX = 5.0
POSTS = (12.0, 24.0, 48.0)


def _trim_root(path_xy, px: float):
    """Move the proximal boundary `px` along the path (drop the start)."""
    rp, sg = resample_polyline(np.asarray(path_xy, float).tolist())
    if float(sg[-1]) <= px + 4.0:
        return None
    idx = int(np.searchsorted(sg, px))
    out = [list(rp[idx])] + [list(q) for q in np.asarray(path_xy, float)]
    return out


def _score(model, clip, prompt, route_crop):
    clip_t = torch.from_numpy(clip).unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        pred = model.forward(clip_t, prompt, route_xy=route_crop)
        q = torch.softmax(pred.front_logits, -1)[0]
        s_grid = pred.front_s.detach().numpy()
        s_hat = float(s_grid[int(q.argmax())])
    return s_hat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(
        REPO / "runs/prototypes/v30/rev10_front_snap25/best_front_ep19.pt"))
    ap.add_argument("--samples", default=str(
        REPO / "runs/prototypes/v30/rev10_front_snap25/samples.json"))
    ap.add_argument("--snapshot", default=str(
        REPO / "runs/prototypes/v30/snapshots/snap25"))
    ap.add_argument("--out", default=str(
        REPO / "runs/prototypes/v30/rev11_support_perturbation.json"))
    a = ap.parse_args()

    samples = json.loads(Path(a.samples).read_text())
    path_entries = [e for e in samples if e.get("kind") == "path_tip"]
    if not path_entries:
        print("no path_tip entries in the saved samples")
        return 1
    snap = Path(a.snapshot)
    man = json.loads((snap / "snapshot_manifest.json").read_text())
    movies = {k: v.get("path") for k, v in (man.get("movies") or {}).items()}
    obs_all = {str(o.obs_uuid): o for o in
               samples_from_snapshot(str(snap))}

    model, info = build_model_from_checkpoint(a.checkpoint)
    model.eval()
    print(f"model: {info['model_kwargs']} | param sha "
          f"{info['parameter_hash'][:12]}")

    from tubetracker.annotation_frames import FrameReader
    report = {"checkpoint": str(a.checkpoint),
              "parameter_hash": info["parameter_hash"],
              "tol_px": TOL_PX, "events": [], "pass": True}
    for e in path_entries:
        entry_id = str(e.get("entry_id", ""))
        movie_key = str(e.get("movie", ""))
        frame = int(e.get("source_frame", -1))
        crop = tuple(int(v) for v in e["crop_xywh"])
        o = obs_all.get(str(e.get("obs_uuid", "")))
        row: dict = {"entry": entry_id, "crop_xywh": list(crop)}
        if o is None or not o.path_xy or not o.tip_xy:
            row["skipped"] = "no human path/tip"
            report["events"].append(row)
            continue
        reader = FrameReader(movies[movie_key])
        try:
            frames = tuple(max(0, frame + off) for off in QUERY_OFFSETS)
            missing = tuple(1 if frame + off < 0 else 0
                            for off in QUERY_OFFSETS)
            clip_np = load_clip_pixels(reader, frames, crop, missing)
        finally:
            reader.close()
        sha = hashlib.sha256(clip_np.tobytes()).hexdigest()[:16]
        row["clip_sha"] = sha
        row["clip_sha_match"] = (sha == str(e.get("clip_sha", "")))
        ox, oy = crop[0], crop[1]

        # the two prompt arms: legacy focus disc (what the saved run
        # trained with) and the rev11 typed detection prompt
        from prototypes.v30_video_apex.inference import build_typed_prompt
        legacy_prompt, _lp = build_typed_prompt(
            str(e.get("obs_uuid", "")), crop_wh=(crop[2], crop[3]),
            crop_origin=(ox, oy), focus_xy=o.focus_xy)
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "tvf_perturb", REPO / "scripts" / "train_v30_front.py")
        _tvf = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_tvf)  # type: ignore[union-attr]
        typed_prompt, _tprov, _pk = _tvf.sample_prompt(
            o, clip_np, ox, oy, crop[2], crop[3], "auto-grain", False)
        row["typed_prompt_kind"] = _tprov["prompt_kind"]

        def _route(post: float, path=o.path_xy):
            return blind_oracle_route(path, o.tip_xy, post_px=post)

        # verify the frozen route at 24 px reproduces exactly
        r24 = None
        try:
            r24 = _route(24.0)
            r_hash = hashlib.sha256(json.dumps(
                r24["route_xy"].tolist(), sort_keys=True,
                default=str).encode()).hexdigest()[:16]
            row["route_hash_match"] = (r_hash == str(e.get("route_hash", "")))
            row["s_star_frozen_match"] = (
                e.get("s_star") is not None
                and abs(float(e["s_star"]) - float(r24["s_star"])) < 1e-9)
        except ValueError as ex:
            row["route_error"] = str(ex)

        variants: dict = {}
        for post in POSTS:
            try:
                r = _route(post)
            except ValueError as ex:
                variants[f"post_{int(post)}"] = {"skip": str(ex)}
                continue
            route_crop = (r["route_xy"]
                          - np.array([ox, oy])).tolist()
            s_sup = float(r["support"])
            base_err = abs((s_sup - 24.0) - float(r["s_star"]))
            for arm, prompt in (("focus", legacy_prompt),
                                ("detected", typed_prompt)):
                s_hat = _score(model, clip_np, prompt, route_crop)
                variants.setdefault(f"post_{int(post)}", {})[arm] = {
                    "s_hat": s_hat, "s_star": float(r["s_star"]),
                    "err_px": abs(s_hat - float(r["s_star"])),
                    "baseline_err_px": base_err}
        # proximal trim (start boundary moved 12 px, cap unmoved)
        trimmed = _trim_root(o.path_xy, 12.0)
        if trimmed is not None and r24 is not None:
            try:
                rt = _route(24.0, trimmed)
                rc = (rt["route_xy"] - np.array([ox, oy])).tolist()
                r24c = (r24["route_xy"] - np.array([ox, oy])).tolist()
                s_t = _score(model, clip_np, typed_prompt, rc)
                s_0 = _score(model, clip_np, typed_prompt, r24c)
                variants["trim12"] = {
                    "s_hat": s_t, "s_star": float(rt["s_star"]),
                    "err_px": abs(s_t - float(rt["s_star"])),
                    "follows_cap": abs((s_t - s_0) - (
                        float(rt["s_star"]) - float(r24["s_star"])))}
            except ValueError as ex:
                variants["trim12"] = {"skip": str(ex)}
        # mean-image ablation at 24 px
        if r24 is not None and "detected" in variants.get("post_24", {}):
            mi = np.full_like(clip_np, float(clip_np.mean()))
            rc = (r24["route_xy"] - np.array([ox, oy])).tolist()
            s_mi = _score(model, mi, typed_prompt, rc)
            variants["mean_image"] = {
                "s_hat": s_mi, "err_px": abs(s_mi - float(r24["s_star"])),
                "delta_from_real": abs(
                    s_mi - variants["post_24"]["detected"]["s_hat"])}
        row["variants"] = variants
        _det_errs = [v.get("detected", {}).get("err_px")
                     for k, v in variants.items()
                     if k.startswith("post_") and "detected" in v]
        row["detected_errs_px"] = _det_errs
        row["ok"] = bool(_det_errs) and all(
            e2 is not None and e2 <= TOL_PX for e2 in _det_errs)
        report["pass"] = report["pass"] and row["ok"]
        report["events"].append(row)
        print(f"{entry_id:>22} clip_sha_ok={row['clip_sha_match']} "
              f"route_ok={row.get('route_hash_match')} "
              f"detected errs={['%.2f' % e2 if e2 is not None else None for e2 in _det_errs]} ok={row['ok']}")
        for k, v in variants.items():
            if "detected" in v:
                print(f"    {k:>9} focus {v['focus']['err_px']:6.2f} | "
                      f"detected {v['detected']['err_px']:6.2f} | "
                      f"baseline {v['detected']['baseline_err_px']:5.2f}")

    Path(a.out).write_text(json.dumps(report, indent=1, default=str))
    print(f"\nperturbation gate pass: {report['pass']} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
