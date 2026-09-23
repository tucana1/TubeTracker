"""rev11 item 5: the decisive front experiment on RUNTIME proposals.

The review's corrected task: supports must come from the DEPLOYMENT
proposer (independent of the ground-truth cap); the full human path and
precise tip are used ONLY to label and evaluate. Presence (does this
route carry the owner's current cap) is trained explicitly and separated
from conditional location; uncertain geometry stays unknown; top-K
separated hypotheses are evaluated.

Modes:

  build — generate the runtime proposal set per developed event (the
          same family the movie runner uses), label each proposal with
          the human path/tip (present / absent / UNCERTAIN), record
          plausibility metrics, origins and source ids, and save the
          manifest. No model is involved in proposal generation.

  train — short deterministic update blocks over the manifest (the
          runtime arm) or over oracle +24 routes (the baseline arm),
          from a documented init checkpoint, saving weights per block.

  eval  — perturbation (support-boundary) + endpoint coverage + top-K
          separated hypotheses for a trained checkpoint, plus the
          image-free `support_length-24` baseline.

Usage:
  .venv/bin/python scripts/rev11_proposal_fit.py --mode build \
      --out runs/prototypes/v30/rev11_proposals.json
  .venv/bin/python scripts/rev11_proposal_fit.py --mode train \
      --manifest runs/prototypes/v30/rev11_proposals.json \
      --init runs/prototypes/v30/rev10_front_snap25/best_front_ep19.pt \
      --supervision runtime --updates 200 --block 50 \
      --out runs/prototypes/v30/rev11_propfit_runtime.json
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
from prototypes.v30_video_apex.eval_route_evidence import (  # noqa: E402
    path_metrics)
from prototypes.v30_video_apex.inference import (  # noqa: E402
    build_typed_prompt)
from prototypes.v30_video_apex.model import build_model  # noqa: E402
from prototypes.v30_video_apex.train import (  # noqa: E402
    LossWeights, load_checkpoint, masked_multihead_loss, save_checkpoint,
    train_step_front)
from prototypes.v30_video_apex.targets import (  # noqa: E402
    blind_oracle_route, front_interval_mask, load_clip_pixels,
    polyline_point_projection, project_to_arclength, resample_polyline,
    samples_from_snapshot)
from prototypes.v30_video_apex import routes as R  # noqa: E402
from prototypes.v30_video_apex import train as _T  # noqa: E402

CS = 288
PRESENT_TOL_PX = 8.0     # <=: route carries the cap (label 1)
ABSENT_TOL_PX = 24.0     # >: decoy (label 0); between: UNCERTAIN (masked)
ENDPOINT_TOL_PX = 5.0    # the endpoint gate (selected cap must be this
                         # close); a corridor-positive route farther than
                         # this is SUPPORT-ONLY, not a wrong selector
FRONT_HALFWIDTH_PX = 8.0


def _param_hash(model) -> str:
    """Deterministic hash of ALL parameters (rev12 P0.3).

    The ACTUAL starting weights are recorded — not a description of the
    intended initialization. Consumers compare hashes mechanically.
    """
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(
            p.detach().cpu().numpy().astype(np.float32)).tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ build
def within_row_hinge(logits, pos_mask, neg_mask, margin: float = 1.0):
    """rev13 W3.5b: hardest-negative within-row hinge.

    Returns clamp(max(neg_logits) - max(pos_logits) + margin, 0):
    zero once the best positive location beats the best negative
    location of the SAME row by `margin` logits. Masks are numpy
    boolean arrays; the caller guarantees both are non-empty.
    """
    import torch as _t
    _max_pos = logits[_t.from_numpy(pos_mask)].max()
    _max_neg = logits[_t.from_numpy(neg_mask)].max()
    return _t.clamp(_max_neg - _max_pos + margin, min=0.0)


def _proposals_for_event(gray, anchor, v1, dev) -> list[dict]:
    """The deployment family — rev13 W2.6: delegated to the ONE shared
    callable (generate_route_hypotheses), so fitting, evaluation and
    deployment cannot drift apart. v1/dev are passed through and their
    actual use is recorded by the callable."""
    from prototypes.v30_video_apex.proposals import (
        generate_route_hypotheses)
    props, ginfo = generate_route_hypotheses(
        gray, anchor, v1=v1, dev=dev, include_body_walks=False)
    return props, (ginfo["grain_center"], ginfo["grain_radius_px"],
                   ginfo["grain_source"])


def mode_build(a) -> int:
    snap = Path(a.snapshot)
    man = json.loads((snap / "snapshot_manifest.json").read_text())
    movies = {k: v.get("path") for k, v in (man.get("movies") or {}).items()}
    rows = samples_from_snapshot(str(snap))
    events = [s for s in rows if s.kind == "path_tip" and s.path_xy
              and s.tip_xy and s.movie in movies]
    if a.event_ref:
        keep = tuple(x.strip() for x in a.event_ref.split(","))
        events = [s for s in events
                  if any(k in str(s.tube_ref) for k in keep)]
    v1, dev = R.load_v1(a.v1_weights)
    from tubetracker.annotation_frames import FrameReader
    out_rows = []
    for s in events:
        r = FrameReader(movies[s.movie])
        try:
            frame = int(s.source_frame)
            fr = r.read(frame).frame
            gray = np.asarray(fr[:, :, 0] if fr.ndim == 3 else fr)
            H, W = gray.shape
            anchor = (float(s.path_xy[0][0]), float(s.path_xy[0][1]))
            props, (gc, gr, gsrc) = _proposals_for_event(gray, anchor,
                                                         v1, dev)
            for pr in props:
                pts = np.asarray(pr["polyline"], float)
                cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                ox = int(min(max(cx - CS / 2, 0), max(0, W - CS)))
                oy = int(min(max(cy - CS / 2, 0), max(0, H - CS)))
                # ---- rev12 P0.3: continuous geometry, one helper ----
                # BEFORE: nearest-VERTEX distance (historical). AFTER:
                # continuous point-to-SEGMENT projection returning
                # (distance, projected arclength, projected position).
                tip_xy = np.asarray(s.tip_xy, float)
                d_tip, s_proj, proj_xy = polyline_point_projection(
                    pts, tip_xy)
                d_tip_vertex = float(np.hypot(
                    *(tip_xy[None, :] - pts).T).min())
                sg = np.asarray(resample_polyline(pts.tolist())[1], float)
                total = float(sg[-1])
                # labels under BOTH geometries, same thresholds
                def _label(dv, sp):
                    if dv <= PRESENT_TOL_PX and sp is not None \
                            and 2.0 < sp < total - 8.0:
                        return 1
                    if dv > ABSENT_TOL_PX:
                        return 0
                    return None
                _sp_v = None
                if d_tip_vertex <= ABSENT_TOL_PX:
                    _sp_v = project_to_arclength(pts, tip_xy)
                present_v = _label(d_tip_vertex, _sp_v)
                present = _label(d_tip, s_proj)
                # foreign-cap detection: the route's endpoint lands on
                # ANOTHER event's certified tip (separate negative class)
                foreign_cap = ""
                for s2 in events:
                    if s2 is s or s2.movie != s.movie \
                            or int(s2.source_frame) != frame:
                        continue
                    d2 = polyline_point_projection(pts, np.asarray(
                        s2.tip_xy, float))[0]
                    if d2 <= PRESENT_TOL_PX:
                        foreign_cap = str(s2.tube_ref)
                        break
                pm = path_metrics(pts, np.asarray(s.path_xy, float)) \
                    if len(pts) >= 2 else {}
                # tolerance reconciliation: a corridor-positive route
                # (<=8 px) that cannot produce an endpoint within the
                # 5 px endpoint gate is SUPPORT-ONLY coverage — not a
                # wrong selector for an impossible endpoint.
                support_only = bool(present == 1
                                    and d_tip > ENDPOINT_TOL_PX)
                out_rows.append({
                    "entry_id": f"{s.movie}|{s.tube_ref}|{frame}|"
                                f"{pr['route_id']}",
                    "movie": s.movie, "frame": frame, "owner": s.tube_ref,
                    "crop_xywh": [ox, oy, CS, CS],
                    "route_native": pts.tolist(),
                    "origin": pr.get("seed", "proposal"),
                    "root_source": pr.get("root_source"),
                    "anchor_native": pr.get("anchor"),
                    "grain_center_native": [float(gc[0]), float(gc[1])],
                    "grain_radius_px": float(gr),
                    "grain_source": gsrc,
                    "source_annotation_ids": [str(s.obs_uuid),
                                              str(s.obs_revision)],
                    "tip_native": [float(s.tip_xy[0]),
                                   float(s.tip_xy[1])],
                    "human_path_start_native": [float(s.path_xy[0][0]),
                                                float(s.path_xy[0][1])],
                    "tip_distance_px": d_tip,
                    "tip_distance_vertex_px": d_tip_vertex,
                    "tip_proj_native": [float(proj_xy[0]),
                                        float(proj_xy[1])],
                    "s_proj_px": s_proj,
                    "support_length_px": total,
                    "present": present,
                    "present_vertex_geometry": present_v,
                    "label_changed": bool(present_v != present),
                    "foreign_cap_owner": foreign_cap,
                    "support_only": support_only,
                    "uncertain": present is None,
                    "path_plausibility": {
                        "precision": pm.get("precision"),
                        "recall": pm.get("recall"),
                        "terminal_err_px": pm.get("terminal_err_px")},
                })
        finally:
            r.close()
        print(f"{s.entry_id}: {sum(1 for x in out_rows if x['frame'] == int(s.source_frame) and x['owner'] == s.tube_ref)} proposals")
    n_present = sum(1 for x in out_rows if x["present"] == 1)
    n_absent = sum(1 for x in out_rows if x["present"] == 0)
    n_unc = sum(1 for x in out_rows if x["present"] is None)
    n_pv = sum(1 for x in out_rows if x["present_vertex_geometry"] == 1)
    n_av = sum(1 for x in out_rows if x["present_vertex_geometry"] == 0)
    n_uv = sum(1 for x in out_rows if x["present_vertex_geometry"] is None)
    # before/after ledger for EVERY label change (rev12 P0.3)
    ledger = []
    for x in out_rows:
        if not x["label_changed"]:
            continue
        ledger.append({
            "entry_id": x["entry_id"],
            "before": ("present" if x["present_vertex_geometry"] == 1
                       else "absent" if x["present_vertex_geometry"] == 0
                       else "unknown"),
            "after": ("present" if x["present"] == 1
                      else "absent" if x["present"] == 0
                      else "unknown"),
            "d_vertex_px": round(x["tip_distance_vertex_px"], 2),
            "d_continuous_px": round(x["tip_distance_px"], 2),
            "s_proj_px": (round(x["s_proj_px"], 2)
                          if x["s_proj_px"] is not None else None),
            "support_length_px": round(x["support_length_px"], 2),
            "foreign_cap_owner": x["foreign_cap_owner"],
        })
    manifest = {"snapshot": str(snap), "n_proposals": len(out_rows),
                "n_present": n_present, "n_absent": n_absent,
                "n_uncertain": n_unc,
                "census_vertex_geometry": {"present": n_pv,
                                           "absent": n_av,
                                           "unknown": n_uv},
                "census_continuous_geometry": {"present": n_present,
                                               "absent": n_absent,
                                               "unknown": n_unc},
                "label_changes": len(ledger),
                "label_change_ledger": ledger,
                "foreign_cap_rows": sum(
                    1 for x in out_rows if x["foreign_cap_owner"]),
                "support_only_rows": sum(
                    1 for x in out_rows if x["support_only"]),
                "tolerances": {"present_le_px": PRESENT_TOL_PX,
                               "absent_gt_px": ABSENT_TOL_PX,
                               "endpoint_le_px": ENDPOINT_TOL_PX},
                "geometry": "continuous point-to-segment projection "
                            "(rev12 P0.3; polyline_point_projection)",
                "note": "supports are runtime proposals; human path/tip "
                        "used ONLY as labels",
                "rows": out_rows}
    Path(a.out).write_text(json.dumps(manifest, indent=1))
    print(f"proposals -> {a.out}: {len(out_rows)} rows "
          f"(present {n_present}, absent {n_absent}, uncertain {n_unc}; "
          f"ledger {len(ledger)} changes)")
    return 0


# ------------------------------------------------------------------ train
def _load_event_clip(movies, movie, frame, crop):
    from tubetracker.annotation_frames import FrameReader
    r = FrameReader(movies[movie])
    try:
        frames = tuple(max(0, frame + o) for o in QUERY_OFFSETS)
        missing = tuple(1 if frame + o < 0 else 0 for o in QUERY_OFFSETS)
        return load_clip_pixels(r, frames, tuple(crop), missing)
    finally:
        r.close()


def mode_train(a) -> int:
    man = json.loads(Path(a.manifest).read_text())
    snap = Path(man.get("snapshot", a.snapshot))
    sman = json.loads((snap / "snapshot_manifest.json").read_text())
    movies = {k: v.get("path") for k, v in (sman.get("movies") or {}).items()}
    from prototypes.v30_video_apex.cap_evidence import CapLabelIndex
    cap_labels = CapLabelIndex.from_snapshot(snap)
    cap_usage = {"positive": 0, "negative": 0, "unknown": 0}

    # rev12 P0.3: seeds BEFORE any model/head construction so the
    # initialization is reproducible and its hash is meaningful.
    import random
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(a.seed)

    model = build_model("temporal", base=a.base, multiscale=True,
                        presence_head=True,
                        local_cap_head=(a.scorer == "local"),
                        cap_window_head=(a.scorer == "capwindow"),
                        cap_window_encoder=bool(
                            a.scorer == "capwindow"
                            and getattr(a, "adapt_encoder", False)))
    gaps = {}
    if a.init:
        info = load_checkpoint(a.init, model, strict=False)
        gaps = {"missing": info.get("missing_keys", []),
                "unexpected": info.get("unexpected_keys", [])}
        print(f"init from {a.init}: gaps {gaps}")
    # the hash of the ACTUAL starting weights (post-load), not a
    # description of the intent
    init_param_hash = _param_hash(model)
    weights = LossWeights(apex=0.0, visibility=0.0, body=0.0,
            route_correct=1.0 if a.route_supervision else 0.0,
                          front=1.0,
                          front_present=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    if a.scorer in ("local", "capwindow"):
        # rev11 section 8B / rev13 W3 arm: train ONLY the native cap
        # scorer (local MLP or the ordered-window scorer). The trunk is
        # frozen (it already carries the owner/body context the review
        # wants conditioning on); this is a bounded head-only
        # experiment, not an LR ladder.
        _pref = "cap_scorer" if a.scorer == "local" else "cw_"
        cap_params = [p for n, p in model.named_parameters()
                      if n.startswith(_pref)]
        assert cap_params, f"{a.scorer} head did not build its scorer"
        head_params = cap_params
        if getattr(a, "adapt_encoder", False):
            # rev13 W3 contingency (declared one-variable comparison):
            # the SAME labels/contracts/updates/losses, with the
            # encoder adapted to the cap task at --lr instead of the
            # frozen-trunk head-only arm. The comparison artifact
            # records trainable_groups for both sides.
            _trunk = [p for n, p in model.named_parameters()
                      if not n.startswith(_pref)]
            opt = torch.optim.AdamW([
                {"params": cap_params, "lr": a.head_lr},
                {"params": _trunk, "lr": a.lr}])
        else:
            opt = torch.optim.AdamW(cap_params, lr=a.head_lr)
    else:
        head_params = [p for n, p in model.named_parameters()
                       if "front" in n or "route_head" in n]
        opt = torch.optim.AdamW([
            {"params": head_params, "lr": a.head_lr},
            {"params": [p for n, p in model.named_parameters()
                        if not any(p is q for q in head_params)],
             "lr": a.lr}])

    rows = man["rows"]
    if a.supervision == "runtime":
        samples = rows
    else:
        samples = _oracle_arm(rows, movies, str(snap))
    # rev11: BALANCED sampling. The proposal pool is ~85% decoys; a
    # uniform order taught the first run's presence head the majority
    # prior (present 0.164 / absent 0.162 — no separation at all). The
    # LOSS is unchanged; only the sampling frequency is balanced so the
    # head sees roughly half positives per block (documented in the
    # output JSON for equal-updates comparisons).
    present_idx = [i for i, x in enumerate(samples)
                   if x.get("present") == 1]
    other_idx = [i for i, x in enumerate(samples)
                 if x.get("present") != 1]
    print(f"sampling: {len(present_idx)} present, {len(other_idx)} "
          f"other; ~50/50 draw")
    order: list[int] = []
    _rng0 = np.random.default_rng(a.seed + 11)
    for _ in range(max(1, a.updates // a.block) * a.block):
        if present_idx and _rng0.random() < 0.5:
            order.append(int(_rng0.choice(present_idx)))
        else:
            order.append(int(_rng0.choice(other_idx)))
    groups: dict[tuple, list] = {}
    for x in samples:
        key = (x["movie"], x["frame"], tuple(x["crop_xywh"]))
        groups.setdefault(key, []).append(x)
    clip_cache: dict = {}

    history, n = [], 0
    done = 0
    for blk in range(max(1, a.updates // a.block)):
        blk_losses, blk_pres, blk_loc = [], [], []
        for _ in range(a.block):
            i = order[done % len(order)]
            done += 1
            x = samples[i]
            key = (x["movie"], x["frame"], tuple(x["crop_xywh"]))
            clip_np = clip_cache.get(key)
            if clip_np is None:
                clip_np = _load_event_clip(movies, x["movie"], x["frame"],
                                           x["crop_xywh"])
                clip_cache[key] = clip_np
            clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)
            ox, oy = x["crop_xywh"][0], x["crop_xywh"][1]
            prompt, _prov = build_typed_prompt(
                str(x["owner"]), crop_wh=(CS, CS), crop_origin=(ox, oy),
                detected_grain_xy=tuple(x["grain_center_native"]),
                detected_radius_px=float(x["grain_radius_px"]))
            route_crop = (np.asarray(x["route_native"], float)
                          - np.array([ox, oy])).tolist()
            present = x.get("present")
            # rev11 item 5: route-level supervision. The route head is
            # trained on the ACTUAL runtime proposals — present routes
            # positive, decoys negative, the 8-24 px band masked
            # (UNKNOWN, never a silent negative). See route_label().
            if a.route_supervision and present is not None:
                route_t = route_label(present)
                route_valid = torch.ones(1)
            else:
                route_t, route_valid = None, None
            if a.scorer in ("local", "capwindow"):
                # Generic cap truth is independent of this owner's route
                # label. Only explicitly reviewed local non-cap is negative.
                loc_crop = resample_polyline(route_crop, step=4.0)[0]
                local_truth = cap_labels.locations(
                    x["movie"], x["frame"], loc_crop + [ox, oy])
                tgt = local_truth["target"]
                val = (local_truth["positive"] | local_truth["negative"]).astype(np.float32)
                tang = np.gradient(loc_crop, axis=0)
                tang = tang / np.clip(
                    np.hypot(tang[:, 0], tang[:, 1])[:, None], 1e-9,
                    None)
                _loc_t = torch.from_numpy(
                    loc_crop).unsqueeze(0).to(torch.float32)
                _tan_t = torch.from_numpy(
                    tang).unsqueeze(0).to(torch.float32)
                if a.scorer == "local":
                    cq = {"locations_xy": _loc_t, "tangents_xy": _tan_t}
                    pred = model.forward(clip, prompt, cap_query=cq)
                    lg = pred.cap_logits.reshape(-1)
                    dxy_px = None
                    logvar = None
                    # rev13 W3.4 (same rule): a location whose
                    # patch would clamp at the frame edge gets
                    # no supervised label.
                    _half = 6.0
                    _inside = ((loc_crop[:, 0] >= _half)
                               & (loc_crop[:, 0] <= CS - 1 - _half)
                               & (loc_crop[:, 1] >= _half)
                               & (loc_crop[:, 1] <= CS - 1 - _half))
                    val = val * _inside.astype(np.float32)
                else:
                    # rev13 W3: the ordered-window scorer, called as the
                    # deployed code calls it; the native XY refinement
                    # and the log-variance are trained here.
                    cw = model.score_cap_window(
                        clip, prompt, _loc_t, _tan_t)
                    lg = cw["logits"].reshape(-1)
                    dxy_px = cw["dxy_px"].reshape(-1, 2)
                    logvar = cw["logvar"].reshape(-1)
                    # rev13 W3.4: invalid (border-clamped)
                    # locations carry no supervised label —
                    # never a silent negative.
                    val = val * cw["valid"].reshape(-1).numpy()                    .astype(np.float32)
                tv = torch.from_numpy(tgt)
                vv = torch.from_numpy(val)
                cap_usage["positive"] += int(((tgt > 0) & (val > 0)).sum())
                cap_usage["negative"] += int(((tgt == 0) & (val > 0)).sum())
                cap_usage["unknown"] += int((val == 0).sum())
                # rev11: the valid locations are ~195:1 negative:positive
                # (74 positives vs 14395 negatives across the whole
                # pool — measured). A plain mean BCE learns the constant
                # no-cap prior and the scorer stops discriminating (the
                # first arm's eval was unchanged from its own checkpoint
                # at 1/4 the updates). pos_weight = n_neg/n_pos of THIS
                # route's valid locations is the single declared fix.
                n_pos = float((tv * vv).sum())
                n_neg = float(((1.0 - tv) * vv).sum())
                pos_w = torch.tensor(
                    max(n_neg / max(n_pos, 1.0), 1.0), dtype=torch.float32)
                bce = (torch.nn.functional
                       .binary_cross_entropy_with_logits(
                           lg, tv, reduction="none", pos_weight=pos_w))
                loss = (bce * vv).sum() / vv.sum().clamp_min(1.0)
                if a.scorer == "capwindow":
                    # rev13 W3.5b: WITHIN-ROW discrimination term.
                    # The certified tips make this supervision
                    # trustworthy (H433: all six verified within
                    # 4.52 px), and per-location BCE alone left
                    # within-row separation at chance (12/23
                    # rows inverted, H432). The hardest-negative
                    # hinge directly demands: the best positive
                    # location beats the best negative location
                    # by `rank_margin` logits, in the SAME image
                    # and route.
                    _pm2 = ((tgt > 0) & (val > 0))
                    _nm2 = ((tgt == 0) & (val > 0))
                    if _pm2.any() and _nm2.any():
                        loss = loss + a.rank_weight * within_row_hinge(
                            lg, _pm2, _nm2, a.rank_margin)
                if a.scorer == "capwindow" and dxy_px is not None:
                    # rev13 W3: native refinement + uncertainty on the
                    # POSITIVE locations only (the human tip is the cap
                    # target): smooth-L1 toward (tip - location) and a
                    # log-variance NLL. Ambiguous/negative locations
                    # carry no refinement claim.
                    pm = torch.from_numpy(
                        ((tgt > 0) & (val > 0)).astype(np.float32))
                    if float(pm.sum()) > 0:
                        dxy_t = torch.from_numpy(
                            local_truth["offset_xy"].astype(np.float32))
                        se = (dxy_px - dxy_t) ** 2
                        lv = logvar.reshape(-1, 1)
                        nll = 0.5 * (torch.exp(-lv) * se + lv)
                        loss = loss + 0.5 * (
                            (nll.sum(dim=-1) * pm).sum()
                            / pm.sum().clamp_min(1.0))
                opt.zero_grad()
                loss.backward()
                opt.step()
                blk_losses.append(float(loss.detach()))
                blk_pres.append(0.0)
                blk_loc.append(float(loss.detach()))
                n += 1
            else:
                with torch.no_grad():
                    p0 = model.forward(clip, prompt, route_xy=route_crop)
                S = int(p0.front_logits.shape[-1])
                if present == 1:
                    interval = torch.from_numpy(front_interval_mask(
                        p0.front_s.detach().numpy(), float(x["s_proj_px"]),
                        FRONT_HALFWIDTH_PX)).to(torch.float32)
                    p_t = torch.ones(1)
                    p_v = torch.ones(1)
                    f_v = torch.ones(1)
                elif present == 0:
                    interval = torch.zeros(1, S)
                    p_t = torch.zeros(1)
                    p_v = torch.ones(1)
                    f_v = torch.zeros(1)     # no-cap route: location masked
                else:
                    interval = torch.zeros(1, S)
                    p_t = torch.zeros(1)
                    p_v = torch.zeros(1)     # uncertain: masked everywhere
                    f_v = torch.zeros(1)
                r = train_step_front(
                    model, opt, clip, prompt, route_crop,
                    torch.zeros(1, 1, CS, CS),
                    {"apex_valid": torch.zeros(1)},
                    interval.unsqueeze(0) if interval.ndim == 2 else interval,
                    f_v, torch.tensor([2]), torch.ones(1), weights,
                    route_t=route_t, route_valid=route_valid,
                    present_t=p_t, present_valid=p_v)
                blk_losses.append(r.get("total", 0.0))
                blk_pres.append(r.get("front_present", 0.0))
                blk_loc.append(r.get("front", 0.0))
                n += 1
        model.eval()
        with torch.no_grad():
            accs = []
            for x in samples[:80]:
                key = (x["movie"], x["frame"], tuple(x["crop_xywh"]))
                clip_np = clip_cache.get(key)
                if clip_np is None:
                    clip_np = _load_event_clip(movies, x["movie"],
                                               x["frame"], x["crop_xywh"])
                    clip_cache[key] = clip_np
                ox, oy = x["crop_xywh"][0], x["crop_xywh"][1]
                prompt, _ = build_typed_prompt(
                    str(x["owner"]), crop_wh=(CS, CS),
                    crop_origin=(ox, oy),
                    detected_grain_xy=tuple(x["grain_center_native"]),
                    detected_radius_px=float(x["grain_radius_px"]))
                rc = (np.asarray(x["route_native"], float)
                      - np.array([ox, oy])).tolist()
                with torch.no_grad():
                    pr = model.forward(
                        torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2),
                        prompt, route_xy=rc)
                p = float(torch.sigmoid(
                    pr.front_present_logit.reshape(-1))[0])
                rp = float(torch.sigmoid(
                    pr.route_logit.reshape(-1))[0]) \
                    if pr.route_logit is not None else float("nan")
                if x.get("present") == 1:
                    accs.append(("present", p, rp))
                elif x.get("present") == 0:
                    accs.append(("absent", p, rp))
            model.train()
        acc_p = np.mean([p for k, p, _ in accs if k == "present"]) if any(
            k == "present" for k, _, _ in accs) else float("nan")
        acc_a = np.mean([p for k, p, _ in accs if k == "absent"]) if any(
            k == "absent" for k, _, _ in accs) else float("nan")
        rp_p = np.mean([r for k, _, r in accs if k == "present"]) if any(
            k == "present" for k, _, _ in accs) else float("nan")
        rp_a = np.mean([r for k, _, r in accs if k == "absent"]) if any(
            k == "absent" for k, _, _ in accs) else float("nan")
        row = {"block": blk, "updates": (blk + 1) * a.block,
               "loss_mean": float(np.mean(blk_losses)),
               "front_present_mean": float(np.mean(blk_pres)),
               "front_mean": float(np.mean(blk_loc)),
               "present_prob_mean": float(acc_p),
               "absent_prob_mean": float(acc_a),
               "route_present_mean": float(rp_p),
               "route_absent_mean": float(rp_a)}
        history.append(row)
        print(f"block {blk}: {row}")
        ck = Path(a.out).with_suffix(f".block{blk}.weights.pt")
        # full metadata (schema/activation/preprocessing + config hash)
        # so the strict factory can rebuild this checkpoint later
        cfg = {"variant": "temporal", "base": a.base, "multiscale": True,
               "presence_head": True, "lr": a.lr, "head_lr": a.head_lr}
        if a.scorer == "local":
            cfg["local_cap_head"] = True
        if a.scorer == "capwindow":
            # rev13 W3: the trained head is part of the DECLARED
            # architecture — the strict factory rebuilds it from this
            # config alone.
            cfg["cap_window_head"] = True
            cfg["cap_window_semantics"] = 2
            cfg["query_index"] = 4
            cfg["source_offsets"] = list(QUERY_OFFSETS)
            if getattr(a, "adapt_encoder", False):
                cfg["cap_window_encoder"] = True
        save_checkpoint(ck, model, config=cfg,
                        manifest={"mode": "proposal_fit",
                                  "scorer": a.scorer,
                                  "supervision": a.supervision,
                                  "init": a.init,
                                  "updates": (blk + 1) * a.block,
                                  "seed": a.seed,
                                  "manifest": a.manifest})
    Path(a.out).write_text(json.dumps({
        "cap_label_scope": cap_labels.summary(),
        "cap_loss_contributions": cap_usage,
        "supervision": a.supervision, "scorer": a.scorer,
        "objective": {
            "per_location_bce": True,
            "within_row_hinge_weight": a.rank_weight,
            "within_row_hinge_margin": a.rank_margin,
            "note": "hardest-negative hinge between the best "
                    "positive and best negative location of "
                    "the same row (same image, same route)"},
        "init": a.init, "seed": a.seed,
        # rev12 P0.3: the actual initialized state and full effective
        # configuration — comparisons are checked mechanically.
        "init_param_hash": init_param_hash,
        "init_gaps": gaps,
        "trainable_groups": {
            "head_params": [n for n, p in model.named_parameters()
                            if any(p is q for q in head_params)],
            "n_head_params": len(head_params),
            "n_total_params": sum(1 for _ in model.parameters())},
        "class_balance": {"present": len(present_idx),
                          "other": len(other_idx),
                          "uncertain": sum(
                              1 for x in samples
                              if x.get("present") is None),
                          "absent": sum(
                              1 for x in samples
                              if x.get("present") == 0)},
        "route_supervision": bool(a.route_supervision),
        "effective_config": {
            "lr": a.lr, "head_lr": a.head_lr, "updates": a.updates,
            "block": a.block, "base": a.base, "scorer": a.scorer,
            "supervision": a.supervision, "seed": a.seed},
        "updates": done, "block": a.block, "head_lr": a.head_lr,
        "lr": a.lr, "history": history,
        "sampling": {"mode": "balanced~50/50 present-vs-other",
                     "n_present": len(present_idx),
                     "n_other": len(other_idx)},
        "manifest": a.manifest, "manifest_rows": len(rows),
        "note": "equal-updates arm for the runtime-proposal experiment"},
        indent=1))
    print(f"-> {a.out}")
    return 0


def _oracle_arm(rows, movies, snapshot: str):
    """Baseline arm: the SAME events, oracle +24 routes (the trained
    regime), one row per event. present=1, s_proj=s_star. The human
    path/tip is a LABEL here too — the comparison isolates the
    supervision source, nothing else."""
    from prototypes.v30_video_apex.targets import samples_from_snapshot
    evs: dict = {}
    for x in rows:
        evs.setdefault((x["movie"], x["frame"], x["owner"]), x)
    obs = {str(s.tube_ref): s for s in samples_from_snapshot(snapshot)
           if s.kind == "path_tip"}
    out = []
    for (movie, frame, owner), x in sorted(evs.items()):
        o = obs.get(str(owner))
        if o is None or not o.path_xy or not o.tip_xy:
            continue
        r = blind_oracle_route(o.path_xy, o.tip_xy, post_px=24.0)
        pts = np.asarray(r["route_xy"], float)
        out.append({**x, "route_native": pts.tolist(),
                    "present": 1, "uncertain": False,
                    "s_proj_px": float(r["s_star"]),
                    "support_length_px": float(r["support"])})
    return out


# ------------------------------------------------------------------ eval
def route_label(present):
    """rev11 item 5 route-supervision label mapping.

    present == 1 -> route carries the owned cap (positive);
    present == 0 -> decoy (negative);
    uncertain    -> None (MASKED — the 8-24 px band is never a silent
                    negative).
    Returns None or a 1-element tensor, ready for train_step_front.
    """
    import torch as _t
    if present == 1:
        return _t.tensor([1.0])
    if present == 0:
        return _t.tensor([0.0])
    return None


def mode_eval(a) -> int:
    man = json.loads(Path(a.manifest).read_text())
    snap = Path(man.get("snapshot", a.snapshot))
    sman = json.loads((snap / "snapshot_manifest.json").read_text())
    movies = {k: v.get("path") for k, v in (sman.get("movies") or {}).items()}
    from prototypes.v30_video_apex.cap_evidence import CapLabelIndex, emit_cap_candidates
    cap_labels = CapLabelIndex.from_snapshot(snap)
    # rev13 W1: evaluation uses the metadata-driven STRICT factory. The
    # declared architecture (presence/local-cap heads included) rebuilds
    # from the checkpoint's own config; missing/unexpected tensors
    # refuse; no untrained random head is added at evaluation time.
    from prototypes.v30_video_apex.model_factory import (
        build_model_from_checkpoint)
    model, _minfo = build_model_from_checkpoint(a.init)
    print(f"eval strict load {a.init}: {_minfo['semantics']} | "
          f"post-load hash {_minfo['parameter_hash']} | "
          f"kwargs {_minfo['model_kwargs']}")
    model.eval()
    groups: dict[tuple, list] = {}
    for x in man["rows"]:
        groups.setdefault((x["movie"], x["frame"], x["owner"]),
                          []).append(x)
    res = []
    n_ev = top1 = top3 = top5 = 0
    top1_r = top3_r = top5_r = 0
    base1 = 0
    n_support = n_current = n_oracle = 0
    n_support_only_events = 0
    clip_cache: dict = {}
    clip_hashes: dict = {}

    def _clip_for(movie, frame, crop):
        # rev12 P0.3: versioned by (movie, frame, crop, native
        # transform); identical keys share pixels AND the recorded hash.
        key = (movie, int(frame), tuple(int(v) for v in crop))
        if key not in clip_cache:
            arr = _load_event_clip(movies, movie, frame, crop)
            clip_cache[key] = arr
            clip_hashes["|".join(map(str, key))] = hashlib.sha256(
                np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
        return clip_cache[key]

    for (movie, frame, owner), rows in sorted(groups.items()):
        n_ev += 1
        first_crop = rows[0]["crop_xywh"]
        tip = np.asarray(rows[0]["tip_native"], float)
        hyps = []      # (score, native_xy)
        hyps_r = []    # rev11 item 5: route_p x q ranking key
        base_pts = []
        event_support = False
        event_current = False
        event_support_only = []
        for x in rows:
            # rev12 P0.3: SHARED construction — every row is evaluated
            # under ITS OWN crop (the one it was trained under); the
            # historical first-row substitution is available only under
            # --crop-mode first for exact reproduction of old numbers.
            crop = (rows[0]["crop_xywh"] if a.crop_mode == "first"
                    else x["crop_xywh"])
            ox, oy = crop[0], crop[1]
            clip_np = _clip_for(movie, frame, crop)
            clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)
            prompt, _ = build_typed_prompt(
                str(owner), crop_wh=(CS, CS), crop_origin=(ox, oy),
                detected_grain_xy=tuple(x["grain_center_native"]),
                detected_radius_px=float(x["grain_radius_px"]))
            pts = np.asarray(x["route_native"], float)
            rc = (pts - np.array([ox, oy])).tolist()
            # image-validity: points outside the crop's pixels are not
            # silently treated as image evidence
            _ptsn = np.asarray(rc, float)
            _in = ((_ptsn[:, 0] >= 0) & (_ptsn[:, 0] < CS)
                   & (_ptsn[:, 1] >= 0) & (_ptsn[:, 1] < CS))
            x["route_in_crop_frac"] = float(_in.mean()) if len(_in) \
                else 0.0
            x["route_pts_out_of_crop"] = int((~_in).sum())
            # native-coordinate round trip (asserted per row)
            _rt = (np.asarray(rc, float) + np.array([ox, oy]))
            assert np.allclose(_rt, pts, atol=1e-6), (
                f"native round trip failed for {x['entry_id']}")
            seg = np.diff(pts, axis=0)
            cum = np.concatenate([[0.0], np.cumsum(np.hypot(
                seg[:, 0], seg[:, 1]))])
            if x.get("present") == 1:
                event_support = True
                _sp = x.get("s_proj_px")
                _tot = float(x.get("support_length_px") or 0.0)
                if _sp is not None and _sp >= _tot - 24.0:
                    event_current = True
                if x.get("support_only"):
                    event_support_only.append(x["entry_id"])
            if a.scorer in ("local", "capwindow"):
                # rev11 section 8B / rev13 W3: hypotheses are local-cap
                # maxima at explicit native locations — no ribbon, no
                # dependence on the support's ends. The ordered-window
                # scorer additionally REFINES each hypothesis by its
                # native XY prediction (the deployed use of dxy_px).
                loc_crop = resample_polyline(rc, step=4.0)[0]
                tang = np.gradient(loc_crop, axis=0)
                tang = tang / np.clip(
                    np.hypot(tang[:, 0], tang[:, 1])[:, None], 1e-9,
                    None)
                _loc_t = torch.from_numpy(
                    loc_crop).unsqueeze(0).to(torch.float32)
                _tan_t = torch.from_numpy(
                    tang).unsqueeze(0).to(torch.float32)
                if a.scorer == "local":
                    cq = {"locations_xy": _loc_t, "tangents_xy": _tan_t}
                    with torch.no_grad():
                        p = model.forward(clip, prompt, cap_query=cq)
                    sc = torch.sigmoid(p.cap_logits.reshape(-1)).numpy()
                    raw_lg = p.cap_logits.reshape(-1).numpy()
                    _valid_e = ((loc_crop >= 6).all(1) & (loc_crop <= CS - 7).all(1))
                    loc_refined = loc_crop
                    x["cap_window"] = None
                else:
                    with torch.no_grad():
                        cw = model.score_cap_window(
                            clip, prompt, _loc_t, _tan_t)
                    sc = torch.sigmoid(cw["logits"].reshape(-1)).numpy()
                    raw_lg = cw["logits"].reshape(-1).numpy()
                    # rev13 W3.4: invalid (border-clamped)
                    # locations are never emitted as hypotheses.
                    _valid_e = cw["valid"].reshape(-1).numpy()
                    x["n_invalid_locations"] = int((~_valid_e).sum())
                    loc_refined = (loc_crop
                                   + cw["dxy_px"].reshape(-1, 2).numpy())
                    x["cap_window"] = {
                        "window": cw["window"],
                        "logvar": [round(float(v), 4) for v in
                                   cw["logvar"].reshape(-1)],
                        "dxy_px": [[round(float(v), 3) for v in row2]
                                   for row2 in
                                   cw["dxy_px"].reshape(-1, 2).numpy()]}
                x["cap_probabilities"] = sc.tolist()
                x["cap_logits"] = raw_lg.tolist()
                x["cap_valid"] = _valid_e.tolist()
                truth = cap_labels.locations(movie, frame, loc_crop + [ox, oy])
                x["cap_targets"] = np.where(truth["positive"], 1,
                                            np.where(truth["negative"], 0, -1)).tolist()
                emitted = emit_cap_candidates(
                    sc, loc_crop + [ox, oy], _valid_e,
                    offsets_xy=loc_refined - loc_crop,
                    bounds_xyxy=(ox, oy, ox + CS, oy + CS),
                    threshold=0.0, movie=movie, source_frame=frame)
                x["emitted_caps"] = emitted
                for cap in emitted:
                    hyps.append((cap["probability"], np.asarray(cap["tip_xy"])))
            else:
                with torch.no_grad():
                    p = model.forward(clip, prompt, route_xy=rc)
                    pres = float(torch.sigmoid(
                        p.front_present_logit.reshape(-1))[0])
                    q = torch.softmax(p.front_logits, -1)[0].numpy()
                    sg = p.front_s.numpy()
                sc = pres * q
                idx = [i for i in range(1, len(sc) - 1)
                       if sc[i] >= sc[i - 1] and sc[i] >= sc[i + 1]]
                idx.sort(key=lambda i: -sc[i])
                # rev12 P0.3: calibration-independent raw evidence per
                # row — the unnormalized cap logits AND the s grid, not
                # only the normalized peak scores.
                x["raw_cap_logits"] = [round(float(v), 4)
                                       for v in p.front_logits.numpy()
                                       .reshape(-1)]
                x["cap_q"] = [round(float(v), 5) for v in q.reshape(-1)]
                x["s_grid_px"] = [round(float(v), 2) for v in sg]
                x["front_present_prob"] = round(pres, 5)
                if p.route_logit is not None:
                    x["route_p"] = round(float(torch.sigmoid(
                        p.route_logit.reshape(-1))[0]), 5)
                for i in idx:
                    s_i = float(sg[i])
                    xy = (np.array([np.interp(s_i, cum, pts[:, 0]),
                                    np.interp(s_i, cum, pts[:, 1])]))
                    hyps.append((float(sc[i]), xy))
                if p.route_logit is not None:
                    rp = float(torch.sigmoid(
                        p.route_logit.reshape(-1))[0])
                    sc_r = rp * q
                    idx_r = [i for i in range(1, len(sc_r) - 1)
                             if sc_r[i] >= sc_r[i - 1]
                             and sc_r[i] >= sc_r[i + 1]]
                    idx_r.sort(key=lambda i: -sc_r[i])
                    for i in idx_r:
                        s_i = float(sg[i])
                        xy = (np.array([
                            np.interp(s_i, cum, pts[:, 0]),
                            np.interp(s_i, cum, pts[:, 1])]))
                        hyps_r.append((float(sc_r[i]), xy))
            # image-free baseline: support_length - 24 along this route
            s_b = float(x.get("support_length_px") or 0.0) - 24.0
            if s_b > 0:
                base_pts.append(np.array([
                    np.interp(s_b, cum, pts[:, 0]),
                    np.interp(s_b, cum, pts[:, 1])]))
        def _kept_topk(lst):
            lst.sort(key=lambda h: -h[0])
            kept: list[np.ndarray] = []
            for scv, xy in lst:
                if all(np.hypot(*(xy - k)) >= 12.0 for k in kept):
                    kept.append(xy)
                if len(kept) >= 5:
                    break
            return kept

        kept = _kept_topk(hyps)
        kept_r = _kept_topk(hyps_r)
        errs = {K: (min(np.hypot(*(k - tip)) for k in kept[:K])
                    if kept else None) for K in (1, 3, 5)}
        errs_r = {K: (min(np.hypot(*(k - tip)) for k in kept_r[:K])
                      if kept_r else None) for K in (1, 3, 5)}
        base_err = min((np.hypot(*(b - tip)) for b in base_pts),
                       default=None)
        # endpoint-oracle coverage: best achievable over ALL proposals
        oracle_err = min((np.hypot(*(h[1] - tip)) for h in hyps),
                         default=None)
        for K in (1, 3, 5):
            ev, evr = errs[K], errs_r[K]
            ok = ev is not None and ev <= 5.0
            ok_r = evr is not None and evr <= 5.0
            if K == 1:
                top1 += int(ok)
                top1_r += int(ok_r)
            elif K == 3:
                top3 += int(ok)
                top3_r += int(ok_r)
            else:
                top5 += int(ok)
                top5_r += int(ok_r)
        base1 += int(base_err is not None and base_err <= 5.0)
        n_support += int(event_support)
        n_oracle += int(oracle_err is not None and oracle_err <= 5.0)
        # rev13 W3.6: current-path coverage comes from the DEPLOYED
        # pool's actual emitted endpoints (the endpoint oracle over the
        # emitted candidates), not from the support-length shortcut.
        # The old support_length-24 rule survives only as the named
        # shortcut baseline below.
        event_current_real = bool(
            oracle_err is not None and oracle_err <= ENDPOINT_TOL_PX)
        n_current += int(event_current_real)
        if event_support_only and not any(
                r.get("present") == 1 and not r.get("support_only")
                for r in rows):
            n_support_only_events += 1
        res.append({"event": f"{movie}|{owner}|{frame}",
                    "n_proposals": len(rows),
                    "support_reach": bool(event_support),
                    "current_path_coverage": event_current_real,
                    "current_path_coverage_shortcut": bool(
                        event_current),
                    "shortcut_name": "support_length_minus24 (NOT "
                                     "current-path coverage)",
                    "endpoint_oracle_within_5px": bool(
                        oracle_err is not None and oracle_err <= 5.0),
                    "endpoint_oracle_err_px": oracle_err,
                    "support_only_rows": len(event_support_only),
                    "top1_endpoint_err_px": errs[1],
                    "top3_endpoint_err_px": errs[3],
                    "top5_endpoint_err_px": errs[5],
                    "route_q_top1_endpoint_err_px": errs_r[1],
                    "route_q_top3_endpoint_err_px": errs_r[3],
                    "route_q_top5_endpoint_err_px": errs_r[5],
                    "support_length_minus24_err_px": base_err})
        print({k: (round(v, 2) if isinstance(v, float) else v)
               for k, v in res[-1].items()})
    # rev12 P0.3 acceptance: calibration-independent confusion over the
    # manifest rows (presence head and route head at 0.5), with the
    # unknown band kept separate — never folded into a negative.
    conf = {"presence": {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
                         "unknown_band": 0, "n_rows": 0},
            "route": {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
                      "unknown_band": 0, "n_rows": 0}}
    # rev13 W3: for the native cap scorers the per-row presence signal
    # is the scorer's own maximum probability at its emitted locations
    # (the new evidence), reported as its own confusion — the
    # front_present head is untrained in a frozen-trunk head-only arm.
    cap_conf = {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "unknown_band": 0, "n_rows": 0, "threshold": 0.5}
    for x in man["rows"]:
        _p = x.get("present")
        if _p is None:
            conf["presence"]["unknown_band"] += 1
            conf["route"]["unknown_band"] += 1
            cap_conf["unknown_band"] += 1
            continue
        _fp_prob = x.get("front_present_prob")
        if _fp_prob is not None:
            _pred = 1 if _fp_prob > 0.5 else 0
            k = ("tp" if (_p == 1 and _pred == 1) else
                 "tn" if (_p == 0 and _pred == 0) else
                 "fp" if (_p == 0 and _pred == 1) else "fn")
            conf["presence"][k] += 1
            conf["presence"]["n_rows"] += 1
        _rp = x.get("route_p")
        if _rp is not None:
            _pred = 1 if _rp > 0.5 else 0
            k = ("tp" if (_p == 1 and _pred == 1) else
                 "tn" if (_p == 0 and _pred == 0) else
                 "fp" if (_p == 0 and _pred == 1) else "fn")
            conf["route"][k] += 1
            conf["route"]["n_rows"] += 1
    # Every licensed location participates, including caps on a route
    # whose owner-correctness label is unknown. Row presence is not truth
    # for a generic cap detector.
    cap_conf["unit"] = "licensed_local_cap_locations"
    for x in man["rows"]:
        for probability, target, valid in zip(x.get("cap_probabilities", []),
                                              x.get("cap_targets", []), x.get("cap_valid", [])):
            if target < 0 or not valid:
                cap_conf["unknown_band"] += 1
                continue
            yes = probability >= 0.5
            key = ("tp" if yes else "fn") if target else ("fp" if yes else "tn")
            cap_conf[key] += 1
            cap_conf["n_rows"] += 1
    out = {"checkpoint": a.init, "manifest": a.manifest,
           "cap_label_scope": cap_labels.summary(),
           "n_events": n_ev,
           "crop_mode": a.crop_mode,
           "panel": "development/fit-only (the six familiar events; "
                    "NOT held-out generalization evidence)",
           "presence_confusion": conf["presence"],
           "route_confusion": conf["route"],
           "cap_scorer_confusion": cap_conf,
           "support_coverage_events": n_support,
           "current_path_coverage_events": n_current,
           "endpoint_oracle_coverage_events": n_oracle,
           "support_only_events": n_support_only_events,
           "top1_within_5px": top1, "top3_within_5px": top3,
           "top5_within_5px": top5,
           "route_q_top1_within_5px": top1_r,
           "route_q_top3_within_5px": top3_r,
           "route_q_top5_within_5px": top5_r,
           "support_length_minus24_within_5px": base1,
           "clip_hashes": clip_hashes,
           "events": res,
           "note": "endpoints are native xy of separated hypotheses; the "
                   "human tip is a LABEL only. Support coverage counts "
                   "events with a corridor-positive route; current-path "
                   "coverage requires the cap within the last 24 px of "
                   "that support; support_only rows are corridor-positive "
                   "but endpoint-impossible and are NOT selection "
                   "failures."}
    outp = Path(a.out) if a.out else Path(a.init).with_suffix(
        ".endpointeval.json")
    # rev12 P0.3: the per-row RAW evidence (unnormalized cap logits,
    # q, s grids, presence/route probabilities) is persisted beside
    # the summary — "save all cap distributions", not only peaks.
    rows_out = outp.with_suffix(".rows.json")
    rows_out.write_text(json.dumps(man["rows"], indent=1))
    out["rows_file"] = str(rows_out)
    if a.scorer == "local":
        # rev11 section 8B acceptance: the local scorer's input at a
        # fixed native location must be IDENTICAL when a distant
        # support endpoint moves. Witness: the same human path under
        # post=12 and post=48 oracle constructions shares its geometry
        # up to the tip, so the tip sample's window and logit must be
        # bit-equal; shared-prefix logits across the two full grids
        # must be bit-equal too (only the moved end adds samples).
        inv = []
        obs = {str(s.tube_ref): s for s in samples_from_snapshot(
            str(snap)) if s.kind == "path_tip"}
        for (movie, frame, owner), rows in sorted(groups.items()):
            o = obs.get(str(owner))
            if o is None or not o.path_xy or not o.tip_xy:
                continue
            crop = rows[0]["crop_xywh"]
            ox, oy = crop[0], crop[1]
            prompt, _ = build_typed_prompt(
                str(owner), crop_wh=(CS, CS), crop_origin=(ox, oy),
                detected_grain_xy=tuple(rows[0]["grain_center_native"]),
                detected_radius_px=float(rows[0]["grain_radius_px"]))
            clip_np = _load_event_clip(movies, movie, frame, crop)
            clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)
            grids, tip_cats = {}, {}
            for post in (12.0, 48.0):
                r = blind_oracle_route(o.path_xy, o.tip_xy, post_px=post)
                rc = (np.asarray(r["route_xy"], float)
                      - np.array([ox, oy]))
                loc = resample_polyline(rc.tolist(), step=4.0)[0]
                tang = np.gradient(loc, axis=0)
                tang = tang / np.clip(
                    np.hypot(tang[:, 0], tang[:, 1])[:, None], 1e-9,
                    None)
                cq = {"locations_xy": torch.from_numpy(loc).unsqueeze(
                          0).to(torch.float32),
                      "tangents_xy": torch.from_numpy(tang).unsqueeze(
                          0).to(torch.float32)}
                with torch.no_grad():
                    pr = model.forward(clip, prompt, cap_query=cq)
                lg = pr.cap_logits.reshape(-1).numpy()
                tip_c = (np.asarray(o.tip_xy, float)
                         - np.array([ox, oy]))
                d = np.hypot(loc[:, 0] - tip_c[0], loc[:, 1] - tip_c[1])
                j = int(np.argmin(d))
                grids[post] = (loc, tang, lg, j)
            la, ta, ga, ja = grids[12.0]
            lb, tb, gb, jb = grids[48.0]
            k = min(len(ga), len(gb))
            prefix_delta = float(np.max(np.abs(ga[:k] - gb[:k]))) \
                if k else None
            # Two layers of the acceptance:
            # L1 (architectural): the scorer's window at a FIXED native
            # location is a pure function of (x, y, tangent) — nothing
            # from any construction enters it. Query the tip point and a
            # point 30 px back with fixed values under both runs.
            # L2 (practical): the per-route sampling lattice can place
            # its nearest-to-tip sample at slightly different points;
            # report both offsets and logit deltas against the ribbon
            # head's measured 11.06 px / 6.92 px localization failures
            # under the same move (H385).
            tipn = np.asarray(o.tip_xy, float)
            tipc = tipn - np.array([ox, oy])
            seg2 = np.asarray(o.path_xy, float)[-1] - np.asarray(
                o.path_xy, float)[-2]
            tau = seg2 / (float(np.hypot(*seg2)) + 1e-9)
            back = tipc - 30.0 * tau
            fixed = {"locations_xy": torch.tensor(
                [[[tipc[0], tipc[1]], [back[0], back[1]]]],
                dtype=torch.float32),
                "tangents_xy": torch.tensor(
                    [[[float(tau[0]), float(tau[1])],
                      [float(tau[0]), float(tau[1])]]],
                    dtype=torch.float32)}
            with torch.no_grad():
                pf = model.forward(clip, prompt, cap_query=fixed)
            lg_fixed = pf.cap_logits.reshape(-1).numpy()
            off_a = float(np.hypot(la[ja, 0] - tipc[0],
                                   la[ja, 1] - tipc[1]))
            off_b = float(np.hypot(lb[jb, 0] - tipc[0],
                                   lb[jb, 1] - tipc[1]))
            inv.append({
                "event": f"{movie}|{owner}|{frame}",
                "fixed_tip_logit": float(lg_fixed[0]),
                "fixed_back30_logit": float(lg_fixed[1]),
                "fixed_query_note": "L1: identical values under both "
                                    "constructions by construction "
                                    "(the construction is not an input)",
                "lattice_offset_post12_px": off_a,
                "lattice_offset_post48_px": off_b,
                "tip_logit_post12": float(ga[ja]),
                "tip_logit_post48": float(gb[jb]),
                "tip_logit_delta": abs(float(ga[ja]) - float(gb[jb])),
                "shared_prefix_max_delta": prefix_delta,
                "n_locations_post12": int(len(ga)),
                "n_locations_post48": int(len(gb)),
            })
            print("invariance", inv[-1])
        out["invariance"] = inv
        out["invariance_note"] = (
            "post=12 vs post=48 oracle constructions share geometry up "
            "to the tip; the local scorer's inputs and shared-prefix "
            "logits must be bit-equal. The ribbon front head under the "
            "same move reported 11.06/6.92 px errors (H385).")
    outp.write_text(json.dumps(out, indent=1))
    print(f"top-1 {top1}/{n_ev} top-3 {top3}/{n_ev} top-5 {top5}/{n_ev} "
          f"| support_length-24 {base1}/{n_ev} -> {outp}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["build", "train", "eval"])
    ap.add_argument("--snapshot", default=str(
        REPO / "runs/prototypes/v30/snapshots/snap25"))
    ap.add_argument("--v1-weights", default=str(
        REPO / "runs/prototypes/timesfm/tip_cnn_v1"
               "/best-point-heatmap-model.pt"))
    ap.add_argument("--event-ref", default="r4-p")
    ap.add_argument("--manifest", default=str(
        REPO / "runs/prototypes/v30/rev12_proposals.json"))
    ap.add_argument("--init", default=str(
        REPO / "runs/prototypes/v30/rev11_front_tailjitter"
               "/best_front_ep15.pt"),
        help="rev12 P0.3: the preserved TAIL-JITTER checkpoint is the "
             "next baseline (the old rev10 base trials stay "
             "historical).")
    ap.add_argument("--supervision", default="runtime",
                    choices=["runtime", "oracle"])
    ap.add_argument("--scorer", default="ribbon",
                    choices=["ribbon", "local", "capwindow"])
    ap.add_argument("--crop-mode", default="per-row",
                    choices=["per-row", "first"],
                    help="rev12 P0.3: evaluation crops. 'per-row' = "
                         "every route under its own (training) crop; "
                         "'first' = the historical first-row "
                         "substitution (reproduction only).")
    ap.add_argument("--route-supervision", action="store_true",
                    help="rev11 item 5: train the ROUTE head on the "
                         "actual runtime proposals — present routes "
                         "(<= 8 px cap) positive, decoys (> 24 px) "
                         "negative, the 8-24 px band masked UNKNOWN. "
                         "Route plausibility learned on the deployment "
                         "proposer's own false routes.")
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--rank-weight", type=float, default=1.0,
                    help="rev13 W3.5b: weight of the within-row "
                         "hardest-negative hinge (0 = the old "
                         "per-location BCE objective)")
    ap.add_argument("--rank-margin", type=float, default=1.0,
                    help="required logit margin between the "
                         "best positive and best negative "
                         "location in the same row")
    ap.add_argument("--adapt-encoder", action="store_true",
                    help="rev13 W3 contingency: adapt the encoder at "
                         "--lr under the SAME labels/contracts/losses "
                         "(declared one-variable comparison vs the "
                         "frozen-trunk arm)")
    ap.add_argument("--base", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "build":
        return mode_build(a)
    if a.mode == "train":
        return mode_train(a)
    return mode_eval(a)


if __name__ == "__main__":
    raise SystemExit(main())
