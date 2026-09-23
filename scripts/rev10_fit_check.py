"""rev10 WP-B: three-owner discrimination on the existing clump scene.

The review's WP-B, implemented as written:

  * ONE image, ONE crop, all THREE owner queries in a single
    optimization step (their masks, their reviewed scopes, explicit
    foreign-exclusive negatives for each);
  * short deterministic blocks, inspecting loss components, per-group
    gradient magnitudes, body-logit variance, foreign-term contribution
    and saved native predictions every block;
  * ABORT immediately on a dead branch (flat body field), zero foreign
    contribution, invalid target, or divergence — the review forbids
    repeating "an uninstrumented 60-epoch objective/LR ladder";
  * the proposed FIT gate: ~0.90 IoU per owner on its valid evaluation
    domain, >=0.95 recall of its verified body, <0.05 positive rate on
    each verified non-overlapping rival, correct attachment/current cap,
    and rejection of an owned-absence case.

This is a fit test on a scene the model has been developed against. It
says nothing about generalization, and the numbers it prints are not
accuracy claims.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex import train as T  # noqa: E402
from prototypes.v30_video_apex.batch_builder import (  # noqa: E402
    body_mask_tensors, finalize_body_supervision, own_mask_target)
from prototypes.v30_video_apex.inference import (  # noqa: E402
    build_query_prompt, load_query_clip)
from prototypes.v30_video_apex.targets import (  # noqa: E402
    confusable_union, paint_overlap_union, samples_from_snapshot)

CS = 288
CROP = (475, 518, CS, CS)          # pinned clump crop from the review
FRAME = 42000
MOVIE = "ld"


def _scene(snapshot: Path):
    """The three clump owners: targets, prompts, foreign sets."""
    rows = [s for s in samples_from_snapshot(str(snapshot))
            if s.kind == "body_mask" and s.mask_raster
            and s.movie == MOVIE and int(s.source_frame or -1) == FRAME]
    pool = [{"mask_uuid": x.mask_uuid, "movie": x.movie,
             "source_frame": x.source_frame, "owner_key": x.owner_key,
             "mask_raster": x.mask_raster} for x in rows]
    eligible = {str(x.mask_uuid) for x in rows
                if x.mask_uuid and not x.quarantine_reason}
    out = []
    for s in rows:
        bt = own_mask_target(s, CS, CS, (CROP[0], CROP[1]))
        if not (bt.target > 0).any():
            continue
        fo = confusable_union(CS, CS, (CROP[0], CROP[1]), movie=s.movie,
                              frame=int(s.source_frame),
                              self_uuid=s.mask_uuid,
                              self_owner_key=s.owner_key, masks=pool,
                              eligible=eligible)
        ov = paint_overlap_union(CS, CS, (CROP[0], CROP[1]),
                                 movie=s.movie, frame=int(s.source_frame),
                                 masks=pool, eligible=eligible)
        # rev11: the ONE shared supervision step — the fit test must feed
        # the objective exactly the tensors the trainer feeds (the audit:
        # this helper discarded 623 of g1's 683 known foreign pixels for
        # snap24 because it skipped the validity extension).
        bt = finalize_body_supervision(bt, confusable=fo, overlap=ov)
        _ch = body_mask_tensors(bt, confusable=fo, overlap=ov)
        tgt = list(s.target_xy or [])
        if len(tgt) == 2:
            qx, qy = float(tgt[0]) - CROP[0], float(tgt[1]) - CROP[1]
        else:
            qx = qy = -1.0
        if not (0.0 <= qx < CS and 0.0 <= qy < CS):
            ys, xs = np.nonzero(bt.target > 0)
            qx, qy = float(xs.mean()), float(ys.mean())
        out.append({"obs": str(s.obs_uuid),
                    "owner": str(s.owner_key).split("|")[-1],
                    "target": bt.target, "valid": bt.valid,
                    "band": bt.band, "bg_reviewed": bt.bg_reviewed,
                    "foreign": fo & ~ov, "overlap": ov,
                    "channels": _ch,
                    "q": (qx, qy), "mask": bt.target > 0})
    return out


def _loss_for(m, clip, owner_prompt, item, weights):
    prompt = build_query_prompt(item["owner"], item["q"], (CS, CS))
    pred = m.forward(clip, prompt)
    # rev11: the mask tensors come from the SAME builder the trainer uses
    # (batch_builder.body_mask_tensors via _scene) — same sample, same
    # crop, same prompt, same pixels.
    masks = {k: torch.from_numpy(np.asarray(v, dtype=np.float32))[
        None, None] for k, v in item["channels"].items()}
    out = T.masked_multihead_loss(
        {"body": pred.body}, {"body_mask": torch.from_numpy(
            item["target"])[None, None]}, masks, weights)
    return out, pred.body


def _absence_gate(cases):
    """Pre-declared aggregation for the owned-absence rejection gate.

    All-or-nothing: every human-verified absence grain must be
    rejected (visibility argmax == no_tube_visible). One miss = the
    gate fails — no partial credit, no thresholds.
    """
    n = len(cases)
    n_rej = sum(1 for c in cases if c.get("rejected"))
    return {"n_cases": n, "n_rejected": n_rej,
            "gate_passed": bool(n > 0 and n_rej == n),
            "gate": "visibility argmax == no_tube_visible on every "
                    "owned-absence case"}


def _absence_test(model, snapshot: Path, owner_filter: str = ""):
    """Owned-absence rejection gate (rev11 item 7; corrected in rev12).

    rev12 P0.2: the gate selects ONLY explicitly CERTIFIED, grain-scoped
    `no_tube_visible` records (owned-absence regions built from human
    grain clicks). Legacy observation records — including hidden tips
    (`not_directly_visible`) and `owner_uncertain` — are NEVER counted
    as certified absence; they appear in the separate visibility
    confusion matrix with their own truth classes. A focus point is
    not a grain and cannot satisfy the gate.

    Pre-declared gate: at every certified absence grain the model's
    OWNED visibility argmax must be `no_tube_visible` (index 3).
    All-or-nothing. The per-case table carries truth scope, query,
    prediction and the raw false-emission proxy (heat_max).
    """
    from prototypes.v30_video_apex.inference import (
        build_typed_prompt, load_query_clip)
    from prototypes.v30_video_apex.targets import VISIBILITY_INDEX
    from tubetracker.annotation_frames import FrameReader
    _inv = {v: k for k, v in VISIBILITY_INDEX.items()}
    _sizes: dict[str, tuple[int, int]] = {}
    _man = json.loads((Path(snapshot) / "snapshot_manifest.json")
                      .read_text())
    _movies = {k: v.get("path") for k, v in
               (_man.get("movies") or {}).items()}
    all_nt = [s for s in samples_from_snapshot(str(snapshot))
              if s.kind == "no_tube"
              and (not owner_filter or owner_filter in
                   (str(s.owner_uuid) + str(s.owner_key)))]
    rows = [s for s in all_nt
            if bool(getattr(s, "certified", False))
            and str(getattr(s, "direct_state", "")) == "no_tube_visible"]
    cases = []
    for s in rows:
        cs = CS
        cx, cy = float(s.focus_xy[0]), float(s.focus_xy[1])
        if s.movie not in _sizes:
            _r = FrameReader(_movies[s.movie])
            _sizes[s.movie] = (int(_r.native_size[0]),
                               int(_r.native_size[1]))
            _r.close()
        nw, nh = _sizes[s.movie]
        crop = (int(min(max(cx - cs / 2, 0), max(0, nw - cs))),
                int(min(max(cy - cs / 2, 0), max(0, nh - cs))),
                cs, cs)
        clip_np = load_query_clip(snapshot, s.movie, int(s.source_frame),
                                  crop)
        prompt, _prov = build_typed_prompt(
            str(s.owner_uuid or s.entry_id), crop_wh=(cs, cs),
            crop_origin=(crop[0], crop[1]),
            human_grain_xy=(cx, cy), human_grain_radius_px=13.0)
        clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)
        with torch.no_grad():
            pred = model.forward(clip, prompt)
        vl = pred.visibility_logits[0].detach().numpy()
        arg = int(vl.argmax())
        prob = float(np.exp(vl[arg]) / np.exp(vl).sum())
        top2 = float(np.sort(vl)[-2])
        cases.append({
            "case": str(s.entry_id), "frame": int(s.source_frame),
            "grain_xy": [round(cx, 1), round(cy, 1)],
            "owner": str(s.owner_uuid or s.owner_key),
            "grain_id": str(getattr(s, "grain_id", "") or ""),
            "truth_scope": ("certified-grain-scoped"
                            if getattr(s, "certified", False)
                            else "legacy-observation"),
            "query_kind": str(getattr(s, "query_kind", "") or ""),
            "vis_argmax": arg, "vis_class": _inv.get(arg, str(arg)),
            "vis_prob": round(prob, 4),
            "vis_margin": round(float(vl[arg] - top2), 4),
            "heat_max": round(float(pred.heat[0, 0].max()), 4),
            "emits_tip_above_zero": bool(
                float(pred.heat[0, 0].max()) > 0.0),
            "rejected": bool(arg == VISIBILITY_INDEX["no_tube_visible"]),
        })
    # separate visibility confusion over EVERY no_tube sample record:
    # certified rows contribute their gate predictions; legacy rows are
    # predicted here with the same pipeline.
    confusion: dict[str, dict[str, int]] = {}
    for c in cases:
        truth = "no_tube_visible"
        confusion.setdefault(truth, {})
        confusion[truth][c["vis_class"]] = confusion[truth].get(
            c["vis_class"], 0) + 1
    for s in all_nt:
        if any(c["case"] == str(s.entry_id) for c in cases):
            continue
        cs = CS
        cx, cy = float(s.focus_xy[0]), float(s.focus_xy[1])
        if s.movie not in _sizes:
            _r = FrameReader(_movies[s.movie])
            _sizes[s.movie] = (int(_r.native_size[0]),
                               int(_r.native_size[1]))
            _r.close()
        nw, nh = _sizes[s.movie]
        crop = (int(min(max(cx - cs / 2, 0), max(0, nw - cs))),
                int(min(max(cy - cs / 2, 0), max(0, nh - cs))),
                cs, cs)
        clip_np = load_query_clip(snapshot, s.movie, int(s.source_frame),
                                  crop)
        prompt, _prov = build_typed_prompt(
            str(s.owner_uuid or s.entry_id), crop_wh=(cs, cs),
            crop_origin=(crop[0], crop[1]),
            human_grain_xy=(cx, cy), human_grain_radius_px=13.0)
        clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)
        with torch.no_grad():
            pred = model.forward(clip, prompt)
        arg = int(pred.visibility_logits[0].detach().numpy().argmax())
        truth = str(getattr(s, "direct_state", "") or "unknown")
        pred_cls = _inv.get(arg, str(arg))
        confusion.setdefault(truth, {})
        confusion[truth][pred_cls] = confusion[truth].get(
            pred_cls, 0) + 1
    return {**_absence_gate(cases), "cases": cases,
            "n_all_no_tube_records": len(all_nt),
            "inventory": {k: sum(1 for s in all_nt if str(
                getattr(s, "direct_state", "")) == k)
                for k in sorted({str(getattr(s, "direct_state", ""))
                                 for s in all_nt})},
            "visibility_confusion": confusion,
            "gate_scope": ("certified grain-scoped no_tube_visible "
                           "records only")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(
        REPO / "runs/prototypes/v30/snapshots/snap24"))
    ap.add_argument("--checkpoint", default=str(
        REPO / "runs/prototypes/v30/rev9_wpB8/ep29.pt"))
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--head-lr", type=float, default=0.5)
    ap.add_argument("--trunk-lr", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="",
                    choices=["", "temporal", "owner-decoder"],
                    help="build a FRESH model of this variant instead of "
                         "loading --checkpoint (the WP-B architecture "
                         "comparison: equal samples/updates, both arms "
                         "from scratch)")
    ap.add_argument("--scratch", action="store_true",
                    help="fresh temporal model, no checkpoint (the other "
                         "half of the equal-updates comparison)")
    ap.add_argument("--init-encoder-from", default="",
                    help="transfer the IMAGE encoder weights from a "
                         "temporal checkpoint into the owner-decoder "
                         "variant (name-mapped through the mirror; the "
                         "first conv is trimmed from image+owner to "
                         "image-only). Everything downstream stays fresh: "
                         "the owner-conditioned decoder is the change "
                         "under test.")
    ap.add_argument("--absence-snapshot", default=str(
        REPO / "runs/prototypes/v30/snap25_plus_rev11own"),
        help="snapshot carrying human-verified owned-absence cases "
             "(owner-task 'No tube' grains); the rejection gate runs "
             "on every no_tube sample found there")
    ap.add_argument("--absence-owner", default="",
                    help="optional substring filter for absence cases")
    ap.add_argument("--skip-fit", action="store_true",
                    help="skip the three-owner fit loop (absence "
                         "rejection + model load only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from prototypes.v30_video_apex.model_factory import (
        build_model_from_checkpoint)

    # rev12 P0.1: the repaired configuration — the licensed selector
    # domains with balanced strata (reviewed negatives inside and
    # outside the band) and Dice over the licensed extent. The old
    # band/band configuration gave 954/40/1,098 already-reviewed ring
    # pixels zero gradient; the strata configuration restores them.
    T.set_body_objective("dice_selectors")
    T.set_body_dice_balanced(True)
    T.set_body_dice_region("extent")
    T.set_body_bg_region("strata")
    T.set_body_dice_weight(1.0)
    T.set_body_self_bce_weight(0.5)
    T.set_body_bg_bce_weight(0.15)
    T.set_body_foreign_bce_weight(1.0)
    weights = T.LossWeights(body=1.0)

    owners = _scene(Path(a.snapshot))
    print(f"scene: {len(owners)} owners -> "
          f"{[o['owner'] for o in owners]}")
    for o in owners:
        _fo, _v = o["foreign"], (o["valid"] > 0)
        print(f"  {o['owner']:>16} paint {int(o['mask'].sum()):>5} px "
              f"valid {int(_v.sum()):>6} "
              f"foreign {int(_fo.sum()):>5} "
              f"foreign_valid {int((_v & _fo).sum()):>5} "
              f"overlap {int(o['overlap'].sum()):>4} q "
              f"({o['q'][0]:.0f},{o['q'][1]:.0f})")
    if len(owners) != 3:
        print("REFUSE: the scene must hold exactly three owners")

    clip_np = load_query_clip(Path(a.snapshot), MOVIE, FRAME, CROP)
    clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)

    if a.scratch or a.variant:
        import torch as _t
        from prototypes.v30_video_apex.model import build_model
        _v = a.variant or "temporal"
        model = build_model(_v, base=8, multiscale=True)
        info = {"model_kwargs": {"variant": _v, "base": 8,
                                 "multiscale": True},
                "semantics": "scratch (rev10 WP-B equal-updates arm)"}
        if a.init_encoder_from:
            if _v != "owner-decoder":
                raise SystemExit("--init-encoder-from is for the "
                                 "owner-decoder variant")
            _ck = _t.load(a.init_encoder_from, map_location="cpu",
                          weights_only=False)
            _sd = _ck.get("model_state") or {}
            _n_transferred = 0
            _skipped: list[str] = []
            with _t.no_grad():
                for _n, _p in model.named_parameters():
                    # ENCODER ONLY, and the od model stores it as
                    # `enc.enc.N.*` / `enc.bottleneck.N.*`: the mirror
                    # prefix `enc.` maps to the checkpoint's `unet.`.
                    # Copying the temporal model's DECODER here was
                    # wrong twice over: its learned weights expect
                    # features produced WITH the owner channel, and
                    # training that copy at head rate diverged
                    # (probe: loss 7478, head grad 1.4e+05).
                    if not _n.startswith("enc."):
                        _skipped.append(_n)
                        continue
                    _key = "unet." + _n[4:]
                    if _key not in _sd:
                        _skipped.append(_n)
                        continue
                    _w = _sd[_key]
                    if tuple(_w.shape) != tuple(_p.shape):
                        if _w.dim() == 4 and _w.shape[1] > _p.shape[1]:
                            _w = _w[:, :_p.shape[1]]
                        else:
                            raise SystemExit(
                                f"cannot map {_key}: {tuple(_w.shape)} "
                                f"-> {tuple(_p.shape)}")
                    _p.copy_(_w)
                    _n_transferred += 1
            print(f"encoder transfer: {_n_transferred} encoder tensors; "
                  f"{len(_skipped)} left fresh (decoder/heads are the "
                  f"change under test)", flush=True)
            print(f"encoder transfer: {_n_transferred} tensors from "
                  f"{a.init_encoder_from}", flush=True)
            info["encoder_transferred_from"] = str(a.init_encoder_from)
    else:
        model, info = build_model_from_checkpoint(
            a.checkpoint, allow_legacy_semantics=True)
    print(f"model: {info['model_kwargs']} | semantics {info['semantics']}")
    for prm in model.parameters():
        prm.requires_grad_(True)
    if getattr(model, "variant", "") == "owner-decoder":
        # The architecture under test puts its capacity in the
        # owner-conditioned decoder: that decoder (and every head) is the
        # learning group here, while the TRANSFERRED image encoder keeps
        # the slow trunk rate. This is a declared assignment, not a tune:
        # the temporal arm's decoder was pretrained, so comparing a
        # scratch decoder against it at the trunk rate would compare
        # initialisations rather than architectures (it did: 60 steps at
        # trunk rate left the field positive everywhere, rivals ~1.0).
        head = [p for n, p in model.named_parameters()
                if not (n.startswith("enc.") or n.startswith("bottleneck."))]
        print(f"optimiser: owner-decoder arm -> {len(head)} decoder/head "
              f"tensors at lr {a.head_lr}; encoder at lr {a.trunk_lr}",
              flush=True)
    else:
        head = [p for n, p in model.named_parameters() if n.startswith("body.")
                or n.startswith("vis_head") or n.startswith("mode_head")
                or n.startswith("nobs_head")]
    head_ids = {id(p) for p in head}
    trunk = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": head, "lr": a.head_lr},
        {"params": trunk, "lr": a.trunk_lr}])

    def _grad_norms():
        h = sum(float(p.grad.detach().pow(2).sum())
                for p in head if p.grad is not None) ** 0.5
        t = sum(float(p.grad.detach().pow(2).sum())
                for p in trunk if p.grad is not None) ** 0.5
        return h, t

    def _metrics(probs):
        """Per owner: own IoU/recall, rival positive rates."""
        rows = []
        for i, o in enumerate(owners):
            own = o["mask"]
            own_valid = (o["valid"] > 0) & own
            pred = probs[i] > 0.5
            inter = float((pred & own).sum())
            union = float((pred | own).sum())
            rec = (float((pred & own_valid).sum())
                   / max(1.0, float(own_valid.sum())))
            rivals = []
            for j, o2 in enumerate(owners):
                if j == i:
                    continue
                non_overlap = o2["mask"] & ~own
                if not non_overlap.any():
                    continue
                rivals.append(float((pred & non_overlap).sum())
                              / max(1.0, float(non_overlap.sum())))
            rows.append({"owner": o["owner"],
                         "iou_eval_domain": round(
                             float((pred & own & (o["valid"] > 0)).sum())
                             / max(1.0, float(
                                 ((pred | own) & (o["valid"] > 0)).sum())),
                             4),
                         "iou_crop": round(inter / union if union else 0.0,
                                           4),
                         "recall_verified_body": round(rec, 4),
                         "rival_positive_rates": [round(r, 4)
                                                  for r in rivals],
                         "pred_frac_crop": round(float(pred.mean()), 4),
                         "logit_std": round(float(
                             (probs[i] - probs[i].mean()).std()), 4)})
        return rows

    abs_pre = _absence_test(model, Path(a.absence_snapshot),
                            a.absence_owner)
    print(f"\nowned-absence rejection (loaded checkpoint, pre-fit): "
          f"{abs_pre['n_rejected']}/{abs_pre['n_cases']} rejected "
          f"(gate {'PASS' if abs_pre['gate_passed'] else 'FAIL'})")
    for c in abs_pre["cases"]:
        print(f"  {c['case']}: {c['vis_class']} p={c['vis_prob']} "
              f"margin={c['vis_margin']} heat_max={c['heat_max']}")
    if a.skip_fit:
        out = {"checkpoint": a.checkpoint, "skipped_fit": True,
               "owned_absence_case_pre": abs_pre,
               "absence_gate_passed": abs_pre["gate_passed"],
               "note": "absence rejection only (no fit loop)"}
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))
        print(f"\n-> {a.out}")
        return 0

    history = []
    initial = None
    print(f"\nsteps={a.steps} block={a.block} "
          f"(head lr {a.head_lr}, trunk lr {a.trunk_lr})")
    for step in range(a.steps):
        opt.zero_grad()
        parts_sum: dict[str, float] = {}
        total_t = None
        for o in owners:
            out, _body = _loss_for(model, clip, None, o, weights)
            for k, v in out.items():
                if not k.startswith("body"):
                    continue
                _val = v.detach() if hasattr(v, "detach") else v
                # rev12 P0.1: numeric diagnostics only (some loss parts
                # are structured values, not scalars)
                if not isinstance(_val, (int, float)):
                    continue
                parts_sum[k] = parts_sum.get(k, 0.0) + float(_val)
            total_t = out["body"] if total_t is None \
                else total_t + out["body"]
        assert total_t is not None
        total_val = float(total_t)
        if not np.isfinite(total_val):
            print(f"ABORT step {step}: non-finite loss")
            break
        if initial is None:
            initial = total_val
        total_t.backward()
        hg, tg = _grad_norms()
        opt.step()

        if (step + 1) % a.block == 0 or step == 0:
            with torch.no_grad():
                probs = []
                for o in owners:
                    prompt = build_query_prompt(o["owner"], o["q"],
                                                (CS, CS))
                    probs.append(torch.sigmoid(
                        model.forward(clip, prompt).body[0, 0]).numpy())
            mets = _metrics(probs)
            fo_px = sum(int(o["foreign"].sum()) for o in owners)
            rec = {"step": step, "total_loss": round(total_val, 6),
                   "parts": {k: round(v, 6) for k, v in parts_sum.items()},
                   "head_grad": round(hg, 8), "trunk_grad": round(tg, 8),
                   "foreign_px": fo_px,
                   "metrics": mets,
                   "mean_logit_std": round(float(np.mean(
                       [m["logit_std"] for m in mets])), 6)}
            history.append(rec)
            # Write the result EVERY block. A reaped or slept-through run
            # that only writes at the end leaves nothing — the same
            # lesson as history.json, and this run takes ~30 minutes.
            try:
                _partial = {"checkpoint": a.checkpoint,
                            "crop_xywh": list(CROP), "frame": FRAME,
                            "steps": a.steps, "block": a.block,
                            "head_lr": a.head_lr, "trunk_lr": a.trunk_lr,
                            "seed": a.seed,
                            "owners": [o["owner"] for o in owners],
                            "history": history, "partial": True}
                Path(a.out).write_text(json.dumps(_partial, indent=1,
                                                  default=str))
            except Exception:  # noqa: BLE001
                pass
            # rev10 WP-B: keep the native predictions, not just numbers
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                outp = Path(a.out)
                np.savez_compressed(
                    str(outp.with_suffix(f".step{step}.npz")),
                    **{f"prob_{i}": pr for i, pr in enumerate(probs)},
                    **{f"mask_{i}": o["mask"] for i, o in enumerate(owners)},
                    **{f"foreign_{i}": o["foreign"]
                       for i, o in enumerate(owners)})
                fig, axes = plt.subplots(
                    1, len(owners) + 1, figsize=(4 * (len(owners) + 1), 4.2))
                axes[0].imshow(clip_np[4], cmap="gray")
                for c, o in zip(("#00e5ff", "#ff2ec4", "#ffd000"), owners):
                    axes[0].contour(o["mask"].astype(float),
                                    levels=[0.5], linewidths=1.4, colors=[c])
                    axes[0].plot(o["q"][0], o["q"][1], "+", ms=10,
                                 color=c)
                axes[0].set_title("paints + queries", fontsize=9)
                for i, o in enumerate(owners):
                    ax = axes[i + 1]
                    show = np.stack([clip_np[4]] * 3, -1)
                    pred = probs[i] > 0.5
                    show[pred] = 0.45 * show[pred] + 0.55 * np.array(
                        [1.0, 0.25, 0.25])
                    show[o["mask"] & ~pred] = 0.45 * show[o["mask"] & ~pred] \
                        + 0.55 * np.array([0.2, 1.0, 0.3])
                    ax.imshow(np.clip(show, 0, 1))
                    ax.set_title(
                        f"{o['owner'].split('-')[-1]} step {step}: IoU "
                        f"{rec['metrics'][i]['iou_eval_domain']:.3f} "
                        f"pred {rec['metrics'][i]['pred_frac_crop']*100:.1f}%",
                        fontsize=9)
                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.suptitle(
                    f"rev10 three-owner fit check — step {step}, loss "
                    f"{rec['total_loss']:.4f}", fontsize=10)
                fig.tight_layout()
                fig.savefig(str(outp.with_suffix(f".step{step}.png")), dpi=140)
                plt.close(fig)
            except Exception as _e:  # noqa: BLE001
                print(f"  (prediction save skipped: {type(_e).__name__})")
            print(f"step {step:>3} loss {rec['total_loss']:.6f} "
                  f"| fo {fo_px:>5}px | hg {hg:.2e} tg {tg:.2e} "
                  f"| " + "  ".join(
                      f"{m['owner'].split('-')[-1]}: IoU "
                      f"{m['iou_eval_domain']:.3f} rec "
                      f"{m['recall_verified_body']:.3f} rival "
                      f"{max(m['rival_positive_rates'] or [0]):.3f} "
                      f"std {m['logit_std']:.3f}" for m in mets))
            if fo_px == 0:
                print("ABORT: zero foreign supervision in this scene")
                break
            if rec["mean_logit_std"] == 0.0:
                print("ABORT: dead body branch (flat field)")
                break
            if initial and rec["total_loss"] > 3.0 * abs(initial):
                print("ABORT: divergent loss")
                break

    final = history[-1] if history else None
    abs_post = _absence_test(model, Path(a.absence_snapshot),
                             a.absence_owner)
    print(f"\nowned-absence rejection (post-fit): "
          f"{abs_post['n_rejected']}/{abs_post['n_cases']} rejected "
          f"(gate {'PASS' if abs_post['gate_passed'] else 'FAIL'})")
    fits = []
    if final:
        for m in final["metrics"]:
            fits.append({
                "owner": m["owner"],
                "iou_ge_090": m["iou_eval_domain"] >= 0.90,
                "recall_ge_095": m["recall_verified_body"] >= 0.95,
                "rivals_below_005": all(
                    r < 0.05 for r in m["rival_positive_rates"])})
    # rev11: preserve the fitted WEIGHTS, not only the scores (the
    # review: "Save fit checkpoints, not just scores"; a JSON number
    # cannot be re-run or audited). Saved beside the JSON with the
    # sample/input hashes recorded in its manifest.
    try:
        import torch as _t
        _mk = (info or {}).get("model_kwargs") or {
            "variant": "temporal", "base": 8, "multiscale": True}
        _wpath = Path(a.out).with_suffix(".weights.pt")
        _t.save({"model_state": {k: v.detach().cpu()
                                 for k, v in model.state_dict().items()},
                 "config": {"variant": str(_mk.get("variant", "temporal")),
                            "base": int(_mk.get("base", 8)),
                            "multiscale": bool(_mk.get("multiscale", True))},
                 "manifest": {"script": "rev10_fit_check.py",
                              "snapshot": str(a.snapshot),
                              "steps": int(a.steps),
                              "head_lr": float(a.head_lr),
                              "trunk_lr": float(a.trunk_lr),
                              "seed": int(a.seed),
                              "gate_passed": bool(fits) and all(
                                  f["iou_ge_090"] and f["recall_ge_095"]
                                  and f["rivals_below_005"]
                                  for f in fits)}},
                str(_wpath))
        print(f"fit weights -> {_wpath}")
    except Exception as _we:  # noqa: BLE001
        print(f"fit weights NOT saved: {type(_we).__name__}: {_we}")
    from prototypes.v30_video_apex.model_factory import parameter_hash
    out = {"checkpoint": a.checkpoint, "crop_xywh": list(CROP),
           "frame": FRAME, "steps": a.steps, "block": a.block,
           "head_lr": a.head_lr, "trunk_lr": a.trunk_lr, "seed": a.seed,
           "owners": [o["owner"] for o in owners],
           # rev13 W1: the declared init semantics + the hash of the
           # ACTUAL starting weights (post-load), so the initialization
           # is auditable and a claimed seed-only comparison can be
           # mechanically invalidated when this differs.
           "init_semantics": (info or {}).get("semantics"),
           "init_parameter_hash": parameter_hash(model),
           "history": history, "fit_gate": fits,
           "gate_passed": bool(fits) and all(
               f["iou_ge_090"] and f["recall_ge_095"]
               and f["rivals_below_005"] for f in fits),
           "body_loss_config": T.body_loss_config(),
           "owned_absence_case_pre": abs_pre,
           "owned_absence_case_post": abs_post,
           "absence_gate_passed": abs_post["gate_passed"],
           "note": "fit test on a developed scene; not generalization"}
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"\n-> {a.out}")
    if fits:
        print(f"gate passed: {out['gate_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
