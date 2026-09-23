"""One inference path for owner queries (evaluator and fit report).

`eval_whole_instance.py` and `fit_criteria.py` disagreed on the SAME
checkpoint, crop and queries (diag 0.224 vs 0.007) because each carried
its own copy of "build the clip, build the prompt, threshold". Copies
drift; the reviewer's whole theme is that a contract must live once.
Both scripts now call this module, so a number can only be wrong once.

rev11: the prompt contract is TYPED. Every consumer (training,
evaluation, runtime) constructs prompts through `build_typed_prompt`,
whose inputs carry declared types — human-confirmed grain center,
detected grain center, annotation focus, declared query point, tube
attachment, first-visible proximal point — and whose provenance records
which one was actually used. A focus/gaze point is never labelled
`auto-grain`; only actual detector output is.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

GRAIN_R_NOMINAL = 16.0

# The declared prompt-geometry types, in priority order. The FIRST one
# an input actually provides is what the prompt renders, and its name is
# what the provenance records.
PROMPT_GEOMETRY_SOURCES = ("human-grain", "auto-grain", "query-declared",
                           "focus", "none")


def grain_disc_mask(h: int, w: int, center_xy, radius_px) -> np.ndarray:
    """A grain disc in CROP coordinates — the one disc renderer."""
    yy, xx = np.mgrid[0:int(h), 0:int(w)].astype(np.float32)
    return (((xx - float(center_xy[0])) ** 2 +
             (yy - float(center_xy[1])) ** 2)
            <= float(radius_px) ** 2).astype(np.float32)


def build_typed_prompt(owner_key: str, *, crop_wh,
                       crop_origin=(0.0, 0.0),
                       human_grain_xy=None, human_grain_radius_px=None,
                       detected_grain_xy=None, detected_radius_px=None,
                       query_xy=None, focus_xy=None,
                       attachment_xy=None, proximal_xy=None,
                       grain_radius_px: float = GRAIN_R_NOMINAL):
    """One construction path for the owner prompt across consumers.

    All center/geometry inputs are in NATIVE frame coordinates and are
    converted to crop coordinates here, once (log the transform, never
    re-derive it per consumer). The chosen provenance is the first
    available of PROMPT_GEOMETRY_SOURCES; attachment/proximal geometry
    is kept only for a "human-grain" or "human" rendering.

    Returns (OwnerPrompt, provenance_dict). The dict records every
    input it was given, the chosen center in both frames, the radius and
    the coordinate transform — an audit can separate a detected center
    from a gaze point without guessing.
    """
    from prototypes.v30_video_apex.model import build_owner_prompt

    ox, oy = float(crop_origin[0]), float(crop_origin[1])
    w, h = int(crop_wh[0]), int(crop_wh[1])
    prov = {"owner": str(owner_key), "crop_origin": [ox, oy],
            "crop_wh": [w, h],
            "transform": "native -> crop: subtract the origin; "
                         "no resampling",
            "inputs": {"human_grain_xy": _xy(human_grain_xy),
                       "detected_grain_xy": _xy(detected_grain_xy),
                       "query_xy": _xy(query_xy),
                       "focus_xy": _xy(focus_xy),
                       "attachment_xy": _xy(attachment_xy),
                       "proximal_n": (len(proximal_xy)
                                      if proximal_xy else 0)}}

    chosen = None
    kind = "none"
    radius = float(grain_radius_px)
    for src, xy, r in (("human-grain", human_grain_xy,
                        human_grain_radius_px),
                       ("auto-grain", detected_grain_xy,
                        detected_radius_px),
                       ("query-declared", query_xy, None),
                       ("focus", focus_xy, None)):
        if xy is not None and len(xy) == 2:
            kind = src
            chosen = (float(xy[0]), float(xy[1]))
            if r is not None:
                radius = float(r)
            break
    if kind == "none" and (attachment_xy is not None or proximal_xy):
        # human-geometry-only rendering (attachment + proximal path,
        # no grain disc) — the historical "human" arm, kept exactly.
        kind = "human"
    gm = None
    if chosen is not None:
        gm = grain_disc_mask(h, w, (chosen[0] - ox, chosen[1] - oy), radius)
        prov["center_native"] = [chosen[0], chosen[1]]
        prov["center_crop"] = [chosen[0] - ox, chosen[1] - oy]
        prov["radius_px"] = radius
    prov["prompt_kind"] = kind
    if kind == "none":
        prov["note"] = ("no declared prompt geometry: honest zeros "
                        "(the old behaviour exactly)")
    attach_c = prox_c = None
    if kind in ("human", "human-grain"):
        if attachment_xy is not None and len(attachment_xy) == 2:
            attach_c = (float(attachment_xy[0]) - ox,
                        float(attachment_xy[1]) - oy)
        if proximal_xy:
            prox_c = [(float(q[0]) - ox, float(q[1]) - oy)
                      for q in proximal_xy]
    prompt = build_owner_prompt(str(owner_key), grain_mask=gm,
                                attachment_xy=attach_c,
                                proximal_xy=prox_c, provenance=kind)
    return prompt, prov


def _xy(v):
    return [float(v[0]), float(v[1])] if v is not None and len(v) == 2 \
        else None


def load_query_clip(snapshot: Path, movie_key: str, frame: int, crop):
    """The 9-frame clip the trainer feeds, in [T, H, W]."""
    from prototypes.v30_video_apex.dataset import QUERY_OFFSETS
    from prototypes.v30_video_apex.targets import load_clip_pixels
    from tubetracker.annotation_frames import FrameReader
    man = json.loads((Path(snapshot) / "snapshot_manifest.json").read_text())
    movies = {k: v.get("path") for k, v in (man.get("movies") or {}).items()}
    reader = FrameReader(movies[movie_key])
    frames = tuple(max(0, frame + o) for o in QUERY_OFFSETS)
    missing = tuple(1 if frame + o < 0 else 0 for o in QUERY_OFFSETS)
    try:
        return load_clip_pixels(reader, frames, crop, missing)
    finally:
        reader.close()


def build_query_prompt(owner_key: str, query_xy, crop_wh, threshold_note=None):
    """Grain disc at query_xy — the evaluator's DECLARED query point.

    rev11: typed `query-declared`, never `auto-grain` (only detector
    output is auto-grain). Coordinates are crop-frame; the typed
    constructor is called with a zero origin so the transform is the
    identity and is recorded as such.
    """
    prompt, _prov = build_typed_prompt(
        str(owner_key).split("|")[-1],
        crop_wh=(int(crop_wh[0]), int(crop_wh[1])),
        crop_origin=(0.0, 0.0), query_xy=query_xy)
    return prompt


def predict_queries(model, clip_np, owners_queries: dict, threshold: float) -> dict:
    """Return {owner: (prob, pred_bool)} through ONE forward per owner.

    clip_np is [T, H, W]; the model takes [B, T, C, H, W].
    """
    # rev10 WP-A: follow the model's device. These helpers used to build
    # CPU tensors unconditionally, which a CUDA/MPS checkpoint path would
    # reject; the canary (scripts/rev10_device_canary.py) verifies parity.
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    clip = torch.from_numpy(np.asarray(clip_np)).unsqueeze(0).unsqueeze(2)
    clip = clip.to(dev)
    h, w = clip_np.shape[1], clip_np.shape[2]
    out = {}
    model.eval()
    for owner, qxy in owners_queries.items():
        prompt = build_query_prompt(owner, qxy, (w, h))
        with torch.no_grad():
            body = model.forward(clip, prompt).body[0, 0]
        body = body.detach().float().cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-body))
        out[owner] = (prob, prob > float(threshold))
    return out
