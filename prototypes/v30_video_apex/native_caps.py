"""Native spatial cap detector with an explicit current-image branch.

The detector sees image pixels only. Grain ownership, route completeness
and grain absence belong to the movie solver, not this appearance head.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cap_evidence import emit_cap_candidates

SCHEMA = "tubetracker.native_caps.v1"
LOCAL_SCHEMA = "tubetracker.native_caps.v2"
OFFSETS = (-2, 0, 2)
QUERY_INDEX = 1
TILE = 128
VALID_MARGIN = 16


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PixelChannelNorm(nn.Module):
    """Normalize channels at each pixel; never mix statistics across a crop."""
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def _block(n_in, n_out, normalization="spatial"):
    norm = lambda: (nn.GroupNorm(4, n_out) if normalization == "spatial"
                    else PixelChannelNorm(n_out))
    return nn.Sequential(nn.Conv2d(n_in, n_out, 3, padding=1),
                         norm(), nn.LeakyReLU(0.1),
                         nn.Conv2d(n_out, n_out, 3, padding=1),
                         norm(), nn.LeakyReLU(0.1))


class _SpatialFeatures(nn.Module):
    def __init__(self, base, normalization="spatial"):
        super().__init__()
        self.enc1 = _block(1, base, normalization)
        self.enc2 = _block(base, base * 2, normalization)
        self.middle = _block(base * 2, base * 4, normalization)
        self.dec2 = _block(base * 6, base * 2, normalization)
        self.dec1 = _block(base * 3, base, normalization)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.middle(F.avg_pool2d(b, 2))
        d = self.dec2(torch.cat([b, F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False)], 1))
        return self.dec1(torch.cat([a, F.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False)], 1))


class NativeCapNet(nn.Module):
    def __init__(self, base=12, temporal=False, normalization="spatial", image_gain=1.):
        super().__init__()
        if normalization not in ("spatial", "pixel"):
            raise ValueError("normalization must be spatial or pixel")
        self.base, self.temporal = int(base), bool(temporal)
        self.normalization = normalization
        self.image_gain = float(image_gain)
        if not np.isfinite(self.image_gain) or self.image_gain <= 0:
            raise ValueError("image gain must be positive and finite")
        if normalization == "spatial" and self.image_gain != 1:
            raise ValueError("legacy spatial checkpoints keep their original preprocessing")
        # 32 px exceeds the encoder/decoder's finite convolutional radius.
        # Pixel normalization makes that radius meaningful; GroupNorm does not.
        self.valid_margin = 32 if normalization == "pixel" else VALID_MARGIN
        self.pooling_lattice = 4 if normalization == "pixel" else 1
        self.features = _SpatialFeatures(self.base, normalization)
        self.context_gain = nn.Parameter(torch.zeros(()))
        self.cap = nn.Conv2d(self.base, 1, 1)
        self.offset = nn.Conv2d(self.base, 2, 1)
        self.logvar = nn.Conv2d(self.base, 1, 1)

    def forward(self, clip, query_index=QUERY_INDEX):
        if clip.ndim != 5 or clip.shape[2] != 1 or not 0 <= query_index < clip.shape[1]:
            raise ValueError("native cap input is B,T,1,H,W with an explicit valid query")
        if self.image_gain != 1:
            clip = (clip - .5) * self.image_gain + .5
        current = self.features(clip[:, query_index])
        if self.temporal:
            others = [self.features(clip[:, i]) for i in range(clip.shape[1]) if i != query_index]
            if others:
                context = torch.stack(others).mean(0) - current
                current = current + torch.tanh(self.context_gain) * context
        return {"logits": self.cap(current),
                "offset_xy": 8.0 * torch.tanh(self.offset(current)),
                "logvar": self.logvar(current).clamp(-2, 5)}

    def config(self):
        result = {"base": self.base, "temporal": self.temporal,
                "input_offsets": list(OFFSETS), "query_index": QUERY_INDEX,
                "consumed_offsets": list(OFFSETS) if self.temporal else [0],
                "tile_px": TILE, "valid_margin_px": self.valid_margin,
                "offset_basis": "native_xy", "offset_bound_px_per_axis": 8,
                "preprocessing": "native grayscale uint8 / 255; no spatial rescaling",
                "ownership_conditioning": False}
        if self.normalization == "pixel":
            result.update(normalization="per_pixel_channels", pooling_lattice_px=4,
                          peak_selection="local_max_7_then_native_nms_6")
        if self.image_gain != 1:
            result['image_contrast_gain_about_midgray'] = self.image_gain
        return result


def native_cap_loss(pred, positive, negative, offset_target, *, hard_negative_weight=0., hard_negative_k=32):
    """Equal class loss mass; all unknown pixels have exactly zero loss."""
    logits = pred["logits"]
    positive, negative = positive.bool(), negative.bool()
    if bool((positive & negative).any()):
        raise ValueError("positive and negative cap domains overlap")
    if not math.isfinite(hard_negative_weight) or hard_negative_weight < 0 or hard_negative_k < 1:
        raise ValueError('hard-negative weight must be finite/nonnegative and k positive')
    zero = logits.sum() * 0.0
    pos = F.softplus(-logits)[positive].mean() if bool(positive.any()) else zero
    neg = F.softplus(logits)[negative].mean() if bool(negative.any()) else zero
    hard = []
    if hard_negative_weight:
        for values, mask in zip(logits, negative):
            licensed = F.softplus(values)[mask]
            if licensed.numel():
                hard.append(licensed.topk(min(hard_negative_k, licensed.numel())).values.mean())
    hard_neg = torch.stack(hard).mean() if hard else zero
    mask2 = positive.expand(-1, 2, -1, -1)
    reg = F.smooth_l1_loss(pred["offset_xy"][mask2], offset_target[mask2]) if bool(positive.any()) else zero
    square = ((pred["offset_xy"] - offset_target) ** 2).mean(1, keepdim=True)
    nll = (0.5 * (torch.exp(-pred["logvar"]) * square + pred["logvar"]))[positive].mean() if bool(positive.any()) else zero
    total = pos + neg + 0.25 * reg + 0.01 * nll + hard_negative_weight * hard_neg
    return total, {"positive_bce": float(pos.detach()), "negative_bce": float(neg.detach()),
                   "hard_negative_bce": float(hard_neg.detach()),
                   "offset_l1": float(reg.detach()), "offset_nll": float(nll.detach()),
                   "positive_mass": int(positive.sum()), "negative_mass": int(negative.sum())}


def save_native_checkpoint(path, model, *, manifest):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = LOCAL_SCHEMA if model.normalization == "pixel" else SCHEMA
    torch.save({"schema": schema, "config": model.config(),
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "manifest": manifest}, path)
    return str(path)


def load_native_checkpoint(path, device="cpu"):
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("schema") not in (SCHEMA, LOCAL_SCHEMA):
        raise ValueError("checkpoint is not a native spatial cap detector")
    cfg = p["config"]
    model = NativeCapNet(base=cfg["base"], temporal=cfg["temporal"],
                        normalization="pixel" if p["schema"] == LOCAL_SCHEMA else "spatial",
                        image_gain=cfg.get('image_contrast_gain_about_midgray', 1.))
    if cfg != model.config():
        raise ValueError("native detector preprocessing/geometry contract mismatch")
    model.load_state_dict(p["model_state"], strict=True)
    model.to(device).eval()
    return model, {"checkpoint": str(Path(path).resolve()), "sha256": file_hash(path),
                   "config": cfg, "manifest": p.get("manifest", {})}


def extract_tile(frames, origin, size=TILE):
    """Image padding carries no validity claim; coordinates remain native."""
    import cv2
    ox, oy = map(int, origin)
    out = []
    for frame in frames:
        h, w = frame.shape[-2:]
        left, top = max(0, -ox), max(0, -oy)
        right, bottom = max(0, ox + size - w), max(0, oy + size - h)
        f = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        out.append(f[oy + top:oy + top + size, ox + left:ox + left + size])
    return np.stack(out)


def score_native_tile(model, clip_uint8, *, origin=(0, 0), image_size=None,
                      movie="", source_frame=0, threshold=0.5, emit=True):
    """The same scorer/emitter is used by fit diagnostics and deployment."""
    clip = np.asarray(clip_uint8)
    if clip.shape != (len(OFFSETS), TILE, TILE):
        raise ValueError(f"expected native tile {(len(OFFSETS), TILE, TILE)}, got {clip.shape}")
    device = next(model.parameters()).device
    tensor = torch.from_numpy(np.ascontiguousarray(clip)).to(device=device, dtype=torch.float32)[None, :, None] / 255.0
    with torch.no_grad():
        pred = model(tensor)
    prob = torch.sigmoid(pred["logits"])[0, 0].cpu().numpy()
    delta = pred["offset_xy"][0].permute(1, 2, 0).cpu().numpy()
    variance = pred["logvar"][0, 0].cpu().numpy()
    y, x = np.mgrid[:TILE, :TILE]
    loc = np.stack([x + origin[0], y + origin[1]], -1)
    margin = model.valid_margin
    valid = ((x >= margin) & (x < TILE - margin)
             & (y >= margin) & (y < TILE - margin))
    bounds = None
    if image_size:
        w, h = image_size
        bounds = (0, 0, w, h)
        valid &= ((loc[..., 0] >= margin) & (loc[..., 0] < w - margin)
                  & (loc[..., 1] >= margin) & (loc[..., 1] < h - margin))
    emit_valid = valid
    if model.normalization == "pixel":
        # Compare against the real neighboring logits, including context
        # outside the valid interior. The crop edge cannot create a maximum.
        import cv2
        emit_valid = valid & (prob >= cv2.dilate(prob, np.ones((7, 7), np.uint8)))
    caps = emit_cap_candidates(prob, loc, emit_valid, offsets_xy=delta, bounds_xyxy=bounds,
                               threshold=threshold, movie=movie, source_frame=source_frame,
                               log_variance=variance) if emit else []
    for cap in caps:
        cap["tile_origin_xy"] = list(map(int, origin))
        cap["unrefined_xy"] = list(cap["location_xy"])
        cap["tile_valid_margin_px"] = margin
        cap["normalization"] = model.normalization
    return {"caps": caps, "probability": prob, "offset_xy": delta,
            "valid": valid, "logvar": variance,
            "input_sha256": hashlib.sha256(np.ascontiguousarray(clip).tobytes()).hexdigest()}


def detect_caps(model, frames, *, movie, source_frame, roi=None, threshold=0.5,
                tile_phase_xy=(0, 0)):
    """Tile one frame's declared context; merge once before querying owners."""
    h, w = frames[QUERY_INDEX].shape
    x0, y0, x1, y1 = map(int, roi or (0, 0, w, h))
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("ROI must be a nonempty native rectangle within the image")
    rows, hashes, origins = [], [], []
    margin, lattice = model.valid_margin, model.pooling_lattice
    stride = TILE - 2 * margin
    if len(tile_phase_xy) != 2 or any(int(v) != v or not 0 <= v < stride for v in tile_phase_xy):
        raise ValueError("tile phase must be two integer offsets within one stride")
    px, py = map(int, tile_phase_xy)
    if model.normalization == "pixel":
        return _detect_local_caps(model, frames, movie=movie, source_frame=source_frame,
                                  roi=(x0, y0, x1, y1), threshold=threshold, phase=(px, py))
    start_x = (x0 - margin - px) // lattice * lattice
    start_y = (y0 - margin - py) // lattice * lattice
    for oy in range(start_y, max(start_y + 1, y1 - margin), stride):
        for ox in range(start_x, max(start_x + 1, x1 - margin), stride):
            clip = extract_tile(frames, (ox, oy))
            out = score_native_tile(model, clip, origin=(ox, oy), image_size=(w, h),
                                    movie=movie, source_frame=source_frame, threshold=threshold)
            rows.extend(out["caps"])
            hashes.append(out["input_sha256"])
            origins.append([ox, oy])
    diagnostics = {"tile_origins_xy": origins, "requested_tile_phase_xy": [px, py],
                   "tile_stride_px": stride, "pooling_lattice_px": lattice,
                   "valid_margin_px": margin, "n_tile_proposals": len(rows),
                   "normalization": model.normalization,
                   "unsupported_native_border_px": margin}
    if not rows:
        return {"caps": [], "tile_hashes": hashes, "tiling": diagnostics}
    caps = emit_cap_candidates([r["probability"] for r in rows], [r["tip_xy"] for r in rows],
                                np.ones(len(rows), bool), bounds_xyxy=(x0, y0, x1, y1),
                                threshold=threshold, movie=movie, source_frame=source_frame)
    for row in caps:
        source = rows[row["sample_index"]]
        row["uncertainty_px"] = source.get("uncertainty_px")
        row["uncertainty_calibrated"] = False
        row["tile_origin_xy"] = source["tile_origin_xy"]
        row["unrefined_xy"] = source["unrefined_xy"]
        row["tile_valid_margin_px"] = margin
    return {"caps": caps, "tile_hashes": hashes, "tiling": diagnostics}


def _detect_local_caps(model, frames, *, movie, source_frame, roi, threshold, phase):
    """Stitch raw valid interiors before one peak/refinement/NMS operation."""
    import cv2
    h, w = frames[QUERY_INDEX].shape
    x0, y0, x1, y1 = roi
    # Peak selection also sees pixels just outside the requested ROI.
    dx0, dy0, dx1, dy1 = max(0, x0-3), max(0, y0-3), min(w, x1+3), min(h, y1+3)
    shape = dy1-dy0, dx1-dx0
    probability = np.zeros(shape, np.float32)
    offset = np.zeros((*shape, 2), np.float32)
    variance = np.zeros(shape, np.float32)
    coverage = np.zeros(shape, bool)
    source_origin = np.zeros((*shape, 2), np.int32)
    margin, lattice = model.valid_margin, model.pooling_lattice
    stride = TILE - 2*margin
    sx = (dx0 - margin - phase[0]) // lattice * lattice
    sy = (dy0 - margin - phase[1]) // lattice * lattice
    hashes, origins = [], []
    for oy in range(sy, max(sy+1, dy1-margin), stride):
        for ox in range(sx, max(sx+1, dx1-margin), stride):
            tile = score_native_tile(model, extract_tile(frames, (ox, oy)),
                                     origin=(ox, oy), image_size=(w, h), emit=False)
            lx, ty = max(dx0, ox+margin), max(dy0, oy+margin)
            rx, by = min(dx1, ox+TILE-margin), min(dy1, oy+TILE-margin)
            if lx >= rx or ty >= by:
                continue
            target = np.s_[ty-dy0:by-dy0, lx-dx0:rx-dx0]
            local = np.s_[ty-oy:by-oy, lx-ox:rx-ox]
            probability[target] = tile['probability'][local]
            offset[target] = tile['offset_xy'][local]
            variance[target] = tile['logvar'][local]
            coverage[target] = tile['valid'][local]
            source_origin[target] = (ox, oy)
            hashes.append(tile['input_sha256']); origins.append([ox, oy])
    yy, xx = np.mgrid[dy0:dy1, dx0:dx1]
    roi_mask = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    eligible = roi_mask & (xx >= margin) & (xx < w-margin) & (yy >= margin) & (yy < h-margin)
    if np.any(eligible & ~coverage):
        raise RuntimeError('Native cap tiling left a hole in the supported ROI')
    peak = probability >= cv2.dilate(probability, np.ones((7, 7), np.uint8))
    valid = coverage & peak & roi_mask
    caps = emit_cap_candidates(probability, np.stack([xx, yy], -1), valid,
        offsets_xy=offset, bounds_xyxy=roi, threshold=threshold,
        movie=movie, source_frame=source_frame, log_variance=variance)
    for cap in caps:
        y, x = divmod(cap['sample_index'], shape[1])
        cap.update(tile_origin_xy=source_origin[y, x].tolist(),
                   unrefined_xy=list(cap['location_xy']), tile_valid_margin_px=margin,
                   uncertainty_calibrated=False)
    return {'caps': caps, 'tile_hashes': hashes,
            'tiling': {'tile_origins_xy': origins, 'requested_tile_phase_xy': list(phase),
                       'tile_stride_px': stride, 'pooling_lattice_px': lattice,
                       'valid_margin_px': margin, 'normalization': model.normalization,
                       'merge': 'raw valid interiors before global peaks and native NMS',
                       'covered_roi_pixels': int((coverage & roi_mask).sum()),
                       'eligible_roi_pixels': int(eligible.sum()),
                       'unsupported_native_border_px': margin,
                       'n_raw_threshold_pixels': int((coverage & roi_mask & (probability >= threshold)).sum()),
                       'n_peaks_before_global_nms': int((valid & (probability >= threshold)).sum())}}
