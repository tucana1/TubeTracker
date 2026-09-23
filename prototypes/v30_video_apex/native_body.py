"""Current-image, grain-conditioned body evidence in native coordinates."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .native_caps import _block, extract_tile, file_hash

SCHEMA = "tubetracker.native_owned_body.v3"
TILE = 288
VALID_MARGIN = 64
POOLING_LATTICE = 8


def body_input(image, origin, grain, radius=13., image_gain=1., *,
               with_root=False, root_xy=None):
    """Native crop channels for the owned-body query.

    rev14 P5 fallback: `with_root` appends an explicit root/exit channel
    (a Gaussian at the prompted root). An unknown root is a ZERO channel,
    so the unknown-root path stays explicit and never invents geometry.
    """
    image = np.asarray(image)
    if image.ndim != 2 or len(grain) != 2 or not np.isfinite(grain).all() or radius <= 0:
        raise ValueError("body evidence requires a finite native grain query")
    y, x = np.mgrid[:image.shape[0], :image.shape[1]]
    dx, dy = x + origin[0] - grain[0], y + origin[1] - grain[1]
    disc = np.exp(-(dx*dx + dy*dy)/(2*float(radius)**2))
    gray = (image.astype(np.float32)/255 - .5)*float(image_gain) + .5
    channels = [gray, disc, np.clip(dx/128., -2, 2), np.clip(dy/128., -2, 2)]
    if with_root:
        if root_xy is not None and len(root_xy) == 2 and np.isfinite(root_xy).all():
            rx, ry = x + origin[0] - float(root_xy[0]), y + origin[1] - float(root_xy[1])
            channels.append(np.exp(-(rx*rx + ry*ry)/(2*(2.*float(radius))**2)))
        else:
            channels.append(np.zeros_like(disc))
    return np.stack(channels).astype(np.float32)


class SpatialQueryResidual(nn.Module):
    """An initially inert image/query interaction, applied after normalization."""
    def __init__(self, channels, output_channels=None):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels + 3, channels, 1), nn.SiLU(),
                                    nn.Conv2d(channels, output_channels or channels, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, query):
        q = F.interpolate(query, size=features.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat([features, q], 1))


class NativeBodyNet(nn.Module):
    def __init__(self, base=12, image_gain=1., normalization="spatial", root_channel=False,
                 query_adapters=False, presence_head=False):
        super().__init__()
        if normalization not in ("spatial", "pixel"):
            raise ValueError("body normalization must be spatial or pixel")
        if not np.isfinite(image_gain) or image_gain <= 0:
            raise ValueError("body image gain must be positive and finite")
        self.base = int(base)
        self.image_gain = float(image_gain)
        self.normalization = normalization
        self.pooling_lattice = POOLING_LATTICE if normalization == "pixel" else 1
        self.root_channel = bool(root_channel)
        self.query_adapters = bool(query_adapters)
        # rev15 task 3: optional scalar owned-presence head (per-grain
        # emergence crop -> visible-tube logit). Off by default; enabling it
        # adds keys only, never alters the dense pixel path.
        self.presence_head = bool(presence_head)
        self.e1 = _block(4 + int(self.root_channel), base, normalization)
        self.e2 = _block(base, base*2, normalization)
        self.e3 = _block(base*2, base*4, normalization)
        self.e4 = _block(base*4, base*8, normalization)
        self.d3 = _block(base*12, base*4, normalization)
        self.d2 = _block(base*6, base*2, normalization)
        self.d1 = _block(base*3, base, normalization)
        self.body = nn.Conv2d(base, 1, 1)
        if self.presence_head:
            self.presence_fc = nn.Linear(base, 1)
            nn.init.zeros_(self.presence_fc.weight)
            nn.init.zeros_(self.presence_fc.bias)
        if self.query_adapters:
            self.query_d3 = SpatialQueryResidual(base*4)
            self.query_d2 = SpatialQueryResidual(base*2)
            self.query_d1 = SpatialQueryResidual(base)
            self.query_body = SpatialQueryResidual(base, 1)

    @staticmethod
    def decoder_query(x):
        # Native input remains unchanged. The finer, explicitly recorded
        # 16-pixel coordinate scale distinguishes close grains without
        # changing historical stem weights or introducing owner IDs.
        return torch.cat([x[:, 1:2], torch.clamp(x[:, 2:4]*8., -8., 8.)], 1)

    def condition_decoder(self, name, value, query):
        if self.query_adapters:
            value = value + getattr(self, 'query_' + name)(value, query)
        return value

    def _trunk_d1(self, x, block):
        """Shared encoder/decoder trunk through d1 (identical for the dense
        path and the presence head). `block` is the normalization-aware
        stage runner owned by forward."""
        a = block('e1', x)
        b = block('e2', F.avg_pool2d(a, 2))
        c = block('e3', F.avg_pool2d(b, 2))
        d = block('e4', F.avg_pool2d(c, 2))
        up = lambda p, skip: torch.cat([skip, F.interpolate(p, size=skip.shape[-2:],
                                        mode="bilinear", align_corners=False)], 1)
        query = self.decoder_query(x) if self.query_adapters else None
        d3 = self.condition_decoder('d3', block('d3', up(d, c)), query)
        d2 = self.condition_decoder('d2', block('d2', up(d3, b)), query)
        d1 = self.condition_decoder('d1', block('d1', up(d2, a)), query)
        return d1

    def presence_logit(self, x, pool_mask=None):
        """Scalar owned-presence logit for grain-centred emergence crops.

        pool_mask (optional HxW bool, crop coordinates) restricts pooling to
        the emergence neighbourhood: whole-crop pooling lets neighbouring
        tubes dominate, making visible and absent crops inseparable.
        """
        if not self.presence_head:
            raise ValueError('presence head is not enabled')
        d1 = self._trunk_d1(x, lambda name, value: getattr(self, name)(value))
        if pool_mask is None:
            pooled = d1.mean(dim=(2, 3))
        else:
            mask = torch.as_tensor(pool_mask, dtype=torch.bool, device=d1.device)
            if mask.shape != d1.shape[-2:]:
                raise ValueError('presence pool mask must match crop size')
            denom = mask.float().sum().clamp_min(1.0)
            pooled = (d1 * mask.float()).sum(dim=(2, 3)) / denom
        return self.presence_fc(pooled).squeeze(1)

    def forward(self, x, *, capture_stats=None, reference_stats=None):
        if capture_stats is not None and reference_stats is not None:
            raise ValueError('capture and reference normalization are mutually exclusive')
        if capture_stats is not None or reference_stats is not None:
            if self.normalization != 'spatial' or any(m.training for m in self.modules()):
                raise ValueError('reference normalization requires a spatial model in evaluation mode')
            if capture_stats is not None and capture_stats:
                raise ValueError('normalization capture requires an empty destination')
            expected = {n for n, m in self.named_modules() if isinstance(m, nn.GroupNorm)}
            if reference_stats is not None and set(reference_stats) != expected:
                raise ValueError('reference statistics do not match the body model')

        def block(name, value):
            module = getattr(self, name)
            if capture_stats is None and reference_stats is None:
                return module(value)
            for i, layer in enumerate(module):
                if isinstance(layer, nn.GroupNorm):
                    key = f'{name}.{i}'
                    grouped = value.reshape(value.shape[0], layer.num_groups,
                                            value.shape[1] // layer.num_groups, *value.shape[2:])
                    if capture_stats is not None:
                        # Use the actual backend's statistics. A separate
                        # variance reduction accumulated >1e-5 probability
                        # error through this trained encoder/decoder.
                        _, mean, inverse_std = torch.ops.aten.native_group_norm.default(
                            value, layer.weight, layer.bias, value.shape[0], value.shape[1],
                            value.shape[2] * value.shape[3], layer.num_groups, layer.eps)
                        shape = (value.shape[0], layer.num_groups, 1, 1, 1)
                        capture_stats[key] = (mean.detach().reshape(shape), inverse_std.detach().reshape(shape))
                        # Capture must not alter the canonical prediction.
                        value = layer(value)
                    else:
                        mean, inverse_std = reference_stats[key]
                        shape = (value.shape[0], layer.num_groups, 1, 1, 1)
                        if (mean.shape != shape or inverse_std.shape != shape
                                or mean.device != value.device or inverse_std.device != value.device
                                or mean.dtype != value.dtype or inverse_std.dtype != value.dtype):
                            raise ValueError('reference normalization shape/device/dtype mismatch')
                        value = ((grouped - mean) * inverse_std).reshape_as(value)
                        if layer.affine:
                            value = value * layer.weight[None, :, None, None] + layer.bias[None, :, None, None]
                else:
                    value = layer(value)
            return value

        a = self._trunk_d1(x, block)
        output = self.body(a)
        if self.query_adapters:
            output = output + self.query_body(a, self.decoder_query(x))
        return output

    def config(self):
        return {"base": self.base, "crop_px": TILE, "current_image_only": True,
                "image_contrast_gain_about_midgray": self.image_gain,
                "normalization": self.normalization,
                "crop_origin_lattice_px": self.pooling_lattice,
                "tile_valid_margin_px": VALID_MARGIN,
                **({"root_channel": True} if self.root_channel else {}),
                **({"query_adapters": {"version": "decoder_residual_v1",
                    "stages": ["d3", "d2", "d1", "body"],
                    "channels": ["grain_gaussian", "grain_dx/16", "grain_dy/16"],
                    "coordinate_clip": 8., "placement": "after_normalization"}}
                   if self.query_adapters else {}),
                "channels": ["native_gray/255", "grain_gaussian", "grain_dx/128", "grain_dy/128"]
                            + (["root_gaussian"] if self.root_channel else []),
                **({"presence_head": True} if self.presence_head else {}),
                "activation": "logits", "native_spatial_rescaling": False}


def body_loss(logits, positive, ordinary, foreign, *, hard_negative_weight=0.0, hard_negative_ratio=3.0):
    if not np.isfinite(hard_negative_ratio) or hard_negative_ratio <= 0:
        raise ValueError('hard negative ratio must be finite and positive')
    positive, ordinary, foreign = [t.bool() for t in (positive, ordinary, foreign)]
    if bool((positive & ordinary | positive & foreign | ordinary & foreign).any()):
        raise ValueError("body supervision domains must be disjoint")
    zero = logits.sum()*0
    p = F.softplus(-logits)[positive].mean() if positive.any() else zero
    n = F.softplus(logits)[ordinary].mean() if ordinary.any() else zero
    f = F.softplus(logits)[foreign].mean() if foreign.any() else zero
    valid = positive | ordinary | foreign
    prob = torch.sigmoid(logits)
    dice = 1 - (2*(prob*positive).sum()+1)/(prob[valid].sum()+positive.sum()+1)
    # Equal domain means can leave a small, high-confidence grain rim
    # almost unpenalized among tens of thousands of easy background pixels.
    # Mine only the already licensed ordinary-background domain.
    negatives = F.softplus(logits)[ordinary]
    k = min(negatives.numel(), max(64, int(np.ceil(hard_negative_ratio * int(positive.sum())))))
    hard = negatives.topk(k).values.mean() if k else zero
    total = p + n + f + dice + float(hard_negative_weight) * hard
    return total, {"positive_bce": float(p.detach()), "background_bce": float(n.detach()),
                  "foreign_bce": float(f.detach()), "dice_loss": float(dice.detach()),
                  "hard_negative_bce": float(hard.detach()), "hard_negative_pixels": k,
                  "positive_mass": int(positive.sum()), "background_mass": int(ordinary.sum()),
                  "foreign_mass": int(foreign.sum())}


def save_body_checkpoint(path, model, manifest):
    from pathlib import Path
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    torch.save({"schema": SCHEMA, "config": model.config(), "manifest": manifest,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)


def load_body_checkpoint(path, device="cpu"):
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("schema") not in (SCHEMA, "tubetracker.native_owned_body.v1", "tubetracker.native_owned_body.v2"):
        raise ValueError("checkpoint is not a native owned-body model")
    cfg = dict(p["config"])
    if p["schema"] == "tubetracker.native_owned_body.v1":
        cfg["image_contrast_gain_about_midgray"] = 1.
    if p["schema"] != SCHEMA:
        # Historical weights retain their spatial statistics and exact
        # unaligned grain crop. Loading does not reinterpret the model.
        cfg.update(normalization="spatial", crop_origin_lattice_px=1,
                   tile_valid_margin_px=VALID_MARGIN)
    model = NativeBodyNet(cfg["base"], cfg["image_contrast_gain_about_midgray"], cfg["normalization"],
                          root_channel=bool(cfg.get("root_channel", False)),
                          query_adapters=bool(cfg.get("query_adapters", False)),
                          presence_head=bool(cfg.get("presence_head", False)))
    if model.config() != cfg:
        raise ValueError("native body preprocessing/query contract mismatch")
    model.load_state_dict(p["model_state"], strict=True)
    model.to(device).eval()
    return model, {"sha256": file_hash(path), "config": model.config(), "manifest": p["manifest"],
                   "historical_schema": p["schema"]}


def predict_owned_body(model, gray, owner, origin=None, root_xy=None):
    lattice = getattr(model, 'pooling_lattice', 1)
    if origin is None:
        origin = np.floor(np.asarray(owner["grain_native"])-TILE/2).astype(int)
        origin = origin // lattice * lattice
    else:
        origin = np.asarray(origin)
        if (origin.shape != (2,) or not np.isfinite(origin).all()
                or np.any(origin != np.floor(origin)) or np.any(origin % lattice)):
            raise ValueError(f'body tile origin must use the {lattice}-pixel native lattice')
        origin = origin.astype(int)
    tile = extract_tile([gray], origin, TILE)[0]
    root = owner.get("root_native") if root_xy is None else root_xy
    x = body_input(tile, origin, owner["grain_native"], owner.get("grain_radius_px", 13),
                   image_gain=model.image_gain,
                   with_root=getattr(model, "root_channel", False), root_xy=root)
    with torch.no_grad():
        logits = model(torch.from_numpy(x)[None].to(next(model.parameters()).device))
    return torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32), origin.tolist()


def _reference_model_hash(model):
    """Bind reference statistics to weights and effective inference settings."""
    h = hashlib.sha256(json.dumps(model.config(), sort_keys=True).encode())
    for name, module in model.named_modules():
        h.update(repr((name, type(module).__qualname__, module.extra_repr())).encode())
    for name, tensor in sorted(model.state_dict().items()):
        h.update(repr((name, str(tensor.dtype), tuple(tensor.shape), str(tensor.device))).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class OwnedBodyReference:
    """Inference statistics bound to one model, current image and physical grain.

    The image and grain query are copied at construction. There is no API
    for supplying another frame/owner while reusing the captured statistics.
    This is an experimental extension of the trained canonical crop, not a
    certificate that the extended body is biologically correct.
    """
    def __init__(self, model, gray, owner, *, root_xy=None):
        if not isinstance(model, NativeBodyNet) or model.normalization != 'spatial':
            raise ValueError('grain-reference statistics require a native spatial body model')
        if any(m.training for m in model.modules()):
            raise ValueError('grain-reference statistics require evaluation mode')
        self._model = model
        self._gray = np.array(gray, copy=True, order='C')
        if self._gray.ndim != 2 or not np.isfinite(self._gray).all():
            raise ValueError('grain reference requires a finite native grayscale image')
        self._gray.setflags(write=False)
        self._grain = tuple(float(v) for v in owner['grain_native'])
        self._radius = float(owner.get('grain_radius_px', 13))
        root = owner.get('root_native') if root_xy is None else root_xy
        self._root = (tuple(float(v) for v in root)
                      if model.root_channel and root is not None else None)
        if self._root is not None and (len(self._root) != 2 or not np.isfinite(self._root).all()):
            raise ValueError('grain reference requires a finite native root or an unknown root')
        if len(self._grain) != 2 or not np.isfinite(self._grain).all() or not np.isfinite(self._radius) or self._radius <= 0:
            raise ValueError('grain reference requires a finite physical grain and positive radius')
        self._origin = tuple(np.floor(np.asarray(self._grain) - TILE / 2).astype(int).tolist())
        self._phase = tuple(v % POOLING_LATTICE for v in self._origin)
        self._model_hash = _reference_model_hash(model)
        self._stats = {}
        inputs = self._input(self._origin)
        with torch.no_grad():
            model(inputs, capture_stats=self._stats)
        if _reference_model_hash(model) != self._model_hash:
            raise RuntimeError('body model changed during reference capture')
        self._metadata = {
            'kind': 'current_grain_reference', 'reference_origin_xy': list(self._origin),
            'pooling_phase_xy': list(self._phase), 'grain_native': list(self._grain),
            'grain_radius_px': self._radius, 'owner_id': str(owner.get('id', '')),
            'root_channel': bool(model.root_channel),
            'root_native': list(self._root) if self._root is not None else None,
            'image_shape': list(self._gray.shape), 'image_dtype': str(self._gray.dtype),
            'image_sha256': hashlib.sha256(self._gray.tobytes()).hexdigest(),
            'reference_input_sha256': hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest(),
            'model_state_sha256': self._model_hash, 'normalization_layers': len(self._stats),
            'statistics_operator': 'aten.native_group_norm', 'torch_version': str(torch.__version__),
            'probability_calibrated': False}

    @property
    def origin(self):
        return self._origin

    @property
    def phase(self):
        return self._phase

    def metadata(self):
        return json.loads(json.dumps(self._metadata))

    def _input(self, origin):
        tile = extract_tile([self._gray], origin, TILE)[0]
        inputs = body_input(tile, origin, self._grain, self._radius, self._model.image_gain,
                            with_root=self._model.root_channel, root_xy=self._root)
        return torch.from_numpy(inputs)[None].to(next(self._model.parameters()).device)

    def predict(self, origin=None):
        if any(m.training for m in self._model.modules()):
            raise ValueError('grain-reference inference requires evaluation mode')
        if _reference_model_hash(self._model) != self._model_hash:
            raise RuntimeError('body model changed after reference capture')
        origin = np.asarray(self._origin if origin is None else origin)
        if (origin.shape != (2,) or not np.isfinite(origin).all()
                or np.any(origin != np.floor(origin))
                or np.any((origin - self._origin) % POOLING_LATTICE)):
            raise ValueError('reference tiles must preserve the canonical 8-pixel pooling phase')
        origin = origin.astype(int)
        with torch.no_grad():
            logits = self._model(self._input(origin), reference_stats=self._stats)
        return torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32), origin.tolist()


def predict_owned_body_region(model, gray, owner, roi_xyxy, *, spatial_context='per_tile', root_xy=None):
    """Cover an explicit native region with the same physical-grain query.

    Only tile interiors are blended. Crop agreement is recorded rather
    than interpreted as calibrated biological confidence.
    """
    h, w = gray.shape
    x0, y0, x1, y1 = map(int, roi_xyxy)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('owned-body ROI must lie within the native image')
    if spatial_context not in ('per_tile', 'grain_reference'):
        raise ValueError('body spatial context must be per_tile or grain_reference')
    reference = OwnedBodyReference(model, gray, owner, root_xy=root_xy) if spatial_context == 'grain_reference' else None
    shape = y1-y0, x1-x0
    total, weight = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    low, high = np.ones(shape, np.float32), np.zeros(shape, np.float32)
    count = np.zeros(shape, np.uint8)
    # The four-level encoder/decoder has a convolutional radius below 64.
    # The historical per-tile spatial normalization can vary with context.
    # Reference statistics remove that moving context without new weights.
    margin, stride, lattice = VALID_MARGIN, 128, POOLING_LATTICE
    phase_x, phase_y = reference.phase if reference else (0, 0)
    axis = np.minimum(np.arange(TILE)+1, TILE-np.arange(TILE)).astype(np.float32)
    blend = np.minimum(axis[:, None], axis[None, :])
    origins, query_outside = [], 0
    for oy in range((y0-margin-phase_y)//lattice*lattice+phase_y, y1-margin, stride):
        for ox in range((x0-margin-phase_x)//lattice*lattice+phase_x, x1-margin, stride):
            body, _ = (reference.predict((ox, oy)) if reference else
                       predict_owned_body(model, gray, owner, (ox, oy), root_xy=root_xy))
            lx, ty = max(x0, ox+margin), max(y0, oy+margin)
            rx, by = min(x1, ox+TILE-margin), min(y1, oy+TILE-margin)
            if lx >= rx or ty >= by:
                continue
            dest = np.s_[ty-y0:by-y0, lx-x0:rx-x0]
            src = np.s_[ty-oy:by-oy, lx-ox:rx-ox]
            value, weights = body[src], blend[src]
            total[dest] += value * weights; weight[dest] += weights
            low[dest] = np.minimum(low[dest], value); high[dest] = np.maximum(high[dest], value)
            count[dest] += 1; origins.append([ox, oy])
            gx, gy = owner['grain_native']
            query_outside += not (ox <= gx < ox+TILE and oy <= gy < oy+TILE)
    if not np.all(weight > 0):
        raise RuntimeError('owned-body tiling left a hole in its declared region')
    overlap = count > 1
    disagreement = high[overlap] - low[overlap]
    return total / weight, [x0, y0], {
        'kind': 'native_region_tiles', 'roi_xyxy': [x0, y0, x1, y1],
        'same_grain_query': list(owner['grain_native']), 'tile_origins_xy': origins,
        'tile_margin_px': margin, 'tile_stride_px': stride,
        'normalization': getattr(model, 'normalization', 'unspecified'),
        'normalization_context': spatial_context,
        'normalization_reference': reference.metadata() if reference else None,
        'tile_origin_lattice_px': lattice,
        'tile_origin_phase_xy': [phase_x, phase_y],
        'covered_pixels': int((weight > 0).sum()), 'overlap_pixels': int(overlap.sum()),
        'max_overlap_probability_disagreement': float(disagreement.max()) if overlap.any() else 0.,
        'mean_overlap_probability_disagreement': float(disagreement.mean()) if overlap.any() else 0.,
        'tiles_without_grain_in_input': int(query_outside), 'probability_calibrated': False}
