"""v30 model: owner-conditioned single-frame + temporal U-Net variants, one interface.

Headless import: torch import is guarded; importing this module never requires
Qt/napari. Forward needs torch (CPU) — raises a clear error otherwise.

rev5 repair: real nn.Module subclasses (registered parameters, .to() and
.state_dict() work), tensor-preserving forward (no float()/detach() on any
supervised output — serialization happens in predict_candidates only), an
explicit query index, and a front head that reads route-aligned ribbon
features instead of globally pooled vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

try:  # guarded: headless import must succeed without torch
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


N_MODES_DEFAULT = 5
NOT_OBSERVED_MODE = -1
QUERY_INDEX_DEFAULT = 4  # query is frame 4 of the 9-frame clip, always


@dataclass
class OwnerPrompt:
    owner_id: str
    grain_mask: Any = None  # (H, W) float array or None
    prefix_weight: float = 1.0
    # rev6: NUMERICAL owner geometry (crop coords). The old code only
    # ever passed an owner string + all-zero channel, so front/route
    # outputs ignored the owner entirely (audit: 0.0 change).
    attachment_xy: Any = None  # (x, y) tube-grain exit point
    proximal_xy: Any = None  # [(x, y), ...] first tube widths of path
    # rev7: prompt provenance travels WITH the prompt. Training may
    # carry human geometry; deployment only ever carries automatically
    # derived geometry. A candidate must not manufacture its own
    # confirming owner evidence — provenance is recorded per scored
    # route so evaluations can separate the two.
    provenance: str = "unknown"  # "human" | "auto-grain" | "none" | ...


def build_owner_prompt(owner_id: str,
                       grain_xy=None, grain_r: float = 0.0,
                       grain_mask=None,
                       attachment_xy=None, proximal_xy=None,
                       provenance: str = "unknown") -> OwnerPrompt:
    """One prompt-construction contract for training AND inference.

    Human geometry (attachment + proximal path) is only admissible
    with provenance="human". Deployment prompts carry automatically
    derived geometry only — typically a grain disc mask from detection
    (grain_mask, crop coords) — with provenance="auto-grain".
    Geometry=None renders zeros = the old behavior exactly, so old
    checkpoints still load.
    """
    return OwnerPrompt(owner_id=str(owner_id),
                       grain_mask=grain_mask,
                       attachment_xy=attachment_xy,
                       proximal_xy=(list(proximal_xy)
                                    if proximal_xy is not None else None),
                       provenance=str(provenance))


def resolve_prompt_kind(requested: str, has_attachment_geometry: bool,
                        has_grain_geometry: bool) -> str:
    """Map a drawn prompt kind onto what a sample can actually render.

    rev8: the prompt mix is drawn per sample, but not every kind is
    renderable for every sample. A body-mask sample has no attachment
    or proximal path (those come from a traced observation), so a
    "human" draw would silently train with NO query at all — 60% of
    the mask queries in the first runs, which is how the head learned
    "where tubes are" instead of "which tube is mine". A kind that
    cannot be rendered falls back to one that can, and only an
    explicit "none" is allowed to mean zeros.
    """
    req = str(requested)
    if req == "human" and not has_attachment_geometry:
        req = "auto-grain" if has_grain_geometry else "none"
    if req == "auto-grain" and not has_grain_geometry:
        req = "human" if has_attachment_geometry else "none"
    return req


@dataclass
class CandidatePrediction:
    """Batched TENSOR outputs; every entry keeps its gradient."""

    modes_xy: Any = None  # (B, K, 2) crop coords
    tip_scores: Any = None  # (B, K) logits
    owner_compat: Any = None  # (B, K) logits
    visibility_logits: Any = None  # (B, 7)
    not_observed_score: Any = None  # (B,) logits
    heat: Any = None  # (B, 1, H, W) dense apex evidence: RAW
    # regression score (unrestricted scalar, trained by MSE toward an
    # apex Gaussian with peak 1.0). NOT a probability: never apply
    # sigmoid to it; compare relatively within/across crops only.
    body: Any = None  # (B, 1, H, W) visible-tube-body LOGITS (rev6:
    # per-instance body evidence; supervised only on visible spans)
    front_logits: Any = None  # (B, S) over ribbon arclength, or None
    front_s: Any = None  # (S,) arclength grid, DETACHED metadata (not supervised)
    route_logit: Any = None  # (B,) route-correctness logit, or None
    # rev11: explicit owned-cap presence/rejection logit per route. A
    # conditional softmax ALWAYS picks a position, so its maximum is not
    # a presence probability; this scalar is trained on route-level
    # present/absent labels (None when the model has no presence head).
    front_present_logit: Any = None  # (B,) or None
    # rev11 section 8B: per-location cap logits from local oriented
    # windows at explicit native locations ((B, K) or None).
    cap_logits: Any = None  # (B, K) or None
    modes: list[dict] = field(default_factory=list)  # inference-side floats


def _require_torch():  # type: ignore[no-untyped-def]
    if torch is None or nn is None:
        raise ImportError("v30 model forward requires torch (CPU build is fine)")


def _owner_channel(frames, prompt: OwnerPrompt):  # type: ignore[no-untyped-def]
    """Owner conditioning: grain mask AND/OR rendered owner geometry.

    rev6: the channel used to be all zeros in every real run (only an
    owner string traveled with the prompt). Now it renders the tube's
    grain attachment (Gaussian disc) and proximal path (thick segment)
    from the prompt's numerical geometry. Same shape as before, so old
    checkpoints still load (geometry=None renders zeros = old
    behavior exactly).
    """
    _require_torch()
    import torch as _t

    b, t, _, h, w = frames.shape
    if prompt.grain_mask is not None:
        m = _t.as_tensor(prompt.grain_mask, dtype=frames.dtype,
                         device=frames.device).reshape(1, 1, h, w)
        m = m.expand(b, t, 1, h, w) * float(prompt.prefix_weight)
    else:
        m = _t.zeros((b, t, 1, h, w), dtype=frames.dtype,
                     device=frames.device)
    if prompt.attachment_xy is not None or prompt.proximal_xy is not None:
        yy, xx = _t.meshgrid(
            _t.arange(h, dtype=frames.dtype, device=frames.device),
            _t.arange(w, dtype=frames.dtype, device=frames.device),
            indexing="ij")
        g = _t.zeros((h, w), dtype=frames.dtype,
                     device=frames.device)
        if prompt.attachment_xy is not None:
            ax, ay = float(prompt.attachment_xy[0]), float(
                prompt.attachment_xy[1])
            g = g + _t.exp(-((xx - ax) ** 2 + (yy - ay) ** 2) / (2 * 6.0 ** 2))
        if prompt.proximal_xy is not None:
            pts = [(float(q[0]), float(q[1]))
                   for q in prompt.proximal_xy]
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                t0 = _t.tensor([x0, y0], dtype=frames.dtype,
                               device=frames.device)
                t1 = _t.tensor([x1, y1], dtype=frames.dtype,
                               device=frames.device)
                seg = _t.hypot(xx - t0[0], yy - t0[1]) + _t.hypot(
                    xx - t1[0], yy - t1[1]) - _t.hypot(
                        t1[0] - t0[0], t1[1] - t0[1])
                g = g + (seg < 3.0).to(frames.dtype)
        g = g.clamp(0, 1).reshape(1, 1, 1, h, w).expand(b, t, 1, h, w)
        m = m + g * float(prompt.prefix_weight)
    return _t.cat([frames, m], dim=2)


def sample_ribbon(clip, route_xy, n_across: int = 7, width_px: float = 6.0,
                  query_index: int = QUERY_INDEX_DEFAULT):  # type: ignore[no-untyped-def]
    """Sample a (T, S, U) ribbon along a route polyline (crop coords).

    Returns dict with ribbon tensor (B, T, C, S, U), arclength vector s
    (S,), and the valid support length. Differentiable w.r.t. the clip.
    Route support is the caller's responsibility (oracle or predicted).
    """
    _require_torch()
    import torch as _t
    import torch.nn.functional as _F

    route = _t.as_tensor(route_xy, dtype=clip.dtype,
                         device=clip.device)
    assert route.ndim == 2 and route.shape[0] >= 2 and route.shape[1] == 2
    b, t, c, h, w = clip.shape
    pts = route
    seg = _t.hypot(pts[1:, 0] - pts[:-1, 0], pts[1:, 1] - pts[:-1, 1])
    s = _t.cat([_t.zeros(1, dtype=clip.dtype, device=clip.device),
                _t.cumsum(seg, 0)])
    total = float(s[-1]) if len(s) > 1 else 0.0
    ns = max(int(total) + 1, 8)
    ss = _t.linspace(0.0, total, ns)
    xs = torch_interp(ss, s, pts[:, 0])
    ys = torch_interp(ss, s, pts[:, 1])
    # Tangent/normal frames.
    dx = _t.gradient(xs)[0] if xs.numel() > 1 else _t.ones_like(xs)
    dy = _t.gradient(ys)[0] if ys.numel() > 1 else _t.zeros_like(ys)
    norm = _t.hypot(dx, dy).clamp_min(1e-6)
    nx, ny = -dy / norm, dx / norm
    uu = _t.linspace(-width_px, width_px, n_across)
    gx = xs[:, None] + nx[:, None] * uu[None, :]
    gy = ys[:, None] + ny[:, None] * uu[None, :]
    gx_n = (gx / max(w - 1, 1)) * 2 - 1
    gy_n = (gy / max(h - 1, 1)) * 2 - 1
    grid = _t.stack([gx_n, gy_n], dim=-1).unsqueeze(0).expand(b * t, ns, n_across, 2)
    flat = clip.reshape(b * t, c, h, w)
    samp = _F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=True)
    ribbon = samp.reshape(b, t, c, ns, n_across)
    return {"ribbon": ribbon, "s": ss, "support": total,
            "query_index": int(query_index)}


def torch_interp(xq, x, y):  # type: ignore[no-untyped-def]
    """Piecewise-linear interp of y(x) at xq (all 1D tensors)."""
    _require_torch()
    import torch as _t

    idx = _t.searchsorted(x, xq.clamp(x[0], x[-1])) .clamp(1, len(x) - 1)
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    w = ((xq - x0) / (x1 - x0).clamp_min(1e-9)).clamp(0, 1)
    return y0 + w * (y1 - y0)


class _TinyUNet(nn.Module):  # type: ignore[no-untyped-def]
    """Small conv encoder-decoder; construction requires torch."""

    def __init__(self, in_ch: int, base: int = 16):  # type: ignore[no-untyped-def]
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(),
        )
        self.heat = nn.Conv2d(base, 1, 1)
        self.feat = nn.Conv2d(base, base, 1)

    def forward(self, x):  # type: ignore[no-untyped-def]
        h, w = x.shape[-2:]
        f = self.dec(self.enc(x))
        # Transpose-conv overshoots on odd inputs (e.g. 57 -> 58);
        # crop back so outputs always match the input grid. The front
        # head's arclength axis must equal the ribbon's S exactly.
        f = f[..., :h, :w]
        return self.heat(f), self.feat(f)


class _MultiScaleUNet(nn.Module):  # type: ignore[no-untyped-def]
    """Crop-scale U-Net: 4 stride-2 levels, dilated bottleneck, skips.

    rev8 step 3. The rev7 encoder (`_TinyUNet`) had a single stride-2
    level and a 1x1 body head; the review measured the body output's
    receptive field at 9x9 px, so no grain or attachment prompt could
    establish distant tube connectivity. This encoder sees the crop:
    three stride-2 levels, a dilated bottleneck, and mirrored skips.
    The owner channel is re-injected at every decoder level so the
    query shapes the whole decoded tube, not just its root.
    """

    def __init__(self, in_ch: int, base: int = 16,
                 depth: int = 3, dilations=(2, 4, 8)):
        super().__init__()
        self.depth = int(depth)
        self.owner_in = in_ch - 1  # first channel(s) = image
        chs = [base]
        for i in range(1, self.depth + 1):
            chs.append(base * min(2 ** i, 4))
        # encoder: level 0 full-res, levels 1..depth at stride 2^i
        enc = []
        c_in = in_ch
        for i in range(self.depth + 1):
            c_out = chs[i]
            if i > 0:
                enc.append(nn.Conv2d(c_in, c_out, 3, stride=2, padding=1))
            else:
                enc.append(nn.Conv2d(c_in, c_out, 3, stride=1, padding=1))
            enc.append(nn.ReLU())
            enc.append(nn.Conv2d(c_out, c_out, 3, stride=1, padding=1))
            enc.append(nn.ReLU())
            c_in = c_out
        self.enc = nn.Sequential(*enc)
        # dilated bottleneck at stride 2^depth
        bot = []
        c_bot = chs[-1]
        for d in dilations:
            bot.append(nn.Conv2d(c_bot, c_bot, 3, padding=int(d),
                                 dilation=int(d)))
            bot.append(nn.ReLU())
        self.bottleneck = nn.Sequential(*bot)
        # decoder: one transposed conv + skip-concat conv per level;
        # the owner channel is concatenated at every level too. Skips
        # pair with the encoder level at the resolution just restored.
        dec = []
        c_prev = c_bot
        for lvl in range(self.depth):
            c_skip = chs[self.depth - 1 - lvl]
            c_out = chs[self.depth - 1 - lvl]
            dec.append(nn.ConvTranspose2d(c_prev, c_out, 4, stride=2,
                                          padding=1))
            dec.append(nn.ReLU())
            dec.append(nn.Conv2d(c_out + c_skip + self.owner_in,
                                 c_out, 3, padding=1))
            dec.append(nn.ReLU())
            c_prev = c_out
        self.dec = nn.Sequential(*dec)
        self.heat = nn.Conv2d(base, 1, 1)
        self.feat = nn.Conv2d(base, base, 1)

    def forward(self, x):  # type: ignore[no-untyped-def]
        import torch as _t

        h, w = x.shape[-2:]
        owner = x[:, -1:]  # last channel = the owner (attachment) map
        feats = []
        f = x
        for i, layer in enumerate(self.enc):
            f = layer(f)
            if (i % 4) == 3:  # end of a level block (conv, relu, conv, relu)
                feats.append(f)
        f = self.bottleneck(f)
        for lvl in range(self.depth):
            skip = feats[-(2 + lvl)]
            f = self.dec[lvl * 4](f)          # upsample
            f = self.dec[lvl * 4 + 1](f)      # relu
            if f.shape[-2:] != skip.shape[-2:]:
                f = _t.nn.functional.interpolate(
                    f, size=skip.shape[-2:], mode="nearest")
            own = _t.nn.functional.interpolate(
                owner, size=f.shape[-2:], mode="nearest")
            f = _t.cat([f, skip, own], dim=1)
            f = self.dec[lvl * 4 + 2](f)
            f = self.dec[lvl * 4 + 3](f)
        f = f[..., :h, :w]
        return self.heat(f), self.feat(f)


class SingleFrameOwnerUNet(nn.Module):  # type: ignore[no-untyped-def]
    """N2: tube-conditioned single-frame native branch (registered module)."""

    variant = "single"

    def __init__(self, in_channels: int = 1, base: int = 16, n_modes: int = N_MODES_DEFAULT,
                 multiscale: bool = False):
        super().__init__()
        _require_torch()
        self.n_modes = n_modes
        self.multiscale = bool(multiscale)
        self.unet = (_MultiScaleUNet(in_channels + 1, base) if multiscale
                     else _TinyUNet(in_channels + 1, base))
        self.mode_head = nn.Linear(base, n_modes * 4)  # x, y, tip, compat per mode
        self.vis_head = nn.Linear(base, 7)
        self.nobs_head = nn.Linear(base, 1)
        # NOTE: body is LAST so pre-existing layers keep their exact
        # init stream (a mid-__init__ insert silently re-initialized
        # the front head under fixed seeds and broke descent tests).
        self.body = nn.Conv2d(base, 1, 1)  # rev6: visible body logits

    def forward(self, clip, prompt: OwnerPrompt,  # type: ignore[no-untyped-def]
                query_index: int = QUERY_INDEX_DEFAULT) -> CandidatePrediction:
        _require_torch()
        x = _owner_channel(clip, prompt)  # (B, T, C+1, H, W)
        if not 0 <= query_index < x.shape[1]:
            raise ValueError(f"query_index {query_index} outside clip "
                             f"with {x.shape[1]} frames")
        heat, feat = self.unet(x[:, query_index])
        pooled = feat.mean(dim=(2, 3))  # (B, base)
        m = self.mode_head(pooled).reshape(-1, self.n_modes, 4)
        return CandidatePrediction(
            modes_xy=m[..., :2],
            tip_scores=m[..., 2],
            owner_compat=m[..., 3],
            visibility_logits=self.vis_head(pooled),
            not_observed_score=self.nobs_head(pooled).reshape(-1),
            heat=heat,
            body=self.body(feat),
            front_logits=None,
        )


class TemporalOwnerUNet(nn.Module):  # type: ignore[no-untyped-def]
    """N3: surrounding-frame native model; front head reads ribbon features."""

    variant = "temporal"

    def __init__(self, in_channels: int = 1, base: int = 16, n_modes: int = N_MODES_DEFAULT,
                 n_frames: int = 9, ribbon_across: int = 7,
                 multiscale: bool = False, presence_head: bool = False,
                 local_cap_head: bool = False,
                 cap_window_head: bool = False,
                 cap_window_encoder: bool = False,
                 cap_window_semantics: int = 2):
        super().__init__()
        _require_torch()
        self.n_modes = n_modes
        self.n_frames = n_frames
        self.ribbon_across = ribbon_across
        self.multiscale = bool(multiscale)
        self.unet = (_MultiScaleUNet(in_channels + 1, base) if multiscale
                     else _TinyUNet(in_channels + 1, base))
        # rev8: spatial context in time when the encoder is multiscale
        # (a 1x1 spatial kernel cannot move structure between frames);
        # shape-preserving either way, so old checkpoints still load.
        self.temporal = (nn.Conv3d(base, base, kernel_size=(3, 3, 3),
                                   padding=(1, 1, 1)) if multiscale
                         else nn.Conv3d(base, base, kernel_size=(3, 1, 1),
                                        padding=(1, 0, 0)))
        self.mode_head = nn.Linear(base, n_modes * 4)
        self.vis_head = nn.Linear(base, 7)
        self.nobs_head = nn.Linear(base, 1)
        # Front head: conv over (arclength, transverse) ribbon features.
        self.front_conv = nn.Sequential(
            nn.Conv2d(base, base, (3, 3), padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(base, base, (3, 1), padding=(1, 0)), nn.ReLU(),
        )
        self.front_head = nn.Linear(base, 1)  # logit per arclength sample
        self.route_head = nn.Linear(base, 1)  # route-correctness logit
        # NOTE: body last — see SingleFrameOwnerUNet for why.
        self.body = nn.Conv2d(base, 1, 1)  # rev6: visible body logits
        # rev11: explicit owned-cap presence/rejection score. Additive
        # and DECLARED: only instantiated when the model config says so,
        # so every pre-rev11 checkpoint loads with an identical state
        # dict (the strict factory rejects new unexpected keys by
        # design). Initialized AFTER every existing module so the
        # baseline arms' parameters are bit-identical under a fixed
        # seed — equal-updates comparisons stay fair.
        self.presence_head = bool(presence_head)
        if self.presence_head:
            self.front_present = nn.Linear(base * 2, 1)
        # rev11 section 8B: native LOCAL-CAP scorer. Per-location cap
        # evidence read from an oriented window around (x, y, tangent)
        # in native crop coordinates — NOT from a ribbon sampled
        # relative to the route's ends. The window at a fixed cap
        # location is pixel-identical when a distant support endpoint
        # moves (the invariance the review asks for is by construction;
        # the experiment tests it). Additive and declared like the
        # presence head.
        self.local_cap_head = bool(local_cap_head)
        if self.local_cap_head:
            self.cap_n_along = 11
            self.cap_n_across = 7
            self.cap_half_along = 10.0
            self.cap_half_across = 6.0
            c_in = base * self.cap_n_along * self.cap_n_across
            self.cap_scorer = nn.Sequential(
                nn.Linear(c_in, 64), nn.LeakyReLU(0.01),
                nn.Linear(64, 1))
        # rev13 W3: the native-resolution cap scorer over an ORDERED
        # time window. A 2D conv stack runs per frame on the NATIVE
        # oriented patch (image + owner channel); a temporal 1D
        # convolution mixes the ordered frames; heads predict the cap
        # logit, a native XY refinement (px) and a log-variance
        # (uncertainty). The single-frame control uses the SAME weights
        # with a one-frame window. This is the required new evidence: a
        # native 2D scorer, not a scalar MLP over pooled features.
        self.cap_window_head = bool(cap_window_head)
        self.cap_window_semantics = int(cap_window_semantics)
        if self.cap_window_semantics not in (1, 2):
            raise ValueError("unsupported cap-window semantics")
        if self.cap_window_head:
            self.cw_patch_px = 24
            self.cw_hidden = 16
            self.cw_conv = nn.Sequential(
                nn.Conv2d(2, self.cw_hidden, 3, padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(self.cw_hidden, self.cw_hidden, 3, padding=1),
                nn.LeakyReLU(0.01))
            self.cw_temporal = nn.Conv1d(self.cw_hidden, self.cw_hidden,
                                         3, padding=1)
            self.cw_logit = nn.Linear(3 * self.cw_hidden, 1)
            self.cw_dxy = nn.Linear(3 * self.cw_hidden, 2)
            self.cw_logvar = nn.Linear(3 * self.cw_hidden, 1)
        # rev13 W3 contingency: the ENCODER-FED variant. The scorer
        # additionally consumes the encoder's per-frame features at its
        # patch locations, so adapting the encoder at --lr is a real
        # variable (the raw-pixel variant cannot be changed by encoder
        # training — measured, H429). Declared architecture: its own
        # module set, so raw-variant checkpoints keep loading strictly.
        self.cap_window_encoder = bool(cap_window_head
                                       and cap_window_encoder)
        if self.cap_window_encoder:
            self.cw_feat = nn.Linear(base, self.cw_hidden)
            self.cw_temporal_e = nn.Conv1d(3 * self.cw_hidden,
                                           self.cw_hidden, 3, padding=1)
            self.cw_logit_e = nn.Linear(4 * self.cw_hidden, 1)
            self.cw_dxy_e = nn.Linear(4 * self.cw_hidden, 2)
            self.cw_logvar_e = nn.Linear(4 * self.cw_hidden, 1)

    def score_cap_locations(self, fused, locations_xy, tangents_xy):  # type: ignore[no-untyped-def]
        """Per-location cap logits from local oriented windows.

        fused: (B, C, H, W) crop-coordinate features (same grid as the
        clip). locations_xy, tangents_xy: (B, K, 2). Returns (B, K)
        logits. The sampled grid for location k depends ONLY on
        (x_k, y_k, tangent_k) — never on route ends — so moving a
        support endpoint cannot change the input at a fixed location.
        """
        _require_torch()
        import torch as _t
        B, C, H, W = fused.shape
        P = self.cap_n_along * self.cap_n_across
        al = _t.linspace(-self.cap_half_along, self.cap_half_along,
                         self.cap_n_along)
        ac = _t.linspace(-self.cap_half_across, self.cap_half_across,
                         self.cap_n_across)
        A, R = _t.meshgrid(al, ac, indexing="ij")
        A = A.reshape(1, 1, -1).to(
            dtype=fused.dtype, device=fused.device)
        R = R.reshape(1, 1, -1).to(
            dtype=fused.dtype, device=fused.device)
        tx = tangents_xy[:, :, 0:1]
        ty = tangents_xy[:, :, 1:2]
        nx = -ty
        ny = tx
        px = locations_xy[:, :, 0:1] + A * tx + R * nx  # (B, K, P)
        py = locations_xy[:, :, 1:2] + A * ty + R * ny
        gx = px / max(W - 1, 1) * 2.0 - 1.0
        gy = py / max(H - 1, 1) * 2.0 - 1.0
        grid = _t.stack([gx, gy], dim=-1).reshape(B, -1, 1, 2)
        samp = _t.nn.functional.grid_sample(
            fused, grid, align_corners=True)  # (B, C, K*P, 1)
        samp = samp.reshape(B, C, locations_xy.shape[1], P)
        feat = samp.permute(0, 2, 1, 3).reshape(
            B * locations_xy.shape[1], C * P)
        return self.cap_scorer(feat).reshape(
            B, locations_xy.shape[1])

    def score_cap_window(self, clip, prompt, locations_xy, tangents_xy,
                         window_indices=None, frame_only=False,
                         query_index=QUERY_INDEX_DEFAULT):  # type: ignore[no-untyped-def]
        """rev13 W3: native-resolution cap evidence over an ORDERED time
        window.

        Samples a (cw_patch_px x cw_patch_px) native patch at every
        location from EVERY frame in `window_indices` (default: all
        frames of the clip), oriented by the tangent — the sampling
        depends only on (x, y, tangent), never on route ends. The
        query frame is explicitly identified by query_index;
        `frame_only=True` is the single-frame control (same weights,
        one-frame window). Returns dict with logits (B,K), native XY
        refinement in px (B,K,2), log-variance (B,K) and the window
        actually used.
        """
        _require_torch()
        import torch as _t
        x = _owner_channel(clip, prompt)  # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        K = int(locations_xy.shape[1])
        P = int(self.cw_patch_px)
        half = (P - 1) / 2.0
        if frame_only:
            idx = [int(query_index)]
        elif window_indices is None:
            idx = list(range(T))
        else:
            idx = [int(i) for i in window_indices]
        if (not idx or len(set(idx)) != len(idx)
                or min(idx) < 0 or max(idx) >= T
                or int(query_index) not in idx):
            raise ValueError("cap window must contain the query exactly once and valid frame indices")
        # Version 1 is a historical replay only. New fits anchor every head
        # on the labelled current image, with an actual path from each
        # declared context frame. Reusing old weights never silently changes
        # their function (the strict factory declares the version).
        legacy = self.cap_window_semantics == 1
        qi = len(idx) - 1 if legacy else idx.index(int(query_index))

        def temporal_at_query(seq, conv):
            y = conv(seq.permute(0, 2, 1))
            if not legacy:
                # The encoder-fed convolution changes channel count, so
                # subsequent mixing uses shifts of the encoded sequence.
                # A distance-weighted ordered sum gives all declared frames
                # a nonzero route to the query without a future-only readout.
                positions = _t.arange(len(idx), device=y.device, dtype=y.dtype)
                delta = positions - qi
                weights = (1.0 + 0.1 * delta / max(1, len(idx))) / (1.0 + delta.abs())
                context = (y * weights[None, None, :]).sum(-1) / weights.sum()
                return y[:, :, qi] + context
            return y[:, :, qi]

        def result(z, logit_head, offset_head, variance_head):
            raw = offset_head(z).reshape(B, K, 2)
            if legacy:
                delta = raw * (P / 8.0)
            else:
                # Predict in the oriented patch basis, then rotate once to
                # native XY. Training targets and emitted positions use XY.
                local = half * _t.tanh(raw)
                delta = _t.stack([local[..., 0] * tx[..., 0] + local[..., 1] * nx[..., 0],
                                  local[..., 0] * ty[..., 0] + local[..., 1] * ny[..., 0]], -1)
            return {"logits": logit_head(z).reshape(B, K),
                    "dxy_px": delta, "logvar": variance_head(z).reshape(B, K),
                    "valid": _valid, "window": idx,
                    "query_index": int(query_index),
                    "consumed_indices": idx[-2:] if legacy else idx,
                    "offset_basis": "legacy-native" if legacy else "tangent-normal-to-native",
                    "semantics": self.cap_window_semantics}
        al = _t.linspace(-half, half, P)
        A, R = _t.meshgrid(al, al, indexing="ij")
        A = A.reshape(1, 1, -1).to(dtype=x.dtype, device=x.device)
        R = R.reshape(1, 1, -1).to(dtype=x.dtype, device=x.device)
        norm = _t.linalg.vector_norm(tangents_xy, dim=-1, keepdim=True)
        if bool((norm < 1e-8).any()):
            raise ValueError("cap patch requires a nonzero tangent")
        tangents_xy = tangents_xy / norm
        tx = tangents_xy[:, :, 0:1]
        ty = tangents_xy[:, :, 1:2]
        nx, ny = -ty, tx
        px = locations_xy[:, :, 0:1] + A * tx + R * nx
        py = locations_xy[:, :, 1:2] + A * ty + R * ny
        gx = px / max(W - 1, 1) * 2.0 - 1.0
        gy = py / max(H - 1, 1) * 2.0 - 1.0
        grid = _t.stack([gx, gy], dim=-1).reshape(B, K * P * P, 1, 2)
        # rev13 W3.4: IMAGE-VALIDITY per location. A patch whose sample
        # points leave the frame is clamped by grid_sample — an edge
        # smear with no evidence. Measured (H431): 25% of decoy argmax
        # locations sat at/outside the crop border and the scorer keyed
        # on the clamp artifact. Invalid locations carry no evidence:
        # no supervised label at training time, no emission at eval.
        _valid = ((px >= 0.0) & (px <= float(W - 1))
                  & (py >= 0.0) & (py <= float(H - 1)))
        _valid = _valid.all(dim=-1).reshape(B, K)  # (B, K) bool
        if getattr(self, "cap_window_encoder", False):
            # rev13 W3 contingency: encoder-fed variant. Per frame: the
            # encoder's feature map is sampled at the SAME oriented
            # patch grid and pooled per location; the raw patch features
            # and the encoder features are fused before the ordered
            # temporal convolution. Adapting the encoder now changes the
            # scorer's outputs (the property the raw variant lacked).
            per_frame = []
            for i in idx:
                _, fmap = self.unet(x[:, i])            # (B, Cf, H, W)
                Cf = int(fmap.shape[1])
                fs = _t.nn.functional.grid_sample(
                    fmap, grid, align_corners=True)     # (B,Cf,K*P*P,1)
                fs = fs.reshape(B, Cf, K, P, P).permute(
                    0, 2, 1, 3, 4).reshape(B * K, Cf, P, P)
                fv = self.cw_feat(fs.mean(dim=(2, 3)))  # (B*K, H)
                two = _t.cat([x[:, i, 0:1], x[:, i, 1:2]], dim=1)
                samp = _t.nn.functional.grid_sample(
                    two, grid, align_corners=True)
                samp = samp.reshape(B, 2, K, P, P).permute(
                    0, 2, 1, 3, 4).reshape(B * K, 2, P, P)
                f = self.cw_conv(samp)
                rp = _t.cat([f.mean(dim=(2, 3)),
                             f.amax(dim=(2, 3))], dim=-1)  # (B*K, 2H)
                per_frame.append(_t.cat([rp, fv], dim=-1))  # (B*K, 3H)
            seq = _t.stack(per_frame, dim=1)            # (B*K, T, 3H)
            mix = temporal_at_query(seq, self.cw_temporal_e)
            z = _t.cat([seq[:, qi], mix], dim=-1)
            return result(z, self.cw_logit_e, self.cw_dxy_e, self.cw_logvar_e)
        feats = []
        for i in idx:
            two = _t.cat([x[:, i, 0:1], x[:, i, 1:2]], dim=1)
            samp = _t.nn.functional.grid_sample(
                two, grid, align_corners=True)      # (B,2,K*P*P,1)
            samp = samp.reshape(B, 2, K, P, P)
            # (B, 2, K, P, P) -> (B*K, 2, P, P) must permute FIRST:
            # reshaping directly would interleave channels with
            # locations and couple one location's patch to another's.
            f = self.cw_conv(samp.permute(0, 2, 1, 3, 4).reshape(
                B * K, 2, P, P))
            feats.append(f.reshape(B, K, self.cw_hidden, P, P))
        vol = _t.stack(feats, dim=2)                # (B,K,T,H,P,P)
        sp = vol.mean(dim=(4, 5))                   # (B,K,T,H)
        sp = sp.reshape(B * K, len(idx), self.cw_hidden)
        # the ORDERED temporal convolution: neighbours in acquisition
        # order, so reversing the window changes the output
        mix = temporal_at_query(sp, self.cw_temporal).reshape(B, K, self.cw_hidden)
        cur = vol[:, :, qi]
        cur_vec = _t.cat([cur.mean(dim=(3, 4)),
                          cur.amax(dim=(3, 4))], dim=-1)  # (B,K,2H)
        z = _t.cat([cur_vec, mix], dim=-1)
        return result(z, self.cw_logit, self.cw_dxy, self.cw_logvar)

    def forward(self, clip, prompt: OwnerPrompt,  # type: ignore[no-untyped-def]
                query_index: int = QUERY_INDEX_DEFAULT,
                route_xy=None, cap_query=None) -> CandidatePrediction:
        _require_torch()
        import torch as _t

        x = _owner_channel(clip, prompt)  # (B, T, C, H, W)
        b, t, c, h, w = x.shape
        if not 0 <= query_index < t:
            raise ValueError(f"query_index {query_index} outside clip "
                             f"with {t} frames")
        feats = []
        for i in range(t):
            _, f = self.unet(x[:, i])
            feats.append(f)
        vol = _t.stack(feats, dim=2)  # (B, base, T, H, W)
        # H347: this relu was a DEAD RELU and it blocked every
        # head. Measured on a real crop: the encoder's feature
        # maps carry spatial std 0.124, the temporal Conv3d's
        # output is then constant, relu zeroes it, and the body
        # head can only return its bias (std 0.0, mean 1.3663 —
        # bit-identical across different crops and prompts). The
        # gradient through a dead relu is exactly zero, so no
        # loss or objective could ever recover it: six 60-epoch
        # arms sat flat for this reason, and the rev9 audit's
        # image-independence finding has the same signature.
        # Leaky relu keeps a slope, so the layer cannot die.
        vol = _t.nn.functional.leaky_relu(self.temporal(vol),
                                          negative_slope=0.01)
        fused = vol.mean(dim=2)  # (B, base, H, W)
        pooled = fused.mean(dim=(2, 3))
        m = self.mode_head(pooled).reshape(-1, self.n_modes, 4)
        cap_logits = None
        if self.local_cap_head and cap_query is not None:
            # rev11 section 8B: per-location cap evidence at explicit
            # native locations. Independent of any route's ends.
            cap_logits = self.score_cap_locations(
                fused, cap_query["locations_xy"],
                cap_query["tangents_xy"])
        front_logits = None
        front_s = None
        route_logit = None
        front_present_logit = None
        if route_xy is not None:
            # NOTE (rev6 Pkg3 negative, H283): fused-temporal fronts
            # were tried — route-sampled pre-fusion features through
            # the temporal conv, no new params. Result: dev fronts
            # stuck at 18-47px over 10 epochs (v12), no trend; mean
            # fusion smears the peak the head keys on (context frames
            # hold the tip elsewhere). Reverted to the localizing
            # query-frame front (1.2/0.2px). Real temporal fusion needs
            # query-gated attention, not mean-fusion — parked behind
            # body masks that can validate it.
            rb = sample_ribbon(x, route_xy,
                               n_across=self.ribbon_across,
                               query_index=query_index)
            # Ribbon image+owner features through the shared encoder
            # (query time).
            rq = rb["ribbon"][:, query_index]  # (B, C+1, S, U)
            qf = self.unet(rq)[1]
            ff = self.front_conv(qf).mean(dim=-1)  # (B, base, S)
            front_logits = self.front_head(
                ff.transpose(1, 2)).reshape(b, -1)  # (B, S)
            front_s = rb["s"].detach()  # metadata grid, never supervised
            # Max over arclength (not mean): the cap is a local pattern
            # occupying a few of ~100 samples; mean-pooling dilutes it
            # 100:1 and true/wrong ribbons become indistinguishable.
            route_logit = self.route_head(
                ff.max(dim=-1).values).reshape(b)  # (B,)
            if self.presence_head:
                # rev11: presence/rejection from the route's own feature
                # profile (max + mean over arclength) — a softmax always
                # picks a position; this scalar says whether the route
                # carries the owner's current cap at all.
                front_present_logit = self.front_present(_t.cat(
                    [ff.max(dim=-1).values, ff.mean(dim=-1)],
                    dim=1)).reshape(b)
        return CandidatePrediction(
            modes_xy=m[..., :2],
            tip_scores=m[..., 2],
            owner_compat=m[..., 3],
            visibility_logits=self.vis_head(pooled),
            not_observed_score=self.nobs_head(pooled).reshape(-1),
            heat=self.unet.heat(fused),
            body=self.body(fused),
            front_logits=front_logits,
            front_s=front_s,
            route_logit=route_logit,
            front_present_logit=front_present_logit,
            cap_logits=cap_logits,
        )


def build_model(variant: str = "single", **kwargs):  # type: ignore[no-untyped-def]
    _require_torch()
    if variant == "single":
        return SingleFrameOwnerUNet(**kwargs)
    if variant == "temporal":
        return TemporalOwnerUNet(**kwargs)
    if variant == "owner-decoder":
        from .owner_decoder import OwnerConditionedDecoderNet
        return OwnerConditionedDecoderNet(**kwargs)
    raise ValueError(f"unknown v30 model variant: {variant}")


def predict_candidates(model, clip, prompt: OwnerPrompt, movie_id: str = "movie",
                       source_frame: int = 0,
                       query_index: int = QUERY_INDEX_DEFAULT,
                       route_xy=None) -> list[dict]:  # type: ignore[no-untyped-def]
    """One candidate interface for both variants; appends explicit not-observed state.

    Serialization (float conversion) happens HERE, never in forward: the
    training path keeps every tensor and its gradient.
    """
    import torch as _t

    from .contracts import ObservationState, Visibility

    was_training = bool(getattr(model, "training", False))
    model.eval()
    with _t.no_grad():
        pred = model.forward(clip, prompt, query_index=query_index,
                             **({"route_xy": route_xy}
                                 if getattr(model, "variant", "") == "temporal"
                                 else {}))
    if was_training:
        model.train()
    sig = _t.sigmoid
    out = []
    xy = pred.modes_xy.detach()
    ts = sig(pred.tip_scores.detach())
    oc = sig(pred.owner_compat.detach())
    for k in range(xy.shape[1]):
        out.append({
            "candidate_id": f"{movie_id}:{source_frame}:{prompt.owner_id}:m{k}",
            "movie_id": movie_id, "owner_id": prompt.owner_id,
            "source_frame": int(source_frame),
            "x_native": float(xy[0, k, 0]), "y_native": float(xy[0, k, 1]),
            "mode": int(k), "tip_score": float(ts[0, k]),
            "owner_compat": float(oc[0, k]),
            "visibility": Visibility.DIRECTLY_VISIBLE,
            "observation": ObservationState.OBSERVED,
            "selected": False,
        })
    out.append({
        "candidate_id": f"{movie_id}:{source_frame}:{prompt.owner_id}:nob",
        "movie_id": movie_id, "owner_id": prompt.owner_id,
        "source_frame": int(source_frame),
        "x_native": None, "y_native": None, "mode": NOT_OBSERVED_MODE,
        "tip_score": float(sig(pred.not_observed_score.detach())[0]),
        "owner_compat": 0.0,
        "visibility": Visibility.HIDDEN,
        "observation": ObservationState.NOT_OBSERVED,
        "selected": False,
    })
    return out


def random_clip(batch: int = 1, frames: int = 9, channels: int = 1, size: int = 64):  # type: ignore[no-untyped-def]
    _require_torch()
    import torch as _t

    return _t.randn(batch, frames, channels, size, size)
