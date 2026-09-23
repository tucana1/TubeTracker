"""rev10 WP-B escalation: image-only cached encoder + owner-conditioned decoder.

The review's exact wording (line 125 of the rev10 review):

  "cache image-only high-resolution features once per frame/clip; provide
   grain/attachment/proximal-mask prompts to a separate owner-conditioned
   decoder; predict owned body and presence with explicit current-front
   evidence. Keep thin structures at sufficient resolution and use crop
   overlap/context for long tubes. Evaluate a residual query-frame feature
   path into the body decoder so temporal fusion cannot erase all spatial
   evidence."

and its constraint (line 127): "Do not turn a metadata owner string into a
memorized label embedding that cannot identify new grains." Ownership here
enters ONLY as spatial mask prompts -- never as an id embedding.

Why this change follows the fit verdict (H363): the repaired objective fits
partially (g1 through the rival gate, IoU 0.24-0.80 against ~0.90), and the
review's decision rule applies when "the corrected current architecture
cannot fit this tiny task". In the current model the owner prompt is
concatenated into the ENCODER input, so every owner's gradient must pass
through one shared trunk -- the multiclass interference the plateau shows.
Here the trunk is image-only (and therefore cacheable across queries),
while the owner geometry conditions the decoder directly at every level.

The encoder is an explicit mirror of `_MultiScaleUNet`'s encoder; a test
asserts layer-for-layer equivalence under identical weights so the mirror
cannot silently drift.
"""

from __future__ import annotations

import torch as _t
import torch.nn as nn

from .model import (N_MODES_DEFAULT, QUERY_INDEX_DEFAULT, CandidatePrediction,
                    OwnerPrompt, _owner_channel, _require_torch,
                    sample_ribbon)


class ImageOnlyEncoder(nn.Module):  # type: ignore[no-untyped-def]
    """Multiscale image encoder with no owner channels.

    Mirrors `_MultiScaleUNet.enc` + `.bottleneck` exactly (same layer
    order, same channel plan) so weights can be transferred, but takes
    the image alone. Its per-level features are safe to cache once per
    frame and reuse for every owner query.
    """

    def __init__(self, in_ch: int = 1, base: int = 16, depth: int = 3,
                 dilations=(2, 4, 8)):
        super().__init__()
        self.depth = int(depth)
        chs = [base]
        for i in range(1, self.depth + 1):
            chs.append(base * min(2 ** i, 4))
        self.chs = chs
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
        bot = []
        c_bot = chs[-1]
        for d in dilations:
            bot.append(nn.Conv2d(c_bot, c_bot, 3, padding=int(d),
                                 dilation=int(d)))
            bot.append(nn.ReLU())
        self.bottleneck = nn.Sequential(*bot)

    def forward(self, x):  # type: ignore[no-untyped-def]
        feats = []
        f = x
        for i, layer in enumerate(self.enc):
            f = layer(f)
            if (i % 4) == 3:
                feats.append(f)
        return self.bottleneck(f), feats


class OwnerConditionedDecoderNet(nn.Module):  # type: ignore[no-untyped-def]
    """`variant = "owner-decoder"`. Interface-compatible with
    TemporalOwnerUNet so the fit check, evaluator and runner can use it
    behind one flag."""

    variant = "owner-decoder"

    def __init__(self, in_channels: int = 1, base: int = 16,
                 n_modes: int = N_MODES_DEFAULT, n_frames: int = 9,
                 ribbon_across: int = 7, multiscale: bool = True):
        super().__init__()
        _require_torch()
        if not multiscale:
            raise ValueError(
                "owner-decoder variant requires multiscale=True: the "
                "point of the change is high-resolution image features "
                "with a decoder that can keep thin structures")
        self.n_modes = n_modes
        self.n_frames = n_frames
        self.ribbon_across = ribbon_across
        self.multiscale = True
        self.n_owner_ch = in_channels  # prompt channels concatenated at DECODER
        base = int(base)
        self.enc = ImageOnlyEncoder(in_channels, base)
        depth = self.enc.depth
        chs = self.enc.chs
        # decoder: prompt channels are concatenated at EVERY level, so the
        # owner geometry shapes the whole tube rather than only its root.
        dec = []
        c_prev = chs[-1]
        for lvl in range(depth):
            c_skip = chs[depth - 1 - lvl]
            c_out = chs[depth - 1 - lvl]
            dec.append(nn.ConvTranspose2d(c_prev, c_out, 4, stride=2,
                                          padding=1))
            # H347's family: a FRESH ReLU stack can die (pre-activations
            # all negative -> exactly zero output -> constant field and
            # zero gradient upstream). Measured here directly: the
            # 300-step fit arm aborted at step 30 with a flat field and
            # trunk grad 0.00e+00. Leaky keeps a slope. The ENCODER
            # keeps plain ReLU on purpose: it is the transferred mirror
            # and must stay layer-exact for the weight transfer to mean
            # what it says.
            dec.append(nn.LeakyReLU(negative_slope=0.01))
            dec.append(nn.Conv2d(c_out + c_skip + self.n_owner_ch,
                                 c_out, 3, padding=1))
            dec.append(nn.LeakyReLU(negative_slope=0.01))
            c_prev = c_out
        self.dec = nn.Sequential(*dec)
        self.feat = nn.Conv2d(base, base, 1)
        self.heat = nn.Conv2d(base, 1, 1)
        # temporal fusion over the decoded features (same shape and role
        # as the temporal variant so the checks transfer) ...
        self.temporal = nn.Conv3d(base, base, kernel_size=(3, 3, 3),
                                  padding=(1, 1, 1))
        # ... BUT the body head also reads the QUERY FRAME's own features:
        # mean-fusion smears the tip and the query frame is where the
        # evidence is. This is the review's "residual query-frame feature
        # path into the body decoder".
        self.body = nn.Conv2d(base * 2, 1, 1)
        self.mode_head = nn.Linear(base, n_modes * 4)
        self.vis_head = nn.Linear(base, 7)
        self.nobs_head = nn.Linear(base, 1)
        self.front_conv = nn.Sequential(
            nn.Conv2d(base, base, (3, 3), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(base, base, (3, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.front_head = nn.Linear(base, 1)
        self.route_head = nn.Linear(base, 1)

    def _decode(self, img, owner_2d):  # type: ignore[no-untyped-def]
        """img: (B, C, H, W) image-only; owner_2d: (B, C_own, H, W)
        prompt channels at native crop resolution."""
        f, feats = self.enc(img)
        h, w = img.shape[-2:]
        depth = self.enc.depth
        for lvl in range(depth):
            skip = feats[-(2 + lvl)]
            f = self.dec[lvl * 4](f)
            f = self.dec[lvl * 4 + 1](f)
            if f.shape[-2:] != skip.shape[-2:]:
                f = _t.nn.functional.interpolate(
                    f, size=skip.shape[-2:], mode="nearest")
            own = _t.nn.functional.interpolate(
                owner_2d, size=f.shape[-2:], mode="nearest")
            f = _t.cat([f, skip, own], dim=1)
            f = self.dec[lvl * 4 + 2](f)
            f = self.dec[lvl * 4 + 3](f)
        return f[..., :h, :w]

    def forward(self, clip, prompt: OwnerPrompt,  # type: ignore[no-untyped-def]
                query_index: int = QUERY_INDEX_DEFAULT,
                route_xy=None) -> CandidatePrediction:
        _require_torch()
        x = _owner_channel(clip, prompt)  # (B, T, C_img + C_own, H, W)
        b, t, c, h, w = x.shape
        if not 0 <= query_index < t:
            raise ValueError(f"query_index {query_index} outside clip "
                             f"with {t} frames")
        n_img = c - self.n_owner_ch
        img = x[:, :, :n_img]
        own = x[:, :, n_img:]
        dec = [self._decode(img[:, i], own[:, i]) for i in range(t)]
        vol = _t.stack(dec, dim=2)              # (B, base, T, H, W)
        vol = _t.nn.functional.leaky_relu(self.temporal(vol),
                                          negative_slope=0.01)
        fused = vol.mean(dim=2)
        # The review: "a residual query-frame feature path into the body
        # decoder so temporal fusion cannot erase all spatial evidence."
        query_feat = vol[:, :, query_index]
        pooled = fused.mean(dim=(2, 3))
        m = self.mode_head(pooled).reshape(-1, self.n_modes, 4)
        front_logits = None
        front_s = None
        route_logit = None
        if route_xy is not None:
            rb = sample_ribbon(x, route_xy, n_across=self.ribbon_across,
                               query_index=query_index)
            rq = rb["ribbon"][:, query_index]
            qf = self._decode(rq[:, :n_img], rq[:, n_img:])
            ff = self.front_conv(qf).mean(dim=-1)   # (B, base, S)
            front_logits = self.front_head(
                ff.transpose(1, 2)).reshape(b, -1)
            front_s = rb["s"].detach()              # metadata, never supervised
            route_logit = self.route_head(
                ff.max(dim=-1).values).reshape(b)
        return CandidatePrediction(
            modes_xy=m[..., :2],
            tip_scores=m[..., 2],
            owner_compat=m[..., 3],
            visibility_logits=self.vis_head(pooled),
            not_observed_score=self.nobs_head(pooled).reshape(-1),
            heat=self.heat(fused),
            body=self.body(_t.cat([fused, query_feat], dim=1)),
            front_logits=front_logits,
            front_s=front_s,
            route_logit=route_logit,
        )
