"""rev8 step 3: the body head must be able to see the crop, and the
owner query must be able to reach the tube's distal end.

The review measured the rev7 body output's receptive field at 9x9 px
(gradient support) and concluded: "a grain or attachment prompt cannot
establish distant tube connectivity through that spatial footprint".
These tests measure both properties directly, on the real model
classes, and keep the old architecture's number as the contrast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.model import (  # noqa: E402
    OwnerPrompt, _owner_channel, build_model)


def _field_span(model, size: int = 288, center: int | None = None) -> float:
    """Gradient support of one central body output pixel, in px."""
    c = size // 2 if center is None else center
    torch.manual_seed(0)
    clip = torch.rand(1, 9, 1, size, size, requires_grad=True)
    prompt = OwnerPrompt(owner_id="o")
    pred = model.forward(clip, prompt)
    pred.body[0, 0, c, c].backward()
    g = clip.grad.abs().sum(dim=(0, 1, 2))
    rows = torch.nonzero(g.sum(dim=1) > 0).flatten()
    cols = torch.nonzero(g.sum(dim=0) > 0).flatten()
    if len(rows) == 0 or len(cols) == 0:
        return 0.0
    return float(max(int(rows.max()) - int(rows.min()),
                     int(cols.max()) - int(cols.min())))


def _body_through_owner_channel(model, oc: torch.Tensor) -> torch.Tensor:
    """The body head's own path: per-frame encoder -> temporal -> body.

    Asserted below to reproduce `model.forward(...).body` exactly, so
    this test cannot drift from the real computation silently.
    """
    feats = [model.unet(oc[:, i])[1] for i in range(oc.shape[1])]
    vol = torch.stack(feats, dim=2)
    # MUST mirror model.py's temporal activation. It was `relu` and that
    # relu was DEAD (H347: pre-activation uniformly negative -> output
    # exactly 0 -> every head returned its bias), which is why the real
    # forward and this replay agreed on a constant. model.py now uses
    # leaky_relu; this replay has to change with it or this test's
    # "must reproduce the model's own body head" assertion fires — which
    # is exactly what it is for.
    vol = torch.nn.functional.leaky_relu(model.temporal(vol),
                                         negative_slope=0.01)
    fused = vol.mean(dim=2)
    return model.body(fused)


def test_multiscale_encoder_sees_the_crop():
    torch.manual_seed(0)
    old = build_model("temporal", base=8)
    old.eval()
    new = build_model("temporal", base=8, multiscale=True)
    new.eval()
    span_old = _field_span(old)
    span_new = _field_span(new)
    print(f"body-head receptive field: rev7={span_old:.0f}px "
          f"multiscale={span_new:.0f}px (288 crop)")
    assert span_old < 32.0, span_old          # the measured 9x9 defect
    assert span_new >= 96.0, span_new         # crop scale, not local
    assert span_new > 4 * span_old, (span_new, span_old)


def test_owner_query_reaches_the_distal_body():
    """With a 9px footprint the distal answer CANNOT depend on the
    attachment prompt; with a crop-scale footprint it must."""
    size = 288
    torch.manual_seed(0)
    clip = torch.rand(1, 9, 1, size, size)
    attach = (40.0, 40.0)          # prompt at the root
    distal = (200, 200)            # output far from the prompt

    def distal_owner_grad(model) -> tuple[float, float]:
        model.eval()
        prompt = OwnerPrompt(owner_id="o", attachment_xy=attach,
                             proximal_xy=[attach, (52.0, 52.0)])
        with torch.no_grad():
            ref = model.forward(clip, prompt).body
        oc = _owner_channel(clip, prompt).detach().requires_grad_(True)
        body = _body_through_owner_channel(model, oc)
        assert torch.allclose(body, ref, atol=1e-5), \
            "test path must reproduce the model's own body head"
        body[0, 0, distal[1], distal[0]].backward()
        g = oc.grad.abs()
        # the claim is whether the prompt's REGION can influence the
        # distal answer. Strided encoders sample the input on a
        # lattice, so a single pixel can read zero even inside the
        # field — probe a neighbourhood, which is the honest test.
        ax, ay = int(attach[0]), int(attach[1])
        at_prompt = float(g[..., max(0, ay - 10):ay + 11,
                             max(0, ax - 10):ax + 11].max())
        # where the distal pixel's footprint actually reaches
        nz = (g.sum(dim=(0, 1, 2)) > 0).nonzero()
        span = 0.0
        if nz.numel():
            span = float(max(int(nz[:, 0].max() - nz[:, 0].min()),
                             int(nz[:, 1].max() - nz[:, 1].min())))
        return at_prompt, span

    g_old, span_old = distal_owner_grad(build_model("temporal", base=8))
    g_new, span_new = distal_owner_grad(build_model("temporal", base=8,
                                                    multiscale=True))
    print(f"|d distal body / d owner prompt at the prompt|: "
          f"rev7={g_old:.3e} (footprint {span_old:.0f}px)  "
          f"multiscale={g_new:.3e} (footprint {span_new:.0f}px)")
    assert g_old == 0.0, g_old
    assert g_new > 0.0, g_new
