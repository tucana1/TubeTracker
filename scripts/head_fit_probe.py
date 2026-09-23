"""Fast testbed: can the body head fit ONE tube with nothing in the way?

Run the diagnosis before spending hours on a full training run. The
trunk is frozen and only `body.weight`/`body.bias` (a 1x1 conv, five
parameters) are optimised against a real mask target, sweeping loss
shapes and reporting crop-IoU and logit spread.

Why it exists (rev8 H311): the body head had been flat in every
checkpoint, and hours of full-scale runs could not tell whether that
was structural (broken path) or dynamical (too slow). This answers it
in two minutes: with the head alone the logit spread climbs steadily
(0.000 -> 0.081 logits in 400 steps at lr 0.05) while crop-IoU stays at
the all-foreground value 0.009 -- so the path is fine and the head is
roughly two orders of magnitude too slow to cross threshold inside a
run's step budget. Sweep here first; only then launch a real run.

Original docstring follows.

Can the body head fit ONE tube if nothing else is in the way?

Freeze the whole model, optimise only `body.weight`/`body.bias` (a 1x1
conv) on the real clip against the real mask target, with the current
loss shape (split BCE + optional Dice). If this cannot localise the
tube, the head/loss path is broken; if it can, the blocker is
optimisation under the multi-head mixture.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/Users/joshjiang/Documents/TubeTracker')
from prototypes.v30_video_apex.model import (  # noqa: E402
    OwnerPrompt, build_model)
from prototypes.v30_video_apex.targets import (  # noqa: E402
    body_mask_from_raster, load_clip_pixels, samples_from_snapshot)
from tubetracker.annotation_frames import FrameReader  # noqa: E402

CS = 288
TAG = 'mask-rev8m-000'
s = next(x for x in samples_from_snapshot(
    'runs/prototypes/v30/snapshots/snap20')
    if x.kind == 'body_mask' and TAG in x.entry_id)
rdr = FrameReader(s.movie_path)
cx, cy = s.focus_xy
ox = int(min(max(cx - CS / 2, 0), max(0, 1280 - CS)))
oy = int(min(max(cy - CS / 2, 0), max(0, 1024 - CS)))
clip = np.asarray(load_clip_pixels(rdr, [int(s.source_frame)],
                                   (ox, oy, CS, CS)), dtype=np.float32)
x = torch.from_numpy(clip).unsqueeze(1).unsqueeze(0)
tgt, val = body_mask_from_raster(CS, CS, (ox, oy), s.mask_raster,
                                complete=s.complete,
                                review_region=s.review_region)
t = torch.from_numpy((tgt > 0).astype(np.float32))[None, None]
v = torch.from_numpy((val > 0).astype(np.float32))[None, None]
print(f'crop=({ox},{oy}) paint={int(t.sum())}px valid={int(v.sum())}px')

model = build_model('temporal', base=4, multiscale=True)
sd = torch.load('runs/prototypes/v30/canfit_j0/front.pt', map_location='cpu')
model.load_state_dict(sd['model_state'], strict=False)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
for p in model.body.parameters():
    p.requires_grad_(True)
prompt = OwnerPrompt(owner_id=TAG,
                     grain_mask=torch.zeros_like(t),
                     prefix_weight=1.0)     # no answer in the prompt

def iou(pred, t):
    inter = float((pred & (t > 0)).sum())
    union = float((pred | (t > 0)).sum())
    return inter / union if union else 0.0


# (label, bce_scale, dice_weight, head_lr, pos_weight, norm)
#   norm='split'  -> each class normalised by its own mass (current code)
#   norm='pixel'  -> normalised by pixel count, positives weighted by
#                    pos_weight (the uniform direction keeps a gradient)
CONFIGS = [
    ('split bce x20 lr0.5', 20.0, 0.0, 0.5, 1.0, 'split'),
    ('pixel pw20 lr0.5', 1.0, 0.0, 0.5, 20.0, 'pixel'),
    ('pixel pw100 lr0.5', 1.0, 0.0, 0.5, 100.0, 'pixel'),
    ('pixel pw20 + dice lr0.5', 1.0, 1.0, 0.5, 20.0, 'pixel'),
]
N_STEPS = 1500
for mode, bce_scale, dice_w, head_lr, pos_w, norm in CONFIGS:
    # reset head to a constant predictor each time
    with torch.no_grad():
        model.body.weight.zero_()
        model.body.bias.zero_()
    opt = torch.optim.Adam(model.body.parameters(), lr=head_lr)
    for i in range(N_STEPS + 1):
        out = model(x, prompt, query_index=0)
        logit = out.body
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, t, reduction='none')
        pos = t * v
        neg = (1.0 - t) * v
        if norm == 'split':
            term = bce_scale * 0.5 * (
                (bce * pos).sum() / pos.sum().clamp_min(1.0)
                + (bce * neg).sum() / neg.sum().clamp_min(1.0))
        else:
            w = (1.0 + (pos_w - 1.0) * t) * v
            term = bce_scale * (bce * w).sum() / v.sum().clamp_min(1.0)
        if dice_w:
            from prototypes.v30_video_apex.train import soft_dice_loss
            term = term + dice_w * soft_dice_loss(
                torch.sigmoid(logit), t, v)
        opt.zero_grad()
        term.backward()
        opt.step()
        if i % 500 == 0 or i == N_STEPS:
            with torch.no_grad():
                p = torch.sigmoid(logit)
                pred = p[0, 0].numpy() >= 0.5
                print(f'  {mode} step {i:3d}: loss={float(term):.4f} '
                      f'logit_mean={float(logit.mean()):+.2f} '
                      f'logit_std={float(logit.std()):.4f} '
                      f'crop_iou={iou(pred, t[0, 0].numpy()):.3f} '
                      f'pred_frac={float(pred.mean()):.3f}')
rdr.close()
