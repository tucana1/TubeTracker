"""rev11: verify the owned-absence records through target builder + loss.

Chain (review item 7): database -> snapshot -> target builder -> loss.
- snapshot: the trial build's regions.json (checked separately; 9
  owned_absence regions incl. the 6 new ones).
- target builder: samples_from_snapshot must emit `no_tube` samples
  for them (visibility-only kind; never a tip target).
- loss: one supervised step with the trainer's own semantics
  (vis target = no_tube_visible index 3, vis_valid=1 because the
  record carries an owner key) must run and stay finite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SNAP = REPO / 'runs/prototypes/v30/snap25_plus_rev11own'
MOVIE = '/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4'


def main() -> int:
    from prototypes.v30_video_apex.targets import (
        VISIBILITY_INDEX, load_clip_pixels, samples_from_snapshot)
    samples = samples_from_snapshot(str(SNAP))
    nt = [s for s in samples if s.kind == 'no_tube']
    print(f'=== target builder: {len(nt)} no_tube samples '
          f'of {len(samples)} total')
    for s in nt:
        print(f"  {s.entry_id} | frame {s.source_frame} | "
              f"grain=({s.focus_xy[0]:.0f},{s.focus_xy[1]:.0f}) | "
              f"owner={s.owner_uuid or s.owner_key}")
    assert nt, 'no no_tube samples built — chain broken'
    print('  visibility target index for no_tube_visible:',
          VISIBILITY_INDEX.get('no_tube_visible'))

    import torch
    from prototypes.v30_video_apex.dataset import QUERY_OFFSETS
    from prototypes.v30_video_apex.inference import build_typed_prompt
    from prototypes.v30_video_apex.model import build_model
    from prototypes.v30_video_apex.train import (
        LossWeights, train_step_front)
    from tubetracker.annotation_frames import FrameReader

    # the new case: ball B at (945.4, 39.1), frame 13650
    s = next((q for q in nt if 'rev11o-002-B' in q.entry_id), nt[0])
    CS = 288
    cx, cy = float(s.focus_xy[0]), float(s.focus_xy[1])
    r = FrameReader(MOVIE)
    NW, NH = r.native_size[0], r.native_size[1]
    ox = int(min(max(cx - CS / 2, 0), max(0, NW - CS)))
    oy = int(min(max(cy - CS / 2, 0), max(0, NH - CS)))
    frames = tuple(max(0, s.source_frame + o) for o in QUERY_OFFSETS)
    missing = tuple(1 if s.source_frame + o < 0 else 0 for o in QUERY_OFFSETS)
    clip_np = load_clip_pixels(r, frames, (ox, oy, CS, CS), missing)
    r.close()
    prompt, prov = build_typed_prompt(
        s.owner_uuid or s.entry_id, crop_wh=(CS, CS), crop_origin=(ox, oy),
        human_grain_xy=(cx, cy), human_grain_radius_px=13.0)
    print(f"=== loss step on {s.entry_id}: prompt kind "
          f"{prov.get('prompt_kind')} | center "
          f"{prov.get('center_native')}")
    model = build_model('temporal', base=8, multiscale=True)
    clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)
    route = [[cx - ox, cy - oy], [cx - ox + 30, cy - oy + 30]]
    with torch.no_grad():
        probe = model.forward(clip, prompt, route_xy=route)
    S = int(probe.front_logits.shape[-1])
    weights = LossWeights(apex=1.0, visibility=0.3, body=0.0,
                          route_correct=0.0, front=0.0)
    opt = torch.optim.AdamW(list(model.parameters())[:2], lr=1e-4)
    rstep = train_step_front(
        model, opt, clip, prompt, route,
        torch.zeros(1, 1, CS, CS), {'apex_valid': torch.zeros(1)},
        torch.zeros(1, 1, S), torch.zeros(1),
        torch.tensor([VISIBILITY_INDEX['no_tube_visible']]),
        torch.ones(1), weights)
    tot = float(rstep.get('total', float('nan')))
    print('  loss parts:', {k: (round(float(v), 5)
                                if not isinstance(v, str) else v)
                            for k, v in rstep.items()})
    assert np.isfinite(tot), 'loss not finite'
    print('  total finite:', tot)
    print('CHAIN OK: database -> snapshot -> target builder -> loss')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
