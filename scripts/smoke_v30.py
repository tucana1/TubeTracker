"""Smoke test for the v30 skeleton: real supervised step + honest scoring.

rev5: no million-pixel tolerances, no surrogate losses. A tiny labelled
fit must actually descend, and selected-output scoring must distinguish
an unselected correct candidate from a selected one.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from prototypes.v30_video_apex.association import beam_search_per_owner
from prototypes.v30_video_apex.dataset import ClipSampler, GroupedManifest, ManifestEntry
from prototypes.v30_video_apex.evaluate import score_probes
from prototypes.v30_video_apex.model import OwnerPrompt, build_model, predict_candidates, random_clip
from prototypes.v30_video_apex.train import LossWeights, masked_multihead_loss


def main() -> int:
    import torch

    torch.manual_seed(0)
    model = build_model("temporal", base=8)
    model.train()
    clip = random_clip(1, 9, 1, 64)
    prompt = OwnerPrompt(owner_id="owner-A")
    route = [[4.0, 4.0], [20.0, 20.0], [40.0, 44.0], [60.0, 60.0]]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    # Labelled synthetic target: apex heat blob + front index + visibility.
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    apex = torch.exp(-((xx - 60.0) ** 2 + (yy - 60.0) ** 2) / (2 * 4.0)).unsqueeze(0)
    # Front support derives from the route itself (never assumed).
    with torch.no_grad():
        probe = model.forward(clip, prompt, query_index=4, route_xy=route)  # type: ignore[call-arg]
    n_s = int(probe.front_logits.shape[1])
    front = torch.zeros(n_s)
    front[int(n_s * 0.7)] = 1.0
    target = {"apex_heat": apex, "front": front.unsqueeze(0),
              "visibility": torch.tensor([0])}
    masks = {"apex_valid": torch.ones_like(apex),
             "front_valid": torch.ones(1),
             "vis_valid": torch.ones(1)}
    first, last = 0.0, 0.0
    _started = False
    for step in range(30):
        opt.zero_grad()
        pred = model.forward(clip, prompt, query_index=4, route_xy=route)  # type: ignore[call-arg]
        assert pred.front_logits is not None
        assert pred.front_logits.shape[1] == n_s, pred.front_logits.shape
        assert pred.heat.requires_grad and pred.front_logits.requires_grad
        losses = masked_multihead_loss(
            {"apex_heat": pred.heat, "front_logits": pred.front_logits,
             "visibility_logits": pred.visibility_logits},
            target, masks, LossWeights(apex=1.0, front=1.0, visibility=0.5))
        losses["total"].backward()
        opt.step()
        last = float(losses["total"])
        if not _started:
            first, _started = last, True
    assert last < first, f"tiny fit did not descend: {first} -> {last}"
    # Honest scoring: unselected-correct scores oracle but not selected.
    cands = predict_candidates(model, clip, prompt, movie_id="m",
                               source_frame=7)
    for c in cands:
        c["tip_score"] = 0.01
    cands[0]["x_native"], cands[0]["y_native"] = 61.0, 60.5
    cands[1]["x_native"], cands[1]["y_native"] = 5.0, 5.0
    cands[1]["tip_score"] = 0.99  # wrong winner
    gold = [{"movie_id": "m", "owner_id": "owner-A", "source_frame": 7,
             "x_native": 61.0, "y_native": 60.0, "observation": "observed",
             "stratum": "clean"}]
    rep = score_probes(cands, gold, tolerance_px=5.0)
    assert rep["oracle_coverage"] == 1.0, rep
    assert rep["correct_owner_coverage"] == 0.0, rep
    beam = beam_search_per_owner([[cands[0]], [cands[1]]], "owner-A")
    assert len(beam) >= 1
    print(f"smoke_v30 OK: fit {first:.4f}->{last:.4f} "
          f"oracle={rep['oracle_coverage']} selected={rep['correct_owner_coverage']} "
          f"beam={len(beam)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
