"""Validity masks: positives win, weak tips ignored, gradient audit."""

import numpy as np

from tubetracker.validity_masks import build_validity, masked_targets


def test_positive_wins_overlap_and_weak_ignored():
    pos, ign = build_validity(64, 64, [(32.0, 32.0)], [(34.0, 32.0)])
    assert pos[32, 32] and not ign[32, 32]
    assert ign[32, 40] and not pos[32, 40]
    assert not ign[0, 0] and not pos[0, 0]


def test_ignored_locations_contribute_zero_gradient():
    import torch
    from tubetracker.cnn_prototype import masked_heatmap_loss

    tgt, valid = masked_targets(64, 64, [(20.0, 20.0)], [(50.0, 50.0)])
    logits = torch.zeros(1, 2, 64, 64, requires_grad=True)
    targets = torch.stack([torch.zeros(64, 64),
                           torch.from_numpy(tgt)]).unsqueeze(0)
    reviewed = torch.ones(1, 2)
    valid = torch.stack([torch.ones(64, 64),
                         torch.from_numpy(valid)]).unsqueeze(0)
    loss = masked_heatmap_loss(logits, targets, reviewed, valid=valid)
    loss.backward()
    g = logits.grad[0, 1].detach().numpy()
    # weak disc around (50,50) with low target must carry ~zero gradient;
    # positive disc around (20,20) must carry nonzero gradient.
    assert abs(g[45:55, 45:55]).max() < 1e-6
    assert abs(g[15:25, 15:25]).max() > 1e-4
