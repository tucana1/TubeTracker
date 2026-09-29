"""Training through the decoder: a CRF over monotone fronts.

With per-cell logits z(t, i), a front f(t) (cells covered) scores sum_t G(t, f(t)) with
G(t, f) = sum_{i < f} z(t, i) (the per-cell Bernoulli log-likelihood up to a constant), over
sequences with 0 <= f(t) - f(t-1) <= vmax. The loss is -log P(f in C | z), C the fronts the labels
allow (f(t) in [lo(t), hi(t)]: one value for exact synthetic truth, a range for partial labels):
logsumexp over all monotone fronts minus logsumexp over the allowed ones, both by the forward
algorithm. Divided by the number of bins.
"""

from __future__ import annotations

import numpy as np
import torch

NEG = -1e4


def bounds(y: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per bin, the allowed front range [lo, hi] (cells covered) from a (target, mask) pair."""
    T, S = y.shape
    idx = np.arange(S)
    pos = (y > 0.5) & (m > 0.5)
    neg = (y < 0.5) & (m > 0.5)
    lo = np.where(pos.any(1), np.max(np.where(pos, idx[None], -1), axis=1) + 1, 0)
    hi = np.where(neg.any(1), np.min(np.where(neg, idx[None], S), axis=1), S)
    bad = lo > hi
    lo[bad], hi[bad] = 0, S
    return lo.astype(np.int64), hi.astype(np.int64)


def _forward(g: torch.Tensor, allowed: torch.Tensor, tvalid: torch.Tensor, vmax: int) -> torch.Tensor:
    """log sum over monotone fronts of exp(sum_t g[t, f(t)]), fronts restricted to ``allowed``."""
    B, T, F = g.shape
    a = torch.where(allowed[:, 0], g[:, 0], torch.full_like(g[:, 0], NEG))
    for t in range(1, T):
        shifted = [a] + [torch.cat([torch.full_like(a[:, :k], NEG), a[:, :-k]], dim=1) for k in range(1, vmax + 1)]
        w = torch.logsumexp(torch.stack(shifted, dim=-1), dim=-1)
        new = torch.where(allowed[:, t], g[:, t] + w, torch.full_like(w, NEG))
        a = torch.where(tvalid[:, t, None], new, a)
    return torch.logsumexp(a, dim=1)


def loss(z: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, tvalid: torch.Tensor, n_cells: torch.Tensor,
         vmax: int = 4, clip: float = 8.0) -> torch.Tensor:
    """Mean over the batch of -log P(front allowed by the labels | z) / (bins)."""
    B, T, S = z.shape
    zc = z.clamp(-clip, clip)
    g = torch.cat([torch.zeros_like(zc[:, :, :1]), zc.cumsum(dim=-1)], dim=-1)          # (B, T, S + 1)
    f = torch.arange(S + 1, device=z.device)
    cap = f[None, :] <= n_cells[:, None]                                                   # padded cells
    allowed_u = cap[:, None, :].expand(B, T, S + 1)
    allowed_c = allowed_u & (f[None, None, :] >= lo[:, :, None]) & (f[None, None, :] <= hi[:, :, None])
    nll = _forward(g, allowed_u, tvalid, vmax) - _forward(g, allowed_c, tvalid, vmax)
    return (nll / tvalid.sum(dim=1).clamp(min=1)).mean()


def front_loss(z: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, tvalid: torch.Tensor, n_cells: torch.Tensor,
               clip: float = 8.0) -> torch.Tensor:
    """The per-bin version (no coupling between bins, which the decoder's DP adds): for each bin a softmax
    over front positions f with score G(t, f), and the loss -log P(f in [lo(t), hi(t)]). Fully vectorised."""
    B, T, S = z.shape
    zc = z.clamp(-clip, clip)
    g = torch.cat([torch.zeros_like(zc[:, :, :1]), zc.cumsum(dim=-1)], dim=-1)          # (B, T, S + 1)
    f = torch.arange(S + 1, device=z.device)
    cap = (f[None, :] <= n_cells[:, None])[:, None, :]
    allowed = cap & (f[None, None, :] >= lo[:, :, None]) & (f[None, None, :] <= hi[:, :, None])
    neg = torch.full_like(g, NEG)
    nll = torch.logsumexp(torch.where(cap, g, neg), -1) - torch.logsumexp(torch.where(allowed, g, neg), -1)
    return ((nll * tvalid).sum(1) / tvalid.sum(1).clamp(min=1)).mean()
