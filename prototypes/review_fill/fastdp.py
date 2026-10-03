"""fill.dp_front compiled with numba (the same fronts, ties broken the same way), for the tuning grids only.

    from prototypes.review_fill import fastdp; fastdp.install()   # fill.read_lengths then uses it
"""
from __future__ import annotations

import numpy as np
from numba import njit

from prototypes.review_fill import fill


@njit(cache=True)
def _dp(gain, vm):
    n_bins, m = gain.shape
    score = gain[0].copy()
    back = np.zeros((n_bins, m), np.int32)
    new = np.empty(m)
    for t in range(1, n_bins):
        v = vm[t]
        for j in range(m):
            best = -np.inf
            bi = j
            for p in range(j - v, j + 1):
                if p >= 0 and score[p] > best:
                    best = score[p]
                    bi = p
            new[j] = gain[t, j] + best
            back[t, j] = bi
        for j in range(m):
            score[j] = new[j]
    front = np.zeros(n_bins, np.int32)
    front[n_bins - 1] = np.argmax(score)
    for t in range(n_bins - 1, 0, -1):
        front[t - 1] = back[t, front[t]]
    return front


def dp_front(evidence, vmax, lo=None, hi=None, unary=None):
    n_bins, n = evidence.shape
    vm = np.ascontiguousarray(np.broadcast_to(np.asarray(vmax, np.int64), (n_bins,)))
    gain = np.concatenate([np.zeros((n_bins, 1)), np.cumsum(evidence, axis=1)], axis=1)
    if unary is not None:
        gain = gain + unary
    idx = np.arange(n + 1)
    if lo is not None or hi is not None:
        lo_ = np.zeros(n_bins, int) if lo is None else np.asarray(lo, int)
        hi_ = np.full(n_bins, n, int) if hi is None else np.asarray(hi, int)
        gain = np.where((idx[None] >= lo_[:, None]) & (idx[None] <= hi_[:, None]), gain, -1e18)
    return _dp(np.ascontiguousarray(gain, np.float64), np.maximum(vm, 0))


def install() -> None:
    fill.dp_front = dp_front


def check(seed: int = 0) -> None:
    """The compiled DP gives the same fronts as fill.dp_front on random problems."""
    rng = np.random.default_rng(seed)
    slow = fill.__dict__.get("_dp_front_numpy", fill.dp_front)
    for _ in range(200):
        nb, n = rng.integers(2, 40), rng.integers(1, 60)
        E = np.round(rng.standard_normal((nb, n)), 1)
        vm = rng.integers(0, 6, nb)
        lo = np.zeros(nb, int)
        hi = np.full(nb, n, int)
        if rng.random() < 0.5:
            i = rng.integers(0, nb)
            lo[i] = hi[i] = rng.integers(0, n + 1)
        a, b = slow(E, vm, lo, hi), dp_front(E, vm, lo, hi)
        assert np.array_equal(a, b), (a, b)
