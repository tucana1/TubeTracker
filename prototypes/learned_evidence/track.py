"""Grain tracking by the grain's own appearance (normalised cross-correlation of its disc and rim), bin by bin,
anchored on a post-landing reference.

Grains still landing during the first bins make the census and the "before" reference a smear: the grain then
spends the movie up to ~17 px from its census position, and a grain frame centred on the census misses its rim.
Here the grain is re-found by its dark rim (``sparsetrack.grains.detect``) in the mean of bins ``anchor`` ..
``anchor + 2`` and tracked from there, forwards and backwards, with a disc-and-rim template. Used by the prefix
decoder (``prefix.py``).
"""

from __future__ import annotations

import cv2
import numpy as np

from sparsetrack import grains as census


def _hp(img, sigma=4.0):
    img = img.astype(np.float32)
    return img - cv2.GaussianBlur(img, (0, 0), sigma)


def _subpix(res, y, x):
    def par(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d
    dy = par(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < res.shape[0] - 1 else 0.0
    dx = par(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < res.shape[1] - 1 else 0.0
    return float(np.clip(dx, -1, 1)), float(np.clip(dy, -1, 1))


def _match(hp_t, tmpl, mask, c, rt, pos, s):
    H, W = hp_t.shape
    ox, oy = int(round(pos[0])), int(round(pos[1]))
    x0, x1 = c + ox - rt - s, c + ox + rt + s + 1
    y0, y1 = c + oy - rt - s, c + oy + rt + s + 1
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None, 0.0
    res = cv2.matchTemplate(np.ascontiguousarray(hp_t[y0:y1, x0:x1]), tmpl, cv2.TM_CCORR_NORMED, mask=mask)
    res = np.nan_to_num(res, nan=-1.0)
    y, x = np.unravel_index(int(np.argmax(res)), res.shape)
    dx, dy = _subpix(res, y, x)
    return np.array([ox + x - s + dx, oy + y - s + dy]), float(res[y, x])


def track_grain(img: np.ndarray, centre: float, gr: float, anchor: int = 8, search: int = 7, pad: float = 3.0,
                recentre: float = 20.0, arrive_ncc: float = 0.8):
    """img: (T, H, W) globally registered crops centred on the census position. Returns (shifts (T, 2): the
    grain's displacement (dx, dy) from the census position in each bin, ncc (T,), anchor offset (2,))."""
    T, H, W = img.shape
    a0 = int(min(anchor, max(0, T - 4)))
    c = int(round(centre))
    rt = int(np.ceil(gr + pad))
    hp = np.stack([_hp(im) for im in img])
    yy, xx = np.mgrid[-rt:rt + 1, -rt:rt + 1]
    mask = (np.hypot(xx, yy) <= gr + pad).astype(np.float32)
    # where is the grain at the anchor bins?  First by its early look (a grain that was already settled)...
    early = hp[:3].mean(axis=0)
    t_early = np.ascontiguousarray(early[c - rt:c + rt + 1, c - rt:c + rt + 1])
    m_anchor, q_anchor = _match(hp[a0:a0 + 3].mean(axis=0), t_early, mask, c, rt, np.zeros(2), int(recentre))
    off = m_anchor if (m_anchor is not None and q_anchor >= arrive_ncc) else np.zeros(2)
    if m_anchor is None or q_anchor < arrive_ncc:
        # ...else it was still landing in the early bins (a smear): re-find it as the census does, by Hough circles
        ref_raw = img[a0:a0 + 3].mean(axis=0)
        w = int(gr + recentre + 10)
        found = census.detect(ref_raw[c - w:c + w, c - w:c + w], r_min=max(5, int(gr - 4)), r_max=int(gr + 4),
                              ring_min=5.0, body_min=15.0)
        if found:
            best = min(found, key=lambda f: np.hypot(f["x"] - w + 0.5, f["y"] - w + 0.5))
            d = np.array([best["x"] - w + 0.5, best["y"] - w + 0.5])
            if np.hypot(*d) <= recentre:
                off = d
    ref = hp[a0:a0 + 3].mean(axis=0)
    m = np.float32([[1, 0, -off[0]], [0, 1, -off[1]]])
    ref_c = cv2.warpAffine(ref, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    tmpl = np.ascontiguousarray(ref_c[c - rt:c + rt + 1, c - rt:c + rt + 1])
    shifts = np.zeros((T, 2))
    ncc = np.zeros(T)
    for order in (range(a0, T), range(a0 - 1, -1, -1)):
        pos = off.copy()
        for t in order:
            new, q = _match(hp[t], tmpl, mask, c, rt, pos, search)
            if new is not None and q > 0.3:
                pos = new
            shifts[t], ncc[t] = pos, q
    med = np.array([np.median(shifts[max(0, t - 2):t + 3], axis=0) for t in range(T)])
    bad = np.hypot(*(shifts - med).T) > 1.5
    shifts = np.where(bad[:, None], med, shifts)
    return shifts, ncc, off
