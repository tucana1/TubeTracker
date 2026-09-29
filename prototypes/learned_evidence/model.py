"""A small U-Net: (bin, before, after) crops -> tube-body and tip logits."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    # BatchNorm: at inference it uses running statistics, so the output does not depend on the
    # tile size or content (GroupNorm/InstanceNorm normalise over the whole tile and would)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin: int = 3, cout: int = 2, widths=(16, 32, 64, 128)):
        super().__init__()
        self.widths = tuple(widths)
        self.enc = nn.ModuleList()
        c = cin
        for w in widths:
            self.enc.append(_block(c, w))
            c = w
        self.dec = nn.ModuleList(_block(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(len(widths) - 1)))
        self.head = nn.Conv2d(widths[0], cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for blk in self.dec:
            s = skips.pop()
            x = blk(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.head(x)


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(path: str, device: str | None = None) -> UNet:
    device = device or best_device()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = UNet(widths=ck.get("widths", (16, 32, 64, 128)))
    net.load_state_dict(ck["state"])
    return net.to(device).eval()


@torch.no_grad()
def _forward(net: UNet, x: np.ndarray) -> np.ndarray:
    _, h, w = x.shape
    m = 2 ** (len(net.widths) - 1)
    ph, pw = (-h) % m, (-w) % m
    dev = next(net.parameters()).device
    xt = torch.from_numpy(np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")).float()[None].to(dev)
    return torch.sigmoid(net(xt))[0, :, :h, :w].float().cpu().numpy()


def _ramp(n: int, overlap: int) -> np.ndarray:
    r = np.ones(n, np.float32)
    k = min(overlap, n // 2)
    if k > 0:
        edge = (np.arange(k, dtype=np.float32) + 1.0) / (k + 1.0)
        r[:k], r[n - k:] = edge, edge[::-1]
    return r


@torch.no_grad()
def predict(net: UNet, x: np.ndarray, tile: int = 256, overlap: int = 48) -> np.ndarray:
    """Probabilities (2, H, W) for a (3, H, W) input of any size: overlapping tiles blended with a
    linear ramp, so tile borders (where the receptive field is truncated) carry little weight."""
    net.eval()
    _, h, w = x.shape
    if h <= tile and w <= tile:
        return _forward(net, x)
    step = tile - overlap

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        s = list(range(0, n - tile + 1, step))
        return s if s[-1] == n - tile else s + [n - tile]

    out = np.zeros((2, h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    for y0 in starts(h):
        for x0 in starts(w):
            p = _forward(net, x[:, y0:y0 + tile, x0:x0 + tile])
            th, tw = p.shape[1:]
            wgt = _ramp(th, overlap if th == tile else 0)[:, None] * _ramp(tw, overlap if tw == tile else 0)[None, :]
            out[:, y0:y0 + th, x0:x0 + tw] += p * wgt
            weight[y0:y0 + th, x0:x0 + tw] += wgt
    return out / np.maximum(weight, 1e-6)
