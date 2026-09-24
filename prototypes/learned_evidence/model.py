"""A small U-Net: (bin, before, after) crops -> tube-body and tip logits."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


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


def load(path: str, device: str = "cpu") -> UNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = UNet(widths=ck.get("widths", (16, 32, 64, 128)))
    net.load_state_dict(ck["state"])
    return net.eval()


@torch.no_grad()
def predict(net: UNet, x: np.ndarray, tile: int = 256, overlap: int = 32) -> np.ndarray:
    """Probabilities (2, H, W) for a (3, H, W) input of any size, tiled with overlap."""
    _, h, w = x.shape
    m = 2 ** (len(net.widths) - 1)
    if h <= tile and w <= tile:
        ph, pw = (-h) % m, (-w) % m
        xt = torch.from_numpy(np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")).float()[None]
        return torch.sigmoid(net(xt))[0, :, :h, :w].numpy()
    out = np.zeros((2, h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    step = tile - overlap

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        s = list(range(0, n - tile + 1, step))
        return s if s[-1] == n - tile else s + [n - tile]

    for y0 in starts(h):
        for x0 in starts(w):
            p = predict(net, x[:, y0:y0 + tile, x0:x0 + tile], tile, overlap)
            out[:, y0:y0 + tile, x0:x0 + tile] += p
            weight[y0:y0 + tile, x0:x0 + tile] += 1.0
    return out / np.maximum(weight, 1e-6)
