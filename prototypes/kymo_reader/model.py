"""KymoNet: a dilated residual CNN over (bin, arc) that reads P(tube has reached arc s by bin t).

The dynamic input (per bin and arc) and the static input (per arc: end state, before image, rim and
neighbour distances) enter through their own first layers and are summed. Ten residual blocks with
dilations up to 32 bins in time and 8 px along the path give a receptive field of ~157 bins x ~57 px.
Normalisation is per position (LayerNorm over channels), so a crop and the whole kymograph are read
alike. ~0.12 M parameters (width 32).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .features import C_DYN, C_STAT

DIL_T = (1, 2, 4, 8, 16, 32, 1, 2, 4, 8)
DIL_S = (1, 1, 2, 2, 4, 4, 1, 2, 4, 8)


class ChanNorm(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, c: int, dt: int, ds: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=(dt, ds), dilation=(dt, ds))
        self.norm = ChanNorm(c)
        self.mix = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, x):
        return x + self.mix(F.gelu(self.norm(self.conv(x))))


class KymoNet(nn.Module):
    def __init__(self, width: int = 32, dil_t=DIL_T, dil_s=DIL_S):
        super().__init__()
        self.cfg = {"width": width, "dil_t": tuple(dil_t), "dil_s": tuple(dil_s)}
        self.dyn_in = nn.Conv2d(C_DYN, width, 3, padding=1)
        self.stat_in = nn.Conv1d(C_STAT, width, 3, padding=1)
        self.norm0 = ChanNorm(width)
        self.blocks = nn.ModuleList(Block(width, dt, ds) for dt, ds in zip(dil_t, dil_s))
        self.norm1 = ChanNorm(width)
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, dyn, stat):
        h = self.dyn_in(dyn) + self.stat_in(stat)[:, :, None, :]
        h = F.gelu(self.norm0(h))
        for b in self.blocks:
            h = b(h)
        return self.head(F.gelu(self.norm1(h)))[:, 0]


def save(net: KymoNet, path, **extra) -> None:
    torch.save({"state": net.state_dict(), "cfg": net.cfg, **extra}, str(path))


def load(path, device: str = "cpu") -> KymoNet:
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    net = KymoNet(**ck["cfg"])
    net.load_state_dict(ck["state"])
    return net.eval().to(device)
