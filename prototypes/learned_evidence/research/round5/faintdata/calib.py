"""Operating-point shift for a model trained with faint data: the constant added to its body logit so that it marks as
many pixels P > 0.5 as v2 does on v2's own ten training shards (first 256 samples each, whole 96 px samples, eval mode).
Writes the shifted model (head bias of the body channel + delta) next to the original.

    python calib.py MODEL.pt OUT.pt
"""
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/user/TubeTracker")
from prototypes.learned_evidence.model import load  # noqa: E402

ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/faintdata")
V2 = "/home/user/TubeTracker/prototypes/learned_evidence/models/unet_v2_sample_field.pt"
SHARDS = sorted(glob.glob(str(ME.parent.parent / "le/shards/train_*.npz")))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    src, out = sys.argv[1], sys.argv[2]
    v2, net = load(V2, "cpu"), load(src, "cpu")
    lv2, lnet = [], []
    for f in SHARDS:
        x = torch.from_numpy(np.load(f)["x"][:256].astype(np.float32))
        lv2.append(v2(x)[:, 0].flatten().numpy())
        lnet.append(net(x)[:, 0].flatten().numpy())
    lv2, lnet = np.concatenate(lv2), np.concatenate(lnet)
    n_v2 = int((lv2 > 0).sum())
    t = float(np.sort(lnet)[::-1][n_v2])  # the logit above which the model marks as many pixels as v2
    delta = -t
    print(f"v2 marks {n_v2} px (P > 0.5) of {lv2.size}; {Path(src).name} marks {int((lnet > 0).sum())}; "
          f"threshold logit {t:+.3f} (P {1 / (1 + np.exp(-t)):.3f}) -> body-logit shift {delta:+.3f}")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck["state"]["head.bias"][0] += delta
    ck["args"] = dict(ck.get("args", {}), calib_delta=delta, calib_rule="match v2's P>0.5 pixel count on v2's 10 shards")
    torch.save(ck, out)
    chk = load(out, "cpu")
    n = sum(int((chk(torch.from_numpy(np.load(f)["x"][:256].astype(np.float32)))[:, 0] > 0).sum()) for f in SHARDS)
    print(f"check: shifted model marks {n} px on the same samples; saved {out}")


if __name__ == "__main__":
    main()
