"""rev10 WP-A: device/parity canary for train, evaluator and runner.

The review: "Current inference helpers construct CPU tensors; a CUDA
checkpoint/device path needs an explicit canary before paying for a GPU
run." This measures one forward+backward at the native useful
resolution, reports time and memory, and checks that the accelerated
device reproduces the CPU result — so a GPU run is entered with a
verified configuration rather than a hope.

Run: python scripts/rev10_device_canary.py --out canary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _devices() -> list[str]:
    out = ["cpu"]
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        out.append("mps")
    if torch.cuda.is_available():
        out.append("cuda")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=8)
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--crop", type=int, default=288)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from prototypes.v30_video_apex.model import build_model, build_owner_prompt

    clip_cpu = torch.rand(1, a.frames, 1, a.crop, a.crop)
    prompt = build_owner_prompt("canary", provenance="none")
    report = {"torch": str(torch.__version__),
              "devices": _devices(), "crop": a.crop,
              "frames": a.frames, "base": a.base, "runs": {}}

    ref = None
    for dev in report["devices"]:
        # the SAME initialization on every device, or the comparison
        # measures two different models (first version did exactly that
        # and reported a 0.404 "parity difference")
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        model = build_model("temporal", base=a.base, multiscale=True).to(dev)
        model.train()
        clip = clip_cpu.to(dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            for _ in range(a.steps):
                model.zero_grad(set_to_none=True)
                out = model.forward(clip, prompt)
                loss = out.body.float().pow(2).mean()
                loss.backward()
            dt = (time.time() - t0) / a.steps
            with torch.no_grad():
                model.eval()
                val = model.forward(clip, prompt).body[0, 0].float().cpu()
            entry = {"s_per_step": round(dt, 4),
                     "body_sum": float(val.sum())}
            if dev == "cuda":
                entry["peak_memory_mb"] = round(
                    torch.cuda.max_memory_allocated() / 1e6, 1)
            if ref is None:
                ref = val
                entry["parity_vs_cpu_max_abs"] = 0.0
            else:
                entry["parity_vs_cpu_max_abs"] = float(
                    (val - ref).abs().max())
            report["runs"][dev] = entry
            print(f"{dev:>5}: {dt:.3f}s/step, body sum "
                  f"{float(val.sum()):.6f}, "
                  f"parity {entry['parity_vs_cpu_max_abs']:.2e}"
                  + (f", peak {entry.get('peak_memory_mb')} MB"
                     if "peak_memory_mb" in entry else ""))
        except Exception as e:  # noqa: BLE001
            report["runs"][dev] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{dev:>5}: FAILED {type(e).__name__}: {str(e)[:90]}")

    # every AVAILABLE device must have run and matched the CPU reference
    ok = True
    for dev in report["devices"]:
        v = report["runs"].get(dev, {})
        if "error" in v or dev not in report["runs"]:
            ok = False
        elif dev != "cpu" and (v.get("parity_vs_cpu_max_abs", 1.0) or 0.0) >= 1e-4:
            ok = False
    report["parity_ok"] = ok
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"\nparity_ok: {report['parity_ok']} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
