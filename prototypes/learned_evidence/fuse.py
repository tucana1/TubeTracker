"""Fused evidence: the shipped network's tube probability, with a thick-tube network's where that one marks a tube too
wide to be thin.

Networks trained to see the real sample movie's thick, dark, hollow tubes see them but lose thin and faint synthetic
tubes; the shipped network is the reverse. The fused map is the shipped network's probability everywhere except where
the thick-tube network marks a wide structure: its mask (P > ``thr``), with thin holes filled (the hollow middle of a
double-walled tube, ``reach.fill_small_holes``), is opened with a disc of radius ``radius`` px, so that only structures
about 2 x ``radius`` + 1 px wide or wider survive; that core, grown by ``grow`` px (the mask's soft edge), takes the
larger of the two probabilities. Thin and faint lines stay the shipped network's, so on thin-tube movies the fused map
barely differs from it.

    fused_cache(base_cache, thick_cache, out_cache)   # both built by evaluate.prob_cache on the same movie

writes a probability cache in the same format (``sparsetrack.stack``, P x ``evaluate.SCALE_P``) that the per-bin
decoder reads like any other. Its meta records both networks' fingerprints and the rule under "fusion" (and carries no
"model_sha1" of its own: it is not one network's evidence); an existing fused cache is used again only for the same
inputs and rule.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from sparsetrack import stack

from .evaluate import SCALE_P
from .reach import fill_small_holes

RULE = {"thr": 0.5, "radius": 3.0, "fill": 4.0, "grow": 1}
THICK = Path(__file__).parent / "models" / "unet_thick_b3.pt"  # the shipped thick-tube network (round 3's B3)


def disc(r: float) -> np.ndarray:
    """Structuring element: the pixels within ``r`` px of the centre."""
    k = int(np.floor(r))
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    return (x * x + y * y <= r * r + 1e-9).astype(np.uint8)


def wide_gate(p_thick: np.ndarray, thr: float = 0.5, radius: float = 3.0, fill: float = 4.0, grow: int = 1,
              scale: float = SCALE_P) -> np.ndarray:
    """Where the thick-tube network marks a structure at least about 2 x ``radius`` + 1 px wide (``p_thick`` in cache
    units, P x ``scale``)."""
    m = np.asarray(p_thick, np.float32) > thr * scale
    if fill > 0:
        m = fill_small_holes(m, fill)
    core = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disc(radius))
    if grow > 0:
        core = cv2.dilate(core, disc(grow) if grow > 1 else np.ones((3, 3), np.uint8))
    return core.astype(bool)


def fuse(p_base: np.ndarray, p_thick: np.ndarray, **rule) -> np.ndarray:
    """One bin: ``p_base`` except inside ``wide_gate(p_thick)``, where the larger of the two."""
    gate = wide_gate(p_thick, **rule)
    return np.where(gate, np.maximum(p_base, p_thick), p_base).astype(np.asarray(p_base).dtype)


def fused_cache(base_cache: str | Path, thick_cache: str | Path, out_cache: str | Path, log=print, **rule) -> Path:
    """A probability cache of the fused evidence (meta and grains of ``base_cache``; see the module docstring)."""
    base_cache, thick_cache, out_cache = Path(base_cache), Path(thick_cache), Path(out_cache)
    rule = {**RULE, **rule}
    pb, mb = stack.load(base_cache)
    pt, mt = stack.load(thick_cache)
    if pb.shape != pt.shape:
        raise ValueError(f"{base_cache} and {thick_cache} are not the same movie ({pb.shape} vs {pt.shape})")
    record = {"base": str(base_cache), "base_sha1": mb.get("model_sha1"), "thick": str(thick_cache),
              "thick_sha1": mt.get("model_sha1"), **rule}
    if (out_cache / "meta.json").exists():
        if json.loads((out_cache / "meta.json").read_text()).get("fusion") == record:
            return out_cache
        log(f"fused cache {out_cache.name} holds other evidence: building it again")
        (out_cache / "meta.json").unlink()  # a stopped rebuild must not look finished
    out_cache.mkdir(parents=True, exist_ok=True)
    started = time.time()
    out = np.lib.format.open_memmap(out_cache / "bins.npy", mode="w+", dtype=np.float16, shape=pb.shape)
    for b in range(pb.shape[0]):
        out[b] = fuse(np.asarray(pb[b]), np.asarray(pt[b]), **rule)
    out.flush()
    del out
    meta = {k: v for k, v in mb.items() if k != "model_sha1"}
    meta.update(evidence=f"fused: {mb.get('evidence', base_cache)}; wide structures from {thick_cache}", fusion=record)
    (out_cache / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.copy(base_cache / "grains.json", out_cache / "grains.json")
    log(f"fused cache {out_cache.name}: {pb.shape[0]} bins in {time.time() - started:.0f} s")
    return out_cache
