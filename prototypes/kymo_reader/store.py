"""Compact on-disk kymograph samples.

Dynamic channels (I_t - B, grey levels) are stored as int8 through a tanh curve (0.31 grey levels
per step near zero, saturating smoothly at about 60), static channels as float16. One ``.npz``
holds many samples: keys ``{i}/{field}`` plus ``{i}/info`` (JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

QSCALE = 40.0


def quant(x: np.ndarray) -> np.ndarray:
    return np.round(127.0 * np.tanh(np.asarray(x, np.float32) / QSCALE)).astype(np.int8)


def dequant(q: np.ndarray) -> np.ndarray:
    return (QSCALE * np.arctanh(np.clip(q.astype(np.float32), -126.0, 126.0) / 127.0)).astype(np.float32)


def pack(ky: dict, info: dict) -> dict:
    """A kymograph (``kymo.extract``) plus targets/metadata as a flat dict of arrays."""
    out = {"dyn": quant(ky["dyn"]), "eb": ky["eb"].astype(np.float16), "bb": ky["bb"].astype(np.float16),
           "rim": ky["rim"].astype(np.float16), "other": ky["other"].astype(np.float16),
           "valid": ky["valid"].all(axis=2).astype(np.uint8), "pts": ky["pts"].astype(np.float32),
           "normal": ky["normal"].astype(np.float32), "s": ky["s"].astype(np.float32)}
    if ky.get("turned"):
        out["dyn0"] = quant(ky["dyn0"])
        out["rot"] = ky["rot_deg"].astype(np.float32)
        out["pivot"] = ky["pivot"].astype(np.float32)
    for k in ("target", "st_len", "st_tip", "lab_y", "lab_m", "origin"):
        if k in ky:
            out[k] = ky[k]
    out["info"] = np.frombuffer(json.dumps({**info, "n_path": int(ky["n_path"]), "path_len": float(ky["path_len"]),
                                            "i0": int(ky.get("i0", 0)),
                                            "rs": int(ky["rs"]), "turned": bool(ky.get("turned"))}).encode(), np.uint8)
    return out


def save(path: str | Path, samples: list[dict]) -> None:
    flat = {f"{i}/{k}": v for i, s in enumerate(samples) for k, v in s.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **flat)
    tmp.rename(path)


def load(path: str | Path) -> list[dict]:
    z = np.load(path)
    out: dict[int, dict] = {}
    for key in z.files:
        i, k = key.split("/", 1)
        out.setdefault(int(i), {})[k] = z[key]
    samples = []
    for i in sorted(out):
        s = out[i]
        s["info"] = json.loads(bytes(s["info"]).decode())
        samples.append(s)
    return samples
