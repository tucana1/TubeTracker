"""Paths and movie lists for the research kit: the synthetic development movies, the sealed held-out set, the real
sample movie, and where probability maps and predictions are kept.

Everything lives under one folder, ``$LE_RESEARCH`` (default ``runs/learned_evidence/research``, which git ignores):

    synth/<movie>_cache/                      image caches (sparsetrack.stack), written by regen.py
    synth/synth_<kind>_s<seed>.mp4, _truth.json   the generator's movie and truth
    sparse/<model>_<movie>.npz                probability maps (sparse.py)
    preds/<movie>/<tag>.json                  per-bin decoder predictions (decode.py)

Movie names are ``<kind>s<seed>``: v5s3 (thin), v5faints30, v5thicks30, v5ws26; "real" is the real sample movie
(``runs/sparsetrack/sample_movie/cache``, which is also the synthetic movies' field).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("LE_RESEARCH", REPO / "runs" / "learned_evidence" / "research"))
SYNTH, SPARSE, PREDS, TMP = ROOT / "synth", ROOT / "sparse", ROOT / "preds", ROOT / "tmp"
FIELD = REPO / "runs" / "sparsetrack" / "sample_movie" / "cache"
MODELS = {"v2": REPO / "prototypes/learned_evidence/models/unet_v2_sample_field.pt",
          "B3": REPO / "prototypes/learned_evidence/models/unet_thick_b3.pt"}

# kind -> (sparsetrack.synth preset, overrides); the variants as ft_synth.py and data.EXTRA_PRESETS made them
KINDS = {"v5": ("v5", {}),
         "v5faint": ("v5", {"amplitude": (0.35, 0.7)}),
         "v5thick": ("v5", {"p_bright_core": 1.0, "width": (2.0, 3.0)}),
         "v5w": ("v5", {"width": (1.3, 2.5)})}
THIN = [f"v5s{s}" for s in (3, 4, 5, 6, 7, 8, 13, 14, 15)]  # 7, 8, 13-15: held out until 27 Sep 2026 (four looks)
FAINT, THICK, WIDE = ["v5faints30", "v5faints31"], ["v5thicks30", "v5thicks31"], ["v5ws26", "v5ws27"]
DEV = [*THIN, *FAINT, *THICK, *WIDE]
# the sealed held-out set of round 5: rendered only for a look at a frozen candidate, never for development
FRESH = ["v5s40", "v5s41", "v5s42", "v5s43", "v5s44", "v5faints40", "v5thicks40", "v5ws40"]
# training data so far (never evaluate on these): v2 on v5 s0, 1, 2, 9, 10, 11, v2 s0, 1, v3 s0, 2; B3 on round 3's
# synthetic negatives (seeds 41-46 and its own); new training movies take seeds >= 50
GROUPS = {"thin (v5 s3-8, s13-15)": THIN, "faint s30+s31": FAINT, "thin+faint pooled": [*THIN, *FAINT],
          "thick s30+s31": THICK, "wide s26+s27": WIDE}
FRESH_GROUPS = {"thin (v5 s40-44)": FRESH[:5], "faint s40": ["v5faints40"],
                "thin+faint pooled": [*FRESH[:5], "v5faints40"], "thick s40": ["v5thicks40"], "wide s40": ["v5ws40"]}


def parse(movie: str) -> tuple[str, int]:
    m = re.fullmatch(r"(v5faint|v5thick|v5w|v5)s(\d+)", movie)
    if not m:
        raise ValueError(f"{movie}: not a synthetic movie name (<kind>s<seed>)")
    return m.group(1), int(m.group(2))


def check(movie: str, sealed: bool = False) -> None:
    """Refuses the sealed held-out set unless the caller is the one look at a frozen candidate (``sealed=True``)."""
    if movie in FRESH and not sealed:
        raise SystemExit(f"{movie} is in the sealed held-out set: never used in development")


def config(movie: str):
    from sparsetrack.synth import preset
    kind, seed = parse(movie)
    base, extra = KINDS[kind]
    return preset(base, seed=seed, **extra)


def image_cache(movie: str, sealed: bool = False) -> Path:
    if movie == "real":
        return FIELD
    check(movie, sealed)
    kind, seed = parse(movie)
    return SYNTH / f"{kind}s{seed}_cache"


def truth_path(movie: str, sealed: bool = False) -> Path:
    check(movie, sealed)
    kind, seed = parse(movie)
    return SYNTH / f"synth_{kind}_s{seed}_truth.json"


def vmax_of(movie: str) -> float:
    return 16.0 if movie == "real" else 4.0


def sparse_path(model: str, movie: str) -> Path:
    return SPARSE / f"{model}_{movie}.npz"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)
