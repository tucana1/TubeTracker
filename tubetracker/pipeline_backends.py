"""rev12 P1.3: explicit pipeline/backend selection (app + CLI).

One selection surface shared by the annotation app and the supported
CLI entry points — v30 no longer lives only in research scripts.

The registry maps a backend name to a loader that returns a callable
`predict(clip_np, prompt) -> Prediction` (or None when the backend is
not configured on this machine — reported, never silently substituted).
"""
from __future__ import annotations

from pathlib import Path

BACKENDS = ("v29-legacy", "v30-strict", "v30-native")

V30_DEFAULT_CHECKPOINT = (
    "runs/prototypes/v30/rev11_front_tailjitter/best_front_ep15.pt")


def available_backends() -> list[str]:
    return list(BACKENDS)


def load_backend(name: str, *, checkpoint: str = "", repo_root: str = ".",
                 analysis_config=None, cache_dir=None):
    """Return (predict_fn_or_None, info_dict).

    - v29-legacy: the historical point-heatmap pipeline (None here —
      it runs inside the app's own legacy path).
    - v30-strict: the metadata-driven STRICT factory. No legacy
      semantics are enabled under this name: a checkpoint that cannot
      declare itself refuses loudly and the backend reports
      `available: False` with the reason — never a silent substitute.
      Checkpoints resolve against repo_root and the info records the
      complete operative state (post-load parameter hash, schema,
      activation, preprocessing, config hash).
    """
    if name not in BACKENDS:
        raise ValueError(f"unknown pipeline backend {name!r}; "
                         f"expected one of {BACKENDS}")
    if name == "v29-legacy":
        return None, {"backend": name,
                      "note": "legacy app path; no external model"}
    if name == "v30-native":
        if not analysis_config:
            raise ValueError("v30-native requires a movie analysis configuration")
        from .movie_analysis import MovieAnalysisService
        for key in ("cap_checkpoint", "body_checkpoint"):
            if not Path(analysis_config[key]).is_file():
                raise ValueError(f"{key} does not exist: {analysis_config[key]}")
        service = MovieAnalysisService(analysis_config["cap_checkpoint"],
            body_checkpoint=analysis_config["body_checkpoint"],
            cache_dir=analysis_config.get("cache_dir") or cache_dir,
            registry_path=analysis_config.get("registry_path"),
            device=analysis_config.get("device", "cpu"))
        return service, {"backend": name, "available": True,
                         "cap_checkpoint": service.cap_checkpoint,
                         "body_checkpoint": service.body_checkpoint}
    root = Path(repo_root).resolve()
    ckpt = Path(checkpoint) if checkpoint else Path(V30_DEFAULT_CHECKPOINT)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if not ckpt.exists():
        return None, {"backend": name, "checkpoint": str(ckpt),
                      "available": False,
                      "reason": "checkpoint not found (resolved against "
                                "repo_root)"}
    import torch  # noqa: F401
    from prototypes.v30_video_apex.model_factory import (
        build_model_from_checkpoint)
    try:
        model, info = build_model_from_checkpoint(str(ckpt))
    except Exception as e:  # explicit failure, never a silent fallback
        return None, {"backend": name, "checkpoint": str(ckpt),
                      "available": False,
                      "reason": f"strict load refused: "
                                f"{type(e).__name__}: {e}"}
    model.eval()

    def predict(clip_np, prompt):
        import torch as _t
        clip = _t.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)
        with _t.no_grad():
            return model.forward(clip, prompt)

    return predict, {"backend": name, "checkpoint": str(ckpt),
                     "available": True,
                     "model_kwargs": info.get("model_kwargs"),
                     "semantics": info.get("semantics"),
                     "parameter_hash": info.get("parameter_hash"),
                     "model_schema": info.get("model_schema"),
                     "activation": info.get("activation"),
                     "preprocessing": info.get("preprocessing"),
                     "config_hash": info.get("config_hash")}
