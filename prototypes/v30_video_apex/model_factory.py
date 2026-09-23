"""Checkpoint-driven model factory (rev9 WP-A.4).

The audit found the movie runner constructing the OLD architecture — it
could not load a current checkpoint (shape mismatch: `base 4` weights
against a `base 8` model, and no multiscale), the trainer reloading with
an unchecked `strict=False`, and the saved config omitting ten knobs
(multiscale, jitter, objective, head LR, prompt mix, refs, confusable
controls, head weights) while recording optimizer groups as an
anonymous `[0.5, 0.003]` list.

Rule: a consumer rebuilds the model from the CHECKPOINT'S OWN metadata.
No architecture guessing, no silent partial loads — a checkpoint that
does not declare its architecture is refused, and a weight gap is an
error unless the caller explicitly declares the exception.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

# Fields a checkpoint must carry for its model to be rebuildable.
REQUIRED_MODEL_KEYS = ("variant", "base")

# rev10 WP-A: semantic versioning. ReLU and LeakyReLU preserve state-
# dict shapes but are NOT the same function, so identical weights do not
# make identical models. Old checkpoints must be replayed under an
# explicitly declared semantics.
CURRENT_MODEL_SCHEMA = 2
CURRENT_ACTIVATION = "leaky_relu:0.01"
CURRENT_PREPROCESSING = ("native/255 grayscale, 9-frame QUERY_OFFSETS "
                         "window, crop origin in native pixels")

# The ONLY permitted legacy weight gap, as an exact key set. The previous
# exception stripped route_head keys and re-loaded with strict=False,
# which also accepted a checkpoint missing `body.weight` (audit fixture
# malformed-checkpoint-probes/missing-body-weight.pt).
LEGACY_WEIGHT_GAP = frozenset({"route_head.weight", "route_head.bias"})


class CheckpointContractError(RuntimeError):
    """The checkpoint cannot be rebuilt/loaded without guessing."""


class LegacyCheckpointError(CheckpointContractError):
    """The checkpoint predates the complete-config/semantics contract.

    rev11: only THIS subclass may be rescued by an explicitly requested
    legacy migration; every other contract failure is final. The movie
    runner used to catch the broad parent and silently rebuild a
    different architecture.
    """


# rev11: schemas whose metadata is understood by this reader. A declared
# schema outside this set is refused even with a legacy flag — its
# semantics are unknown, so loading it here would be guessing.
SUPPORTED_MODEL_SCHEMAS = (2,)


def parameter_hash(model_or_state) -> str:
    """Deterministic sha256 over a model's (or state dict's) parameters.

    rev11 loader contract: the runner records the POST-LOAD hash of the
    actual model and the audit compares it with the checkpoint; a loader
    that silently rewrites trained weights cannot pass that comparison.
    """
    state = (model_or_state.state_dict()
             if hasattr(model_or_state, "state_dict") else model_or_state)
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name]
        if not torch.is_tensor(t):
            continue
        t = t.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        try:
            h.update(t.numpy().tobytes())
        except TypeError:  # bfloat16 and other numpy-less dtypes
            h.update(t.to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def read_checkpoint_payload(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise CheckpointContractError(f"no checkpoint at {p}")
    try:
        payload = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # pragma: no cover - torch-specific
        raise CheckpointContractError(f"unreadable checkpoint {p}: {e}")
    if not isinstance(payload, dict) or "model_state" not in payload:
        keys = sorted(payload)[:8] if isinstance(payload, dict) else \
            type(payload).__name__
        raise CheckpointContractError(
            f"{p} has no model_state (keys={keys})")
    return payload


def checkpoint_config(payload: dict) -> dict:
    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise CheckpointContractError("checkpoint carries no config dict")
    return cfg


def model_kwargs_from_config(cfg: dict) -> dict:
    """Validated kwargs for `build_model`, or a clear refusal."""
    missing = [k for k in REQUIRED_MODEL_KEYS if k not in cfg]
    if missing:
        raise LegacyCheckpointError(
            f"checkpoint config is missing {missing}; it predates the "
            f"complete-config contract (WP-A.4) and its architecture "
            f"cannot be rebuilt without guessing")
    kwargs = {"variant": str(cfg["variant"]), "base": int(cfg["base"])}
    if "multiscale" in cfg:
        kwargs["multiscale"] = bool(cfg["multiscale"])
    if "presence_head" in cfg:
        # rev11: the explicit presence head is part of the declared
        # architecture — a checkpoint that trained it rebuilds it.
        kwargs["presence_head"] = bool(cfg["presence_head"])
    if "local_cap_head" in cfg:
        # rev11 section 8B: native local-cap scorer, same discipline.
        kwargs["local_cap_head"] = bool(cfg["local_cap_head"])
    if "cap_window_head" in cfg:
        # rev13 W3: native cap scorer over an ordered time window —
        # declared like every other head, rebuilt by the strict factory.
        kwargs["cap_window_head"] = bool(cfg["cap_window_head"])
        if cfg["cap_window_head"]:
            kwargs["cap_window_semantics"] = int(cfg.get("cap_window_semantics", 1))
    if "cap_window_encoder" in cfg:
        # rev13 W3 contingency: the encoder-fed variant of that scorer.
        kwargs["cap_window_encoder"] = bool(cfg["cap_window_encoder"])
    return kwargs


def build_model_from_checkpoint(path: str | Path,
                                device: str | None = None,
                                allow_route_head_gap: bool = False,
                                allow_legacy_semantics: bool = False,
                                extra_kwargs: dict | None = None,
                                require_multiscale_declared: bool = True,
                                ):
    """Build the model a checkpoint describes and load it strictly.

    Returns (model, info). `info` carries the config, its hash, the
    manifest, the load report and the POST-LOAD parameter hash.

    Raises CheckpointContractError when the architecture is not
    declared or the weights do not fit exactly. `allow_route_head_gap`
    is the single declared exception (pre-route_head checkpoints): it
    zeroes the route head ONLY when that head is genuinely absent in
    full — present trained weights are always preserved (rev11).
    """
    from prototypes.v30_video_apex.model import build_model

    payload = read_checkpoint_payload(path)
    cfg = checkpoint_config(payload)
    if require_multiscale_declared and "multiscale" not in cfg:
        raise LegacyCheckpointError(
            "checkpoint does not declare `multiscale`; refusing to guess "
            "the encoder architecture (WP-A.4)")
    kwargs = model_kwargs_from_config(cfg)
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    # ---- rev11 metadata validation: schema, activation and
    # preprocessing are checked against supported definitions, not
    # merely compared as free strings. A WRONG declared value is final
    # (no flag rescues it); only fully absent metadata needs a declared
    # legacy replay.
    declared_schema = payload.get("model_schema", cfg.get("model_schema"))
    declared_act = payload.get("activation", cfg.get("activation"))
    declared_pre = payload.get("preprocessing", cfg.get("preprocessing"))
    schema_i: int | None = None
    if declared_act is not None and str(declared_act) != CURRENT_ACTIVATION:
        raise CheckpointContractError(
            f"refuse: checkpoint activation {declared_act!r} is not the "
            f"current {CURRENT_ACTIVATION!r}; replay it under its own "
            f"declared architecture instead of this one")
    if declared_schema is not None:
        try:
            schema_i = int(declared_schema)
        except (TypeError, ValueError):
            raise CheckpointContractError(
                f"refuse: model_schema {declared_schema!r} is not an "
                f"integer")
        if schema_i not in SUPPORTED_MODEL_SCHEMAS:
            raise CheckpointContractError(
                f"refuse: model_schema {schema_i} is not a supported "
                f"schema {list(SUPPORTED_MODEL_SCHEMAS)}; its semantics "
                f"are unknown to this reader")
    if declared_pre is not None and str(declared_pre) != CURRENT_PREPROCESSING:
        raise CheckpointContractError(
            f"refuse: checkpoint preprocessing {declared_pre!r} is not "
            f"the current {CURRENT_PREPROCESSING!r}; a model built "
            f"under other input conventions is not this model")
    if declared_schema is None or declared_act is None or declared_pre is None:
        if not allow_legacy_semantics:
            raise LegacyCheckpointError(
                "refuse: checkpoint declares no model_schema/activation/"
                "preprocessing. ReLU and LeakyReLU keep identical weight "
                "shapes but are different functions, so old weights need "
                "an explicitly declared replay (pass "
                "allow_legacy_semantics=True).")
        semantics = "legacy-declared-replay"
    else:
        semantics = f"{declared_act} (schema {schema_i})"

    model = build_model(**kwargs)
    if device:
        model = model.to(device)
    state = payload["model_state"]
    # Load permissively ONCE, then judge the report against an exact
    # allowlist. A shape mismatch still raises (that is not a key gap).
    try:
        report = model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        raise CheckpointContractError(
            f"refuse: checkpoint weights incompatible with the declared "
            f"model ({e})") from e
    missing = list(report.missing_keys)
    unexpected = list(report.unexpected_keys)
    route_head_fallback: str | None = None
    if not allow_route_head_gap:
        if missing or unexpected:
            raise CheckpointContractError(
                f"refuse: missing={missing} unexpected={unexpected}; pass "
                f"the declared legacy-gap exception to accept "
                f"{sorted(LEGACY_WEIGHT_GAP)} only")
    else:
        bad_missing = sorted(set(missing) - LEGACY_WEIGHT_GAP)
        if bad_missing or unexpected:
            raise CheckpointContractError(
                f"refuse: the legacy exception covers only "
                f"{sorted(LEGACY_WEIGHT_GAP)}; this checkpoint also has "
                f"missing={bad_missing} unexpected={unexpected}")
        # rev11 fix: zero the head ONLY when it is genuinely absent in
        # full. A checkpoint carrying its trained route head keeps every
        # loaded parameter — the previous code zeroed PRESENT weights
        # (audit: route weight abs sum 1.28663 -> 0.0, bias -> 0.0, and
        # runtime turned that probability into a constant 0.5).
        if missing:
            if set(missing) != set(LEGACY_WEIGHT_GAP):
                raise CheckpointContractError(
                    f"refuse: partial legacy route head "
                    f"missing={sorted(missing)}; the declared gap is the "
                    f"COMPLETE key set {sorted(LEGACY_WEIGHT_GAP)} and a "
                    f"partial head has no declared meaning")
            for _p in getattr(model, "route_head",
                              torch.nn.Module()).parameters():
                with torch.no_grad():
                    _p.zero_()
            route_head_fallback = "zeroed-absent-legacy-head"
    info = {"checkpoint": str(path), "config": cfg,
            "semantics": semantics,
            "model_schema": declared_schema,
            "activation": declared_act,
            "preprocessing": declared_pre,
            "config_hash": payload.get("config_hash", ""),
            "manifest": payload.get("manifest", {}),
            "optimizer_group_lrs": cfg.get("optimizer_group_lrs", {}),
            "missing_keys": missing, "unexpected_keys": unexpected,
            "route_head_fallback": route_head_fallback,
            "parameter_hash": parameter_hash(model),
            "model_kwargs": kwargs}
    return model, info


def optimizer_groups_summary(cfg: dict) -> str:
    """Human-readable group LR line for logs (named, never a bare list)."""
    groups = cfg.get("optimizer_group_lrs")
    if isinstance(groups, dict) and groups:
        return ", ".join(f"{k}={v:g}" for k, v in sorted(groups.items()))
    lr = cfg.get("lr")
    return f"all={lr:g}" if isinstance(lr, (int, float)) else "unknown"
