"""v30 infer: per-frame feature cache, batched owner queries, raw candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class FeatureCache:
    """Shared features decoded once per frame; keyed by frame+transform+model hash."""

    _store: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @staticmethod
    def key(movie_id: str, source_frame: int, transform: str,
            model_hash: str, owner_id: str = "") -> str:
        # rev5: keys are movie-qualified (frame ids collide across
        # movies). Owner stays in the key because conditioning fuses
        # early; sharing happens at the decoded-clip level, honestly.
        return f"{movie_id}|{source_frame}|{transform}|{model_hash}|{owner_id}"

    def get(self, movie_id: str, source_frame: int, transform: str,
            model_hash: str, owner_id: str = "") -> Any:
        k = self.key(movie_id, source_frame, transform, model_hash, owner_id)
        if k in self._store:
            self.hits += 1
            return self._store[k]
        self.misses += 1
        return None

    def put(self, movie_id: str, source_frame: int, transform: str,
            model_hash: str, features: Any, owner_id: str = "") -> None:
        self._store[self.key(movie_id, source_frame, transform,
                             model_hash, owner_id)] = features

    def recall_diagnostics(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "entries": len(self._store)}


def batched_owner_queries(model, clip, prompts: Sequence, batch_size: int = 4,  # type: ignore[no-untyped-def]
                          movie_id: str = "movie", source_frame: int = 0) -> list[dict]:
    """Run one shared clip through the model for many owners in batches."""
    from .model import predict_candidates

    out: list[dict] = []
    for i in range(0, len(prompts), batch_size):
        for prompt in prompts[i: i + batch_size]:
            out.extend(predict_candidates(model, clip, prompt, movie_id, source_frame))
    return out


def decode_raw_candidates(predictions: Sequence[dict]) -> list[dict]:
    """Export raw observation candidates BEFORE any smoothing/selection.

    Smoothing lives in association.py; this function must not sort, merge, or drop.
    """
    from .contracts import validate_candidate_schema

    return [validate_candidate_schema(dict(p)).__dict__ for p in predictions]
