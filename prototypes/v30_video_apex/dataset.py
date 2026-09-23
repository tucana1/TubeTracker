"""v30 dataset: grouped manifests, distinct-frame clip sampler, stable crop keys.

Headless: stdlib + numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class ManifestEntry:
    entry_id: str
    movie_id: str
    acquisition_group: str
    source_frame: int
    owner_id: str
    frame_path: str = ""
    has_gold: bool = True  # gold existence is explicit, never an "accepted" flag
    # rev8: the supervision kind this row carries. One owner at one
    # frame can legitimately hold SEVERAL kinds (a path/apex sample and
    # a painted body mask are different heads, not duplicates), so
    # duplicate detection keys on kind as well.
    kind: str = ""

    def validate(self) -> None:
        if not self.entry_id or not self.movie_id or not self.owner_id:
            raise ValueError("ManifestEntry ids must be non-empty")
        if not self.acquisition_group:
            raise ValueError("ManifestEntry.acquisition_group must be non-empty")
        if self.source_frame < 0:
            raise ValueError("ManifestEntry.source_frame must be >= 0")


@dataclass
class GroupedManifest:
    entries: list[ManifestEntry] = field(default_factory=list)

    def validate(self) -> None:
        seen = set()
        for e in self.entries:
            e.validate()
            if e.entry_id in seen:
                raise ValueError(f"duplicate entry_id: {e.entry_id}")
            seen.add(e.entry_id)

    def groups(self) -> dict[str, list[ManifestEntry]]:
        out: dict[str, list[ManifestEntry]] = {}
        for e in self.entries:
            out.setdefault(e.acquisition_group, []).append(e)
        return out

    def save(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [e.__dict__ for e in sorted(self.entries, key=lambda x: x.entry_id)]
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GroupedManifest":
        rows = json.loads(Path(path).read_text())
        return cls(entries=[ManifestEntry(**r) for r in rows])

    def split_by_group(
        self, train_groups: Sequence[str], dev_groups: Sequence[str], test_groups: Sequence[str]
    ) -> dict[str, "GroupedManifest"]:
        """Grouped split: a group (acquisition session) lives in exactly one split."""
        tr, dv, te = set(train_groups), set(dev_groups), set(test_groups)
        if tr & dv or tr & te or dv & te:
            raise ValueError("split groups must be disjoint")
        out = {"train": GroupedManifest(), "dev": GroupedManifest(), "test": GroupedManifest()}
        known = tr | dv | te
        for e in self.entries:
            if e.acquisition_group not in known:
                raise ValueError(f"entry {e.entry_id} group {e.acquisition_group} in no split")
            if e.acquisition_group in tr:
                out["train"].entries.append(e)
            elif e.acquisition_group in dv:
                out["dev"].entries.append(e)
            else:
                out["test"].entries.append(e)
        return out


def assert_movie_qualified(entries: Sequence[ManifestEntry]) -> None:
    """Every entry_id must start with its movie_id + '|' and no two entries
    may share a (movie_id, source_frame, owner_id, kind) key.

    Source-frame ids collide across movies; an unqualified join would mix
    labels between recordings. The same owner/frame may appear more than
    once ONLY for different kinds of supervision (a path sample and a
    painted body mask are different heads); a same-kind repeat is a real
    duplicate and raises. Raises ValueError on violation.
    """
    seen_keys: set[tuple] = set()
    for e in entries:
        e.validate()
        if not e.entry_id.startswith(e.movie_id + "|"):
            raise ValueError(
                f"entry_id {e.entry_id!r} is not movie-qualified "
                f"(must start with {e.movie_id + '|'!r})")
        key = (e.movie_id, int(e.source_frame), e.owner_id,
               str(getattr(e, "kind", "")))
        if key in seen_keys:
            raise ValueError(f"duplicate movie/frame/owner/kind key: {key}")
        seen_keys.add(key)


def stable_crop_key(entry_id: str, crop: Sequence[int], seed: int) -> str:
    """Stable crop key: same (entry, crop, seed) -> same key, always."""
    h = hashlib.sha256()
    h.update(entry_id.encode())
    h.update(json.dumps(list(crop)).encode())
    h.update(str(int(seed)).encode())
    return h.hexdigest()[:16]


QUERY_INDEX = 4  # the query is frame 4 of 9, always
QUERY_OFFSETS = (-8, -6, -4, -2, 0, 2, 4, 6, 8)  # distinct source frames


@dataclass(frozen=True)
class ClipSpec:
    entry_id: str
    movie_id: str
    owner_id: str
    center_frame: int
    frame_ids: tuple  # 9 distinct source frames
    crop_xywh: tuple  # (x, y, w, h) native coords
    seed: int
    missing_mask: tuple = ()  # 1 = padding/absent frame (not a repeated pixel)
    time_offsets: tuple = ()

    N_FRAMES: int = 9

    def validate(self) -> None:
        if len(self.frame_ids) != 9:
            raise ValueError("ClipSpec requires exactly 9 frame ids")
        live = [f for f, m in zip(self.frame_ids, self.missing_mask or (0,) * 9)
                if not m]
        if len(set(live)) != len(live):
            raise ValueError("ClipSpec live frames must be distinct "
                             "(missing frames carry missing_mask=1)")
        if len(self.crop_xywh) != 4 or self.crop_xywh[2] <= 0 or self.crop_xywh[3] <= 0:
            raise ValueError("ClipSpec.crop_xywh must be (x, y, w>0, h>0)")

    @property
    def crop_key(self) -> str:
        return stable_crop_key(self.entry_id, list(self.crop_xywh), self.seed)


class ClipSampler:
    """Reproducible clip sampler: RNG seeded by (seed, entry) — stable across epochs."""

    def __init__(self, manifest: GroupedManifest, seed: int = 0, span: int = 4):
        self.manifest = manifest
        self.seed = int(seed)
        self.span = int(span)

    def _rng(self, entry_id: str, epoch: int):  # type: ignore[no-untyped-def]
        import numpy as np

        key = stable_crop_key(entry_id, [epoch], self.seed)
        return np.random.default_rng(int(key[:8], 16))

    def sample(self, entry: ManifestEntry, epoch: int = 0,
                 owner_xy: tuple[float, float] | None = None,
                 image_hw: tuple[int, int] = (1024, 1280)) -> ClipSpec:
        rng = self._rng(entry.entry_id, epoch)
        center = int(entry.source_frame)
        # Fixed symmetric offsets: query is ALWAYS index 4. No sorting
        # that would move it (rev5: never label frame 100 while frame
        # 102 is evaluated as the query).
        frames = tuple(center + o for o in QUERY_OFFSETS)
        missing = tuple(1 if f < 0 else 0 for f in frames)
        frames = tuple(max(0, f) for f in frames)
        # Owner-centered crop clipped to image bounds (rev5: crops are
        # bounded and owner-aware, never random coordinates).
        cw, ch = 320, 320
        H, W = image_hw
        if owner_xy is not None:
            cx = int(min(max(owner_xy[0] - cw / 2
                             + float(rng.uniform(-24, 24)), 0), max(0, W - cw)))
            cy = int(min(max(owner_xy[1] - ch / 2
                             + float(rng.uniform(-24, 24)), 0), max(0, H - ch)))
        else:
            cx = int(rng.integers(0, max(1, W - cw + 1)))
            cy = int(rng.integers(0, max(1, H - ch + 1)))
        crop = (cx, cy, min(cw, W), min(ch, H))
        offsets = tuple(float(f - center) for f in frames)
        spec = ClipSpec(
            entry_id=entry.entry_id,
            movie_id=entry.movie_id,
            owner_id=entry.owner_id,
            center_frame=center,
            frame_ids=frames,
            crop_xywh=crop,
            seed=self.seed,
            missing_mask=missing,
            time_offsets=offsets,
        )
        spec.validate()
        return spec

    def sample_all(self, entries: Optional[Sequence[ManifestEntry]] = None, epoch: int = 0) -> list[ClipSpec]:
        return [self.sample(e, epoch) for e in (entries or self.manifest.entries)]
