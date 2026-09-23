"""Exact source-frame reading (P0A, Qt-free).

`FrameReader` opens a movie once and returns decoded frames by SOURCE
frame index with identity verification: OpenCV seeks are codec-
approximate, so after `set(POS_FRAMES, i)` we cross-check
`get(POS_FRAMES)` and, on mismatch beyond tolerance, resync by
sequential grab from the last known-good position. Every returned
frame carries its verified source id; callers must key annotations by
that id, never by a display slider index.

A PyAV-backed reader with container-accurate timestamps can replace
the backend later behind the same `read(i) -> (frame, verified_id)`
contract; the verification logic stays.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ReadResult:
    frame: np.ndarray
    requested_id: int
    verified_id: int
    exact: bool


class FrameReader:
    """Source-frame-exact video reader with resync fallback."""

    def __init__(self, path: str, resync_window: int = 600):
        self.path = path
        self.resync_window = resync_window
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise OSError(f"cannot open movie: {path}")
        self._known_id: int | None = None

    def __len__(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def native_size(self) -> tuple[int, int]:
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def _pos(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    def read(self, source_id: int) -> ReadResult:
        """Return the decoded frame for source_id, verified or resynced."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_id))
        ok, frame = self._cap.read()
        if not ok:
            raise OSError(f"decode failed at source frame {source_id}")
        back = self._pos() - 1
        if back == int(source_id):
            self._known_id = int(source_id)
            return ReadResult(frame, int(source_id), int(source_id), True)
        # Resync: sequential grab from min(known, requested) forward.
        start = int(source_id)
        if self._known_id is not None and self._known_id <= source_id:
            start = max(self._known_id, int(source_id) - self.resync_window)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame = None
        for _ in range(int(source_id) - start + 1):
            ok, frame = self._cap.read()
            if not ok:
                raise OSError(f"resync decode failed near {source_id}")
        verified = self._pos() - 1
        self._known_id = int(verified)
        assert frame is not None
        return ReadResult(frame, int(source_id), int(verified),
                          verified == int(source_id))

    def close(self) -> None:
        self._cap.release()
