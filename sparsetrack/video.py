"""Keyframe-only access to movie files through ffmpeg/ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


@dataclass(frozen=True)
class MovieInfo:
    path: str
    width: int
    height: int
    n_frames: int
    fps: float
    keyframes: tuple[int, ...]  # source-frame index of every keyframe, ascending

    @property
    def keyframe_interval(self) -> int | None:
        gaps = np.diff(self.keyframes)
        if len(gaps) and np.all(gaps == gaps[0]):
            return int(gaps[0])
        return None


def keyframe_indices(timestamps: list[str], fps: float) -> tuple[int, ...]:
    """Convert ffprobe keyframe timestamps (seconds) to source-frame indices.

    Accepts raw ffprobe CSV lines, which can carry trailing separators ("0.857143,").
    """
    fields = [t.split(",")[0].strip() for t in timestamps]
    frames = [int(round(float(t) * fps)) for t in fields if t and t != "N/A"]
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError("keyframe timestamps are not strictly increasing")
    return tuple(frames)


def probe(path: str | Path) -> MovieInfo:
    path = str(path)
    stream = json.loads(subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    stamps = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
         "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.split()
    keyframes = keyframe_indices(stamps, fps)
    n_frames = int(stream.get("nb_frames") or 0) or (keyframes[-1] + 1)
    return MovieInfo(path, int(stream["width"]), int(stream["height"]), n_frames, fps, keyframes)


def iter_keyframes(info: MovieInfo, crop: tuple[int, int, int, int] | None = None
                   ) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (source_frame, grey uint8 image) for every keyframe, in order.

    ``crop`` is (x, y, width, height) in source pixels.
    """
    width, height = info.width, info.height
    filters: list[str] = []
    if crop is not None:
        x, y, width, height = crop
        filters = ["-vf", f"crop={width}:{height}:{x}:{y}"]
    cmd = [FFMPEG, "-v", "error", "-skip_frame", "nokey", "-i", info.path, *filters,
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    size = width * height
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    decoded = 0
    try:
        for frame in info.keyframes:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            decoded += 1
            yield frame, np.frombuffer(buf, np.uint8).reshape(height, width)
    finally:
        proc.kill()
        proc.wait()
    if decoded != len(info.keyframes):
        raise RuntimeError(f"decoded {decoded} of {len(info.keyframes)} keyframes from {info.path}")
