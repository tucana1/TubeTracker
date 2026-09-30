"""Time and length units for one movie.

A movie's frames are converted to minutes with the seconds per frame, which the app derives from how long the
movie took (its duration, entered at setup) and its frame count; lengths in pixels are converted to micrometres
with the pixel size. Either may be unknown: times then stay in frames and lengths in pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def seconds_per_frame(duration_s: float | None, n_frames: int | None) -> float | None:
    """How long one frame stands for: the movie's duration over its frame count."""
    if not duration_s or not n_frames or duration_s <= 0 or n_frames <= 0:
        return None
    return float(duration_s) / float(n_frames)


def duration_seconds(hours: float | str | None = 0, minutes: float | str | None = 0,
                     seconds: float | str | None = 0) -> float | None:
    """Hours, minutes and seconds as entered (blank counts as 0) in seconds; None when all are blank or zero."""
    total = 0.0
    for value, scale in ((hours, 3600.0), (minutes, 60.0), (seconds, 1.0)):
        if value in (None, ""):
            continue
        v = float(value)
        if v < 0 or not math.isfinite(v):
            raise ValueError("a duration cannot be negative")
        total += v * scale
    return total or None


def format_minutes(minutes: float | None) -> str:
    """``143 min (2 h 23 min)``; plain minutes under an hour."""
    if minutes is None:
        return "-"
    m = float(minutes)
    if abs(m) < 60:
        return f"{m:.0f} min" if abs(m) >= 10 else f"{m:.1f} min"
    h, rest = divmod(round(m), 60)
    return f"{m:.0f} min ({h} h {rest:02d} min)"


def format_duration(seconds: float | None) -> str:
    """``7 h 18 min`` (or ``45 min``, ``50 s``)."""
    if not seconds:
        return "-"
    s = round(float(seconds))
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min" + (f" {sec} s" if sec else "")
    return f"{sec} s"


@dataclass(frozen=True)
class Units:
    """Seconds per source frame and micrometres per pixel (either may be None: unknown)."""

    s_per_frame: float | None = None
    um_per_px: float | None = None

    @classmethod
    def from_setup(cls, setup: dict | None, n_frames: int | None = None) -> "Units":
        """From a movie's setup: seconds per frame from its duration and frame count (else as entered), pixel size."""
        setup = setup or {}
        spf = seconds_per_frame(setup.get("duration_s"), n_frames or setup.get("n_frames")) or setup.get("s_per_frame")
        um = setup.get("um_per_px")
        return cls(float(spf) if spf else None, float(um) if um else None)

    @property
    def timed(self) -> bool:
        return bool(self.s_per_frame)

    @property
    def scaled(self) -> bool:
        return bool(self.um_per_px)

    def minutes(self, frame: float | None) -> float | None:
        if frame is None or not self.s_per_frame:
            return None
        return float(frame) * self.s_per_frame / 60.0

    def um(self, px: float | None) -> float | None:
        if px is None or not self.um_per_px:
            return None
        return float(px) * self.um_per_px

    def time(self, frame: float | None) -> float | None:
        """A time in the movie's display unit (minutes, else frames)."""
        if frame is None:
            return None
        return self.minutes(frame) if self.timed else float(frame)

    def length(self, px: float | None) -> float | None:
        """A length in the display unit (um, else px)."""
        if px is None:
            return None
        return self.um(px) if self.scaled else float(px)

    def rate(self, px_per_frame: float | None) -> float | None:
        """A growth rate in the display unit: um/min or px/min, else px per 1000 frames."""
        if px_per_frame is None:
            return None
        r = float(px_per_frame) * (self.um_per_px or 1.0)
        return r * 60.0 / self.s_per_frame if self.timed else r * 1000.0

    @property
    def time_unit(self) -> str:
        return "min" if self.timed else "frame"

    @property
    def length_unit(self) -> str:
        return "µm" if self.scaled else "px"

    @property
    def rate_unit(self) -> str:
        return f"{self.length_unit}/{'min' if self.timed else '1000 frames'}"

    def pair(self) -> tuple[float, float] | None:
        """(um per px, s per frame) when both are known: what SparseTrack's own tables take."""
        return (self.um_per_px, self.s_per_frame) if self.timed and self.scaled else None

    def to_json(self) -> dict:
        return {"s_per_frame": self.s_per_frame, "um_per_px": self.um_per_px, "time_unit": self.time_unit,
                "length_unit": self.length_unit, "rate_unit": self.rate_unit}
