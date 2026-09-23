"""One acquisition clock for length, growth and emergence measurements."""
from dataclasses import dataclass
import math


def validate_acquisition(metadata):
    for key, source in (("seconds_per_source_frame", "cadence_source"),
                        ("micrometres_per_pixel", "calibration_source")):
        value = metadata.get(key)
        if value is not None:
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{key} must be positive and finite")
            if not str(metadata.get(source, "")).strip():
                raise ValueError(f"{key} requires {source}")
    times = metadata.get("source_times_s") or {}
    if times:
        if not str(metadata.get("cadence_source", "")).strip():
            raise ValueError("acquisition timestamps require their source")
        ordered = []
        for frame, time in times.items():
            if int(frame) != float(frame) or int(frame) < 0:
                raise ValueError("acquisition frame indices must be nonnegative integers")
            ordered.append((int(frame), float(time)))
        ordered.sort()
        if len({f for f, _ in ordered}) != len(ordered):
            raise ValueError("duplicate acquisition frame index")
        if (not all(math.isfinite(t) for _, t in ordered)
                or any(b[1] <= a[1] for a, b in zip(ordered, ordered[1:]))):
            raise ValueError("acquisition timestamps must increase with source frames")


@dataclass(frozen=True)
class AcquisitionClock:
    cadence_s: float | None
    timestamps_s: dict
    source: str

    @classmethod
    def from_metadata(cls, metadata):
        validate_acquisition(metadata)
        cadence = metadata.get('seconds_per_source_frame')
        return cls(float(cadence) if cadence is not None else None,
                   {int(f): float(t) for f, t in (metadata.get('source_times_s') or {}).items()},
                   str(metadata.get('cadence_source') or ''))

    def at(self, frame):
        if frame is None:
            return None
        # Explicit timestamps take precedence. Missing timestamps stay unknown;
        # neither playback FPS nor interpolation supplies experiment timing.
        if self.timestamps_s:
            return self.timestamps_s.get(int(frame))
        return int(frame) * self.cadence_s if self.cadence_s is not None else None
