"""Public analysis API for TubeTracker."""

# H235: lazy legacy imports — heavy tracking deps (laptrack, resources)
# load on first attribute access so lightweight modules (annotation
# schema/store/frames/export/tasks, validity) import without them.
# Direct `tubetracker.analysis` imports are unaffected.

_LAZY = {
    "Detections": (".analysis", "Detections"),
    "Tracker": (".analysis", "Tracker"),
    "bbox_center_squared_distance": (".analysis", "bbox_center_squared_distance"),
    "bbox_iou_distance": (".analysis", "bbox_iou_distance"),
    "Point": (".models", "Point"),
    "ROI": (".models", "ROI"),
    "Track": (".models", "Track"),
}

__all__ = [
    "Detections",
    "Point",
    "ROI",
    "Track",
    "Tracker",
    "bbox_center_squared_distance",
    "bbox_iou_distance",
]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name, __name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
