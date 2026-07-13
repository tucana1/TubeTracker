#!/usr/bin/env python3
"""Backward-compatible entry point for the modular TubeTracker package."""

from tubetracker import (
    Detections,
    Point,
    ROI,
    Track,
    Tracker,
    bbox_center_squared_distance,
    bbox_iou_distance,
)
from tubetracker.gui import Screen, Screen_Control, Tracker_GUI, main

__all__ = [
    "Detections",
    "Point",
    "ROI",
    "Screen",
    "Screen_Control",
    "Track",
    "Tracker",
    "Tracker_GUI",
    "bbox_center_squared_distance",
    "bbox_iou_distance",
    "main",
]


if __name__ == "__main__":
    main()
