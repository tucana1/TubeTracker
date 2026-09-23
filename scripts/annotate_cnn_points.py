#!/usr/bin/env python3
"""Create or resume a click-only pollen-center and tube-tip annotation set."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
import wx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tubetracker.cnn_prototype import (  # noqa: E402
    CNN_LABELS,
    FrameReview,
    PointLabel,
    load_dataset_manifest,
    read_frame_reviews,
    read_point_labels,
    write_dataset_manifest,
    write_frame_reviews,
    write_point_labels,
)
from tubetracker.curve_prototype import (  # noqa: E402
    CurveTraceConfig,
    detect_grain_candidates,
)


def parse_args():
    """Parse source-video, sampling, and dataset-directory settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=1000)
    parser.add_argument(
        "--suggest-pollen",
        action="store_true",
        help="Pre-populate unreviewed frames with conservative OpenCV/Hough pollen centers",
    )
    return parser.parse_args()


def initialize_dataset(video, dataset_dir, sample_count, start_frame, end_frame, width):
    """Export evenly spaced, aspect-preserving frames for resumable annotation."""
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.is_file():
        return load_dataset_manifest(manifest_path)
    capture = cv.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    original_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    original_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv.CAP_PROP_FPS))
    final_frame = frame_count - 1 if end_frame <= 0 else min(end_frame, frame_count - 1)
    if start_frame < 0 or final_frame < start_frame:
        capture.release()
        raise ValueError("Invalid annotation frame range")
    sample_count = min(max(1, sample_count), final_frame - start_frame + 1)
    source_frames = np.linspace(
        start_frame, final_frame, sample_count, dtype=int
    ).tolist()
    image_height = int(round(original_height * width / original_width))
    frames_dir = dataset_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, source_frame in enumerate(source_frames):
        capture.set(cv.CAP_PROP_POS_FRAMES, source_frame)
        success, frame = capture.read()
        if not success:
            continue
        image_name = f"frame_{index:04d}_source_{source_frame:06d}.png"
        resized = cv.resize(frame, (width, image_height), interpolation=cv.INTER_AREA)
        if not cv.imwrite(str(frames_dir / image_name), resized):
            capture.release()
            raise RuntimeError(f"Could not write annotation frame: {image_name}")
        records.append({"image_name": image_name, "source_frame": source_frame})
    capture.release()
    manifest = {
        "source_video": str(video),
        "source_frame_count": frame_count,
        "source_fps": fps,
        "original_width": original_width,
        "original_height": original_height,
        "image_width": width,
        "image_height": image_height,
        "scale_x": width / original_width,
        "scale_y": image_height / original_height,
        "frames": records,
    }
    write_dataset_manifest(manifest_path, manifest)
    write_point_labels(dataset_dir / "labels.csv", [])
    write_frame_reviews(
        dataset_dir / "frame_reviews.csv",
        [
            FrameReview(record["image_name"], record["source_frame"])
            for record in records
        ],
    )
    return load_dataset_manifest(manifest_path)


def add_pollen_suggestions(dataset_dir, manifest):
    """Pre-populate blank pollen channels with conservative Hough suggestions."""
    labels_path = dataset_dir / "labels.csv"
    labels = read_point_labels(labels_path)
    completed_images = {
        review.image_name
        for review in read_frame_reviews(dataset_dir / "frame_reviews.csv")
        if review.grain_reviewed
    }
    records = [
        record
        for record in manifest["frames"]
        if record["image_name"] not in completed_images
    ]
    if not records:
        return 0
    frames = [
        cv.imread(str(dataset_dir / "frames" / record["image_name"]))
        for record in records
    ]
    if any(frame is None for frame in frames):
        raise RuntimeError("Could not load every frame for pollen suggestions")
    config = CurveTraceConfig(
        preprocessing="background",
        min_grain_circle_score=0.45,
    )
    candidates_by_frame = detect_grain_candidates(frames, config)
    added = 0
    for record, candidates in zip(records, candidates_by_frame):
        existing = [
            label
            for label in labels
            if label.image_name == record["image_name"]
            and label.label_type == "grain"
        ]
        for candidate in candidates:
            if any(
                math.hypot(label.x - candidate.gv3.x, label.y - candidate.gv3.y)
                <= 8.0
                for label in existing
            ):
                continue
            suggestion = PointLabel(
                image_name=record["image_name"],
                source_frame=int(record["source_frame"]),
                label_type="grain",
                x=float(candidate.gv3.x),
                y=float(candidate.gv3.y),
                provenance="opencv_hough_suggestion",
                confidence=float(getattr(candidate, "circle_score", 0.0)),
            )
            labels.append(suggestion)
            existing.append(suggestion)
            added += 1
    write_point_labels(labels_path, labels)
    manifest.setdefault("suggestion_runs", []).append(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "opencv_hough_circle_with_background_correction",
            "minimum_circle_score": config.min_grain_circle_score,
            "frames_processed": len(records),
            "suggestions_added": added,
        }
    )
    write_dataset_manifest(dataset_dir / "manifest.json", manifest)
    return added


class AnnotationCanvas(wx.Panel):
    """Display one sampled frame and translate clicks into image coordinates."""

    def __init__(self, parent, controller):
        super().__init__(parent, style=wx.BORDER_SIMPLE)
        self.controller = controller
        self.frame = None
        self.bitmap = None
        self.origin = (0, 0)
        self.display_size = (1, 1)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_click)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_click)

    def set_frame(self, frame):
        """Replace the source frame and rebuild its display bitmap."""
        self.frame = frame
        self.rebuild_bitmap()

    def rebuild_bitmap(self):
        """Fit the image inside the current canvas without changing aspect ratio."""
        if self.frame is None:
            return
        canvas_width, canvas_height = self.GetClientSize()
        if canvas_width <= 1 or canvas_height <= 1:
            return
        height, width = self.frame.shape[:2]
        scale = min(canvas_width / width, canvas_height / height)
        display_width = max(1, int(round(width * scale)))
        display_height = max(1, int(round(height * scale)))
        rgb = cv.cvtColor(
            cv.resize(self.frame, (display_width, display_height), interpolation=cv.INTER_AREA),
            cv.COLOR_BGR2RGB,
        )
        self.bitmap = wx.Bitmap(wx.Image(display_width, display_height, rgb.tobytes()))
        self.origin = (
            (canvas_width - display_width) // 2,
            (canvas_height - display_height) // 2,
        )
        self.display_size = (display_width, display_height)
        self.Refresh()

    def image_point(self, event):
        """Convert a canvas click to annotation-image coordinates."""
        if self.frame is None:
            return None
        x = event.GetX() - self.origin[0]
        y = event.GetY() - self.origin[1]
        display_width, display_height = self.display_size
        if x < 0 or y < 0 or x >= display_width or y >= display_height:
            return None
        height, width = self.frame.shape[:2]
        return x * width / display_width, y * height / display_height

    def on_left_click(self, event):
        """Add a point of the currently selected class."""
        point = self.image_point(event)
        if point is not None:
            self.controller.add_label(*point)

    def on_right_click(self, event):
        """Remove the nearest point from the current frame."""
        point = self.image_point(event)
        if point is not None:
            self.controller.remove_nearest_label(*point)

    def on_size(self, event):
        """Rebuild the fitted bitmap after resizing."""
        self.rebuild_bitmap()
        event.Skip()

    def on_paint(self, event):
        """Draw the image and current grain/tip point overlays."""
        painter = wx.AutoBufferedPaintDC(self)
        painter.SetBackground(wx.Brush(wx.Colour(24, 29, 33)))
        painter.Clear()
        if self.bitmap is None or self.frame is None:
            return
        painter.DrawBitmap(self.bitmap, *self.origin)
        height, width = self.frame.shape[:2]
        display_width, display_height = self.display_size
        for label in self.controller.current_labels():
            x = self.origin[0] + label.x * display_width / width
            y = self.origin[1] + label.y * display_height / height
            if label.label_type == "grain":
                color = (
                    wx.Colour(240, 170, 35)
                    if label.provenance == "opencv_hough_suggestion"
                    else wx.Colour(40, 210, 90)
                )
                painter.SetPen(wx.Pen(color, 2))
                painter.SetBrush(wx.TRANSPARENT_BRUSH)
                painter.DrawCircle(round(x), round(y), 6)
            else:
                painter.SetPen(wx.Pen(wx.Colour(230, 60, 190), 2))
                painter.DrawLine(round(x) - 6, round(y), round(x) + 6, round(y))
                painter.DrawLine(round(x), round(y) - 6, round(x), round(y) + 6)


class AnnotationFrame(wx.Frame):
    """Provide a simple resumable workflow for click-only point labels."""

    def __init__(self, dataset_dir, manifest):
        super().__init__(None, title="TubeTracker CNN Point Annotation", size=(1280, 850))
        self.dataset_dir = dataset_dir
        self.manifest = manifest
        self.records = manifest["frames"]
        self.labels = read_point_labels(dataset_dir / "labels.csv")
        reviews = read_frame_reviews(dataset_dir / "frame_reviews.csv")
        self.reviews = {review.image_name: review for review in reviews}
        self.index = 0

        root = wx.Panel(self)
        layout = wx.BoxSizer(wx.VERTICAL)
        toolbar = wx.BoxSizer(wx.HORIZONTAL)
        self.previous_button = wx.Button(root, label="Previous")
        self.next_button = wx.Button(root, label="Next")
        self.save_button = wx.Button(root, label="Save")
        self.class_choice = wx.RadioBox(
            root,
            label="Point type",
            choices=("Pollen center", "Tube tip"),
            majorDimension=2,
            style=wx.RA_SPECIFY_COLS,
        )
        self.grain_reviewed = wx.CheckBox(root, label="Pollen centers complete")
        self.tip_reviewed = wx.CheckBox(root, label="Tube tips complete")
        self.position_text = wx.StaticText(root, label="")
        for control in (
            self.previous_button,
            self.next_button,
            self.save_button,
            self.class_choice,
            self.grain_reviewed,
            self.tip_reviewed,
            self.position_text,
        ):
            toolbar.Add(control, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)
        layout.Add(toolbar, 0, wx.EXPAND)
        self.canvas = AnnotationCanvas(root, self)
        layout.Add(self.canvas, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        root.SetSizer(layout)

        self.previous_button.Bind(wx.EVT_BUTTON, lambda event: self.move(-1))
        self.next_button.Bind(wx.EVT_BUTTON, lambda event: self.move(1))
        self.save_button.Bind(wx.EVT_BUTTON, lambda event: self.persist())
        self.grain_reviewed.Bind(wx.EVT_CHECKBOX, self.on_review_change)
        self.tip_reviewed.Bind(wx.EVT_CHECKBOX, self.on_review_change)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Maximize(True)
        self.show_frame()

    @property
    def record(self):
        """Return metadata for the currently displayed frame."""
        return self.records[self.index]

    def current_labels(self):
        """Return every point belonging to the current frame."""
        image_name = self.record["image_name"]
        return [label for label in self.labels if label.image_name == image_name]

    def show_frame(self):
        """Load one annotation image and synchronize controls."""
        frame = cv.imread(str(self.dataset_dir / "frames" / self.record["image_name"]))
        if frame is None:
            raise RuntimeError(f"Could not load {self.record['image_name']}")
        review = self.reviews[self.record["image_name"]]
        self.grain_reviewed.SetValue(review.grain_reviewed)
        self.tip_reviewed.SetValue(review.tip_reviewed)
        grain_count = sum(label.label_type == "grain" for label in self.current_labels())
        tip_count = sum(label.label_type == "tip" for label in self.current_labels())
        suggested_count = sum(
            label.label_type == "grain"
            and label.provenance == "opencv_hough_suggestion"
            for label in self.current_labels()
        )
        self.position_text.SetLabel(
            f"Frame {self.index + 1}/{len(self.records)}   "
            f"Pollen {grain_count} ({suggested_count} suggested)   Tips {tip_count}"
        )
        self.previous_button.Enable(self.index > 0)
        self.next_button.Enable(self.index + 1 < len(self.records))
        self.canvas.set_frame(frame)

    def add_label(self, x, y):
        """Add a pollen-center or tube-tip point and save immediately."""
        label_type = CNN_LABELS[self.class_choice.GetSelection()]
        self.labels.append(
            PointLabel(
                image_name=self.record["image_name"],
                source_frame=int(self.record["source_frame"]),
                label_type=label_type,
                x=float(x),
                y=float(y),
                provenance="manual",
                confidence=1.0,
            )
        )
        self.persist()
        self.show_frame()

    def remove_nearest_label(self, x, y):
        """Remove the nearest current-frame point within a small click radius."""
        candidates = self.current_labels()
        if not candidates:
            return
        distances = [math.hypot(label.x - x, label.y - y) for label in candidates]
        closest = int(np.argmin(distances))
        if distances[closest] <= 18.0:
            self.labels.remove(candidates[closest])
            self.persist()
            self.show_frame()

    def on_review_change(self, event):
        """Persist channel-completeness flags for the current frame."""
        current = self.reviews[self.record["image_name"]]
        self.reviews[current.image_name] = replace(
            current,
            grain_reviewed=self.grain_reviewed.GetValue(),
            tip_reviewed=self.tip_reviewed.GetValue(),
        )
        self.persist()

    def move(self, offset):
        """Save and display the adjacent sampled frame."""
        self.persist()
        self.index = int(np.clip(self.index + offset, 0, len(self.records) - 1))
        self.show_frame()

    def persist(self):
        """Save point labels and review state atomically enough for resuming."""
        write_point_labels(self.dataset_dir / "labels.csv", self.labels)
        write_frame_reviews(
            self.dataset_dir / "frame_reviews.csv", self.reviews.values()
        )

    def on_close(self, event):
        """Save annotations before closing the window."""
        self.persist()
        self.Destroy()


def main():
    """Initialize the annotation dataset and launch its desktop editor."""
    args = parse_args()
    video = args.video.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = initialize_dataset(
        video,
        dataset_dir,
        args.sample_count,
        args.start_frame,
        args.end_frame,
        args.image_width,
    )
    if args.suggest_pollen:
        added = add_pollen_suggestions(dataset_dir, manifest)
        print(f"Added {added} unreviewed pollen suggestions", flush=True)
        manifest = load_dataset_manifest(dataset_dir / "manifest.json")
    app = wx.App()
    frame = AnnotationFrame(dataset_dir, manifest)
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()
