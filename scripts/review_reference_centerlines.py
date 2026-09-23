#!/usr/bin/env python3
"""Prepare, annotate, finalize, and score blinded centerline references."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
import wx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tubetracker.reference_validation import (  # noqa: E402
    POSITIVE_REFERENCE_OUTCOMES,
    finalize_reference_kit,
    load_reference_kit,
    prepare_reference_kit,
    save_annotations,
    score_reference_kit,
)


OUTCOME_LABELS = (
    "Not reviewed",
    "No visible tube",
    "Visible tube",
    "Tube leaves image",
    "Target pollen not visible",
    "Incorrect target",
    "Ambiguous",
)
OUTCOME_VALUES = (
    "unreviewed",
    "no_visible_tube",
    "visible_tube",
    "field_censored",
    "owner_not_visible",
    "incorrect_target",
    "ambiguous",
)


def parse_args() -> argparse.Namespace:
    """Parse validation-kit preparation and review commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Create a blinded review kit")
    prepare.add_argument("movie", type=Path)
    prepare.add_argument("--identity-report", type=Path, required=True)
    prepare.add_argument("--prediction-run", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--track-ids", default="")
    prepare.add_argument("--uniform-samples", type=int, default=8)
    prepare.add_argument("--challenge-samples", type=int, default=4)
    prepare.add_argument("--crop-size", type=int, default=420)
    prepare.add_argument("--seed", type=int, default=2706)
    prepare.add_argument(
        "--identity-centers-only",
        action="store_true",
        help="Use sparse identity interpolation instead of registered owner tracks",
    )

    annotate = commands.add_parser("annotate", help="Open the blinded editor")
    annotate.add_argument("kit", type=Path)

    finalize = commands.add_parser("finalize", help="Lock completed references")
    finalize.add_argument("kit", type=Path)
    finalize.add_argument("--annotator", default="")

    score = commands.add_parser("score", help="Score predictions after locking")
    score.add_argument("kit", type=Path)
    score.add_argument("--prediction-run", type=Path, required=True)
    return parser.parse_args()


class CenterlineCanvas(wx.Panel):
    """Display an unmodified crop and collect a root-to-tip polyline."""

    def __init__(self, parent: wx.Window, controller: "ReferenceFrame") -> None:
        super().__init__(parent, style=wx.BORDER_SIMPLE)
        self.controller = controller
        self.image: np.ndarray | None = None
        self.bitmap: wx.Bitmap | None = None
        self.origin = (0, 0)
        self.display_size = (1, 1)
        self.SetMinSize((560, 560))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_click)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_click)

    def set_image(self, image: np.ndarray) -> None:
        """Show a new raw crop and rebuild the fitted bitmap."""

        self.image = image
        self.rebuild_bitmap()

    def rebuild_bitmap(self) -> None:
        """Fit the source crop inside the available canvas."""

        if self.image is None:
            return
        canvas_width, canvas_height = self.GetClientSize()
        if canvas_width <= 1 or canvas_height <= 1:
            return
        height, width = self.image.shape[:2]
        scale = min(canvas_width / width, canvas_height / height)
        display_width = max(1, int(round(width * scale)))
        display_height = max(1, int(round(height * scale)))
        resized = cv.resize(
            self.image,
            (display_width, display_height),
            interpolation=cv.INTER_NEAREST if scale > 1.0 else cv.INTER_AREA,
        )
        rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
        self.bitmap = wx.Bitmap(wx.Image(display_width, display_height, rgb.tobytes()))
        self.origin = (
            (canvas_width - display_width) // 2,
            (canvas_height - display_height) // 2,
        )
        self.display_size = (display_width, display_height)
        self.Refresh()

    def crop_point(self, event: wx.MouseEvent) -> tuple[float, float] | None:
        """Convert a mouse event to original crop coordinates."""

        if self.image is None:
            return None
        x = event.GetX() - self.origin[0]
        y = event.GetY() - self.origin[1]
        display_width, display_height = self.display_size
        if not 0 <= x < display_width or not 0 <= y < display_height:
            return None
        height, width = self.image.shape[:2]
        return x * width / display_width, y * height / display_height

    def display_point(self, source_xy: list[float]) -> tuple[int, int]:
        """Map a source-image point into the fitted crop display."""

        record = self.controller.record
        crop_x = source_xy[0] - record["crop_origin_x"]
        crop_y = source_xy[1] - record["crop_origin_y"]
        image_height, image_width = self.image.shape[:2]
        display_width, display_height = self.display_size
        x = self.origin[0] + crop_x * display_width / image_width
        y = self.origin[1] + crop_y * display_height / image_height
        return round(x), round(y)

    def on_left_click(self, event: wx.MouseEvent) -> None:
        """Append one centerline point in root-to-tip order."""

        point = self.crop_point(event)
        if point is not None:
            self.controller.add_point(*point)

    def on_right_click(self, event: wx.MouseEvent) -> None:
        """Remove the nearest centerline point."""

        point = self.crop_point(event)
        if point is not None:
            self.controller.remove_nearest_point(*point)

    def on_size(self, event: wx.SizeEvent) -> None:
        """Rebuild the bitmap after a window resize."""

        self.rebuild_bitmap()
        event.Skip()

    def on_paint(self, event: wx.PaintEvent) -> None:
        """Render the target marker and current manual centerline."""

        painter = wx.AutoBufferedPaintDC(self)
        painter.SetBackground(wx.Brush(wx.Colour(25, 29, 32)))
        painter.Clear()
        if self.bitmap is None or self.image is None:
            return
        painter.DrawBitmap(self.bitmap, *self.origin)
        record = self.controller.record
        if record["target_in_frame"]:
            target_x, target_y = self.display_point(
                [record["target_source_x"], record["target_source_y"]]
            )
            scale = self.display_size[0] / self.image.shape[1]
            offset = max(10, round((record["target_radius_px"] + 4.0) * scale))
            arm = max(6, round(6.0 * scale))
            marker_color = (
                wx.Colour(30, 160, 235)
                if record.get("target_ring_verified")
                else wx.Colour(235, 155, 30)
            )
            painter.SetPen(wx.Pen(marker_color, 3))
            for horizontal, vertical in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                corner_x = target_x + horizontal * offset
                corner_y = target_y + vertical * offset
                painter.DrawLine(
                    corner_x,
                    corner_y,
                    corner_x - horizontal * arm,
                    corner_y,
                )
                painter.DrawLine(
                    corner_x,
                    corner_y,
                    corner_x,
                    corner_y - vertical * arm,
                )

        points = [self.display_point(point) for point in self.controller.points]
        if len(points) >= 2:
            painter.SetPen(wx.Pen(wx.Colour(235, 55, 175), 3))
            for start, end in zip(points, points[1:]):
                painter.DrawLine(*start, *end)
        for index, point in enumerate(points):
            color = wx.Colour(35, 210, 90) if index == len(points) - 1 else wx.Colour(245, 210, 45)
            painter.SetPen(wx.Pen(color, 2))
            painter.SetBrush(wx.Brush(color))
            painter.DrawCircle(*point, 4)


class ReferenceFrame(wx.Frame):
    """Provide a blinded and resumable centerline-review workflow."""

    def __init__(self, kit_dir: Path) -> None:
        super().__init__(None, title="TubeTracker Reference Review", size=(1250, 900))
        self.kit_dir = kit_dir.expanduser().resolve()
        self.manifest, self.annotations = load_reference_kit(self.kit_dir)
        self.locked = (self.kit_dir / "reference_lock.json").exists()
        self.records = [
            record
            for case in self.manifest["cases"]
            for record in case["records"]
        ]
        self.case_positions = {
            record["record_id"]: (position + 1, len(case["records"]))
            for case in self.manifest["cases"]
            for position, record in enumerate(case["records"])
        }
        self.index = self.first_incomplete_index()

        root = wx.Panel(self)
        layout = wx.BoxSizer(wx.VERTICAL)
        header = wx.BoxSizer(wx.VERTICAL)
        command_bar = wx.BoxSizer(wx.HORIZONTAL)
        self.previous_button = self.icon_button(root, wx.ART_GO_BACK, "Previous sample")
        self.next_button = self.icon_button(root, wx.ART_GO_FORWARD, "Next sample")
        self.next_incomplete_button = wx.Button(root, label="Next unfinished")
        self.undo_button = self.icon_button(root, wx.ART_UNDO, "Undo last point")
        self.clear_button = self.icon_button(root, wx.ART_DELETE, "Clear centerline")
        self.complete_button = wx.Button(root, label="Complete and continue")
        self.outcome = wx.RadioBox(
            root,
            label="Observation",
            choices=OUTCOME_LABELS,
            majorDimension=len(OUTCOME_LABELS),
            style=wx.RA_SPECIFY_COLS,
        )
        self.position = wx.StaticText(root, label="")
        self.position.SetMinSize((540, -1))
        for control in (
            self.previous_button,
            self.next_button,
            self.next_incomplete_button,
            self.undo_button,
            self.clear_button,
            self.complete_button,
        ):
            command_bar.Add(control, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        command_bar.AddStretchSpacer()
        command_bar.Add(self.position, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 8)
        header.Add(command_bar, 0, wx.EXPAND)
        header.Add(self.outcome, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        layout.Add(header, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        self.canvas = CenterlineCanvas(root, self)
        layout.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 8)
        root.SetSizer(layout)

        self.previous_button.Bind(wx.EVT_BUTTON, lambda event: self.move(-1))
        self.next_button.Bind(wx.EVT_BUTTON, lambda event: self.move(1))
        self.next_incomplete_button.Bind(wx.EVT_BUTTON, self.go_to_next_incomplete)
        self.undo_button.Bind(wx.EVT_BUTTON, lambda event: self.undo_point())
        self.clear_button.Bind(wx.EVT_BUTTON, lambda event: self.clear_points())
        self.complete_button.Bind(wx.EVT_BUTTON, self.complete_and_continue)
        self.outcome.Bind(wx.EVT_RADIOBOX, self.on_outcome)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Maximize(True)
        self.show_record()

    @staticmethod
    def icon_button(parent: wx.Window, art: str, tooltip: str) -> wx.BitmapButton:
        """Create one compact familiar icon command with a tooltip."""

        bitmap = wx.ArtProvider.GetBitmap(art, wx.ART_BUTTON, (20, 20))
        button = wx.BitmapButton(parent, bitmap=bitmap, size=(36, 34))
        button.SetToolTip(tooltip)
        return button

    @property
    def record(self) -> dict:
        """Return immutable metadata for the displayed review sample."""

        return self.records[self.index]

    @property
    def annotation(self) -> dict:
        """Return mutable annotation state for the displayed sample."""

        return self.annotations["records"][self.record["record_id"]]

    @property
    def points(self) -> list[list[float]]:
        """Return the current source-coordinate centerline points."""

        return self.annotation["points_source_xy"]

    def first_incomplete_index(self) -> int:
        """Start or resume at the first unfinished sample."""

        for index, record in enumerate(self.records):
            if not self.annotations["records"][record["record_id"]]["completed"]:
                return index
        return 0

    def show_record(self) -> None:
        """Load one crop and synchronize all review controls."""

        image = cv.imread(str(self.kit_dir / self.record["crop_image"]))
        if image is None:
            raise RuntimeError(f"could not read {self.record['crop_image']}")
        outcome = self.annotation["outcome"]
        self.outcome.SetSelection(OUTCOME_VALUES.index(outcome))
        completed = sum(
            annotation["completed"]
            for annotation in self.annotations["records"].values()
        )
        state = "locked" if self.locked else (
            "complete" if self.annotation["completed"] else "unfinished"
        )
        case_position, case_count = self.case_positions[self.record["record_id"]]
        if not self.record["target_in_frame"]:
            visibility = "  |  target outside field"
        elif not self.record.get("target_ring_verified"):
            visibility = "  |  target needs confirmation"
        else:
            visibility = ""
        self.position.SetLabel(
            f"Case {self.record['case_id']}  |  "
            f"view {case_position}/{case_count}  |  "
            f"total {self.index + 1}/{len(self.records)}  |  "
            f"{completed} complete  |  {state}{visibility}"
        )
        self.previous_button.Enable(self.index > 0)
        self.next_button.Enable(self.index + 1 < len(self.records))
        for control in (
            self.outcome,
            self.undo_button,
            self.clear_button,
            self.complete_button,
        ):
            control.Enable(not self.locked)
        self.canvas.set_image(image)
        self.Layout()

    def persist(self) -> None:
        """Save annotation progress unless the kit is locked."""

        if not self.locked:
            save_annotations(self.kit_dir, self.annotations)

    def add_point(self, crop_x: float, crop_y: float) -> None:
        """Append one source-coordinate point and switch to visible tube."""

        if self.locked:
            return
        if self.annotation["outcome"] not in POSITIVE_REFERENCE_OUTCOMES:
            self.annotation["outcome"] = "visible_tube"
            self.outcome.SetSelection(OUTCOME_VALUES.index("visible_tube"))
        self.annotation["completed"] = False
        self.points.append(
            [
                float(crop_x + self.record["crop_origin_x"]),
                float(crop_y + self.record["crop_origin_y"]),
            ]
        )
        self.persist()
        self.show_record()

    def remove_nearest_point(self, crop_x: float, crop_y: float) -> None:
        """Remove a nearby point without changing the selected outcome."""

        if self.locked or not self.points:
            return
        source_x = crop_x + self.record["crop_origin_x"]
        source_y = crop_y + self.record["crop_origin_y"]
        distances = [math.hypot(x - source_x, y - source_y) for x, y in self.points]
        nearest = int(np.argmin(distances))
        if distances[nearest] <= 18.0:
            self.points.pop(nearest)
            self.annotation["completed"] = False
            self.persist()
            self.show_record()

    def undo_point(self) -> None:
        """Remove the most recently added centerline point."""

        if not self.locked and self.points:
            self.points.pop()
            self.annotation["completed"] = False
            self.persist()
            self.show_record()

    def clear_points(self) -> None:
        """Clear the current centerline and mark the sample unfinished."""

        if not self.locked:
            self.annotation["points_source_xy"] = []
            self.annotation["completed"] = False
            self.persist()
            self.show_record()

    def on_outcome(self, event: wx.CommandEvent) -> None:
        """Store the selected observation and clear incompatible points."""

        if self.locked:
            return
        outcome = OUTCOME_VALUES[self.outcome.GetSelection()]
        self.annotation["outcome"] = outcome
        self.annotation["completed"] = False
        if outcome not in POSITIVE_REFERENCE_OUTCOMES:
            self.annotation["points_source_xy"] = []
        self.persist()
        self.show_record()

    def complete_and_continue(self, event: wx.CommandEvent) -> None:
        """Validate the current sample, mark it complete, and advance."""

        outcome = self.annotation["outcome"]
        if outcome == "unreviewed":
            wx.MessageBox("Choose an observation before completing this sample.")
            return
        if outcome in POSITIVE_REFERENCE_OUTCOMES and len(self.points) < 2:
            wx.MessageBox("Mark the tube from its pollen attachment to its visible tip.")
            return
        self.annotation["completed"] = True
        self.persist()
        self.go_to_next_incomplete(event)

    def move(self, offset: int) -> None:
        """Display the adjacent sample without changing its completion state."""

        self.persist()
        self.index = int(np.clip(self.index + offset, 0, len(self.records) - 1))
        self.show_record()

    def go_to_next_incomplete(self, event: wx.CommandEvent) -> None:
        """Advance cyclically to the next unfinished sample."""

        for step in range(1, len(self.records) + 1):
            index = (self.index + step) % len(self.records)
            record = self.records[index]
            if not self.annotations["records"][record["record_id"]]["completed"]:
                self.index = index
                self.show_record()
                return
        self.show_record()
        wx.MessageBox("All samples are complete and ready to finalize.")

    def on_close(self, event: wx.CloseEvent) -> None:
        """Persist the current state before closing the editor."""

        self.persist()
        self.Destroy()


def main() -> None:
    """Run the selected preparation, annotation, locking, or scoring step."""

    args = parse_args()
    if args.command == "prepare":
        manifest = prepare_reference_kit(
            args.movie,
            args.identity_report,
            args.prediction_run,
            args.output,
            track_ids=args.track_ids or None,
            uniform_samples=args.uniform_samples,
            challenge_samples=args.challenge_samples,
            crop_size=args.crop_size,
            seed=args.seed,
            refine_centers=not args.identity_centers_only,
        )
        print(
            f"Prepared {manifest['record_count']} blinded samples across "
            f"{manifest['case_count']} cases in {args.output.resolve()}"
        )
    elif args.command == "annotate":
        app = wx.App()
        frame = ReferenceFrame(args.kit)
        frame.Show()
        app.MainLoop()
    elif args.command == "finalize":
        print(json.dumps(finalize_reference_kit(args.kit, args.annotator), indent=2))
    elif args.command == "score":
        print(json.dumps(score_reference_kit(args.kit, args.prediction_run), indent=2))


if __name__ == "__main__":
    main()
