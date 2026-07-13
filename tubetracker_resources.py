"""Validated access to TubeTracker's versioned runtime assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2 as cv


PROJECT_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = PROJECT_ROOT / "assets"
TIP_TEMPLATE_DIR = ASSET_ROOT / "tip_templates"
TIP_TEMPLATE_MANIFEST = TIP_TEMPLATE_DIR / "manifest.json"
LOGO_PATH = PROJECT_ROOT / "logo.png"


class ResourceError(RuntimeError):
	"""Raised when a required TubeTracker resource is absent or invalid."""


def load_logo():
	"""Load the application logo independently of the current directory."""
	logo = cv.imread(str(LOGO_PATH), cv.IMREAD_COLOR)
	if logo is None:
		raise ResourceError(f"Could not load application logo: {LOGO_PATH}")
	return logo


def load_tip_templates():
	"""Load and verify the legacy grayscale tip-matching templates."""
	try:
		manifest = json.loads(TIP_TEMPLATE_MANIFEST.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError) as exc:
		raise ResourceError(
			f"Could not read tip-template manifest: {TIP_TEMPLATE_MANIFEST}"
		) from exc

	if manifest.get("schema_version") != 1:
		raise ResourceError("Unsupported tip-template manifest schema")

	entries = manifest.get("templates")
	if not isinstance(entries, list) or not entries:
		raise ResourceError("Tip-template manifest contains no templates")

	templates = []
	seen_files = set()
	for index, entry in enumerate(entries):
		if not isinstance(entry, dict):
			raise ResourceError(f"Invalid tip-template entry at index {index}")

		filename = entry.get("file")
		if not isinstance(filename, str) or Path(filename).name != filename:
			raise ResourceError(f"Invalid tip-template filename at index {index}")
		if filename in seen_files:
			raise ResourceError(f"Duplicate tip-template filename: {filename}")
		seen_files.add(filename)

		path = TIP_TEMPLATE_DIR / filename
		image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
		if image is None:
			raise ResourceError(f"Could not load tip template: {path}")

		expected_shape = entry.get("shape")
		if list(image.shape) != expected_shape:
			raise ResourceError(
				f"Tip template {filename} has shape {list(image.shape)}; "
				f"expected {expected_shape}"
			)

		actual_hash = hashlib.sha256(image.tobytes()).hexdigest()
		if actual_hash != entry.get("pixel_sha256"):
			raise ResourceError(f"Tip template checksum mismatch: {filename}")

		templates.append(image)

	return templates
