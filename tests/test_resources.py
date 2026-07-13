"""Tests for versioned image resources and path-independent loading."""

import hashlib
import json
import os
import unittest
from pathlib import Path

import TubeTracker as tt
from tubetracker_resources import (
    LOGO_PATH,
    TIP_TEMPLATE_MANIFEST,
    load_logo,
    load_tip_templates,
)


class ResourceTests(unittest.TestCase):
    """Verify integrity and location independence of bundled image assets."""

    def test_tip_templates_match_the_versioned_manifest(self):
        """Every decoded template should match its declared shape and hash."""
        manifest = json.loads(TIP_TEMPLATE_MANIFEST.read_text(encoding="utf-8"))
        templates = load_tip_templates()

        self.assertEqual(len(templates), len(manifest["templates"]))
        for image, entry in zip(templates, manifest["templates"]):
            self.assertEqual(list(image.shape), entry["shape"])
            self.assertEqual(
                hashlib.sha256(image.tobytes()).hexdigest(),
                entry["pixel_sha256"],
            )

    def test_tracker_uses_the_validated_template_loader(self):
        """The tracker should obtain templates through the resource loader."""
        templates = tt.Tracker().tip_templates()
        self.assertEqual(len(templates), 23)

    def test_resources_load_outside_the_repository_directory(self):
        """Runtime assets should not depend on the process working directory."""
        original_directory = Path.cwd()
        try:
            os.chdir(original_directory.parent)
            self.assertEqual(len(load_tip_templates()), 23)
            self.assertGreater(load_logo().size, 0)
        finally:
            os.chdir(original_directory)

    def test_logo_exists_at_the_declared_location(self):
        """The application logo should exist where the loader expects it."""
        self.assertTrue(LOGO_PATH.is_file())

    def test_templates_are_not_embedded_in_the_python_source(self):
        """Legacy template pixel arrays should remain outside source code."""
        source = Path(tt.__file__).read_text(encoding="utf-8")
        self.assertNotIn("weigths =", source)


if __name__ == "__main__":
    unittest.main()
