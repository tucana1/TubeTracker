"""Guard project architecture, compatibility, and packaged resources."""

import hashlib
import inspect
import json
import os
import re
import unittest
from pathlib import Path

import TubeTracker as legacy
import tubetracker
from tubetracker import (
    analysis,
    anchored_tracing,
    cnn_prototype,
    curve_prototype,
    grain_pose,
    growth_front,
    material_curve_tracking,
    material_ribbon,
    models,
    pollen_anchored_chain,
    pollen_motion,
    sampling,
    topology_aware_tracing,
)
from tubetracker_resources import (
    LOGO_PATH,
    TIP_TEMPLATE_MANIFEST,
    load_logo,
    load_tip_templates,
)


class ArchitectureTests(unittest.TestCase):
    """Keep compatibility while preventing the original monolith from returning."""

    def test_legacy_module_reexports_the_package_api(self):
        """Existing scripts should receive the same classes from either import path."""
        self.assertIs(legacy.Detections, tubetracker.Detections)
        self.assertIs(legacy.ROI, tubetracker.ROI)
        self.assertIs(legacy.Track, tubetracker.Track)
        self.assertIs(legacy.Tracker, tubetracker.Tracker)

    def test_analysis_layers_do_not_import_wxpython(self):
        """Headless analysis modules should remain independent of desktop widgets."""
        for module in (
            analysis,
            anchored_tracing,
            cnn_prototype,
            curve_prototype,
            grain_pose,
            growth_front,
            material_curve_tracking,
            material_ribbon,
            models,
            pollen_anchored_chain,
            pollen_motion,
            sampling,
            topology_aware_tracing,
        ):
            source = Path(module.__file__).read_text(encoding="utf-8")
            self.assertNotRegex(source, r"(^|\n)\s*(import wx|from wx)")

    def test_internal_methods_have_descriptive_names(self):
        """Public analysis classes should not regain opaque numbered methods."""
        for cls in (models.ROI, models.Track, analysis.Tracker):
            method_names = [name for name, _ in inspect.getmembers(cls, inspect.isfunction)]
            opaque_names = [name for name in method_names if re.fullmatch(r"f\d+", name)]
            self.assertEqual(opaque_names, [], cls.__name__)

    def test_legacy_module_remains_a_small_facade(self):
        """The compatibility entry point should not accumulate implementation code."""
        line_count = len(Path(legacy.__file__).read_text(encoding="utf-8").splitlines())
        self.assertLess(line_count, 80)


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
        templates = legacy.Tracker().tip_templates()
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
        source = Path(analysis.__file__).read_text(encoding="utf-8")
        self.assertNotIn("weigths =", source)

    def test_unmaintained_motpy_dependency_is_removed(self):
        """Tracking should use LapTrack without retaining Motpy imports."""
        source = Path(analysis.__file__).read_text(encoding="utf-8").lower()
        project = (Path(legacy.__file__).parent / "pyproject.toml").read_text(
            encoding="utf-8"
        ).lower()
        self.assertNotIn("motpy", source)
        self.assertNotIn("motpy", project)
        self.assertIn("laptrack", source)
        self.assertIn("laptrack", project)


if __name__ == "__main__":
    unittest.main()
