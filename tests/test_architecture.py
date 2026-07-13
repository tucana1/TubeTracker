"""Guard the public API and separation between analysis and desktop code."""

import inspect
import re
import unittest
from pathlib import Path

import TubeTracker as legacy
import tubetracker
from tubetracker import analysis, models


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
        for module in (analysis, models):
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


if __name__ == "__main__":
    unittest.main()
