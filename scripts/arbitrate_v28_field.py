#!/usr/bin/env python3
"""Apply cross-owner tube arbitration to a completed v28 field validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tubetracker.field_arbitration import arbitrate_validation_directory


def parse_args() -> argparse.Namespace:
    """Parse the completed validation directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    """Write field-safe statuses, coordinates, and arbitration evidence."""

    output = arbitrate_validation_directory(parse_args().validation_dir)
    report = json.loads(output.read_text())
    print(
        json.dumps(
            {
                "manifest": str(output),
                "status_counts": report["field_status_counts"],
                "ownership_conflict_ids": report["ownership_conflict_ids"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
