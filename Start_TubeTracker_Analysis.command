#!/bin/bash
set -e
cd "$(dirname "$0")"
exec ./.venv-annotator/bin/python scripts/launch_tubetracker_analysis.py "$@"
