#!/bin/zsh
# Double-click to open TubeTracker. Quit it from its menu (or close this window).
cd "$(dirname "$0")" || exit 1
exec ./Start_TubeTracker_local "$@"
