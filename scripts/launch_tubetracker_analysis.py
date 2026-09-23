"""Prepare or launch the supported native TubeTracker analysis project."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tubetracker.analysis_project import bootstrap_project, release_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', default=str(Path.home()/'Documents/TubeTracker-annotator-projects/rev14analysis'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--print-command', action='store_true', help='Read-only preview; does not create a project')
    parser.add_argument('--verification-only', action='store_true', help='Use a separate project whose edits are workflow tests')
    args = parser.parse_args()
    project = Path(args.project_dir).resolve()
    if args.print_command:
        db = project/'annotations.db'
        saved = None
        if db.exists():
            with sqlite3.connect(db.as_uri()+'?mode=ro', uri=True) as connection:
                row = connection.execute("select data from entities where uuid='analysis-configuration'").fetchone()
                saved = json.loads(row[0]) if row else None
        config = saved or release_configuration(project, verification_only=args.verification_only)
    else:
        config = bootstrap_project(project, verification_only=args.verification_only)
    command = [str(ROOT/'.venv-annotator/bin/python'), str(ROOT/'scripts/run_annotation_app.py'),
               '--movie', config['request']['movie_path'], '--project-dir', str(project), '--pipeline', 'v30-native',
               '--actor', 'workflow-test' if config.get('verification_only') else 'reviewer']
    if args.print_command:
        print(shlex.join(command))
    elif args.prepare_only:
        print(json.dumps({'project': str(project), 'configuration': str(project/'analysis-config.json'),
                          'queue': str(project/'requested-annotations.json'), 'command': command}))
    else:
        os.chdir(ROOT)
        os.execv(command[0], command)


if __name__ == '__main__':
    main()
