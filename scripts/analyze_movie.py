"""Supported native movie-analysis CLI; the app calls the same service."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tubetracker.movie_analysis import MovieAnalysisService, export_analysis
from tubetracker.annotation_store import AnnotationStore
from tubetracker.analysis_dependencies import dependency_fingerprint, require_same_dependencies


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--config", help="JSON with request, cap_checkpoint and body_checkpoint")
    choice.add_argument("--job", help="Immutable app job with captured review revisions")
    p.add_argument("--project-dir", help="Project containing live annotations.db")
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"))
    args = p.parse_args(argv)
    job = json.loads(Path(args.job).read_text()) if args.job else None
    config = job["configuration"] if job else json.loads(Path(args.config).read_text())
    if job and "dependencies" in job:
        require_same_dependencies(job["dependencies"], dependency_fingerprint(config))
    root = Path(args.project_dir) if args.project_dir else Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    db = root / "annotations.db"
    reviews, tombstones = [], []
    if job:
        reviews = job["entities"]
        tombstones = job.get("tombstones", [])
    elif db.exists():
        store = AnnotationStore(db)
        try:
            reviews = store.entities()
            tombstones = store.tombstones()
        finally:
            store.close()
    service = MovieAnalysisService(config["cap_checkpoint"],
        body_checkpoint=config["body_checkpoint"],
        cache_dir=config.get("cache_dir") or str(root / "analysis_cache"),
        registry_path=config.get("registry_path") or str(root / "grain_registry.sqlite"),
        device=args.device or config.get("device", "cpu"))
    result = service.analyze(config["request"], review_entities=reviews,
        review_source=job["review_source"] if job else str(db),
        review_tombstones=tombstones,
        progress=lambda item: print(json.dumps(item), flush=True))
    if job and "dependencies" in job:
        require_same_dependencies(job["dependencies"], result["dependencies"])
    paths = export_analysis(result, args.out)
    print(json.dumps({"artifacts": paths, "rows": len(result["rows"]),
                     "cache": result["cache"], "elapsed_s": result["elapsed_wall_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
