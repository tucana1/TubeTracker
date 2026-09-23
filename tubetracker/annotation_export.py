"""Legacy reference-kit import + v2 export helpers (P0A, Qt-free).

`import_legacy_kit` reads the v1 blinded kit (manifest + annotations)
without modifying the originals and returns v2 entities: one Owner per
case (pipeline track_id preserved as alias), one EventTask per record
(pending unless the v1 record is completed), and Observations only for
completed v1 records. Unreviewed records become pending tasks with NO
observation — never a fabricated coordinate.
"""

from __future__ import annotations

import json
from pathlib import Path

from .annotation_schema import (
    NOT_DIRECTLY_VISIBLE,
    EventTask,
    Observation,
    Owner,
    new_uuid,
)


def import_legacy_kit(
    manifest_path: str | Path, annotations_path: str | Path
) -> dict:
    """Import a v1 kit into v2 entities (read-only on the originals)."""
    manifest = json.loads(Path(manifest_path).read_text())
    annotations = json.loads(Path(annotations_path).read_text())
    records = annotations.get("records", {})
    movie = manifest.get("source_video", "")
    movie_uuid = new_uuid()

    owners: list[Owner] = []
    tasks: list[EventTask] = []
    observations: list[Observation] = []
    for case in manifest.get("cases", []):
        owner = Owner(
            uuid=new_uuid(),
            movie_uuid=movie_uuid,
            pipeline_aliases=[int(case["track_id"])],
        )
        owners.append(owner)
        for rec in case.get("records", []):
            task = EventTask(
                uuid=new_uuid(),
                movie_uuid=movie_uuid,
                owner_uuid=owner.uuid,
                source_start=int(rec["source_frame"]),
                source_end=int(rec["source_frame"]),
                query_frames=[int(rec["source_frame"])],
                stratum=str(rec.get("stratum", "")),
                role="reference",
                task_type="apex",
                completed=False,
            )
            tasks.append(task)
            ann = records.get(rec["record_id"], {})
            if ann.get("completed"):
                pts = ann.get("points_source_xy", [])
                observations.append(
                    Observation(
                        uuid=new_uuid(),
                        owner_uuid=owner.uuid,
                        source_frame=int(rec["source_frame"]),
                        direct_state=NOT_DIRECTLY_VISIBLE,
                        direct_xy=None,
                        annotator="legacy-import",
                    )
                )
                task.completed = True
    return {
        "movie_uuid": movie_uuid,
        "movie": movie,
        "owners": owners,
        "tasks": tasks,
        "observations": observations,
    }
