"""SQLite annotation store (P0A, Qt-free, stdlib only).

Single-writer project database with transactional saves, append-only
revision history, optimistic revision checks, and immutable finalized
snapshots. Large images stay outside the DB (paths only). GUI layers
talk to this module; training/eval import from it headlessly.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS entities (
    uuid TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    data TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    updated_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT NOT NULL,
    kind TEXT NOT NULL,
    data TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created_utc TEXT NOT NULL,
    actor TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS idx_revisions_uuid ON revisions(uuid);
CREATE TABLE IF NOT EXISTS snapshots (
    uuid TEXT PRIMARY KEY,
    manifest TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    finalized INTEGER NOT NULL DEFAULT 0,
    created_utc TEXT NOT NULL);
"""


def _utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class AnnotationStore:
    """Single-writer SQLite store with visible save results."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._db = sqlite3.connect(str(self.path), timeout=30.0)
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.executescript(SCHEMA_SQL)
        self._db.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema','v2.0')")
        # rev10: COMMIT the schema marker. Without this the INSERT opens
        # a deferred write transaction that stays open for the whole
        # session, holding the DB's write lock and making every offline
        # writer (builders, task staging) fail with 'database is locked'
        # while the app merely sits open.
        self._db.commit()

    def save(self, kind: str, uuid: str, data: dict,
             actor: str = "") -> int:
        """Upsert one entity transactionally; returns the new revision.

        Raises/sqlite3.OperationalError surfaces visibly on lock/full-disk
        — callers must show Saving/Saved/Save-failed from the outcome.
        """
        return self.save_many([(kind, uuid, data)], actor=actor)[0]

    def save_many(self, records, actor: str = "", *, create_only=False,
                  expected_revisions=None) -> list[int]:
        """Commit linked review changes together, with append-only history."""
        prepared = [(kind, uuid, json.dumps(data, sort_keys=True))
                    for kind, uuid, data in records]
        if len({uuid for _, uuid, _ in prepared}) != len(prepared):
            raise ValueError("one change per entity in an atomic save")
        revisions = []
        with self._db:
            if expected_revisions is not None and not self._db.in_transaction:
                self._db.execute('BEGIN IMMEDIATE')
            for kind, uuid, payload in prepared:
                if expected_revisions is not None:
                    if uuid not in expected_revisions:
                        raise ValueError('every conditional save requires an expected revision')
                    current = self._db.execute('SELECT revision FROM entities WHERE uuid=?', (uuid,)).fetchone()
                    if (current[0] if current else None) != expected_revisions[uuid]:
                        raise ValueError('entity changed before conditional save: '+uuid)
                # Reconfirming an undone record must not reuse revision 1.
                row = self._db.execute(
                    "SELECT MAX(revision) FROM (SELECT revision FROM entities WHERE uuid=? "
                    "UNION ALL SELECT revision FROM revisions WHERE uuid=?)", (uuid, uuid)).fetchone()
                revision = max(int(row[0] or 0), 0) + 1
                now = _utc()
                self._db.execute(
                    ("INSERT" if create_only else "INSERT OR REPLACE") + " INTO entities(uuid, kind, data, revision,"
                    " updated_utc) VALUES (?,?,?,?,?)",
                    (uuid, kind, payload, revision, now))
                self._db.execute(
                    "INSERT INTO revisions(uuid, kind, data, revision,"
                    " created_utc, actor) VALUES (?,?,?,?,?,?)",
                    (uuid, kind, payload, revision, now, actor))
                revisions.append(revision)
        return revisions

    def withdraw_task_review(self, uuid: str, actor: str = "") -> dict:
        """Retain geometry as history while withdrawing this task's evidence.

        Other independently reviewed tasks remain authoritative, even when
        they reference the withdrawn observation as viewing context.
        """
        task = self.load(uuid)
        if task is None or task['kind'] != 'task':
            raise ValueError('Unknown annotation task')
        withdrawal = {'task_uuid': uuid, 'reason': 'cannot_judge'}
        changes = []
        for entity in self.entities():
            data = entity['data']
            if (entity['kind'] not in ('task', 'session')
                    and (data.get('task_uuid') == uuid or data.get('source_task_uuid') == uuid)):
                changes.append((entity['kind'], entity['uuid'], dict(data,
                    review_status='withdrawn', review_withdrawal=withdrawal)))
        data = dict(task['data'], completed=True, review_verdict='unresolved',
                    review_status='withdrawn', review_withdrawal=withdrawal)
        if data.get('task_type') == 'census':
            data.update(census_complete=False, census_membership_resolved=False)
        changes.append(('task', uuid, data))
        self.save_many(changes, actor=actor)
        return data

    def load(self, uuid: str) -> dict | None:
        cur = self._db.execute(
            "SELECT kind, data, revision FROM entities WHERE uuid=?", (uuid,))
        row = cur.fetchone()
        if row is None:
            return None
        return {"kind": row[0], "data": json.loads(row[1]),
                "revision": row[2]}

    def history(self, uuid: str) -> list[dict]:
        cur = self._db.execute(
            "SELECT data, revision, created_utc, actor FROM revisions"
            " WHERE uuid=? ORDER BY revision", (uuid,))
        return [{"data": json.loads(d), "revision": r,
                 "created_utc": t, "actor": a}
                for d, r, t, a in cur.fetchall()]

    def entities(self, kind: str | None = None) -> list[dict]:
        """Latest committed entities, with revisions for analysis invalidation.

        Materialize on the UI thread before starting a background analysis;
        the SQLite connection itself is not shared between threads.
        """
        query = "SELECT uuid, kind, data, revision, updated_utc FROM entities"
        params = ()
        if kind is not None:
            query += " WHERE kind=?"
            params = (kind,)
        query += " ORDER BY updated_utc, uuid"
        return [{"uuid": u, "kind": k, "data": json.loads(d), "revision": r, "updated_utc": t}
                for u, k, d, r, t in self._db.execute(query, params).fetchall()]

    def unfinished_tasks(self, limit: int = 200) -> list[dict]:
        cur = self._db.execute(
            "SELECT uuid, data FROM entities WHERE kind='task' LIMIT ?",
            (limit * 5,))
        out = []
        for uid, data in cur.fetchall():
            d = json.loads(data)
            if not d.get("completed"):
                out.append({"uuid": uid, **d})
                if len(out) >= limit:
                    break
        return out

    def count_tasks(self) -> int:
        """Total task entities (completed or not). P0A: distinguish a
        truly new project from a fully completed one on reopen."""
        cur = self._db.execute(
            "SELECT COUNT(*) FROM entities WHERE kind='task'")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def delete(self, kind: str, uuid: str, actor: str = "") -> bool:
        """Delete one entity (undo support). The append-only revisions
        table keeps the history, so the deletion itself stays audited and
        exportable as a tombstone (a later save supersedes it).
        Returns True when a row existed."""
        with self._db:
            row = self._db.execute(
                "SELECT data FROM entities WHERE kind=? AND uuid=?",
                (kind, uuid)).fetchone()
            cur = self._db.execute(
                "DELETE FROM entities WHERE kind=? AND uuid=?",
                (kind, uuid))
            self._db.execute(
                "INSERT INTO revisions(uuid, kind, data, revision,"
                " created_utc, actor) VALUES (?,?,?,?,?,?)",
                (uuid, kind, json.dumps({"deleted": True, "prior_kind": kind,
                    "prior": json.loads(row[0]) if row else None}), -1,
                 _utc(), actor))
            return (cur.rowcount or 0) > 0

    def tombstones(self) -> list[dict]:
        """Entities whose newest revision is a deletion (undo support).

        A re-save after a deletion writes a newer positive revision, so a
        tombstone is only current while nothing newer exists."""
        cur = self._db.execute(
            "SELECT uuid, kind, created_utc, actor FROM revisions"
            " WHERE rowid IN (SELECT MAX(rowid) FROM revisions GROUP BY uuid)"
            "   AND revision = -1")
        return [{"uuid": u, "kind": k, "deleted_utc": t, "actor": a}
                for u, k, t, a in cur.fetchall()]

    def finalize_snapshot(self, uuid: str, manifest: dict) -> str:
        """Lock a snapshot: immutable once finalized (H-plan contract)."""
        import hashlib
        payload = json.dumps(manifest, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self._db:
            row = self._db.execute(
                "SELECT finalized FROM snapshots WHERE uuid=?",
                (uuid,)).fetchone()
            if row is not None:
                if row[0]:
                    raise ValueError("finalized snapshot rejects mutation")
                self._db.execute(
                    "UPDATE snapshots SET manifest=?, content_hash=?,"
                    " finalized=1 WHERE uuid=?", (payload, digest, uuid))
            else:
                self._db.execute(
                    "INSERT INTO snapshots(uuid, manifest, content_hash,"
                    " finalized, created_utc) VALUES (?,?,?,?,?)",
                    (uuid, payload, digest, 1, _utc()))
        return digest

    def close(self) -> None:
        self._db.close()
