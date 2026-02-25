"""
db_helper.py
============
Single persistent SQLite connection for archiving person records.

Stores records ONLY for people who have left the frame (archived).
The live session gallery (deduplication) lives in entry_db.py.

Design
------
* One shared connection (check_same_thread=False) — safe because
  ThreadPoolExecutor uses max_workers=1 so only one writer exists.
* WAL journal mode — no read/write contention.
* PRAGMA synchronous=NORMAL — durable enough, faster than FULL.
* Index on first_seen — keeps date-range queries fast after many rows.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Optional

_DB_PATH: str = os.path.join("output", "metadata.db")
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS person (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id   INTEGER,
                best_conf  REAL,
                first_seen REAL,
                last_seen  REAL,
                image_path TEXT
            )
        """)
        _conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_person_first_seen
            ON person(first_seen)
        """)
        _conn.commit()
    return _conn


def save_person_metadata(
    track_id: int,
    best_conf: float,
    first_seen: float,
    last_seen: float,
    image_path: str,
) -> None:
    """Insert one archived-person record. Called from background write thread."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO person (track_id, best_conf, first_seen, last_seen, image_path)
           VALUES (?, ?, ?, ?, ?)""",
        (track_id, best_conf, first_seen, last_seen, image_path),
    )
    conn.commit()


def close() -> None:
    """Flush and close. Call on clean shutdown."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None