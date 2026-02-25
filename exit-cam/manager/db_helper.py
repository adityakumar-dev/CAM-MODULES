"""
db_helper.py
============
Single persistent SQLite connection for the emotion zone counter.

Schema additions vs the original project
-----------------------------------------
* person.zone         — which zone the person was in when captured
* zone_counts table   — running total of entries per zone per day
* live_buffer.zone    — zone at time of capture (for crash recovery)

Everything else is identical to the original db_helper.py.
"""
import os
import sqlite3
from typing import Any, Dict, List, Optional

_DB_PATH: str = os.path.join("./", "metadata.db")
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row

        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")

        # ── person table ──────────────────────────────────────────────────────
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS person (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id      INTEGER,
                best_conf     REAL,
                first_seen    REAL,
                last_seen     REAL,
                image_path    TEXT,
                emotion       TEXT,
                emotion_score REAL,
                zone          TEXT
            )
        """)
        _conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_person_first_seen
            ON person(first_seen)
        """)

        # ── zone_counts table — one row per (zone, day) ───────────────────────
        # Incremented atomically each time a new track is confirmed in a zone.
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS zone_counts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                zone      TEXT    NOT NULL,
                day       TEXT    NOT NULL,   -- YYYY-MM-DD
                count     INTEGER NOT NULL DEFAULT 0,
                UNIQUE(zone, day)
            )
        """)

        # ── live_buffer table — tracks images currently in output/best/ ───────
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS live_buffer (
                cid        INTEGER PRIMARY KEY,
                image_path TEXT    NOT NULL,
                best_conf  REAL    NOT NULL,
                first_seen REAL    NOT NULL,
                last_seen  REAL    NOT NULL,
                crop_h     INTEGER NOT NULL DEFAULT 0,
                zone       TEXT
            )
        """)
        _conn.commit()

    return _conn


# ── person ────────────────────────────────────────────────────────────────────

def save_person_metadata(
    track_id:    int,
    best_conf:   float,
    first_seen:  float,
    last_seen:   float,
    image_path:  str,
    emotion:     Optional[str]   = None,
    emotion_score: Optional[float] = None,
    zone:        Optional[str]   = None,
) -> None:
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO person
            (track_id, best_conf, first_seen, last_seen, image_path,
             emotion, emotion_score, zone)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (track_id, best_conf, first_seen, last_seen, image_path,
         emotion, emotion_score, zone),
    )
    conn.commit()


# ── zone_counts ───────────────────────────────────────────────────────────────

def increment_zone_count(zone: str, day: str) -> None:
    """
    Atomically increment the entry counter for (zone, day).
    `day` should be a string like '2026-02-25'.
    """
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO zone_counts (zone, day, count) VALUES (?, ?, 1)
        ON CONFLICT(zone, day) DO UPDATE SET count = count + 1
        """,
        (zone, day),
    )
    conn.commit()


def get_zone_counts_today(day: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT zone, count FROM zone_counts WHERE day = ? ORDER BY zone",
        (day,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_zone_counts_all() -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT zone, day, count FROM zone_counts ORDER BY day DESC, zone"
    ).fetchall()
    return [dict(r) for r in rows]


# ── live_buffer ───────────────────────────────────────────────────────────────

def upsert_live_buffer(
    cid:        int,
    image_path: str,
    best_conf:  float,
    first_seen: float,
    last_seen:  float,
    crop_h:     int,
    zone:       Optional[str] = None,
) -> None:
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO live_buffer (cid, image_path, best_conf, first_seen, last_seen, crop_h, zone)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cid) DO UPDATE SET
            image_path = excluded.image_path,
            best_conf  = excluded.best_conf,
            last_seen  = excluded.last_seen,
            crop_h     = excluded.crop_h,
            zone       = excluded.zone
        """,
        (cid, image_path, best_conf, first_seen, last_seen, crop_h, zone),
    )
    conn.commit()


def delete_live_buffer(cid: int) -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM live_buffer WHERE cid = ?", (cid,))
    conn.commit()


def get_all_live_buffer() -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM live_buffer").fetchall()
    return [dict(r) for r in rows]


# ── lifecycle ─────────────────────────────────────────────────────────────────

def close() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None