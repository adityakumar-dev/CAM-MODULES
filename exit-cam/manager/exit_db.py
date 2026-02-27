"""
ExitDB  —  Session Gallery
============================
Persistent deduplication store for the exit pipeline.

Schema
------
  cid          INTEGER PRIMARY KEY  canonical person ID
  embedding    BLOB                 290-dim float32 (weighted, L2-normalised)
  image_path   TEXT                 best-frame archive path (dashboard thumbnail)
  first_exit   REAL                 unix timestamp of first ever recorded exit
  last_seen    REAL                 unix timestamp of most recent sighting
  exit_count   INTEGER              total confirmed unique exits

Design
------
  Entire table loaded into RAM at startup. All mid-frame lookups are
  pure numpy matmul -- zero DB reads during the frame loop.

  find_match() enforces two conditions before returning a match:
    a) best cosine similarity >= threshold
    b) gap between best and second-best similarity >= MIN_SIM_GAP (0.06)
  If the match is ambiguous (two people look similar), treat as new
  person rather than risk false suppression.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Dict, Optional

import numpy as np

_DEFAULT_DB_PATH  = os.path.join("output", "exit_session.db")
_GALLERY_BLEND_ALPHA: float = 0.3
_MIN_SIM_GAP:         float = 0.06   # minimum gap between best and 2nd best


class ExitDB:

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_table()

        # In-memory caches
        self._embeddings:   Dict[int, np.ndarray] = {}
        self._exit_counts:  Dict[int, int]        = {}
        self._first_exits:  Dict[int, float]      = {}
        self._image_paths:  Dict[int, str]        = {}

        self._load_cache()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS session_gallery (
                cid          INTEGER PRIMARY KEY,
                embedding    BLOB    NOT NULL,
                image_path   TEXT    NOT NULL DEFAULT '',
                first_exit   REAL    NOT NULL,
                last_seen    REAL    NOT NULL,
                exit_count   INTEGER NOT NULL DEFAULT 1
            )
        """)
        # Safe migration for older schemas without image_path
        try:
            self._conn.execute(
                "ALTER TABLE session_gallery ADD COLUMN image_path TEXT NOT NULL DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

    # ── Startup cache load ────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        cur = self._conn.execute(
            "SELECT cid, embedding, image_path, first_exit, exit_count "
            "FROM session_gallery"
        )
        for cid, blob, image_path, first_exit, count in cur.fetchall():
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            self._embeddings[cid]  = emb
            self._exit_counts[cid] = count
            self._first_exits[cid] = first_exit
            self._image_paths[cid] = image_path or ""
        print(f"[ExitDB] Loaded {len(self._embeddings)} person(s) from session gallery.")

    # ── Lookup (threshold + gap check) ───────────────────────────────────────

    def find_match(self, embedding: np.ndarray, threshold: float) -> Optional[int]:
        """
        Vectorised cosine similarity vs all cached embeddings.

        Returns cid of best match only when BOTH conditions hold:
          1. best similarity >= threshold
          2. gap between best and second-best >= _MIN_SIM_GAP

        Condition 2 prevents ambiguous matches (two similar-looking people)
        from causing false suppression of a genuine new exit.
        """
        if not self._embeddings:
            return None

        ids   = list(self._embeddings.keys())
        mat   = np.stack(list(self._embeddings.values()), axis=0)
        norms = np.linalg.norm(mat, axis=1)
        eq    = np.linalg.norm(embedding)

        if eq == 0:
            return None

        mask = norms > 0
        sims = np.zeros(len(ids), dtype=np.float32)
        if mask.any():
            sims[mask] = (mat[mask] @ embedding) / (norms[mask] * eq)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < threshold:
            return None

        if len(sims) > 1:
            sorted_sims = np.sort(sims)[::-1]
            gap = best_sim - float(sorted_sims[1])
            if gap < _MIN_SIM_GAP:
                return None   # ambiguous — too close to second best

        return ids[best_idx]

    def contains(self, cid: int) -> bool:
        return cid in self._embeddings

    # ── Write operations ──────────────────────────────────────────────────────

    def upsert(
        self,
        cid: int,
        embedding: np.ndarray,
        image_path: str = "",
        now: Optional[float] = None,
    ) -> None:
        """
        New exit  -> insert row.
        Return visit -> blend embedding + increment count.
        Updates RAM cache immediately.
        """
        cid = int(cid)  # ensure Python native int — numpy int64 causes datatype mismatch on INTEGER PK
        now = now or time.time()

        if cid in self._embeddings:
            blended = (
                _GALLERY_BLEND_ALPHA * embedding
                + (1 - _GALLERY_BLEND_ALPHA) * self._embeddings[cid]
            )
            self._embeddings[cid]  = blended
            self._exit_counts[cid] += 1
            if image_path:
                self._image_paths[cid] = image_path

            self._conn.execute(
                """UPDATE session_gallery
                   SET embedding=?, image_path=?, last_seen=?, exit_count=?
                   WHERE cid=?""",
                (
                    blended.astype(np.float32).tobytes(),
                    self._image_paths[cid],
                    now,
                    self._exit_counts[cid],
                    cid,
                ),
            )
        else:
            self._embeddings[cid]  = embedding.copy()
            self._exit_counts[cid] = 1
            self._first_exits[cid] = now
            self._image_paths[cid] = image_path

            self._conn.execute(
                """INSERT INTO session_gallery
                   (cid, embedding, image_path, first_exit, last_seen, exit_count)
                   VALUES (?, ?, ?, ?, ?, 1)""",
                (cid, embedding.astype(np.float32).tobytes(), image_path, now, now),
            )

        self._conn.commit()

    def update_last_seen(self, cid: int, now: Optional[float] = None) -> None:
        cid = int(cid)
        now = now or time.time()
        self._conn.execute(
            "UPDATE session_gallery SET last_seen=? WHERE cid=?", (now, cid)
        )
        self._conn.commit()

    def update_image_path(self, cid: int, image_path: str) -> None:
        """Update thumbnail when a better-conf frame is archived."""
        if cid not in self._embeddings:
            return
        self._image_paths[cid] = image_path
        self._conn.execute(
            "UPDATE session_gallery SET image_path=? WHERE cid=?",
            (image_path, cid),
        )
        self._conn.commit()

    # ── Management ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._conn.execute("DELETE FROM session_gallery")
        self._conn.commit()
        self._embeddings.clear()
        self._exit_counts.clear()
        self._first_exits.clear()
        self._image_paths.clear()
        print("[ExitDB] Session gallery reset.")

    def total_unique(self) -> int:
        return len(self._embeddings)

    def exit_count(self, cid: int) -> int:
        return self._exit_counts.get(cid, 0)

    def get_image_path(self, cid: int) -> str:
        return self._image_paths.get(cid, "")

    def close(self) -> None:
        self._conn.close()
