"""
IdentityManager  —  Entry Edition  v5
======================================
Image lifecycle (ONE image per person, ever):
  1. Person detected in capture zone (detector or inside)
     → output/best/id_<cid>.jpg created / replaced if conf improves
  2. Person leaves frame → enters _lost buffer (30 s window)
     → image file stays in output/best/, NOT archived yet
     → if they return within 30 s: ReID reclaim, same PersonState,
       image keeps improving, no duplicate
  3. Lost buffer expires (30 s with no re-detection)
     → THEN and only then: move image to output/best/day_*/hour_*/ archive
     → DB row written
     → image_path updated in session gallery

Counting rules (three paths, in priority order):
  Path 1 — Zone crossing event fires (detector→inside)
  Path 2 — Person's very first detected zone is already "inside"
            (fast walker / detection starts mid-crossing)
  Path 3 — Person exits frame while zone==inside, never triggered paths 1/2

De-duplication:
  - state.counted flag on PersonState survives ReID reclaims within the
    30 s lost buffer → same person re-detected = no recount
  - After lost buffer expires, new PersonState created on next detection
    → gallery.find_match() at threshold 0.78 catches same person returning
    → match found = update existing row, no new count
"""
from __future__ import annotations

import base64
import os
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from .db_helper    import save_person_metadata, close as _db_close
from .entry_db     import EntryDB
from .zone_manager import ZoneManager

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_BUFFER_SECONDS : float = float(getattr(config, "ENTRY_REID_BUFFER_SECONDS", 30.0))
_REID_SIM       : float = float(getattr(config, "ENTRY_REID_SIM_THRESHOLD",   0.65))
_SESSION_SIM    : float = float(getattr(config, "ENTRY_SESSION_THRESHOLD",     0.78))
_HIST_SKIP_PX   : int   = int  (getattr(config, "ENTRY_HIST_SKIP_PX",          4))
_COUNT_TRIGGER  : str   =       getattr(config, "ENTRY_COUNT_TRIGGER", "detector\u2192inside")
_EMB_ALPHA      : float = float(getattr(config, "REID_EMBEDDING_ALPHA",        0.3))
_TRAIL_LEN      : int   = int  (getattr(config, "TRAIL_MAX_LEN",               30))
_MERGE_PX       : int   = int  (getattr(config, "ENTRY_TEMP_MERGE_PX",         60))

_DETECTOR_ZONE  : str       = _COUNT_TRIGGER.split("\u2192")[0].strip()
_INSIDE_ZONE    : str       = _COUNT_TRIGGER.split("\u2192")[1].strip()
_CAPTURE_ZONES  : frozenset = frozenset({_DETECTOR_ZONE, _INSIDE_ZONE})
# Callbacks registered by dashboard so background threads can forward events
# to the backend WS.  Set via: identity_manager._on_captured = fn, etc.
_on_captured: Optional[callable] = None  # fired when a person image is archived
_on_reentry:  Optional[callable] = None  # fired when a known person re-enters

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _spatial_embedding(crop: np.ndarray, frame_h: int, bins: int = 32) -> np.ndarray:
    dim = bins * 3 * 3 + 2
    if crop is None or crop.size == 0:
        return np.zeros(dim, dtype=np.float32)
    h, w = crop.shape[:2]
    if h < 6 or w < 2:
        return np.zeros(dim, dtype=np.float32)
    strips, weights = [crop[:h//3], crop[h//3:2*h//3], crop[2*h//3:]], (1.5, 1.0, 0.7)
    parts = []
    for strip, weight in zip(strips, weights):
        if strip.size == 0:
            parts.append(np.zeros(bins * 3, dtype=np.float32))
            continue
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        hists = []
        for ch, rng in zip(range(3), [(0, 180), (0, 256), (0, 256)]):
            h_arr = cv2.calcHist([hsv], [ch], None, [bins], list(rng)).flatten().astype(np.float32)
            s = h_arr.sum()
            hists.append(h_arr / s if s > 0 else h_arr)
        parts.append(np.concatenate(hists) * weight)
    aspect     = np.float32(min(w / (h + 1e-5), 2.0) / 2.0)
    rel_height = np.float32(min(h / (frame_h + 1e-5), 1.0))
    vec  = np.concatenate([*parts, [aspect, rel_height]])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _crop_frame(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    return frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]


def _box_moved(old_box: Optional[Tuple], new_box: Tuple, px: int) -> bool:
    if old_box is None:
        return True
    return any(abs(a - b) > px for a, b in zip(old_box, new_box))


def _centroid_dist(box: Tuple, cx: int, cy: int) -> float:
    return (((box[0]+box[2])/2 - cx)**2 + ((box[1]+box[3])/2 - cy)**2) ** 0.5


# ──────────────────────────────────────────────────────────────────────────────
# LostEntry — what we keep in the lost buffer
# ──────────────────────────────────────────────────────────────────────────────

class LostEntry:
    """
    Holds everything needed to either:
      a) Reclaim a person back into _active (if re-detected within buffer)
      b) Finalize them on expiry (archive image, write DB row)

    was_in_capture_zone: True if person had a best-frame image captured,
      meaning they were physically in the detector or inside zone.
      Safety-net: on expiry, if not yet counted but was in capture zone,
      count them — catches people who crossed while briefly in _lost buffer.
    """
    __slots__ = ("live_embedding", "best_embedding", "best_conf",
                 "image_path", "first_seen", "last_seen",
                 "counted", "was_in_capture_zone", "t_lost")

    def __init__(self, state: "PersonState", image_path: str, t_lost: float):
        self.live_embedding      = state.live_embedding.copy()
        self.best_embedding      = state.best_embedding.copy()
        self.best_conf           = state.best_conf
        self.image_path          = image_path
        self.first_seen          = state.first_seen
        self.last_seen           = state.last_seen
        self.counted             = state.counted
        self.was_in_capture_zone = bool(image_path)
        self.t_lost              = t_lost


# ──────────────────────────────────────────────────────────────────────────────
# PersonState
# ──────────────────────────────────────────────────────────────────────────────

class PersonState:
    __slots__ = (
        "track_id", "first_seen", "last_seen",
        "centre_history", "zone", "prev_zone",
        "live_embedding", "best_embedding",
        "box", "conf", "best_conf", "best_frame",
        "was_in_detector", "counted",
        "_prev_box",
    )

    def __init__(self, track_id: Any, centre: Tuple[int, int], box: Tuple,
                 conf: float, embedding: np.ndarray):
        self.track_id              = track_id
        self.first_seen            = time.time()
        self.last_seen             = time.time()
        self.centre_history: deque = deque(maxlen=_TRAIL_LEN)
        self.centre_history.append(centre)
        self.zone                  = None
        self.prev_zone             = None
        self.live_embedding        = embedding.copy()
        self.best_embedding        = embedding.copy()
        self.box                   = box
        self.conf                  = conf
        self.best_conf             = conf
        self.best_frame: Optional[np.ndarray] = None
        self.was_in_detector       = False
        self.counted               = False
        self._prev_box             = box

    def restore_from_lost(self, entry: LostEntry) -> None:
        """Restore counted + embedding state from a reclaimed LostEntry.
        prev_zone is deliberately NOT restored — after a reclaim gap,
        the zone sequence restarts so Path 2 (appeared_inside) can fire
        correctly if the person re-enters inside zone directly.
        """
        self.counted        = entry.counted
        self.live_embedding = entry.live_embedding.copy()
        self.best_embedding = entry.best_embedding.copy()
        self.best_conf      = max(self.best_conf, entry.best_conf)
        self.first_seen     = entry.first_seen  # keep original first_seen
        # prev_zone = None (default) so zone-transition logic resets cleanly

    def update(self, centre: Tuple[int, int], box: Tuple, conf: float,
               embedding: Optional[np.ndarray], crop: np.ndarray, frame_h: int,
               in_detector: bool, in_capture: bool) -> None:
        self.last_seen = time.time()
        self.prev_zone = self.zone
        self.centre_history.append(centre)
        self.box  = box
        self.conf = conf

        if embedding is not None:
            self.live_embedding = _EMB_ALPHA * embedding + (1 - _EMB_ALPHA) * self.live_embedding
            self._prev_box = box

        if in_detector:
            self.was_in_detector = True

        # Snapshot BEFORE any update — both checks use the same baseline
        prev_best = self.best_conf

        # Best frame: any capture zone, replace only if better conf
        if in_capture and (self.best_frame is None or conf > prev_best):
            self.best_conf  = conf
            self.best_frame = crop.copy()

        # Best embedding: detector zone only (clearest angle for gallery matching)
        # Uses prev_best so this check is independent of the frame update above
        if in_detector and (self.best_embedding is None or conf > prev_best):
            self.best_embedding = _spatial_embedding(crop, frame_h)


# ──────────────────────────────────────────────────────────────────────────────
# IdentityManager
# ──────────────────────────────────────────────────────────────────────────────

class IdentityManager:

    def __init__(self):
        self._active:   Dict[Any, PersonState] = {}
        self._id_map:   Dict[Any, Any]         = {}
        # cid → LostEntry  (replaces old (embedding, time) tuple)
        self._lost:     Dict[Any, LostEntry]   = {}
        # cid → current live image_path in output/best/
        self._img_path: Dict[Any, str]         = {}

        zones = getattr(config, "ENTRY_ZONES", {})
        if not zones:
            print("[IdentityManager] WARNING: ENTRY_ZONES not set. Run tools/draw_zones.py")
        self._zones = ZoneManager(zones)

        db_path = getattr(config, "ENTRY_DB_PATH", "output/entry_session.db")
        self._gallery = EntryDB(db_path=db_path)
        self.unique_entry_count: int = self._gallery.total_unique()
        print(f"[IdentityManager] Started. Known persons: {self.unique_entry_count}")

        self._best_dir = os.path.join("output", "best")
        os.makedirs(self._best_dir, exist_ok=True)
        self._archive_dir_cache: Dict[str, str] = {}
        self._write_pool = ThreadPoolExecutor(max_workers=1)

    # ── Public ────────────────────────────────────────────────────────────────

    def draw_zones(self, frame: np.ndarray) -> None:
        self._zones.draw(frame)

    def active_count(self) -> int:
        return len(self._active)

    def get_state(self) -> Dict[Any, PersonState]:
        return self._active

    def reset_session(self) -> None:
        self._gallery.reset()
        self.unique_entry_count = 0
        for s in self._active.values():
            s.counted = False
        print("[IdentityManager] Session reset.")

    # ── Archive dir ───────────────────────────────────────────────────────────

    def _archive_dir(self, dt: datetime) -> str:
        key = dt.strftime("day_%Y%m%d/hour_%H")
        if key not in self._archive_dir_cache:
            full = os.path.join(self._best_dir, *key.split("/"))
            os.makedirs(full, exist_ok=True)
            self._archive_dir_cache[key] = full
        return self._archive_dir_cache[key]

    # ── Short-term ReID ───────────────────────────────────────────────────────

    def _reid_lookup(self, embedding: np.ndarray) -> Optional[Any]:
        """Match against live embeddings in lost buffer."""
        if not self._lost or embedding is None:
            return None
        ids  = list(self._lost.keys())
        embs = [e.live_embedding for e in self._lost.values()]
        mat  = np.stack(embs)
        norms = np.linalg.norm(mat, axis=1)
        eq    = np.linalg.norm(embedding)
        sims  = np.zeros(len(ids), dtype=np.float32)
        mask  = (norms > 0) & (eq > 0)
        if mask.any():
            sims[mask] = (mat[mask] @ embedding) / (norms[mask] * eq)
        best = int(np.argmax(sims))
        return ids[best] if sims[best] > _REID_SIM else None

    # ── Temp ID merge ─────────────────────────────────────────────────────────

    def _try_merge_temp(self, real_id: int, cx: int, cy: int) -> None:
        best_tmp, best_dist = None, float("inf")
        for cid, state in self._active.items():
            if not isinstance(cid, str) or not cid.startswith("tmp_"):
                continue
            dist = _centroid_dist(state.box, cx, cy)
            if dist < _MERGE_PX and dist < best_dist:
                best_dist, best_tmp = dist, cid
        if best_tmp is None:
            return

        old          = self._active.pop(best_tmp)
        old.track_id = real_id
        self._active[real_id] = old

        for bt, cid in list(self._id_map.items()):
            if cid == best_tmp:
                self._id_map[bt] = real_id

        # Transfer live image path
        if best_tmp in self._img_path:
            old_path = self._img_path.pop(best_tmp)
            new_path = os.path.join(self._best_dir, f"id_{real_id}.jpg")
            if os.path.exists(old_path):
                self._write_pool.submit(_rename_worker, old_path, new_path)
            self._img_path[real_id] = new_path

        self._zones.transfer(best_tmp, real_id)
        print(f"[IdentityManager] Merged tmp={best_tmp} → {real_id} ({best_dist:.0f}px)")

    # ── Count person ──────────────────────────────────────────────────────────

    def _count_person(self, cid: Any, state: PersonState, reason: str) -> None:
        """Single entry point for all counting. Deduplication via gallery."""
        if state.counted:
            return

        state.counted = True
        img_path      = self._img_path.get(cid, "")
        best_emb      = state.best_embedding
        emb_norm      = float(np.linalg.norm(best_emb))

        # Warn if embedding is zero/garbage — gallery match won't work reliably
        if emb_norm < 0.1:
            print(f"[Entry] WARNING cid={cid} reason={reason}: zero embedding — "
                  f"person may not have been in detector zone long enough. "
                  f"Will count but gallery dedup may be unreliable.")

        match_cid = self._gallery.find_match(best_emb, threshold=_SESSION_SIM)
        if match_cid is not None:
            self._gallery.upsert(match_cid, best_emb, image_path=img_path)
            print(f"[Entry] {reason}: cid={cid} → matched {match_cid} — returning visitor")
            if _on_reentry is not None:
                try:
                    _on_reentry(match_cid, self._gallery.entry_count(match_cid))
                except Exception:
                    pass
        else:
            self._gallery.upsert(cid, best_emb, image_path=img_path)
            self.unique_entry_count = self._gallery.total_unique()
            print(f"[Entry] {reason}: NEW  cid={cid}  emb_norm={emb_norm:.3f}  total={self.unique_entry_count}")

    # ── Best frame: capture-and-replace (ONE file per person) ────────────────

    def _update_best(self, cid: Any, conf: float, crop: np.ndarray,
                     state: PersonState) -> None:
        """
        Write/overwrite output/best/id_<cid>.jpg only when conf improves.
        Single file per person — same filename, replaced in-place.
        Archive happens later (on lost-buffer expiry), never here.
        """
        img_path = os.path.join(self._best_dir, f"id_{str(cid).replace('tmp_', 'tmp')}.jpg")
        is_new   = cid not in self._img_path

        # Compare against state.best_conf which PersonState.update() already snapshotted
        # We only write if this is new OR current conf genuinely exceeds stored best
        stored_conf = 0.0 if is_new else state.best_conf
        if is_new or conf >= stored_conf:
            self._write_pool.submit(cv2.imwrite, img_path, crop.copy())
            self._img_path[cid] = img_path
            if self._gallery.contains(cid):
                self._write_pool.submit(self._gallery.update_image_path, cid, img_path)

    # ── Finalize on lost-buffer expiry ────────────────────────────────────────

    def _finalize(self, cid: Any, entry: LostEntry) -> None:
        """
        Called ONLY when lost buffer expires (30 s no re-detection).
        Single point where archive + DB write happen.
        Counted persons → archive image + write DB row.
        Uncounted persons → silently delete live image file.

        SAFETY-NET: if person was in a capture zone (had image captured) but
        was never counted — count them now. This catches the brief-leave case:
        person disappears for 1-2 frames mid-crossing, ZoneManager lost their
        state, they reappeared in outside zone so no path fired, but they DID
        physically pass through the metal detector (image proves it).
        """
        src = self._img_path.pop(cid, entry.image_path)

        if not entry.counted:
            if entry.was_in_capture_zone and src and os.path.exists(src):
                # Person was in security zone but never triggered a counting path.
                # Count them now before finalizing.
                print(f"[Entry] safety_net: cid={cid} — was in capture zone, counting on expiry")
                match_cid = self._gallery.find_match(entry.best_embedding, threshold=_SESSION_SIM)
                if match_cid is not None:
                    self._gallery.upsert(match_cid, entry.best_embedding, image_path=src)
                    print(f"[Entry] safety_net: cid={cid} matched {match_cid} — returning visitor")
                else:
                    self._gallery.upsert(cid, entry.best_embedding, image_path=src)
                    self.unique_entry_count = self._gallery.total_unique()
                    print(f"[Entry] safety_net: NEW  cid={cid}  total={self.unique_entry_count}")
                # Fall through to archive since they are now counted
            else:
                # Truly never in security zone — discard silently
                if src and os.path.exists(src):
                    self._write_pool.submit(_discard_worker, src)
                return

        if not src or not os.path.exists(src):
            return

        # Use best_conf from entry (highest observed across entire visit)
        dst_dir = self._archive_dir(datetime.now())
        safe    = str(cid).replace("tmp_", "tmp")
        dst     = os.path.join(dst_dir, f"id_{safe}_conf_{entry.best_conf:.2f}.jpg")
        self._write_pool.submit(
            _archive_worker, src, dst, cid,
            entry.best_conf, entry.first_seen, entry.last_seen,
        )

    # ── Main process loop ─────────────────────────────────────────────────────

    def process(self, frame: np.ndarray, raw_tracks: List[dict]) -> List[dict]:
        frame_h     = frame.shape[0]
        active_cids = set()

        for t in raw_tracks:
            bt_id           = t["track_id"]
            is_temp         = t.get("is_temp_id", False)
            x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
            conf            = t["conf"]
            box             = (x1, y1, x2, y2)
            cx, cy          = (x1 + x2) // 2, (y1 + y2) // 2
            crop            = _crop_frame(frame, x1, y1, x2, y2)

            # ── Temp → real ID merge ──────────────────────────────────────────
            if not is_temp and isinstance(bt_id, int) and bt_id not in self._id_map:
                self._try_merge_temp(bt_id, cx, cy)
                self._id_map[bt_id] = bt_id

            cid = self._id_map.get(bt_id, bt_id)

            # ── Embedding ─────────────────────────────────────────────────────
            prev_box = self._active[cid]._prev_box if cid in self._active else None
            need_emb = (cid not in self._active) or _box_moved(prev_box, box, _HIST_SKIP_PX)
            emb      = _spatial_embedding(crop, frame_h) if need_emb else None

            # ── ReID reclaim from lost buffer ─────────────────────────────────
            reclaimed_entry: Optional[LostEntry] = None
            if cid not in self._active and not is_temp and emb is not None:
                reclaimed_cid = self._reid_lookup(emb)
                if reclaimed_cid is not None:
                    reclaimed_entry = self._lost.pop(reclaimed_cid)
                    # Remap bt_id to the reclaimed canonical ID
                    self._id_map[bt_id] = reclaimed_cid
                    cid = reclaimed_cid

            # ── Zone update ───────────────────────────────────────────────────
            event        = self._zones.update(cid, (cx, cy))
            current_zone = self._zones.current_zone(cid)
            in_detector  = current_zone == _DETECTOR_ZONE
            in_inside    = current_zone == _INSIDE_ZONE
            in_capture   = current_zone in _CAPTURE_ZONES

            # ── Create or update PersonState ──────────────────────────────────
            if cid not in self._active:
                init_emb = emb if emb is not None else _spatial_embedding(crop, frame_h)
                self._active[cid] = PersonState(cid, (cx, cy), box, conf, init_emb)
                # Restore state from lost buffer if reclaimed
                if reclaimed_entry is not None:
                    self._active[cid].restore_from_lost(reclaimed_entry)
                    # Restore live image path so the same file keeps being overwritten
                    if reclaimed_entry.image_path:
                        self._img_path[cid] = reclaimed_entry.image_path
            else:
                self._active[cid].update(
                    (cx, cy), box, conf, emb, crop, frame_h, in_detector, in_capture
                )

            state = self._active[cid]
            state.zone = current_zone
            active_cids.add(cid)

            # ── Best frame capture-and-replace (only in capture zones) ────────
            if in_capture:
                self._update_best(cid, conf, crop, state)

            # ── COUNTING ──────────────────────────────────────────────────────
            # Path 1: proper zone-crossing event (detector→inside)
            if event == _COUNT_TRIGGER and not state.counted:
                self._count_person(cid, state, "zone_cross")

            # Path 2a: appeared already inside — first zone assignment is inside.
            # Covers fast walkers and reclaimed persons re-entering inside directly.
            elif in_inside and state.prev_zone is None and not state.counted:
                self._count_person(cid, state, "appeared_inside")

            # Path 2b: person is in inside zone, was previously in detector zone,
            # but a brief leave/re-enter broke the crossing event chain.
            # was_in_detector=True proves they physically passed through the gate.
            # prev_zone being set (not None) means Path 2a already missed them.
            elif (in_inside
                  and state.was_in_detector
                  and state.prev_zone is not None
                  and state.prev_zone != _DETECTOR_ZONE  # not a normal cross (Path 1 handles)
                  and not state.counted):
                self._count_person(cid, state, "was_in_detector_now_inside")

        # ── Disappeared → lost buffer (no archive yet) ────────────────────────
        for gone_cid in set(self._active) - active_cids:
            s = self._active.pop(gone_cid)

            # Path 3: was in inside zone when last seen, never counted via events
            # Use s.zone (the gone person's actual zone), not loop variable
            if s.zone == _INSIDE_ZONE and not s.counted:
                self._count_person(gone_cid, s, "exit_inside")

            img_path = self._img_path.get(gone_cid, "")
            self._lost[gone_cid] = LostEntry(s, img_path, time.time())
            self._zones.remove(gone_cid)

        # ── Prune _id_map ──────────────────────────────────────────────────────
        live_cids = set(self._active) | set(self._lost)
        for bt_id in list(self._id_map):
            if self._id_map[bt_id] not in live_cids:
                del self._id_map[bt_id]

        # ── Expire lost buffer → FINALIZE (archive + DB) ──────────────────────
        now = time.time()
        for lost_cid in list(self._lost):
            entry = self._lost[lost_cid]
            if now - entry.t_lost > _BUFFER_SECONDS:
                self._finalize(lost_cid, entry)
                del self._lost[lost_cid]

        # ── Result list ────────────────────────────────────────────────────────
        results = []
        for cid in active_cids:
            s = self._active[cid]
            results.append(dict(
                track_id           = cid,
                x1=s.box[0], y1=s.box[1], x2=s.box[2], y2=s.box[3],
                conf               = s.conf,
                best_conf          = s.best_conf,
                centre             = s.centre_history[-1],
                centre_history     = s.centre_history,
                zone               = s.zone,
                first_seen         = s.first_seen,
                last_seen          = s.last_seen,
                unique_entry_count = self.unique_entry_count,
            ))
        return results

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        # Finalize all active persons
        for cid, s in list(self._active.items()):
            img_path = self._img_path.get(cid, "")
            entry    = LostEntry(s, img_path, time.time())
            self._finalize(cid, entry)
        # Finalize all lost-buffer persons
        for lost_cid, entry in list(self._lost.items()):
            self._finalize(lost_cid, entry)
        self._write_pool.shutdown(wait=True)
        self._gallery.close()
        _db_close()


# ──────────────────────────────────────────────────────────────────────────────
# Background workers
# ──────────────────────────────────────────────────────────────────────────────

def _rename_worker(src: str, dst: str) -> None:
    try:
        os.rename(src, dst)
    except OSError:
        try:
            shutil.copy2(src, dst)
            os.remove(src)
        except OSError:
            pass


def _discard_worker(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _archive_worker(src: str, dst: str, track_id: Any,
                    best_conf: float, first_seen: float, last_seen: float) -> None:
    try:
        os.rename(src, dst)
    except OSError:
        shutil.copy2(src, dst)
        try:
            os.remove(src)
        except OSError:
            pass
    save_person_metadata(track_id, best_conf, first_seen, last_seen, dst)

    # Forward to backend WS (non-blocking — dashboard registers this callback)
    if _on_captured is not None:
        try:
            img_b64 = None
            try:
                with open(dst, "rb") as _f:
                    img_b64 = base64.b64encode(_f.read()).decode()
            except Exception:
                pass
            _on_captured({
                "event":    "captured",
                "track_id": track_id,
                "image":    img_b64,
                "ts":       time.time(),
            })
        except Exception:
            pass