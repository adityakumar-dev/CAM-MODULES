"""
identity_manager.py
===================
Same core logic as the original IdentityManager, with these additions:

  Zone tracking
  -------------
  • Each PersonState now tracks which zone(s) the person has been counted in.
  • Zone detection uses foot-point (bottom-centre of bounding box) + a
    configurable dwell guard (ZONE_DWELL_FRAMES consecutive frames) to avoid
    single-frame blips incrementing the counter.
  • Zone entry is counted once per track per zone per visit.
  • zone_counts DB table is incremented atomically on confirmed entry.

  Zone-gated capture
  ------------------
  • If config.CAPTURE_ZONES is non-empty, best-frame crops are ONLY saved
    while the person is inside one of those zones.
  • If config.CAPTURE_ZONES is empty, crops are saved everywhere (same as
    original behaviour).

  Re-ID scope
  -----------
  • Re-ID only covers the current session's _lost buffer (same-visit occlusion
    recovery, exactly as the original).
  • NO cross-session / gallery duplicate checking.

  Everything else — async write pool, archive dir structure, crash recovery,
  live_buffer, emotion analysis — is identical to the original.
"""
from __future__ import annotations

import os
import shutil
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

import config
from manager.db_helper import (
    save_person_metadata,
    upsert_live_buffer,
    delete_live_buffer,
    get_all_live_buffer,
    increment_zone_count,
    close as _db_close,
)

# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants (read once from config)
# ──────────────────────────────────────────────────────────────────────────────

_SOURCE_FPS: int = max(1, getattr(config, "SOURCE_FPS", 30))

if hasattr(config, "REID_BUFFER_FRAMES"):
    _BUFFER_SECONDS: float = config.REID_BUFFER_FRAMES / _SOURCE_FPS
elif hasattr(config, "REID_BUFFER_SECONDS"):
    _BUFFER_SECONDS = float(config.REID_BUFFER_SECONDS)
else:
    _BUFFER_SECONDS = 3.0

_HIST_SKIP_PX: int  = getattr(config, "HIST_SKIP_PX",  4)
_MIN_CROP_PX:  int  = getattr(config, "MIN_CROP_PX",   20)

_REID_WITHIN_BUFFER_THRESHOLD: float = getattr(config, "REID_SAME_VISIT_THRESHOLD", 0.50)
_REID_CROSS_VISIT_THRESHOLD:   float = getattr(config, "REID_SIM_THRESHOLD",        0.75)
_ID_MAP_PRUNE_SECONDS: float = _BUFFER_SECONDS * 2.0

_HAPPY_THRESHOLD:      float = getattr(config, "EMOTION_HAPPY_THRESHOLD",      0.40)
_VERY_HAPPY_THRESHOLD: float = getattr(config, "EMOTION_VERY_HAPPY_THRESHOLD", 0.75)
_SAD_THRESHOLD:        float = getattr(config, "EMOTION_SAD_THRESHOLD",        0.40)
_EMOTIEFF_MODEL: str  = getattr(config, "EMOTIEFF_MODEL", "enet_b0_8_best_afew")

_ZONES: dict          = getattr(config, "ZONES", {})
_CAPTURE_ZONES: list  = getattr(config, "CAPTURE_ZONES", [])
_ZONE_DWELL_FRAMES: int = getattr(config, "ZONE_DWELL_FRAMES", 3)

print(f"[IDMGR] Config | buffer={_BUFFER_SECONDS:.1f}s | "
      f"zones={list(_ZONES.keys())} | capture_zones={_CAPTURE_ZONES} | "
      f"dwell_guard={_ZONE_DWELL_FRAMES}f | min_crop={_MIN_CROP_PX}px")


# ──────────────────────────────────────────────────────────────────────────────
# Emotion recognizer — lazy singleton (same as original)
# ──────────────────────────────────────────────────────────────────────────────

_fer_instance: Optional[object] = None


def _get_fer():
    global _fer_instance
    if _fer_instance is None:
        try:
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer
            _fer_instance = EmotiEffLibRecognizer(
                engine="torch", model_name=_EMOTIEFF_MODEL, device="cpu"
            )
        except Exception as exc:
            print(f"[IdentityManager] EmotiEffLib unavailable: {exc}. "
                  "Emotion analysis disabled.")
            _fer_instance = False
    return _fer_instance if _fer_instance is not False else None


# ──────────────────────────────────────────────────────────────────────────────
# Pure helpers (identical to original)
# ──────────────────────────────────────────────────────────────────────────────

def _crop(frame: np.ndarray,
          x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    fh, fw = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(fw, x2), min(fh, y2)
    if x2c - x1c < _MIN_CROP_PX or y2c - y1c < _MIN_CROP_PX:
        return None
    return frame[y1c:y2c, x1c:x2c].copy()


def _colour_hist(crop: np.ndarray, bins: int = 32) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.zeros(bins * 3, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = []
    for ch, rng in zip(range(3), [(0, 180), (0, 256), (0, 256)]):
        h = cv2.calcHist([hsv], [ch], None, [bins], list(rng))
        h = h.flatten().astype(np.float32)
        s = h.sum()
        hist.append(h / s if s > 0 else h)
    return np.concatenate(hist)


def _box_moved(old_box: Optional[Tuple], new_box: Tuple, px: int) -> bool:
    if old_box is None:
        return True
    return any(abs(a - b) > px for a, b in zip(old_box, new_box))


def _point_in_zone(x: int, y: int, polygon: list) -> bool:
    """Ray-casting point-in-polygon via OpenCV (handles concave polygons)."""
    if len(polygon) < 3:
        return False
    pts = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


def _current_zone(foot_x: int, foot_y: int) -> Optional[str]:
    """Return the first zone name whose polygon contains (foot_x, foot_y)."""
    for name, poly in _ZONES.items():
        if _point_in_zone(foot_x, foot_y, poly):
            return name
    return None


def _analyse_emotion(image_path: str,
                     crop_h: int = 0) -> Tuple[Optional[str], Optional[float]]:
    fer = _get_fer()
    if fer is None:
        return None, None
    try:
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            return None, None
        h = img.shape[0]
        head_bottom = max(int(h * 0.40), min(h, 60))
        head_img = img[:head_bottom, :]
        if head_img.size == 0:
            head_img = img
        head_rgb = cv2.cvtColor(head_img, cv2.COLOR_BGR2RGB)
        emotion_labels, scores = fer.predict_emotions(head_rgb, logits=False)
        if not emotion_labels or len(scores) == 0:
            return None, None
        face_scores = scores[0] if hasattr(scores[0], "__len__") else scores
        names = ["angry", "disgust", "fear", "happy",
                 "sad", "surprise", "neutral", "contempt"]
        sd = {names[i]: float(face_scores[i])
              for i in range(min(len(names), len(face_scores)))}
        pos = sd.get("happy", 0.0) + sd.get("surprise", 0.0)
        neg = (sd.get("sad", 0.0)     + sd.get("angry", 0.0) +
               sd.get("fear", 0.0)    + sd.get("disgust", 0.0) +
               sd.get("contempt", 0.0))
        if pos >= neg and pos >= _HAPPY_THRESHOLD:
            return ("Very Happy" if pos >= _VERY_HAPPY_THRESHOLD else "Happy"), pos
        if neg > pos and neg >= _SAD_THRESHOLD:
            return "Sad", neg
        return None, None
    except Exception:
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# PersonState — extended with zone tracking fields
# ──────────────────────────────────────────────────────────────────────────────

class PersonState:
    __slots__ = (
        "track_id", "first_seen", "last_seen",
        "centre_history", "zone", "embedding",
        "box", "conf", "_prev_box",
        # zone tracking
        "counted_zones",        # set[str]  — zones already counted this visit
        "_zone_dwell_name",     # str|None  — zone currently being dwelt in
        "_zone_dwell_frames",   # int       — consecutive frames inside _zone_dwell_name
    )

    def __init__(self, track_id, centre, box, conf, embedding):
        self.track_id   = track_id
        self.first_seen = time.time()
        self.last_seen  = time.time()
        self.centre_history: deque = deque(maxlen=config.TRAIL_MAX_LEN)
        self.centre_history.append(centre)
        self.zone      = None
        self.embedding = embedding
        self.box       = box
        self.conf      = conf
        self._prev_box = box
        # zone tracking
        self.counted_zones: Set[str] = set()
        self._zone_dwell_name: Optional[str] = None
        self._zone_dwell_frames: int = 0

    def update(self, centre, box: Tuple, conf: float,
               embedding: Optional[np.ndarray]) -> None:
        self.last_seen = time.time()
        self.centre_history.append(centre)
        self.box       = box
        self.conf      = conf
        self._prev_box = box
        if embedding is not None:
            a = config.REID_EMBEDDING_ALPHA
            self.embedding = a * embedding + (1 - a) * self.embedding

    def update_zone_dwell(self, zone_name: Optional[str]) -> Optional[str]:
        """
        Feed the zone the person is in this frame.
        Returns the zone name once the dwell guard has been satisfied
        (i.e. the zone should now be counted), otherwise returns None.

        Rules:
          - If the person is in a new zone, reset the dwell counter.
          - Once they have been in the same zone for _ZONE_DWELL_FRAMES
            consecutive frames AND that zone hasn't been counted yet,
            return the zone name so the caller can count it.
        """
        if zone_name is None:
            # Not in any zone — reset dwell streak
            self._zone_dwell_name   = None
            self._zone_dwell_frames = 0
            return None

        if zone_name != self._zone_dwell_name:
            # Entered a different zone — start fresh dwell counter
            self._zone_dwell_name   = zone_name
            self._zone_dwell_frames = 1
            return None

        # Same zone as last frame
        self._zone_dwell_frames += 1

        if (self._zone_dwell_frames >= _ZONE_DWELL_FRAMES
                and zone_name not in self.counted_zones):
            self.counted_zones.add(zone_name)
            return zone_name   # signal: count this zone entry now

        return None


# ──────────────────────────────────────────────────────────────────────────────
# IdentityManager
# ──────────────────────────────────────────────────────────────────────────────

class IdentityManager:

    def __init__(self):
        self._active:   Dict[int, PersonState]              = {}
        self._id_map:   Dict[int, int]                      = {}
        self._id_map_last_seen: Dict[int, float]            = {}
        self._lost:     Dict[int, Tuple[np.ndarray, float]] = {}
        self._best_info: Dict[int, dict]                    = {}
        self._write_gen: Dict[int, int]                     = {}

        self._best_dir = os.path.join("output", "best")
        os.makedirs(self._best_dir, exist_ok=True)

        self._archive_dir_cache: Dict[str, str] = {}
        self._write_pool = ThreadPoolExecutor(max_workers=1)
        self._last_prune_time: float = time.time()

        print(f"[IDMGR] IdentityManager initialised | best_dir={self._best_dir}")

        self._recover_live_buffer()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _archive_dir(self, dt: datetime) -> str:
        key = dt.strftime("day_%Y%m%d/hour_%H")
        if key not in self._archive_dir_cache:
            full = os.path.join(self._best_dir, *key.split("/"))
            os.makedirs(full, exist_ok=True)
            self._archive_dir_cache[key] = full
        return self._archive_dir_cache[key]

    def _expire_lost(self) -> None:
        now = time.time()
        for lost_cid in list(self._lost):
            _, t_lost = self._lost[lost_cid]
            if now - t_lost > _BUFFER_SECONDS:
                self._move_best_to_archive(lost_cid)
                del self._lost[lost_cid]

    def _prune_id_map(self) -> None:
        now = time.time()
        if now - self._last_prune_time < 1.0:
            return
        self._last_prune_time = now
        stale = [bt for bt, t in self._id_map_last_seen.items()
                 if now - t > _ID_MAP_PRUNE_SECONDS]
        for bt in stale:
            self._id_map.pop(bt, None)
            self._id_map_last_seen.pop(bt, None)

    def _reid_lookup(self, embedding: np.ndarray) -> Optional[int]:
        """Same-session lost-track re-association only (no gallery search)."""
        if not self._lost:
            return None
        now = time.time()
        valid_ids, valid_embs, ages = [], [], []
        for lid, (emb, t_lost) in self._lost.items():
            valid_ids.append(lid)
            valid_embs.append(emb)
            ages.append(now - t_lost)
        mat   = np.stack(valid_embs, axis=0)
        norms = np.linalg.norm(mat, axis=1)
        eq    = np.linalg.norm(embedding)
        sims  = np.zeros(len(valid_ids), dtype=np.float32)
        mask  = (norms > 0) & (eq > 0)
        if mask.any():
            sims[mask] = (mat[mask] @ embedding) / (norms[mask] * eq)
        best_idx  = int(np.argmax(sims))
        threshold = (_REID_WITHIN_BUFFER_THRESHOLD
                     if ages[best_idx] <= _BUFFER_SECONDS
                     else _REID_CROSS_VISIT_THRESHOLD)
        return valid_ids[best_idx] if sims[best_idx] >= threshold else None

    # ── Main process loop ─────────────────────────────────────────────────────

    def process(self, frame: np.ndarray, raw_tracks: List[dict]) -> List[dict]:
        self._expire_lost()
        self._prune_id_map()

        active_cids: set    = set()
        results: List[dict] = []

        for t in raw_tracks:
            bt_id = t.get("track_id")

            # Drop unconfirmed (negative synthetic) IDs from YoloDetector
            if bt_id is None or bt_id <= 0:
                continue

            x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
            conf            = float(t["conf"])
            box             = (x1, y1, x2, y2)

            crop = _crop(frame, x1, y1, x2, y2)

            self._id_map_last_seen[bt_id] = time.time()

            # ── Resolve canonical ID (same-session re-ID only) ─────────────
            if bt_id not in self._id_map:
                emb_for_reid = _colour_hist(crop) if crop is not None else None
                reclaimed    = self._reid_lookup(emb_for_reid) \
                               if emb_for_reid is not None else None
                self._id_map[bt_id] = reclaimed if reclaimed is not None else bt_id
            cid = self._id_map[bt_id]

            if cid in self._lost:
                del self._lost[cid]

            # ── Update / create PersonState ────────────────────────────────
            prev_box = self._active[cid]._prev_box if cid in self._active else None
            need_emb = _box_moved(prev_box, box, _HIST_SKIP_PX) and crop is not None
            emb      = _colour_hist(crop) if need_emb else None

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cid in self._active:
                self._active[cid].update((cx, cy), box, conf, emb)
            else:
                init_emb = (emb if emb is not None
                            else (_colour_hist(crop) if crop is not None
                                  else np.zeros(96, dtype=np.float32)))
                self._active[cid] = PersonState(cid, (cx, cy), box, conf, init_emb)

            active_cids.add(cid)

            # ── Zone detection ─────────────────────────────────────────────
            # Use foot-point (bottom-centre of bounding box)
            foot_x = cx
            foot_y = y2
            current_zone = _current_zone(foot_x, foot_y)
            self._active[cid].zone = current_zone

            # Check dwell guard and count zone entries
            triggered_zone = self._active[cid].update_zone_dwell(current_zone)
            if triggered_zone is not None:
                today_str = date.today().isoformat()
                self._write_pool.submit(increment_zone_count, triggered_zone, today_str)
                print(f"[ZONE] cid={cid} COUNTED in zone='{triggered_zone}' | {today_str}")

            # ── Best-frame capture (zone-gated if CAPTURE_ZONES set) ───────
            if crop is not None:
                should_capture = (
                    not _CAPTURE_ZONES                          # no restriction
                    or current_zone in _CAPTURE_ZONES           # inside a capture zone
                )
                if should_capture:
                    self._update_best(cid, conf, crop, current_zone)

        # ── Gone persons → _lost ───────────────────────────────────────────
        for gone_cid in set(self._active) - active_cids:
            s = self._active.pop(gone_cid)
            self._lost[gone_cid] = (s.embedding, time.time())

        # ── Build results ──────────────────────────────────────────────────
        for cid in active_cids:
            s = self._active[cid]
            results.append(dict(
                track_id       = cid,
                x1=s.box[0], y1=s.box[1], x2=s.box[2], y2=s.box[3],
                conf           = s.conf,
                best_conf      = self._best_info.get(cid, {}).get("conf", s.conf),
                centre         = s.centre_history[-1],
                centre_history = s.centre_history,
                zone           = s.zone,
                counted_zones  = list(s.counted_zones),
                first_seen     = s.first_seen,
                last_seen      = s.last_seen,
            ))

        return results

    # ── _update_best ──────────────────────────────────────────────────────────

    def _update_best(self, cid: int, conf: float,
                     crop: np.ndarray, zone: Optional[str] = None) -> None:
        if crop is None or crop.size == 0:
            if cid in self._best_info and cid in self._active:
                self._best_info[cid]["last_seen"] = self._active[cid].last_seen
            return

        image_path = os.path.join(self._best_dir, f"id_{cid}.jpg")
        is_new     = cid not in self._best_info
        is_better  = not is_new and conf > self._best_info[cid]["conf"]

        if not (is_new or is_better):
            if cid in self._best_info and cid in self._active:
                self._best_info[cid]["last_seen"] = self._active[cid].last_seen
            return

        s = self._active.get(cid)
        first_seen = s.first_seen if s else time.time()
        last_seen  = s.last_seen  if s else time.time()
        crop_h     = crop.shape[0]

        self._best_info[cid] = {
            "conf"      : conf,
            "first_seen": first_seen,
            "last_seen" : last_seen,
            "image_path": image_path,
            "crop_h"    : crop_h,
            "zone"      : zone,
        }

        gen = self._write_gen.get(cid, 0) + 1
        self._write_gen[cid] = gen

        self._write_pool.submit(
            _write_best_worker,
            image_path, crop.copy(), cid, conf,
            first_seen, last_seen, crop_h, zone,
            self._write_gen, gen,
        )

    # ── _move_best_to_archive ─────────────────────────────────────────────────

    def _move_best_to_archive(self, cid: int) -> None:
        if cid not in self._best_info:
            self._write_gen.pop(cid, None)
            return

        info = self._best_info.pop(cid)
        self._write_gen.pop(cid, None)
        src  = info["image_path"]

        # Wait up to 1 s for the async write to land
        if not os.path.exists(src):
            deadline = time.time() + 1.0
            while not os.path.exists(src) and time.time() < deadline:
                time.sleep(0.02)

        if not os.path.exists(src):
            print(f"[ARCHIVE] cid={cid} | ❌ FILE MISSING after 1s — no DB record.")
            return

        dst_dir = self._archive_dir(datetime.now())
        dst     = os.path.join(dst_dir, f"id_{cid}_conf_{info['conf']:.2f}.jpg")

        self._write_pool.submit(
            _archive_worker,
            src, dst, cid,
            info["conf"],
            info["first_seen"],
            info["last_seen"],
            info.get("crop_h", 0),
            info.get("zone"),
        )

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        print(f"[IDMGR] shutdown() | active={len(self._active)} | lost={len(self._lost)}")
        for cid in list(self._active):
            self._move_best_to_archive(cid)
        self._active.clear()
        for lost_cid in list(self._lost):
            self._move_best_to_archive(lost_cid)
        self._lost.clear()
        self._write_pool.shutdown(wait=True)
        print("[IDMGR] shutdown() complete — all tasks drained")
        _db_close()

    # ── Orphan recovery (identical to original) ───────────────────────────────

    def _recover_live_buffer(self) -> None:
        stale = get_all_live_buffer()
        if not stale:
            return
        print(f"[IdentityManager] Recovering {len(stale)} orphaned image(s) from previous run…")
        now = datetime.now()
        for row in stale:
            cid = row["cid"]
            src = row["image_path"]
            if not os.path.exists(src):
                self._write_pool.submit(delete_live_buffer, cid)
                continue
            dst_dir = self._archive_dir(now)
            dst = os.path.join(
                dst_dir,
                f"id_{cid}_conf_{row['best_conf']:.2f}_recovered.jpg",
            )
            self._write_pool.submit(
                _archive_worker,
                src, dst, cid,
                row["best_conf"], row["first_seen"], row["last_seen"],
                row.get("crop_h", 0),
                row.get("zone"),
            )
        print("[IdentityManager] Recovery tasks submitted.")

    # ── Public accessors ──────────────────────────────────────────────────────

    def get_state(self) -> Dict[int, PersonState]:
        return self._active

    def active_count(self) -> int:
        return len(self._active)


# ──────────────────────────────────────────────────────────────────────────────
# Background workers
# ──────────────────────────────────────────────────────────────────────────────

def _write_best_worker(
    path: str,
    crop: np.ndarray,
    cid: int,
    conf: float,
    first_seen: float,
    last_seen: float,
    crop_h: int,
    zone: Optional[str],
    write_gen: dict,
    gen_token: int,
) -> None:
    if write_gen.get(cid, 0) != gen_token:
        return   # stale write — a newer crop superseded this one
    ok = cv2.imwrite(path, crop)
    if ok:
        upsert_live_buffer(cid, path, conf, first_seen, last_seen, crop_h, zone)


def _archive_worker(
    src: str,
    dst: str,
    track_id: int,
    best_conf: float,
    first_seen: float,
    last_seen: float,
    crop_h: int = 0,
    zone: Optional[str] = None,
) -> None:
    try:
        os.rename(src, dst)
    except OSError:
        try:
            shutil.copy2(src, dst)
            os.remove(src)
        except OSError:
            pass

    emotion, emotion_score = _analyse_emotion(dst, crop_h)

    save_person_metadata(
        track_id      = track_id,
        best_conf     = best_conf,
        first_seen    = first_seen,
        last_seen     = last_seen,
        image_path    = dst,
        emotion       = emotion,
        emotion_score = emotion_score,
        zone          = zone,
    )

    delete_live_buffer(track_id)
    print(f"[ARCHIVE_WORKER] track_id={track_id} | zone={zone} | "
          f"emotion={emotion} ({emotion_score}) | ✅ archived")