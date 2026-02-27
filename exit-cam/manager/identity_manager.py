"""
IdentityManager  —  Exit Edition
==================================
Image lifecycle (ONE image per person, ever):
  1. Person enters exit zone
     → output/best/id_<cid>.jpg created / replaced if conf improves
  2. Person leaves frame → enters _lost buffer (30 s window)
     → image file stays in output/best/, NOT archived yet
     → if they return within 30 s: ReID reclaim, same PersonState,
       image keeps improving, no duplicate
  3. Lost buffer expires (30 s with no re-detection)
     → THEN and only then: emotion analysed, image moved to archive
     → DB row written

Counting rule (single zone):
  Person is counted as a unique exit when they were inside the exit zone
  at any point during their visit and then disappear from frame.
  Safety-net: on lost-buffer expiry, if was_in_exit_zone and not yet
  counted → count (catches brief-leave / cross-frame-boundary cases).

De-duplication:
  - state.counted flag survives ReID reclaims within the 30 s lost buffer
    → same person re-detected = no recount
  - After lost buffer expires, gallery.find_match() at threshold 0.92
    catches the same person returning → update row, no new count
"""
from __future__ import annotations

import base64
import os
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from .db_helper  import save_person_metadata, increment_zone_count, close as _db_close
from .exit_db    import ExitDB
from .zone_manager import ZoneManager


def _atomic_imwrite(path: str, img) -> bool:
    """
    Write an image atomically: encode to JPEG bytes, write to .tmp, then
    os.replace() into place.  Using imencode avoids OpenCV choosing the wrong
    codec based on the '.tmp' extension.
    On Windows, os.replace() raises PermissionError if the destination is
    currently open — retried up to 3 times with a short sleep.
    """
    try:
        import cv2 as _cv2
        import time as _time
        if img is None or img.size == 0:
            print(f"[Exit][imwrite] SKIP — empty crop for {path}")
            return False
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ok, buf = _cv2.imencode(".jpg", img)
        if not ok:
            print(f"[Exit][imwrite] FAIL — imencode returned False  shape={img.shape}")
            return False
        tmp = path + ".tmp"
        with open(tmp, "wb") as _f:
            _f.write(buf.tobytes())
        for _attempt in range(3):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if _attempt < 2:
                    _time.sleep(0.05)
                else:
                    raise
        print(f"[Exit][imwrite] OK  → {path}  shape={img.shape}")
        return True
    except Exception as _e:
        print(f"[Exit][imwrite] EXCEPTION for {path}: {_e}")
        return False

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_BUFFER_SECONDS : float = float(getattr(config, "EXIT_REID_BUFFER_SECONDS", 30.0))
_REID_SIM       : float = float(getattr(config, "EXIT_REID_SIM_THRESHOLD",  0.65))
_SESSION_SIM    : float = float(getattr(config, "EXIT_SESSION_THRESHOLD",   0.92))
_HIST_SKIP_PX   : int   = int  (getattr(config, "EXIT_HIST_SKIP_PX",        4))
_EMB_ALPHA      : float = float(getattr(config, "REID_EMBEDDING_ALPHA",     0.3))
_TRAIL_LEN      : int   = int  (getattr(config, "TRAIL_MAX_LEN",            30))
_MERGE_PX       : int   = 60   # centroid distance threshold for temp→real ID merge

_EXIT_ZONE      : str   = getattr(config, "EXIT_ZONE_NAME", "exit")

_HAPPY_THRESHOLD:      float = getattr(config, "EMOTION_HAPPY_THRESHOLD",      0.25)
_VERY_HAPPY_THRESHOLD: float = getattr(config, "EMOTION_VERY_HAPPY_THRESHOLD", 0.65)
_SAD_THRESHOLD:        float = getattr(config, "EMOTION_SAD_THRESHOLD",        0.25)
_MIN_SCORE:            float = getattr(config, "EMOTION_MIN_SCORE",            0.25)
_EMOTIEFF_MODEL: str          = getattr(config, "EMOTIEFF_MODEL", "enet_b0_8_best_afew")

# Callback registered by dashboard so background archive threads can forward
# the emotion result to the backend WS.  Set via: identity_manager._on_archived = fn
_on_archived: Optional[callable] = None


# ──────────────────────────────────────────────────────────────────────────────
# Emotion recogniser — lazy singleton (loaded once on first archive, not live)
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


def _analyse_emotion(image_path: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Run emotion recognition on an archived person-crop image.

    Diagnostic confirmed (CCTV footage, 222×493 person crops):
      - Haar cascade detects 0 faces — useless on compressed CCTV frames.
      - emotiefflib's EfficientNet-B0 preprocesses the image itself; it does
        NOT need a pre-cropped face.  Passing the top-30 % head strip gives
        correct predictions (e.g. 'Surprise' 0.34, 'Neutral' 0.26).
      - Full-body crop yields low-confidence Neutral — incorrect.
      - Therefore: slice the top 30 % of the bounding-box crop and pass
        that directly to predict_emotions().

    Mapping (AffectNet 8-class model output → 3 display labels):
        happy              → Happy / Very Happy  (split by confidence)
        surprise           → Happy
        sad / angry / anger
        fear / disgust
        contempt           → Sad
        neutral / other    → None  (no label stored = "Undetected" in UI)
    """
    fer = _get_fer()
    if fer is None:
        return None, None

    try:
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            return None, None

        h = img.shape[0]

        # Top-30 % of person crop = head + shoulder region.
        # emotiefflib handles its own internal preprocessing.
        top_h      = max(h * 30 // 100, 48)
        head_strip = img[:top_h, :]
        head_rgb   = cv2.cvtColor(head_strip, cv2.COLOR_BGR2RGB)

        emotion_labels, scores = fer.predict_emotions(head_rgb, logits=False)

        if not emotion_labels or scores is None or scores.size == 0:
            return None, None

        predicted   = emotion_labels[0].lower().strip()
        # scores shape: (1, n_classes) for a single image
        face_scores = scores[0] if scores.ndim > 1 else scores
        top_conf    = float(np.max(face_scores))

        print(f"[Emotion] {os.path.basename(image_path)} → {predicted} ({top_conf:.2f})")

        # Below minimum confidence — don't assign any label
        if top_conf < _MIN_SCORE:
            return None, None

        if predicted == "happy":
            label = "Very Happy" if top_conf >= _VERY_HAPPY_THRESHOLD else "Happy"
            return label, top_conf

        if predicted in ("surprise", "neutral"):
            return "Happy", top_conf

        if predicted in ("sad", "angry", "anger", "fear", "disgust", "contempt"):
            return "Sad", top_conf

        # any other label → Happy (default positive assumption)
        return "Happy", top_conf

    except Exception as exc:
        print(f"[Emotion] ERROR {os.path.basename(image_path)}: {exc}")
        return None, None


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
    strips  = [crop[:h//3], crop[h//3:2*h//3], crop[2*h//3:]]
    weights = (1.5, 1.0, 0.7)
    parts   = []
    for strip, weight in zip(strips, weights):
        if strip.size == 0:
            parts.append(np.zeros(bins * 3, dtype=np.float32))
            continue
        hsv   = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        hists = []
        for ch, rng in zip(range(3), [(0, 180), (0, 256), (0, 256)]):
            h_arr = cv2.calcHist([hsv], [ch], None, [bins], list(rng)).flatten().astype(np.float32)
            s     = h_arr.sum()
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
      a) Reclaim a person back into _active (re-detected within buffer)
      b) Finalize them on expiry (archive image, write DB row)

    was_in_exit_zone: True iff an image was actually captured for this person.
      Images are only written when the person is inside the exit zone, so
      bool(image_path) is the reliable evidence. Mirrors entry-cam's
      was_in_capture_zone = bool(image_path).
    """
    __slots__ = ("live_embedding", "best_embedding", "best_conf",
                 "image_path", "first_seen", "last_seen",
                 "counted", "was_in_exit_zone", "last_zone", "t_lost")

    def __init__(self, state: "PersonState", image_path: str, t_lost: float):
        self.live_embedding   = state.live_embedding.copy()
        self.best_embedding   = state.best_embedding.copy()
        self.best_conf        = state.best_conf
        self.image_path       = image_path
        self.first_seen       = state.first_seen
        self.last_seen        = state.last_seen
        self.counted          = state.counted
        self.was_in_exit_zone = bool(image_path)   # evidence-based, mirrors entry-cam
        self.last_zone        = state.zone
        self.t_lost           = t_lost


# ──────────────────────────────────────────────────────────────────────────────
# PersonState
# ──────────────────────────────────────────────────────────────────────────────

class PersonState:
    __slots__ = (
        "track_id", "first_seen", "last_seen",
        "centre_history", "zone", "prev_zone",
        "live_embedding", "best_embedding",
        "box", "conf", "best_conf", "best_frame",
        "was_in_exit_zone", "counted",
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
        self.was_in_exit_zone      = False
        self.counted               = False
        self._prev_box             = box

    def restore_from_lost(self, entry: LostEntry) -> None:
        """Restore counted + embedding state from a reclaimed LostEntry."""
        self.counted          = entry.counted
        self.live_embedding   = entry.live_embedding.copy()
        self.best_embedding   = entry.best_embedding.copy()
        self.best_conf        = max(self.best_conf, entry.best_conf)
        self.first_seen       = entry.first_seen
        self.was_in_exit_zone = entry.was_in_exit_zone

    def update(self, centre: Tuple[int, int], box: Tuple, conf: float,
               embedding: Optional[np.ndarray], crop: np.ndarray, frame_h: int,
               in_exit: bool) -> None:
        self.last_seen = time.time()
        self.prev_zone = self.zone
        self.centre_history.append(centre)
        self.box  = box
        self.conf = conf

        if embedding is not None:
            self.live_embedding = _EMB_ALPHA * embedding + (1 - _EMB_ALPHA) * self.live_embedding
            self._prev_box = box

        if in_exit:
            self.was_in_exit_zone = True

        prev_best = self.best_conf

        # Best frame + best embedding: exit zone only.
        # Prevents outside-zone high-conf frames from blocking zone captures.
        # Mirrors entry-cam's in_capture guard for best_frame.
        if in_exit and (self.best_frame is None or conf > prev_best):
            self.best_conf      = conf
            self.best_frame     = crop.copy()
            self.best_embedding = _spatial_embedding(crop, frame_h)


# ──────────────────────────────────────────────────────────────────────────────
# IdentityManager
# ──────────────────────────────────────────────────────────────────────────────

class IdentityManager:

    def __init__(self):
        self._active:   Dict[Any, PersonState] = {}
        self._id_map:   Dict[Any, Any]         = {}
        self._lost:     Dict[Any, LostEntry]   = {}
        self._img_path: Dict[Any, str]         = {}

        zones = getattr(config, "EXIT_ZONES", {})
        if not zones:
            print("[IdentityManager] WARNING: EXIT_ZONES not set. Run manager/draw_zone.py")
        self._zones = ZoneManager(zones)

        db_path = getattr(config, "EXIT_DB_PATH", "output/exit_session.db")
        self._gallery = ExitDB(db_path=db_path)
        self.unique_exit_count: int = self._gallery.total_unique()
        print(f"[IdentityManager] Started. Known persons: {self.unique_exit_count}")

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
        self.unique_exit_count = 0
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
        ids   = list(self._lost.keys())
        embs  = [e.live_embedding for e in self._lost.values()]
        mat   = np.stack(embs)
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
        """Single entry point for all exit counting. Deduplication via gallery."""
        if state.counted:
            return
        state.counted = True

        img_path = self._img_path.get(cid, "")
        best_emb = state.best_embedding
        emb_norm = float(np.linalg.norm(best_emb))

        if emb_norm < 0.1:
            print(f"[Exit] WARNING cid={cid} reason={reason}: zero embedding — "
                  "person may not have been in exit zone long enough.")

        match_cid = self._gallery.find_match(best_emb, threshold=_SESSION_SIM)
        if match_cid is not None:
            self._gallery.upsert(match_cid, best_emb, image_path=img_path)
            print(f"[Exit] {reason}: cid={cid} → matched {match_cid} — returning visitor")
        else:
            # session_gallery.cid is INTEGER PRIMARY KEY — convert tmp_ strings
            gallery_cid = cid if isinstance(cid, int) else abs(hash(str(cid))) & 0x7FFFFFFF
            self._gallery.upsert(gallery_cid, best_emb, image_path=img_path)
            self.unique_exit_count = self._gallery.total_unique()
            print(f"[Exit] {reason}: NEW  cid={cid}  emb_norm={emb_norm:.3f}  "
                  f"total={self.unique_exit_count}")

        today_str = date.today().isoformat()
        self._write_pool.submit(increment_zone_count, _EXIT_ZONE, today_str)

    # ── Best frame: capture-and-replace (ONE file per person) ────────────────

    def _update_best(self, cid: Any, conf: float, crop: np.ndarray,
                     state: PersonState) -> None:
        """
        Write/overwrite output/best/id_<cid>.jpg only when conf improves.
        Single file per person — same filename, replaced in-place.
        Archive happens later (on lost-buffer expiry), never here.
        Only called when person is inside the exit zone.
        """
        img_path = os.path.join(self._best_dir, f"id_{str(cid).replace('tmp_', 'tmp')}.jpg")
        is_new   = cid not in self._img_path

        stored_conf = 0.0 if is_new else state.best_conf
        if is_new or conf >= stored_conf:
            print(f"[Exit][capture] cid={cid}  conf={conf:.3f}  stored={stored_conf:.3f}  new={is_new}  → {img_path}")
            self._write_pool.submit(_atomic_imwrite, img_path, crop.copy())
            self._img_path[cid] = img_path
            if self._gallery.contains(cid):
                self._write_pool.submit(self._gallery.update_image_path, cid, img_path)

    # ── Finalize on lost-buffer expiry ────────────────────────────────────────

    def _finalize(self, cid: Any, entry: LostEntry) -> None:
        """
        Called ONLY when lost buffer expires (30 s no re-detection).
        Archive image, run emotion analysis in background, write DB row.

        SAFETY-NET: if person was in exit zone but never explicitly counted
        (e.g. disappeared between frames mid-zone), count them now.
        """
        src = self._img_path.pop(cid, entry.image_path)

        if not entry.counted:
            if entry.was_in_exit_zone and src and os.path.exists(src):
                print(f"[Exit] safety_net: cid={cid} — was in exit zone, counting on expiry")
                match_cid = self._gallery.find_match(entry.best_embedding, threshold=_SESSION_SIM)
                if match_cid is not None:
                    self._gallery.upsert(match_cid, entry.best_embedding, image_path=src)
                    print(f"[Exit] safety_net: cid={cid} matched {match_cid} — returning visitor")
                else:
                    gallery_cid = cid if isinstance(cid, int) else abs(hash(str(cid))) & 0x7FFFFFFF
                    self._gallery.upsert(gallery_cid, entry.best_embedding, image_path=src)
                    self.unique_exit_count = self._gallery.total_unique()
                    print(f"[Exit] safety_net: NEW  cid={cid}  total={self.unique_exit_count}")
                today_str = date.today().isoformat()
                self._write_pool.submit(increment_zone_count, _EXIT_ZONE, today_str)
                # fall through to archive
            else:
                # Never in exit zone — discard image silently
                if src and os.path.exists(src):
                    self._write_pool.submit(_discard_worker, src)
                return

        if not src or not os.path.exists(src):
            return

        dst_dir     = self._archive_dir(datetime.now())
        safe        = str(cid).replace("tmp_", "tmp")
        dst         = os.path.join(dst_dir, f"id_{safe}_conf_{entry.best_conf:.2f}.jpg")
        db_track_id = cid if isinstance(cid, int) else abs(hash(str(cid))) & 0x7FFFFFFF
        self._write_pool.submit(
            _archive_worker, src, dst, db_track_id,
            entry.best_conf, entry.first_seen, entry.last_seen,
            entry.last_zone,
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
                    self._id_map[bt_id] = reclaimed_cid
                    cid = reclaimed_cid

            # ── Zone update ───────────────────────────────────────────────────
            self._zones.update(cid, (cx, cy))
            current_zone = self._zones.current_zone(cid)
            in_exit      = current_zone == _EXIT_ZONE

            # ── Create or update PersonState ──────────────────────────────────
            if cid not in self._active:
                init_emb = emb if emb is not None else _spatial_embedding(crop, frame_h)
                self._active[cid] = PersonState(cid, (cx, cy), box, conf, init_emb)
                if reclaimed_entry is not None:
                    self._active[cid].restore_from_lost(reclaimed_entry)
                    if reclaimed_entry.image_path:
                        self._img_path[cid] = reclaimed_entry.image_path
                # First-frame zone flag
                if in_exit:
                    self._active[cid].was_in_exit_zone = True
            else:
                self._active[cid].update(
                    (cx, cy), box, conf, emb, crop, frame_h, in_exit
                )

            state      = self._active[cid]
            state.zone = current_zone
            active_cids.add(cid)

            # ── Best frame: ONLY capture when inside exit zone ────────────────
            # Mirrors entry-cam's in_capture guard exactly.
            # Person's bounding-box crop is saved; full frame is never written.
            if in_exit:
                if cid not in self._img_path:
                    print(f"[Exit][zone] cid={cid} ENTERED exit zone")
                self._update_best(cid, conf, crop, state)

        # ── Disappeared → lost buffer (no archive yet) ────────────────────────
        for gone_cid in set(self._active) - active_cids:
            s = self._active.pop(gone_cid)

            # Count: person was in exit zone at some point, now left the frame
            if s.was_in_exit_zone and not s.counted:
                self._count_person(gone_cid, s, "left_after_exit_zone")

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
            s = self._active.get(cid)  # guard: may have been removed by merge
            if s is None:
                continue
            results.append(dict(
                track_id          = cid,
                x1=s.box[0], y1=s.box[1], x2=s.box[2], y2=s.box[3],
                conf              = s.conf,
                best_conf         = s.best_conf,
                centre            = s.centre_history[-1],
                centre_history    = s.centre_history,
                zone              = s.zone,
                first_seen        = s.first_seen,
                last_seen         = s.last_seen,
                unique_exit_count = self.unique_exit_count,
            ))
        return results

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        for cid, s in list(self._active.items()):
            img_path = self._img_path.get(cid, "")
            entry    = LostEntry(s, img_path, time.time())
            self._finalize(cid, entry)
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
        shutil.move(src, dst)
    except OSError:
        pass


def _discard_worker(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _archive_worker(src: str, dst: str, track_id: Any,
                    best_conf: float, first_seen: float, last_seen: float,
                    zone: Optional[str] = None) -> None:
    try:
        shutil.move(src, dst)
    except OSError:
        pass

    emotion, emotion_score = _analyse_emotion(dst)

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

    # Forward to backend WS (non-blocking — dashboard registers this callback)
    if _on_archived is not None:
        try:
            img_b64 = None
            try:
                with open(dst, "rb") as _f:
                    img_b64 = base64.b64encode(_f.read()).decode()
            except Exception:
                pass
            _on_archived({
                "event":         "archived",
                "track_id":      track_id,
                "emotion":       emotion,
                "emotion_score": round(float(emotion_score), 3) if emotion_score is not None else None,
                "zone":          zone,
                "image":         img_b64,
                "ts":            time.time(),
            })
        except Exception:
            pass
