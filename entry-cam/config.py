"""
config.py
=========
Central configuration for the entire pipeline.
Edit values here; no other file needs changing for tuning.
"""
import os

# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH           = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF", "0.35"))
YOLO_IOU_THRESHOLD        = float(os.getenv("YOLO_IOU",  "0.45"))
YOLO_DEVICE               = os.getenv("YOLO_DEVICE", "cpu")   # "cpu" | "cuda" | "mps"
YOLO_IMGSZ                = int(os.getenv("YOLO_IMGSZ", "320"))  # inference resolution — 320 is ~3× faster than 640 on CPU


# ---------------------------------------------------------------------------
# Backend WebSocket forwarding
# Camera connects OUT to your backend — no port needs to be exposed.
# ---------------------------------------------------------------------------
# URL of your backend WS endpoint, e.g. "wss://yourbackend.com/cam/stream"
# Leave empty to disable all backend forwarding.
BACKEND_WS_URL   = os.getenv("BACKEND_WS_URL",   "wss://enabled-flowing-bedbug.ngrok-free.app/cam/stream")
# Shared secret sent as Bearer token in the WS upgrade header.
BACKEND_WS_TOKEN = os.getenv("BACKEND_WS_TOKEN", "token_for_linmar_backend_ws_auth")

# Identifier sent with every message so your backend knows which camera.
BACKEND_CAM_ID   = os.getenv("BACKEND_CAM_ID",   "entry-cam")
# Live frames forwarded per second. Set to 0 to disable frame forwarding.
BACKEND_FRAME_FPS = float(os.getenv("BACKEND_FRAME_FPS", "1"))

# ---------------------------------------------------------------------------
# Re-ID buffer
# ---------------------------------------------------------------------------
REID_BUFFER_FRAMES   = int(os.getenv("REID_BUFFER", "90"))   # frames to keep lost embedding
REID_SIM_THRESHOLD   = float(os.getenv("REID_SIM",  "0.75")) # min cosine sim to reclaim ID
REID_EMBEDDING_ALPHA = float(os.getenv("REID_ALPHA", "0.3")) # EMA weight for new embedding

# ---------------------------------------------------------------------------
# Trail
# ---------------------------------------------------------------------------
# (TRAIL_MAX_LEN is set in the Entry Module section below)

# ---------------------------------------------------------------------------
# Zones  –  legacy placeholder, not used by entry-cam pipeline
# ---------------------------------------------------------------------------
ZONES: dict = {}

# ---------------------------------------------------------------------------
# Source / display
# ---------------------------------------------------------------------------
SOURCE_FPS   = os.getenv("SOURCE_FPS", None)

# ══════════════════════════════════════════════════════════════════════════════
# Entry Module — add these to your existing config.py
# ══════════════════════════════════════════════════════════════════════════════

# ── Display ───────────────────────────────────────────────────────────────────
SHOW_WINDOW = False  # Set True only for local debug — saves ~2 ms/frame on pipeline

# ── Performance tuning ────────────────────────────────────────────────────────
# Run YOLO only on 1 of every N frames; return cached tracks on skipped frames.
# 1 = off (every frame), 2 = half, 3 = third. Good default for 15-20 fps RTSP: 2.
YOLO_SKIP_FRAMES = int(os.getenv("YOLO_SKIP_FRAMES", "2"))

# Max frames-per-second at which push_frame() re-encodes the JPEG for the
# MJPEG stream and backend. Lower = less encoding CPU on the pipeline thread.
PUSH_FRAME_FPS = float(os.getenv("PUSH_FRAME_FPS", "10"))

# ── Tracking ──────────────────────────────────────────────────────────────────
TRAIL_MAX_LEN = 30

# ── Embedding ─────────────────────────────────────────────────────────────────
# EMA weight when blending new live embedding into existing one (0–1).
# Lower = slower to change = more stable but slower to adapt.
REID_EMBEDDING_ALPHA = 0.3

# ── Short-term lost buffer ────────────────────────────────────────────────────
# How long (seconds) to remember a disappeared track before archiving it.
ENTRY_MIN_DWELL_SECONDS = 0.20

# Cosine similarity threshold for short-term re-association.
# 0.65 is safe for spatial HSV embeddings.
ENTRY_REID_SIM_THRESHOLD = 0.65

# ── Session gallery ───────────────────────────────────────────────────────────
# Cosine similarity threshold for cross-session / cross-day deduplication.
#
# HSV colour histograms can reach 0.82 between two unrelated people wearing
# similar-toned clothing (dark suits, uniforms, etc.).  0.92 sits above the
# typical inter-person similarity so only a genuine same-person match passes.
#
# If you see the same person counted twice on a restart → lower this slightly.
# If different people are suppressed → raise it further toward 0.95.
ENTRY_SESSION_THRESHOLD = 0.92

# Skip recomputing embedding if bbox moved less than this many pixels.
ENTRY_HIST_SKIP_PX = 2

# SQLite path for the persistent session gallery.
ENTRY_DB_PATH = "output/entry_session.db"

# ── Zone config ───────────────────────────────────────────────────────────────
# Run:  python tools/draw_zones.py --source 0
# Click to draw polygons, press S to print config, paste below.
#
# Example for a 1280x720 side-view camera:
ENTRY_ZONES = {
    "detector": [(1355, 587), (2314, 623), (2314, 1090), (1268, 1076)],
    "inside": [(673, 1208), (2303, 1171), (2326, 1497), (695, 1505)],
}
# The zone transition that triggers a unique-entry count increment.
ENTRY_COUNT_TRIGGER = "detector\u2192inside"