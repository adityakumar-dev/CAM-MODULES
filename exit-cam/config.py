"""
config.py
=========
Central configuration for the exit-cam pipeline.
Edit values here; no other file needs changing for tuning.
"""
import os

# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH           = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF", "0.15"))
YOLO_IOU_THRESHOLD        = float(os.getenv("YOLO_IOU",  "0.45"))
YOLO_DEVICE               = os.getenv("YOLO_DEVICE", "cpu")   # "cpu" | "cuda" | "mps"
YOLO_IMGSZ                = int(os.getenv("YOLO_IMGSZ", "640"))  # inference resolution

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
BACKEND_CAM_ID   = os.getenv("BACKEND_CAM_ID",   "exit-cam")
# Live frames forwarded per second. Set to 0 to disable frame forwarding.
BACKEND_FRAME_FPS = float(os.getenv("BACKEND_FRAME_FPS", "1"))

# ---------------------------------------------------------------------------
# Emotion thresholds  (analysis runs at archive time, not live)
# ---------------------------------------------------------------------------
EMOTION_HAPPY_THRESHOLD      = 0.40
EMOTION_VERY_HAPPY_THRESHOLD = 0.75
EMOTION_SAD_THRESHOLD        = 0.40
EMOTIEFF_MODEL               = os.getenv("EMOTIEFF_MODEL", "enet_b0_8_best_afew")

# ---------------------------------------------------------------------------
# Re-ID buffer
# ---------------------------------------------------------------------------
REID_EMBEDDING_ALPHA = float(os.getenv("REID_ALPHA", "0.3"))

# ---------------------------------------------------------------------------
# Trail
# ---------------------------------------------------------------------------
TRAIL_MAX_LEN = 30

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
SHOW_WINDOW = True

# ══════════════════════════════════════════════════════════════════════════════
# Exit Module
# ══════════════════════════════════════════════════════════════════════════════

# ── Short-term lost buffer ────────────────────────────────────────────────────
# How long (seconds) to remember a disappeared track before finalising it.
EXIT_REID_BUFFER_SECONDS = float(os.getenv("EXIT_REID_BUFFER_SECONDS", "30.0"))

# Cosine similarity threshold for short-term re-association within lost buffer.
EXIT_REID_SIM_THRESHOLD = float(os.getenv("EXIT_REID_SIM", "0.65"))

# ── Session gallery (cross-restart dedup) ─────────────────────────────────────
# Cosine similarity threshold for matching a person against the session gallery.
# Raise toward 0.95 if different people are suppressed; lower if same person
# is double-counted across restarts.
EXIT_SESSION_THRESHOLD = float(os.getenv("EXIT_SESSION_THRESHOLD", "0.92"))

# Skip recomputing embedding if bbox moved less than this many pixels.
EXIT_HIST_SKIP_PX = int(os.getenv("EXIT_HIST_SKIP_PX", "2"))

# SQLite path for the persistent session gallery.
EXIT_DB_PATH = "output/exit_session.db"

# ── Zone config ───────────────────────────────────────────────────────────────
# Single-zone concept: person is counted as a unique exit when they enter
# the exit zone (appear inside it) and then leave the frame.
# Run:  python manager/draw_zone.py --source 0
# Click to draw the polygon, press S to print config, paste below.
#
# EXIT_ZONES  — dict with exactly one zone (the exit gate / door area).
EXIT_ZONES = {
    "exit": [(1355, 587), (2314, 623), (2314, 1090), (1268, 1076)],
}

# The zone name that triggers a unique-exit count when person leaves frame.
EXIT_ZONE_NAME = "exit"