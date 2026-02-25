"""
config.py
=========
Central configuration for the emotion zone counter pipeline.
Edit values here; no other file needs changing for tuning.
"""
import os

# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH           = os.getenv("YOLO_MODEL_PATH", "yolo11s.pt")
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF", "0.20"))
YOLO_IOU_THRESHOLD        = float(os.getenv("YOLO_IOU",  "0.45"))
YOLO_DEVICE               = os.getenv("YOLO_DEVICE", "cpu")   # "cpu" | "cuda" | "mps"

# ---------------------------------------------------------------------------
# Emotion thresholds
# ---------------------------------------------------------------------------
EMOTION_HAPPY_THRESHOLD      = 0.40
EMOTION_VERY_HAPPY_THRESHOLD = 0.75
EMOTION_SAD_THRESHOLD        = 0.40
EMOTIEFF_MODEL               = os.getenv("EMOTIEFF_MODEL", "enet_b0_8_best_afew")

# ---------------------------------------------------------------------------
# Re-ID buffer (within-session only — lost track re-association)
# ---------------------------------------------------------------------------
# How long to keep a lost track before archiving it.
# Keeps re-ID window in sync with bytetrack.yaml track_buffer.
# At 30 fps, 90 frames = 3 s.  Person occluded up to this long → same visit.
REID_BUFFER_FRAMES   = int(os.getenv("REID_BUFFER", "90"))
REID_BUFFER_SECONDS  = float(os.getenv("REID_BUFFER_SECONDS", "3.0"))
REID_SIM_THRESHOLD   = float(os.getenv("REID_SIM",  "0.75"))
REID_SAME_VISIT_THRESHOLD = float(os.getenv("REID_SAME_VISIT", "0.50"))
REID_EMBEDDING_ALPHA = float(os.getenv("REID_ALPHA", "0.3"))

# ---------------------------------------------------------------------------
# Trail
# ---------------------------------------------------------------------------
TRAIL_MAX_LEN = int(os.getenv("TRAIL_LEN", "60"))

# ---------------------------------------------------------------------------
# Zones
# Define as many zones as needed.  Use draw_zone.py to get coordinates.
# The zone name is stored in the DB so you can filter captures by zone.
#
# CAPTURE_ZONES  — crops are saved only while the person is in one of these.
#                  Leave empty to capture everywhere.
# ---------------------------------------------------------------------------
ZONES: dict = {
    # Paste output from draw_zone.py here, e.g.:    
    "entrance": [(1355, 587), (2314, 623), (2314, 1090), (1268, 1076)],
    "lobby": [(673, 1208), (2303, 1171), (2326, 1497), (695, 1505)],

}

# Optional: restrict best-frame capture to these zone names only.
# If empty, captures happen regardless of zone.
CAPTURE_ZONES: list = []   # e.g. ["entrance"]

# Minimum consecutive frames a track must be inside a zone before
# that zone-entry is counted (prevents single-frame blips).
ZONE_DWELL_FRAMES = int(os.getenv("ZONE_DWELL", "3"))

# ---------------------------------------------------------------------------
# Misc helper constants used by identity_manager
# ---------------------------------------------------------------------------
HIST_SKIP_PX = int(os.getenv("HIST_SKIP_PX", "4"))
MIN_CROP_PX  = int(os.getenv("MIN_CROP_PX",  "20"))

# ---------------------------------------------------------------------------
# Source / display
# ---------------------------------------------------------------------------
SOURCE_FPS  = int(os.getenv("SOURCE_FPS", "30"))
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------
DEBUG = os.getenv("DEBUG", "false").lower() == "true"