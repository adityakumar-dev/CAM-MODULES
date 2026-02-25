"""Manager utilities for safe_test_3 (entry-cam).

Modules exposed:
  cv2_manager      – OpenCV run-loop wrapper
  yolo_detect      – YOLO + ByteTrack detector
  identity_manager – ReID + zone crossing + entry counting
  zone_manager     – Polygon zone tracker
  entry_db         – Persistent session gallery (deduplication across restarts)
  db_helper        – SQLite archive connection (person metadata)
  draw_zones       – Interactive zone coordinate picker
"""