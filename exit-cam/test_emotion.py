import sys
sys.path.insert(0, "/home/linmar/temp/project/test-2/exit-cam")
import config
import cv2
import numpy as np

IMG = "/home/linmar/temp/project/test-2/exit-cam/output/best/day_20260226/hour_11/id_10_conf_0.84.jpg"

img = cv2.imread(IMG)
print(f"Image shape: {img.shape}  dtype: {img.dtype}")
h, w = img.shape[:2]

# ── Haar cascade ──────────────────────────────────────────────────────────
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)
print(f"Cascade loaded: {not cascade.empty()}")

for label, region in [
    ("full image",        img),
    ("upper 50%",         img[:max(h//2,60), :]),
    ("upper 35%",         img[:max(h*35//100,40), :]),
    ("upper 35% @2x",     cv2.resize(img[:max(h*35//100,40),:], None, fx=2, fy=2)),
    ("upper 25% @3x",     cv2.resize(img[:max(h*25//100,30),:], None, fx=3, fy=3)),
]:
    gray  = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(15,15))
    print(f"  Haar [{label:20s}] → {len(faces) if not isinstance(faces, tuple) else 0} face(s)"
          + (f"  best={max(faces, key=lambda r:r[2]*r[3])}" if len(faces) > 0 else ""))

# ── emotiefflib direct ────────────────────────────────────────────────────
print("\n--- emotiefflib tests ---")
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
fer = EmotiEffLibRecognizer(engine="torch", model_name="enet_b0_8_best_afew", device="cpu")

for label, region in [
    ("full image RGB",    img),
    ("top 50% RGB",       img[:max(h//2,60), :]),
    ("top 30% RGB",       img[:max(h*30//100,40), :]),
    ("top 20% RGB @3x",   cv2.resize(img[:max(h//5,30),:], None, fx=3, fy=3)),
]:
    rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    labels, scores = fer.predict_emotions(rgb, logits=False)
    top = float(np.max(scores)) if len(scores) > 0 else 0.0
    print(f"  [{label:22s}] shape={rgb.shape[:2]}  labels={labels}  top={top:.3f}")

print("\nDONE")

# ── Direct test of _analyse_emotion from module ───────────────────────────
print("\n--- _analyse_emotion() direct call ---")
from manager.identity_manager import _analyse_emotion
for path in [
    "/home/linmar/temp/project/test-2/exit-cam/output/best/day_20260226/hour_11/id_10_conf_0.84.jpg",
    "/home/linmar/temp/project/test-2/exit-cam/output/best/day_20260226/hour_11/id_12_conf_0.81.jpg",
    "/home/linmar/temp/project/test-2/exit-cam/output/best/day_20260226/hour_11/id_13_conf_0.84.jpg",
]:
    result = _analyse_emotion(path)
    print(f"  {path.split('/')[-1]:35s} → {result}")
