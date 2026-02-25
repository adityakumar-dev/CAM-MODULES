"""
debug_emotion.py
----------------
Run from project root to diagnose why emotion detection returns NULL.
    python debug_emotion.py
"""
import os
import glob
import cv2
import numpy as np

# ── Find a test image ─────────────────────────────────────────────────────────
images = glob.glob("output/best/**/*.jpg", recursive=True)
if not images:
    print("[ERROR] No archived images found in output/best/")
    exit(1)

# Pick the one with highest conf in filename if possible, else first found
images.sort(reverse=True)
test_img_path = images[0]
print(f"Testing on: {test_img_path}\n")

# ── Load image ────────────────────────────────────────────────────────────────
img = cv2.imread(test_img_path)
if img is None:
    print("[ERROR] cv2.imread returned None — file missing or corrupt")
    exit(1)

h, w = img.shape[:2]
print(f"Full crop size : {w}x{h} px")

head_bottom = max(int(h * 0.40), min(h, 60))
head_img    = img[:head_bottom, :]
print(f"Head crop size : {w}x{head_bottom} px")

# Save head crop so you can visually inspect it
cv2.imwrite("debug_head_crop.jpg", head_img)
print("Head crop saved → debug_head_crop.jpg  (open this to check it looks like a face)\n")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading EmotiEffLibRecognizer...")
try:
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer
    fer = EmotiEffLibRecognizer(model_name="enet_b0_8_best_afew", device="cpu")
    print("Model loaded OK\n")
except Exception as e:
    print(f"[ERROR] Model load failed: {e}")
    exit(1)

# ── Test 1: full crop ─────────────────────────────────────────────────────────
print("── Test 1: full body crop ───────────────────────────────────")
try:
    label, scores = fer.predict_emotions(img, logits=False)
    label = label[0] if isinstance(label, list) else label
    print(f"  Label : {label}")
    print(f"  Score : {float(np.max(scores)):.3f}")
except Exception as e:
    print(f"  [EXCEPTION] {type(e).__name__}: {e}")

# ── Test 2: head crop (top 40%) ───────────────────────────────────────────────
print("\n── Test 2: head crop (top 40%) ─────────────────────────────")
try:
    label, scores = fer.predict_emotions(head_img, logits=False)
    label = label[0] if isinstance(label, list) else label
    print(f"  Label : {label}")
    print(f"  Score : {float(np.max(scores)):.3f}")
except Exception as e:
    print(f"  [EXCEPTION] {type(e).__name__}: {e}")

# ── Test 3: try all archived images ──────────────────────────────────────────
print("\n── Test 3: scan all archived images ────────────────────────")
detected = 0
for path in images[:10]:   # cap at 10
    try:
        img_i = cv2.imread(path)
        if img_i is None:
            continue
        hi = img_i.shape[0]
        head = img_i[:max(int(hi * 0.40), min(hi, 60)), :]
        lbl, sc = fer.predict_emotions(head, logits=False)
        lbl = lbl[0] if isinstance(lbl, list) else lbl
        score = float(np.max(sc))
        status = "✓" if score >= 0.40 else "✗ low score"
        print(f"  {os.path.basename(path):40s}  {lbl:12s}  {score:.3f}  {status}")
        if score >= 0.40:
            detected += 1
    except Exception as e:
        print(f"  {os.path.basename(path):40s}  [EXCEPTION] {type(e).__name__}: {e}")

print(f"\n{detected}/{min(len(images), 10)} images had emotion score >= 0.40")
print("\nIf all scores are low or all exceptions:")
print("  → The crops likely don't contain a visible face (back of head, blurry, too far)")
print("  → Or the video test source has people too small / far from camera")