import cv2
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.vision.ocr_engine import OCREngine

img_path = ROOT / "debug/recordings/1785429486181/ocr_debug/quantity_strip_0010_merged.png"
img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
print(f"Image shape: {img.shape}")

vis = img.copy()
h, w = img.shape[:2]

ocr = OCREngine()
det_results = ocr.detect_text(img)
print(f"Detected {len(det_results)} boxes")

for i, (bbox, det_conf) in enumerate(det_results):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x1, x2 = max(0, int(min(xs))), min(w, int(max(xs)))
    y1, y2 = max(0, int(min(ys))), min(h, int(max(ys)))
    if x2 <= x1 or y2 <= y1:
        continue
    crop = img[y1:y2, x1:x2]
    ocr_results = ocr.recognize(crop, min_confidence=0.3)
    texts = [f"{text}({conf:.2f})" for _, (text, conf) in ocr_results]
    raw_text = ",".join(texts) if texts else "none"

    # Check if has digits
    has_digit = any(any(c.isdigit() for c in text) for _, (text, _) in ocr_results)
    color = (0, 255, 0) if has_digit else (0, 0, 255)
    pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], True, color, 2)

    label = f"{raw_text}"
    cv2.putText(vis, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    print(f"box {i}: x=[{x1},{x2}] det_conf={det_conf:.2f} ocr={raw_text}")

out_path = ROOT / "debug/quantity_strip_0010_merged_visualization.png"
cv2.imencode(".png", vis)[1].tofile(str(out_path))
print(f"Saved visualization to {out_path}")
