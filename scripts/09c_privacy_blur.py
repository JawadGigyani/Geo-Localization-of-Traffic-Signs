# Privacy scrub: blur license plates and faces in the saved crops/frames.
#
#   python scripts/09c_privacy_blur.py
#
# IMPORTANT: a COCO-pretrained YOLO has no 'license plate' or 'face' class. It
# only knows 'person'. Blurring 'person' is not plate anonymisation, so the
# script reports exactly which categories it covered.
#
# To do this properly, point WEIGHTS_PRIVACY at a plate/face detector, e.g. a
# YOLO model fine-tuned on an open plate dataset, then set PRIVACY_CLASSES to
# that model's class ids (see the printed class list on first run).
#
# Env:
#   INPUT_DIR=data/pipeline_crops
#   OUTPUT_DIR=data/pipeline_crops_blurred
#   WEIGHTS_PRIVACY=yolo11n.pt
#   PRIVACY_CLASSES=0,1          comma-separated class ids to blur (optional)

import os
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = Path(os.environ.get("INPUT_DIR", ROOT / "data" / "pipeline_crops"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT / "data" / "pipeline_crops_blurred"))
WEIGHTS = os.environ.get("WEIGHTS_PRIVACY", "yolo11n.pt")
CONF = float(os.environ.get("PRIVACY_CONF", "0.35"))

if not INPUT_DIR.is_dir():
    raise SystemExit(f"INPUT_DIR not found: {INPUT_DIR}. Run scripts/03_run_pipeline.py first.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model = YOLO(WEIGHTS)
class_names = model.names
print("model:", WEIGHTS)
print("classes:", class_names)

# Work out what we can actually anonymise with the loaded model.
target_ids = set()
env_classes = os.environ.get("PRIVACY_CLASSES", "").strip()
if env_classes:
    target_ids = {int(c) for c in env_classes.split(",") if c.strip()}
    print("blurring class ids from PRIVACY_CLASSES:", sorted(target_ids))
else:
    wanted = ("plate", "licence", "license", "face", "head", "person")
    for cid, name in class_names.items():
        if any(w in str(name).lower() for w in wanted):
            target_ids.add(int(cid))
    print("auto-selected class ids:", {i: class_names[i] for i in sorted(target_ids)})

covers_plates = any("plate" in str(class_names[i]).lower() for i in target_ids)
covers_faces = any(w in str(class_names[i]).lower() for i in target_ids for w in ("face", "head"))

if not target_ids:
    raise SystemExit("No privacy-relevant classes in this model. Set PRIVACY_CLASSES explicitly.")
if not covers_plates or not covers_faces:
    print("\n" + "!" * 68)
    print("! LIMITED ANONYMISATION")
    print(f"!   license plates covered: {covers_plates}")
    print(f"!   faces covered:          {covers_faces}")
    print("! This model cannot detect the missing categories, so the output is not")
    print("! fully anonymised. Use a plate/face detector for real use.")
    print("!" * 68 + "\n")

paths = sorted(p for p in INPUT_DIR.iterdir()
               if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
print("images:", len(paths))

blurred_regions = 0
images_touched = 0

for path in paths:
    img = cv2.imread(str(path))
    if img is None:
        continue
    results = model.predict(source=img, conf=CONF, verbose=False)
    hits = 0
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            if int(box.cls.item()) not in target_ids:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            # kernel scaled to the region, and always odd
            k = max(9, (min(roi.shape[0], roi.shape[1]) // 3) | 1)
            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
            hits += 1
    cv2.imwrite(str(OUTPUT_DIR / path.name), img)
    blurred_regions += hits
    images_touched += 1 if hits else 0

report = {
    "model": WEIGHTS,
    "target_classes": {int(i): str(class_names[i]) for i in sorted(target_ids)},
    "covers_license_plates": covers_plates,
    "covers_faces": covers_faces,
    "images_processed": len(paths),
    "images_with_redactions": images_touched,
    "regions_blurred": blurred_regions,
}
with open(OUTPUT_DIR / "privacy_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
print("Done ->", OUTPUT_DIR)
