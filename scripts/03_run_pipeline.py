# Run the inventory pipeline over a folder of frames.
#
#   python scripts/03_run_pipeline.py
#
# Env:
#   SEQUENCE_DIR=data/demo_sequence      folder of frames (+ optional frames_meta.json)
#   WEIGHTS_PATH=weights/best.pt
#   SKIP_UPLOAD=1                        dry run, no Supabase writes
#   ALLOW_SYNTHETIC_GPS=1                demo only: place GPS-less signs on a fake
#                                        Burlington point, tagged as synthetic
#   ASSOCIATION=tracker                  run the BoT-SORT baseline instead
#   TRACKER=bytetrack.yaml               ...or the ByteTrack baseline
#   HFOV_DEG / SIZE_K / RADIUS_M         override weights/calibration.json
#   CALIBRATE=1                          refit the camera constants on THIS folder's
#                                        ground truth (in-sample; off by default)
#
# The camera constants come from weights/calibration.json, which
# scripts/07_batch_evaluate.py fits on held-out calibration sequences, so the
# pipeline needs no ground truth to run on new photos.
#
# All real work lives in pipeline_core.py, which the API imports too.

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import pipeline_core
from ultralytics import YOLO

SEQUENCE_DIR = os.environ.get("SEQUENCE_DIR", str(ROOT / "data" / "demo_sequence"))
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", str(ROOT / "weights" / "best.pt"))
SKIP_UPLOAD = os.environ.get("SKIP_UPLOAD", "0") == "1"
ALLOW_SYNTHETIC_GPS = os.environ.get("ALLOW_SYNTHETIC_GPS", "0") == "1"
CONF = float(os.environ.get("CONF", "0.25"))
TRACKER = os.environ.get("TRACKER", "botsort.yaml")

# "geometric" projects each detection onto the map and clusters there -- the
# only thing that works at 1 Hz. "tracker" is the BoT-SORT baseline.
ASSOCIATION = os.environ.get("ASSOCIATION", "geometric").strip().lower()
CALIBRATION_PATH = os.environ.get("CALIBRATION_PATH", str(ROOT / "weights" / "calibration.json"))
HFOV, SIZE_K, RADIUS_M, GEOMETRY_SOURCE = pipeline_core.geometry_settings(CALIBRATION_PATH)
CALIBRATE = os.environ.get("CALIBRATE", "0") == "1"

# Optional second stage: re-reads each crop to pick the right variant within the
# family the detector chose. Set CLS_WEIGHTS=none to disable.
CLS_WEIGHTS = os.environ.get("CLS_WEIGHTS", str(ROOT / "weights" / "crop_classifier.pt"))
CLS_MIN_CONF = float(os.environ.get("CLS_MIN_CONF", "0.8"))

print("SEQUENCE_DIR =", SEQUENCE_DIR)
print("WEIGHTS_PATH =", WEIGHTS_PATH)

if not os.path.isdir(SEQUENCE_DIR):
    raise SystemExit(f"SEQUENCE_DIR not found: {SEQUENCE_DIR}")

frames = pipeline_core.list_frames(SEQUENCE_DIR)
print("Frames:", len(frames))
if not frames:
    raise SystemExit("No images in SEQUENCE_DIR")

meta = pipeline_core.load_frames_meta(SEQUENCE_DIR)
print("frames_meta.json:", "loaded" if meta else "absent (GPS will come from XML / .gps.txt only)")

if not os.path.isfile(WEIGHTS_PATH):
    print("WARNING: no fine-tuned weights at", WEIGHTS_PATH)
    print("         falling back to COCO yolo11n.pt -- it does NOT know sign classes.")
    WEIGHTS_PATH = "yolo11n.pt"

model = YOLO(WEIGHTS_PATH)

IMGSZ = int(os.environ.get("IMGSZ", "0")) or pipeline_core.training_imgsz(model)
print("IMGSZ        =", IMGSZ, "(must match training resolution)")
print("ASSOCIATION  =", ASSOCIATION)

if ASSOCIATION == "tracker":
    # Baseline. Fails on 1 Hz imagery: the camera moves ~8 m between frames, so
    # IoU-based association never confirms a track. Kept for comparison.
    # A second copy of the detector counts every box, tracked or not.
    stats = {}
    tracks = pipeline_core.run_tracking(
        model, frames, tracker=TRACKER, conf=CONF, meta=meta, imgsz=IMGSZ,
        stats=stats, count_model=YOLO(WEIGHTS_PATH),
    )
    print(f"Tracker {TRACKER}: {stats['with_id']} of {stats['detections']} "
          f"detections were given an identity")
else:
    cls_model, families = None, None
    if CLS_WEIGHTS.lower() != "none" and os.path.isfile(CLS_WEIGHTS):
        cls_model = YOLO(CLS_WEIGHTS)
        families = pipeline_core.build_families(list(model.names.values()))
        print(f"Second stage: {os.path.basename(CLS_WEIGHTS)} over "
              f"{len(families)} confusable families")
    else:
        print("Second stage: disabled (no crop_classifier.pt)")

    dets = pipeline_core.detect_all(model, frames, conf=CONF, imgsz=IMGSZ, meta=meta,
                                    cls_model=cls_model, families=families,
                                    cls_min_conf=CLS_MIN_CONF)
    print("Raw detections:", len(dets))
    if cls_model is not None:
        changed = sum(1 for d in dets if d.get("cls_refined"))
        print(f"  stage 2 changed {changed}/{len(dets)} labels "
              f"({100*changed/max(1,len(dets)):.1f}%)")

    hfov, size_k = HFOV, SIZE_K
    print(f"Geometry: hfov={hfov:.0f} deg  size_k={size_k:.3f} m  radius={RADIUS_M:.0f} m  "
          f"(from {GEOMETRY_SOURCE})")
    if CALIBRATE:
        # In-sample: fits on the same folder it then scores. Useful to compare
        # cameras, never for reporting results.
        fit_hfov, fit_k, n = pipeline_core.calibrate_geometry(dets)
        if fit_hfov:
            hfov, size_k = fit_hfov, fit_k
            print(f"CALIBRATE=1: refitted on this folder's {n} ground-truth matches: "
                  f"hfov={hfov:.0f} deg  size_k={size_k:.3f} m (in-sample)")
        else:
            print(f"CALIBRATE=1 but only {n} ground-truth matches; keeping the stored constants")

    tracks = pipeline_core.associate_geometric(
        dets, hfov_deg=hfov, size_k=size_k, radius_m=RADIUS_M
    )
    multi = sum(1 for t in tracks.values() if t.get("observations", 1) > 1)
    print(f"Clustered into {len(tracks)} physical signs "
          f"({multi} seen in more than one frame)")

print("Unique tracks kept:", len(tracks))

sources = {}
for info in tracks.values():
    sources[info["gps_source"]] = sources.get(info["gps_source"], 0) + 1
print("GPS sources:", sources)

if ALLOW_SYNTHETIC_GPS:
    n = pipeline_core.apply_synthetic_gps(tracks)
    if n:
        print(f"NOTE: {n} track(s) given SYNTHETIC demo coordinates (gps_source=synthetic_demo)")

seq_name = os.path.basename(os.path.abspath(SEQUENCE_DIR))
out_crops = ROOT / "data" / "pipeline_crops"
pipeline_core.save_crops(tracks, str(out_crops), seq_name)

# Always write a local record, so results exist even without Supabase.
report = []
for tid, info in tracks.items():
    report.append({k: v for k, v in info.items() if k != "crop"})
report_path = ROOT / "data" / f"pipeline_result_{seq_name}.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("Local result ->", report_path)

if SKIP_UPLOAD:
    print("SKIP_UPLOAD=1 -- not writing to Supabase.")
    raise SystemExit(0)

url = os.environ.get("SUPABASE_URL", "")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
if not url or not key or "YOUR_PROJECT" in url:
    print("Supabase not configured in .env -- crops saved locally only.")
    raise SystemExit(0)

from supabase import create_client

supabase = create_client(url, key)
inserted = pipeline_core.upload_and_insert(supabase, tracks, seq_name)
print("Done. Inserted rows:", len(inserted))
