# Check everything the pipeline needs BEFORE running it.
#
#   python scripts/00_preflight.py
#
# Verifies: dependencies, trained weights, demo frames + GPS metadata,
# .env, and live Supabase connectivity (table, functions, storage buckets).

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

problems = []
warnings = []


def ok(msg):
    print(f"  [ok]   {msg}")


def bad(msg, fix=None):
    print(f"  [FAIL] {msg}")
    problems.append((msg, fix))


def warn(msg, fix=None):
    print(f"  [warn] {msg}")
    warnings.append((msg, fix))


print("=" * 66)
print("1. Python dependencies")
print("=" * 66)
try:
    import torch
    import cv2
    import ultralytics
    from ultralytics import YOLO
    ok(f"torch {torch.__version__}, ultralytics {ultralytics.__version__}, cv2 {cv2.__version__}")
except ImportError as e:
    bad(f"missing dependency: {e}", "pip install -r requirements.txt")
    print("\nCannot continue without dependencies.")
    raise SystemExit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    ok("python-dotenv")
except ImportError:
    bad("python-dotenv missing", "pip install -r requirements.txt")

print()
print("=" * 66)
print("2. Trained weights")
print("=" * 66)
weights = ROOT / "weights" / "best.pt"
if not weights.is_file():
    bad(f"no model at {weights}",
        "download best.pt from Drive: traffic-sign-inventory/export/best.pt")
else:
    size_mb = weights.stat().st_size / 1e6
    if size_mb < 1:
        bad(f"best.pt is only {size_mb:.1f} MB - truncated download?", "re-download it")
    else:
        try:
            model = YOLO(str(weights))
            names = model.names
            ok(f"best.pt loads: {len(names)} classes, {size_mb:.1f} MB")
            print(f"         first classes: {[names[i] for i in sorted(names)[:6]]}")
            if len(names) == 80:
                bad("model has 80 classes - this is COCO yolo11n, not the fine-tuned model",
                    "download the real best.pt from Drive")
        except Exception as e:
            bad(f"best.pt will not load: {e}", "re-download it")

classifier = ROOT / "weights" / "crop_classifier.pt"
if not classifier.is_file():
    warn(f"no crop classifier at {classifier} - the second stage will be skipped",
         "restore weights/crop_classifier.pt from git")
else:
    try:
        cls_names = YOLO(str(classifier)).names
        ok(f"crop_classifier.pt loads: {len(cls_names)} classes")
        if len(cls_names) != 34:
            warn(f"expected 34 look-alike classes, found {len(cls_names)}")
    except Exception as e:
        bad(f"crop_classifier.pt will not load: {e}", "restore it from git")

import pipeline_core

calibration = ROOT / "weights" / "calibration.json"
if calibration.is_file():
    cal = pipeline_core.load_calibration(calibration)
    ok(f"calibration.json: hfov={cal['hfov_deg']:.0f} deg  size_k={cal['size_k_m']:.3f} m  "
       f"radius={cal['radius_m']:.0f} m")
else:
    warn("no weights/calibration.json - using the built-in defaults (80 deg, 0.88 m, 60 m)",
         "restore it from git, or run scripts/07_batch_evaluate.py to refit it")

print()
print("=" * 66)
print("3. Demo sequence")
print("=" * 66)

seq_dir = os.environ.get("SEQUENCE_DIR", str(ROOT / "data" / "demo_sequence"))
if not os.path.isdir(seq_dir):
    bad(f"no sequence directory at {seq_dir}",
        "run scripts/02_convert_arts_to_yolo.py then scripts/05_export_sequences.py "
        "(or unzip the notebook's demo_sequence.zip there)")
else:
    frames = pipeline_core.list_frames(seq_dir)
    meta = pipeline_core.load_frames_meta(seq_dir)

    synthetic = [f for f in frames if os.path.basename(f).startswith("frame_")]
    if synthetic:
        bad(f"{len(synthetic)} synthetic placeholder frames still present",
            f'delete them:  rm "{seq_dir}"/frame_*')

    real = [f for f in frames if f not in synthetic]
    if not real:
        bad("no real frames found",
            "run scripts/05_export_sequences.py, or unzip demo_sequence.zip into data/demo_sequence/")
    else:
        ok(f"{len(real)} real frames")

    if not meta:
        bad("frames_meta.json missing - no GPS, no ground truth, no metrics",
            "it should be inside demo_sequence.zip")
    else:
        ok(f"frames_meta.json: {len(meta)} entries")
        with_gps = sum(1 for m in meta.values() if m.get("camera_lat") is not None)
        with_sign = sum(1 for m in meta.values()
                        if any(o.get("sign_lat") for o in m.get("objects", [])))
        with_id = sum(1 for m in meta.values()
                      if any(o.get("sign_id") for o in m.get("objects", [])))
        print(f"         camera GPS: {with_gps}   sign GPS: {with_sign}   sign ids: {with_id}")
        if with_gps == 0:
            bad("no camera GPS in metadata - the map will be empty")
        if with_sign == 0:
            warn("no ground-truth sign GPS - geo-localization error cannot be measured")
        if with_id == 0:
            warn("no physical sign ids - dedupe accuracy cannot be measured")

        ids = set()
        for m in meta.values():
            for o in m.get("objects", []):
                if o.get("sign_id"):
                    ids.add(o["sign_id"])
        if ids:
            print(f"         {len(ids)} distinct physical signs in this sequence")
            print(f"         (the pipeline should produce roughly this many rows)")

print()
print("=" * 66)
print("4. Environment file")
print("=" * 66)
if not (ROOT / ".env").is_file():
    bad(".env not found", "copy .env.example to .env and fill in Supabase details")
else:
    ok(".env present")

url = os.environ.get("SUPABASE_URL", "")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
configured = bool(url and key and "YOUR_PROJECT" not in url and "your_service" not in key)
if not configured:
    warn("Supabase not configured - a dry run still works with SKIP_UPLOAD=1",
         "fill SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")

print()
print("=" * 66)
print("5. Supabase (live check)")
print("=" * 66)
if not configured:
    print("  [skip] not configured")
else:
    try:
        from supabase import create_client
        supabase = create_client(url, key)
        ok(f"client created for {url}")

        try:
            res = supabase.table("traffic_signs").select("id").limit(1).execute()
            ok(f"table traffic_signs exists ({len(res.data or [])} row sampled)")
        except Exception as e:
            bad(f"table traffic_signs unreachable: {e}",
                "run scripts/04_seed_supabase.sql in the Supabase SQL editor")

        try:
            supabase.table("traffic_signs").select(
                "camera_lat,camera_lon,observations,gt_sign_id"
            ).limit(1).execute()
            ok("schema is current (camera position + observation count present)")
        except Exception:
            bad("schema is out of date - missing camera_lat / observations",
                "re-run the UPDATED scripts/04_seed_supabase.sql in the SQL editor")

        try:
            res = supabase.rpc("signs_in_bbox", {
                "min_lng": -74.0, "min_lat": 42.0, "max_lng": -71.0, "max_lat": 45.5,
            }).execute()
            ok(f"function signs_in_bbox works ({len(res.data or [])} rows in Vermont)")
            if res.data and "error_m" not in res.data[0]:
                bad("signs_in_bbox is the old version (no error_m column)",
                    "re-run the UPDATED scripts/04_seed_supabase.sql")
        except Exception as e:
            bad(f"signs_in_bbox failed: {e}", "re-run scripts/04_seed_supabase.sql")

        try:
            buckets = {b.name if hasattr(b, "name") else b["name"]
                       for b in supabase.storage.list_buckets()}
            for needed in ("sign-crops", "sign-frames"):
                if needed in buckets:
                    ok(f"bucket {needed}")
                else:
                    bad(f"bucket {needed} missing",
                        f"Supabase dashboard -> Storage -> New bucket '{needed}' (public)")
        except Exception as e:
            warn(f"could not list buckets: {e}", "check them manually in the dashboard")

    except Exception as e:
        bad(f"Supabase connection failed: {e}", "check SUPABASE_URL and the service_role key")

print()
print("=" * 66)
if problems:
    print(f"NOT READY - {len(problems)} problem(s)")
    print("=" * 66)
    for msg, fix in problems:
        print(f"  * {msg}")
        if fix:
            print(f"      fix: {fix}")
else:
    print("READY")
    print("=" * 66)
    print("  next:  python scripts/03_run_pipeline.py")
    print("  or dry-run first, without touching Supabase:")
    print("         SKIP_UPLOAD=1 python scripts/03_run_pipeline.py")

if warnings:
    print()
    print(f"{len(warnings)} warning(s):")
    for msg, fix in warnings:
        print(f"  * {msg}")
        if fix:
            print(f"      fix: {fix}")

raise SystemExit(1 if problems else 0)
