# Evaluate the two things that make this an inventory system rather than a
# YOLO demo, using ARTSv2 ground truth, on ONE sequence:
#
#   1. Geo-localization error  - metres between the map pin and the true sign GPS
#   2. Deduplication           - did N frames of one physical sign collapse into
#                                exactly one inventory row?
#
# Run AFTER scripts/03_run_pipeline.py, which writes data/pipeline_result_*.json
#
#   python scripts/06_evaluate_geo_and_dedupe.py
#
# The metrics come from pipeline_core.score_rows, the same code
# scripts/07_batch_evaluate.py uses, so one sequence and many are scored alike:
# ids of the same class within CANON_M metres count as one physical post.
#
# 03 places signs with the stored, held-out constants in weights/calibration.json.
# When 03 ran with CALIBRATE=1, the numbers below are in-sample.
#
# Env:
#   RESULT_JSON=data/pipeline_result_demo_sequence.json
#   SEQUENCE_DIR=data/demo_sequence
#   CANON_M=2.0

import os
import sys
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_core

SEQUENCE_DIR = Path(os.environ.get("SEQUENCE_DIR", ROOT / "data" / "demo_sequence"))
seq_name = SEQUENCE_DIR.name
RESULT_JSON = Path(os.environ.get("RESULT_JSON", ROOT / "data" / f"pipeline_result_{seq_name}.json"))
CANON_M = float(os.environ.get("CANON_M", "2.0"))

if not RESULT_JSON.is_file():
    raise SystemExit(f"No pipeline result at {RESULT_JSON}. Run scripts/03_run_pipeline.py first.")

with open(RESULT_JSON, "r", encoding="utf-8") as f:
    tracks = json.load(f)
print(f"Loaded {len(tracks)} rows from {RESULT_JSON.name}")

meta = pipeline_core.load_frames_meta(str(SEQUENCE_DIR))
print(f"Ground-truth frames: {len(meta)}")

score = pipeline_core.score_rows(tracks, meta, tol_m=CANON_M)
pooled = pipeline_core.pool_scores([score])
errors, baseline = score["errors"], score["baseline"]

# ---------------------------------------------------------------- METRIC 1
print("\n=== 1. Geo-localization error (map pin vs true sign GPS) ===")
if not errors:
    print("No rows have BOTH a real GPS fix and matched ground-truth sign GPS.")
    print("Causes: the dataset has no per-sign GPS, or frames_meta.json is missing,")
    print("or no predicted box overlapped a ground-truth box at IoU >= 0.3.")
else:
    print(f"scored rows       : {len(errors)}")
    print(f"mean error        : {sum(errors) / len(errors):.1f} m")
    print(f"median error      : {pipeline_core.percentile(errors, 0.5):.1f} m")
    print(f"90th percentile   : {pipeline_core.percentile(errors, 0.9):.1f} m")
    print(f"worst             : {max(errors):.1f} m")
    if baseline:
        # what the error would be if each sign were pinned at the camera
        b = pipeline_core.percentile(baseline, 0.5)
        a = pipeline_core.percentile(errors, 0.5)
        print()
        print(f"  median error, camera pin (no projection) : {b:.1f} m")
        print(f"  median error, geometric projection       : {a:.1f} m")
        print(f"  improvement                              : {100 * (b - a) / b:+.0f}%")

# ---------------------------------------------------------------- METRIC 2
print("\n=== 2. Deduplication ===")
if not score["posts"]:
    print("No sign ids in the ground truth -- skipping.")
    print(f"(For reference: {len(meta)} frames collapsed into {len(tracks)} inventory rows.)")
else:
    print(f"ground-truth ids / posts : {score['raw_ids']} / {score['posts']}")
    print(f"inventory rows produced  : {score['rows']}")
    print(f"posts found              : {score['found']}  ({100 * pooled['recall']:.1f}% recall)")
    print(f"fragmented (1 post -> many rows) : {score['fragmented']}")
    print(f"merged (many posts -> 1 row)     : {score['merged']}")
    if score["found"]:
        print(f"dedupe score             : {100 * pooled['dedupe']:.1f}%   "
              "(found - fragmented - merged) / found")

# ---------------------------------------------------------------- METRIC 3
print("\n=== 3. Classification agreement on matched rows ===")
agree = Counter()
for t in tracks:
    if t.get("gt_sign_id") is None:
        continue
    gt_name = t.get("gt_class")
    if gt_name is None:
        frame = meta.get(Path(t["frame_path"]).stem, {})
        for obj in frame.get("objects", []):
            if str(obj.get("sign_id")) == str(t["gt_sign_id"]):
                gt_name = obj["name"]
                break
    if gt_name is None:
        continue
    agree["correct" if gt_name.replace(":", "-") == t["cls_name"] else "wrong"] += 1
total = agree["correct"] + agree["wrong"]
if total:
    print(f"matched rows: {total}   correct class: {agree['correct']} "
          f"({100 * agree['correct'] / total:.1f}%)")
else:
    print("No class-comparable matches.")

report_path = ROOT / "data" / f"evaluation_{seq_name}.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump({
        "sequence": seq_name,
        "rows": len(tracks),
        "frames": len(meta),
        "geo_error_m": {
            "n": len(errors),
            "mean": (sum(errors) / len(errors)) if errors else None,
            "median": pipeline_core.percentile(errors, 0.5),
            "p90": pipeline_core.percentile(errors, 0.9),
            "baseline_median": pipeline_core.percentile(baseline, 0.5),
        },
        "dedupe": {k: pooled[k] for k in ("raw_ids", "posts", "rows", "found", "fragmented",
                                          "merged", "recall", "dedupe")},
        "class_agreement": dict(agree),
    }, f, indent=2)
print("\nSaved ->", report_path)
