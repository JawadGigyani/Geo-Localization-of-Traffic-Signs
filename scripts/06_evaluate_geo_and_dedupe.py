# Evaluate the two things that make this an inventory system rather than a
# YOLO demo, using ARTSv2 ground truth:
#
#   1. Geo-localization error  - metres between the map pin and the true sign GPS
#   2. Deduplication accuracy  - did tracking collapse N frames of one physical
#                                sign into exactly one inventory row?
#
# Run AFTER scripts/03_run_pipeline.py, which writes data/pipeline_result_*.json
#
#   python scripts/06_evaluate_geo_and_dedupe.py
#
# Env:
#   RESULT_JSON=data/pipeline_result_demo_sequence.json
#   SEQUENCE_DIR=data/demo_sequence

import os
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]

SEQUENCE_DIR = Path(os.environ.get("SEQUENCE_DIR", ROOT / "data" / "demo_sequence"))
seq_name = SEQUENCE_DIR.name
RESULT_JSON = Path(os.environ.get("RESULT_JSON", ROOT / "data" / f"pipeline_result_{seq_name}.json"))

if not RESULT_JSON.is_file():
    raise SystemExit(f"No pipeline result at {RESULT_JSON}. Run scripts/03_run_pipeline.py first.")

with open(RESULT_JSON, "r", encoding="utf-8") as f:
    tracks = json.load(f)
print(f"Loaded {len(tracks)} tracks from {RESULT_JSON.name}")

meta_path = SEQUENCE_DIR / "frames_meta.json"
meta = {}
if meta_path.is_file():
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f).get("frames", {})
print(f"Ground-truth frames: {len(meta)}")


def haversine_m(lat1, lon1, lat2, lon2):
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dlon / 2) ** 2
    return 6371000.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------- METRIC 1
print("\n=== 1. Geo-localization error (map pin vs true sign GPS) ===")
errors = []
for t in tracks:
    if t.get("lat") is None or t.get("gt_sign_lat") is None:
        continue
    if t.get("gps_source") == "synthetic_demo":
        continue  # never score against invented coordinates
    errors.append((
        haversine_m(t["lat"], t["lon"], t["gt_sign_lat"], t["gt_sign_lon"]),
        t["cls_name"],
    ))

# Baseline: what the error would be if we simply pinned each sign at the
# camera, i.e. skipped projection entirely. The gap is the value added.
baseline = []
for t in tracks:
    if t.get("camera_lat") is None or t.get("gt_sign_lat") is None:
        continue
    baseline.append(haversine_m(t["camera_lat"], t["camera_lon"],
                                t["gt_sign_lat"], t["gt_sign_lon"]))

if not errors:
    print("No tracks have BOTH a real GPS fix and matched ground-truth sign GPS.")
    print("Causes: the dataset has no per-sign GPS, or frames_meta.json is missing,")
    print("or no predicted box overlapped a ground-truth box at IoU >= 0.3.")
else:
    values = [e for e, _ in errors]
    print(f"paired detections : {len(values)}")
    print(f"mean error        : {sum(values)/len(values):.1f} m")
    print(f"median error      : {percentile(values, 0.5):.1f} m")
    print(f"90th percentile   : {percentile(values, 0.9):.1f} m")
    print(f"worst             : {max(values):.1f} m")

    if baseline:
        b = percentile(baseline, 0.5)
        a = percentile(values, 0.5)
        print()
        print("  median error, camera pin (no projection) : "
              f"{b:.1f} m")
        print(f"  median error, geometric projection       : {a:.1f} m")
        print(f"  improvement                              : {100*(b-a)/b:+.0f}%")

# ---------------------------------------------------------------- METRIC 2
print("\n=== 2. Deduplication accuracy (tracking) ===")
gt_sign_ids = set()
for frame in meta.values():
    for obj in frame.get("objects", []):
        if obj.get("sign_id") is not None:
            gt_sign_ids.add(str(obj["sign_id"]))

matched = [t for t in tracks if t.get("gt_sign_id") is not None]
by_gt = defaultdict(set)
by_track = defaultdict(set)
for t in tracks:
    # every ground-truth sign the cluster absorbed, so a merge is observable
    ids = t.get("member_gt_ids") or (
        [str(t["gt_sign_id"])] if t.get("gt_sign_id") else [])
    for gid in ids:
        by_gt[gid].add(t["track_id"])
        by_track[t["track_id"]].add(gid)

if not gt_sign_ids:
    print("No unique sign ids in the ground truth -- skipping.")
    print(f"(For reference: {len(meta)} frames collapsed into {len(tracks)} inventory rows,")
    print(f" a {len(meta)/max(1,len(tracks)):.1f}x reduction.)")
else:
    fragmented = sum(1 for ids in by_gt.values() if len(ids) > 1)
    merged = sum(1 for gts in by_track.values() if len(gts) > 1)
    covered = len(by_gt)
    print(f"ground-truth unique signs : {len(gt_sign_ids)}")
    print(f"inventory rows produced   : {len(tracks)}")
    print(f"GT signs detected         : {covered}  ({100*covered/len(gt_sign_ids):.1f}% recall)")
    print(f"fragmented (1 sign -> many rows) : {fragmented}")
    print(f"merged (many signs -> 1 row)     : {merged}")
    denom = covered + fragmented + merged
    if denom:
        print(f"dedupe accuracy           : {100*(covered-fragmented-merged)/covered:.1f}% "
              "(detected signs mapped to exactly one row)")

# ---------------------------------------------------------------- METRIC 3
print("\n=== 3. Classification agreement on matched detections ===")
agree = Counter()
for t in matched:
    gt_name = None
    stem = Path(t["frame_path"]).stem
    frame = meta.get(stem, {})
    for obj in frame.get("objects", []):
        if str(obj.get("sign_id")) == str(t["gt_sign_id"]):
            gt_name = obj["name"]
            break
    if gt_name is None:
        continue
    agree["correct" if gt_name.replace(":", "-") == t["cls_name"] else "wrong"] += 1
total = agree["correct"] + agree["wrong"]
if total:
    print(f"matched detections: {total}   correct class: {agree['correct']} "
          f"({100*agree['correct']/total:.1f}%)")
else:
    print("No class-comparable matches.")

report_path = ROOT / "data" / f"evaluation_{seq_name}.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump({
        "sequence": seq_name,
        "tracks": len(tracks),
        "frames": len(meta),
        "geo_error_m": {
            "n": len(errors),
            "mean": (sum(e for e, _ in errors) / len(errors)) if errors else None,
            "median": percentile([e for e, _ in errors], 0.5) if errors else None,
            "p90": percentile([e for e, _ in errors], 0.9) if errors else None,
        },
        "dedupe": {
            "gt_unique_signs": len(gt_sign_ids),
            "detected": len(by_gt),
            "fragmented": sum(1 for ids in by_gt.values() if len(ids) > 1),
            "merged": sum(1 for g in by_track.values() if len(g) > 1),
        },
        "class_agreement": dict(agree),
    }, f, indent=2)
print("\nSaved ->", report_path)
