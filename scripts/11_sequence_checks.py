# Baselines and sanity checks on one road sequence (default: the demo sequence).
#
#   python scripts/11_sequence_checks.py
#
# Measures, and writes to results/sequence_checks.json:
#   1. spacing between consecutive photos (why video trackers fail on this data)
#   2. video-tracker baselines: how many detections BoT-SORT and ByteTrack give an
#      identity, against position clustering
#   3. the camera constants fitted on this one sequence (in-sample) next to the
#      stored held-out fit in weights/calibration.json
#   4. position error split into the part along the line of sight (range) and the
#      part across it (direction), using the stored constants
#   5. which ground-truth ids collapse into one post (same class within CANON_M)
#      and whether duplicate ids always appear in the same photos
#
# Runs on a CPU in a few minutes. Nothing is written to the database.
#
# Env:
#   SEQUENCE_DIR=data/demo_sequence   WEIGHTS_PATH   CLS_WEIGHTS   CLS_MIN_CONF (0.8)
#   CONF (0.25)   CANON_M (2.0)   CALIBRATION_PATH   RESULTS_DIR=results

import os
import sys
import json
import math
import platform
import statistics
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import pipeline_core
from ultralytics import YOLO

SEQUENCE_DIR = Path(os.environ.get("SEQUENCE_DIR", ROOT / "data" / "demo_sequence"))
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", str(ROOT / "weights" / "best.pt"))
CLS_WEIGHTS = os.environ.get("CLS_WEIGHTS", str(ROOT / "weights" / "crop_classifier.pt"))
CLS_MIN_CONF = float(os.environ.get("CLS_MIN_CONF", "0.8"))
CONF = float(os.environ.get("CONF", "0.25"))
CANON_M = float(os.environ.get("CANON_M", "2.0"))
CALIBRATION_PATH = os.environ.get("CALIBRATION_PATH", str(ROOT / "weights" / "calibration.json"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))

frames = pipeline_core.list_frames(str(SEQUENCE_DIR))
meta = pipeline_core.load_frames_meta(str(SEQUENCE_DIR))
if not frames or not meta:
    raise SystemExit(f"No photos with frames_meta.json in {SEQUENCE_DIR}")
stems = [Path(f).stem for f in frames]


def r(x, digits=2):
    return None if x is None else round(float(x), digits)


report = {"sequence": SEQUENCE_DIR.name, "frames": len(frames)}
print(f"sequence: {SEQUENCE_DIR.name} ({len(frames)} photos)")

# ------------------------------------------------------------ 1. photo spacing
gaps = []
for a, b in zip(stems, stems[1:]):
    ma, mb = meta.get(a, {}), meta.get(b, {})
    if None not in (ma.get("camera_lat"), ma.get("camera_lon"),
                    mb.get("camera_lat"), mb.get("camera_lon")):
        gaps.append(pipeline_core.haversine_m(ma["camera_lat"], ma["camera_lon"],
                                              mb["camera_lat"], mb["camera_lon"]))
report["photo_spacing_m"] = {
    "gaps": len(gaps), "median": r(pipeline_core.percentile(gaps, 0.5)),
    "p10": r(pipeline_core.percentile(gaps, 0.1)), "p90": r(pipeline_core.percentile(gaps, 0.9)),
}
print(f"1. photo spacing: median {report['photo_spacing_m']['median']} m over {len(gaps)} gaps")

# ------------------------------------------------------------ detection
model = YOLO(WEIGHTS_PATH)
imgsz = pipeline_core.training_imgsz(model)
cls_model, families = None, None
if CLS_WEIGHTS.lower() != "none" and os.path.isfile(CLS_WEIGHTS):
    cls_model = YOLO(CLS_WEIGHTS)
    families = pipeline_core.build_families(list(model.names.values()))
dets = pipeline_core.detect_all(model, frames, conf=CONF, imgsz=imgsz, meta=meta,
                                keep_crops=False, cls_model=cls_model, families=families,
                                cls_min_conf=CLS_MIN_CONF)

# ------------------------------------------------------------ 3. in-sample fit
hfov, size_k, radius, source = pipeline_core.geometry_settings(CALIBRATION_PATH)
fit_hfov, fit_k, n_fit = pipeline_core.calibrate_geometry(dets)
report["camera_fit"] = {
    "stored_held_out": {"hfov_deg": hfov, "size_k_m": r(size_k, 4), "radius_m": radius},
    "this_sequence_in_sample": {"hfov_deg": fit_hfov, "size_k_m": r(fit_k, 4), "matches": n_fit},
}
print(f"3. camera fit: stored {hfov:.0f} deg / {size_k:.3f} m; "
      f"this sequence alone {fit_hfov} deg / {r(fit_k, 3)} m ({n_fit} matches)")

# ------------------------------------------------------------ 4. error components
rows = pipeline_core.associate_geometric(dets, hfov_deg=hfov, size_k=size_k, radius_m=radius)
total, along, across = [], [], []
for row in rows.values():
    if row.get("gt_sign_lat") is None or row.get("camera_lat") is None:
        continue
    bearing = math.radians(pipeline_core.bearing_deg(row["camera_lat"], row["camera_lon"],
                                                     row["lat"], row["lon"]))
    north = (row["gt_sign_lat"] - row["lat"]) * 111320.0
    east = (row["gt_sign_lon"] - row["lon"]) * 111320.0 * math.cos(math.radians(row["lat"]))
    total.append(math.hypot(north, east))
    along.append(abs(north * math.cos(bearing) + east * math.sin(bearing)))
    across.append(abs(-north * math.sin(bearing) + east * math.cos(bearing)))
report["position_error_m"] = {
    "scored_rows": len(total),
    "median": r(statistics.median(total)) if total else None,
    "median_along_line_of_sight": r(statistics.median(along)) if along else None,
    "median_across_line_of_sight": r(statistics.median(across)) if across else None,
    "mean_along_line_of_sight": r(statistics.mean(along)) if along else None,
    "mean_across_line_of_sight": r(statistics.mean(across)) if across else None,
}
pe = report["position_error_m"]
print(f"4. position error over {pe['scored_rows']} rows: median {pe['median']} m "
      f"(along the line of sight {pe['median_along_line_of_sight']} m, "
      f"across {pe['median_across_line_of_sight']} m)")

# ------------------------------------------------------------ 2. tracker baselines
count_model = YOLO(WEIGHTS_PATH)
baselines = {}
for tracker in ("botsort.yaml", "bytetrack.yaml"):
    stats = {}
    # a fresh model per tracker, so no tracker state carries over between runs
    tracks = pipeline_core.run_tracking(YOLO(WEIGHTS_PATH), frames, tracker=tracker,
                                        conf=CONF, meta=meta, imgsz=imgsz,
                                        stats=stats, count_model=count_model)
    baselines[tracker.split(".")[0]] = {"detections": stats["detections"],
                                        "with_identity": stats["with_id"],
                                        "tracks": len(tracks)}
projected = sum(1 for d in dets if d.get("projected"))
baselines["position_clustering"] = {"detections": len(dets), "with_identity": projected,
                                    "tracks": len(rows)}
report["association_baselines"] = baselines
for name, b in baselines.items():
    print(f"2. {name:20s} {b['with_identity']} of {b['detections']} detections given an identity")

# ------------------------------------------------------------ 5. id merges
signs, photos_of = {}, defaultdict(set)
for stem, m in meta.items():
    for o in m.get("objects", []):
        gid = o.get("sign_id")
        if gid and o.get("sign_lat") is not None:
            signs[str(gid)] = (o["name"], o["sign_lat"], o["sign_lon"])
            photos_of[str(gid)].add(stem)
canon = pipeline_core.canonical_gt_map(meta, CANON_M)
merged = sorted(g for g, c in canon.items() if g != c)
distances = [pipeline_core.haversine_m(signs[g][1], signs[g][2],
                                       signs[canon[g]][1], signs[canon[g]][2]) for g in merged]
report["ground_truth_merges"] = {
    "canon_m": CANON_M, "ids": len(canon), "posts": len(set(canon.values())),
    "merged_ids": len(merged),
    "max_merge_distance_m": r(max(distances)) if distances else None,
    "merged_ids_in_same_photos_as_their_post": sum(1 for g in merged
                                                   if photos_of[g] == photos_of[canon[g]]),
    "examples": [[g, canon[g]] for g in merged[:5]],
}
gm = report["ground_truth_merges"]
print(f"5. {gm['ids']} ids -> {gm['posts']} posts; {gm['merged_ids']} merged ids, "
      f"max distance {gm['max_merge_distance_m']} m, "
      f"{gm['merged_ids_in_same_photos_as_their_post']} appear in exactly the same photos")

report["versions"] = {"python": platform.python_version()}
for name in ("ultralytics", "torch"):
    try:
        report["versions"][name] = __import__(name).__version__
    except Exception:
        report["versions"][name] = None

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out = RESULTS_DIR / "sequence_checks.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
    f.write("\n")
print("\nsaved ->", out)
