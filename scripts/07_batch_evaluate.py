# Run the pipeline over many sequences and pool the results.
#
#   python scripts/07_batch_evaluate.py
#
# Why this exists: one 69-frame sequence yields ~35 signs, which is an anecdote.
# Pooling a few hundred signs across independent road stretches turns the
# headline numbers into measurements.
#
# It also removes the calibration caveat. The camera field of view and the
# width-to-range constant are fitted on a HELD-OUT set of sequences and then
# applied unchanged to the rest, so no evaluated sequence contributes to the
# constants used to score it.
#
# Env:
#   SEQ_ROOT=data/sequences        folder of folders, each a sequence
#   CALIB_FRACTION=0.4             share of sequences reserved for calibration
#   CONF=0.25
#   UPLOAD=1                       also write the evaluation sequences to Supabase

import os
import sys
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import pipeline_core
from ultralytics import YOLO

SEQ_ROOT = Path(os.environ.get("SEQ_ROOT", ROOT / "data" / "sequences"))
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", str(ROOT / "weights" / "best.pt"))
CONF = float(os.environ.get("CONF", "0.25"))
CALIB_FRACTION = float(os.environ.get("CALIB_FRACTION", "0.4"))
RADIUS_M = float(os.environ.get("RADIUS_M", "60"))
UPLOAD = os.environ.get("UPLOAD", "0") == "1"
SWEEP = os.environ.get("SWEEP", "0") == "1"
CLS_WEIGHTS = os.environ.get("CLS_WEIGHTS", str(ROOT / "weights" / "crop_classifier.pt"))
CLS_MIN_CONF = float(os.environ.get("CLS_MIN_CONF", "0.8"))

if not SEQ_ROOT.is_dir():
    raise SystemExit(
        f"No sequence folder at {SEQ_ROOT}\n"
        "Run notebook Phase F, then unzip sequences.zip into data/sequences/"
    )

seq_dirs = sorted(d for d in SEQ_ROOT.iterdir()
                  if d.is_dir() and (d / "frames_meta.json").is_file())
if not seq_dirs:
    raise SystemExit(f"No sequences with frames_meta.json under {SEQ_ROOT}")

print(f"sequences found: {len(seq_dirs)}")
if not os.path.isfile(WEIGHTS_PATH):
    raise SystemExit(f"No model at {WEIGHTS_PATH}")

model = YOLO(WEIGHTS_PATH)
IMGSZ = int(os.environ.get("IMGSZ", "0")) or pipeline_core.training_imgsz(model)
print(f"weights: {WEIGHTS_PATH}   imgsz: {IMGSZ}")

cls_model, families, CLS_TAG = None, None, "stage1"
if CLS_WEIGHTS.lower() != "none" and os.path.isfile(CLS_WEIGHTS):
    cls_model = YOLO(CLS_WEIGHTS)
    families = pipeline_core.build_families(list(model.names.values()))
    CLS_TAG = "stage2"
    print(f"second stage: {os.path.basename(CLS_WEIGHTS)}, "
          f"{len(families)} confusable families")
else:
    print("second stage: disabled")


CANON_M = float(os.environ.get("CANON_M", "2.0"))


def canonical_gt_map(meta, tol_m=CANON_M):
    """Collapse ground-truth ids that describe the same physical post.

    ARTSv2's <id> is not a clean physical-sign key: some entries appear twice
    with a numeric prefix at IDENTICAL coordinates (measured: 0.0 m apart), and
    sign assemblies put several plaques on one post metres apart. Scoring
    against the raw ids asks the system to separate signs that no geometric
    method could, so ids of the same class within tol_m are treated as one post.

    Returns {raw_id: canonical_id}."""
    signs = {}
    for m in meta.values():
        for o in m.get("objects", []):
            gid = o.get("sign_id")
            if gid and o.get("sign_lat") is not None:
                signs[str(gid)] = (o["name"], o["sign_lat"], o["sign_lon"])

    canon, mapping = [], {}
    for gid, (name, lat, lon) in sorted(signs.items()):
        hit = None
        for c in canon:
            if c["name"] == name and pipeline_core.haversine_m(
                    c["lat"], c["lon"], lat, lon) <= tol_m:
                hit = c
                break
        if hit is None:
            canon.append({"name": name, "lat": lat, "lon": lon, "id": gid})
            mapping[gid] = gid
        else:
            mapping[gid] = hit["id"]
    return mapping


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return s[int(k)] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------- detect once
print()
print("=" * 70)
print("DETECTION")
print("=" * 70)

# Detection is the expensive part (CPU, 1280 px). Cache it so parameter sweeps
# are instant instead of a fresh half-hour inference run each time. Crops are
# numpy arrays and are re-extracted on demand when uploading.
CACHE = ROOT / "data" / f"detections_cache_conf{CONF}_{CLS_TAG}.json"
cached = {}
if CACHE.is_file():
    with open(CACHE, "r", encoding="utf-8") as f:
        cached = json.load(f)
    print(f"  cache: {CACHE.name} ({len(cached)} sequences)")

per_seq = {}
dirty = False
for seq in seq_dirs:
    frames = pipeline_core.list_frames(str(seq))
    meta = pipeline_core.load_frames_meta(str(seq))
    if seq.name in cached and len(cached[seq.name]["frames"]) == len(frames):
        dets = cached[seq.name]["dets"]
        source = "cached"
    else:
        dets = pipeline_core.detect_all(model, frames, conf=CONF, imgsz=IMGSZ,
                                        meta=meta, keep_crops=False,
                                        cls_model=cls_model, families=families,
                                        cls_min_conf=0.0)
        cached[seq.name] = {"frames": [os.path.basename(f) for f in frames],
                            "dets": dets}
        dirty = True
        source = "detected"
    # Apply the stage-2 confidence threshold here rather than at detection
    # time: the cache stores stage-1 name, stage-2 name and stage-2 confidence,
    # so any threshold can be evaluated without re-running inference.
    if CLS_MIN_CONF > 0:
        reverted = 0
        for d in dets:
            c = d.get("cls_stage2_conf")
            if c is not None and c < CLS_MIN_CONF and d.get("cls_refined"):
                d["cls_name"] = d.get("cls_name_stage1", d["cls_name"])
                d["cls_refined"] = False
                reverted += 1

    raw_ids = {str(o["sign_id"]) for m in meta.values()
               for o in m.get("objects", []) if o.get("sign_id")}
    canon = canonical_gt_map(meta)
    gt_ids = {canon.get(g, g) for g in raw_ids}
    per_seq[seq.name] = {"dir": seq, "frames": frames, "meta": meta,
                         "dets": dets, "gt_ids": gt_ids,
                         "raw_ids": raw_ids, "canon": canon}
    print(f"  {seq.name:<28s} {len(frames):4d} frames  {len(dets):4d} detections  "
          f"{len(raw_ids):3d} ids -> {len(gt_ids):3d} posts  [{source}]")

if dirty:
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cached, f)
    print(f"  cache written -> {CACHE.name}")

# ------------------------------------------------------- split calib / eval
n_calib = max(1, int(len(seq_dirs) * CALIB_FRACTION))
calib_names = [s.name for s in seq_dirs[:n_calib]]
eval_names = [s.name for s in seq_dirs[n_calib:]]
if not eval_names:
    raise SystemExit("Not enough sequences to hold any back for evaluation. "
                     "Lower CALIB_FRACTION or export more sequences.")

print()
print("=" * 70)
print("CALIBRATION  (held out from every number reported below)")
print("=" * 70)
calib_dets = [d for n in calib_names for d in per_seq[n]["dets"]]
hfov, size_k, n_used = pipeline_core.calibrate_geometry(calib_dets)
if hfov is None:
    raise SystemExit(f"Only {n_used} ground-truth matches in the calibration set.")
print(f"  fitted on {len(calib_names)} sequences, {n_used} matched detections")
print(f"  hfov   = {hfov:.0f} deg")
print(f"  size_k = {size_k:.3f} m")
print(f"  evaluating on the other {len(eval_names)} sequences")

# ---------------------------------------------------------------- sweep
def score(names, radius):
    """Pooled metrics at a given clustering radius, over the given sequences."""
    errors = []
    gt = rows = found = frag = merge = correct = 0
    for name in names:
        info = per_seq[name]
        tracks = pipeline_core.associate_geometric(
            info["dets"], hfov_deg=hfov, size_k=size_k, radius_m=radius
        )
        by_gt, by_track = defaultdict(set), defaultdict(set)
        for t in tracks.values():
            if t.get("gt_sign_lat") is not None:
                errors.append(pipeline_core.haversine_m(
                    t["lat"], t["lon"], t["gt_sign_lat"], t["gt_sign_lon"]))
            canon = info["canon"]
            for gid in (t.get("member_gt_ids")
                        or ([str(t["gt_sign_id"])] if t.get("gt_sign_id") else [])):
                gid = canon.get(gid, gid)
                by_gt[gid].add(t["track_id"])
                by_track[t["track_id"]].add(gid)
        gt += len(info["gt_ids"])
        rows += len(tracks)
        found += len(by_gt)
        frag += sum(1 for v in by_gt.values() if len(v) > 1)
        merge += sum(1 for v in by_track.values() if len(v) > 1)
        # accumulate INSIDE the loop: by_gt/by_track are per-sequence, and
        # track ids restart at 1 in every sequence so they cannot be pooled
        for gid, tset in by_gt.items():
            if len(tset) == 1 and len(by_track[next(iter(tset))]) == 1:
                correct += 1
    # One-to-one resolution: a sign counts only if it maps to exactly one row
    # AND that row maps to exactly one sign. Plain "recall" is gameable -- widen
    # the radius until one blob swallows every sign and it reads 100%.
    prec = 100 * correct / rows if rows else 0
    rec = 100 * correct / gt if gt else 0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0
    return {"gt": gt, "rows": rows, "found": found, "frag": frag,
            "merge": merge, "errors": errors, "correct": correct,
            "precision": prec, "recall_1to1": rec, "f1": f1,
            "recall": 100 * found / gt if gt else 0,
            "dedupe": 100 * (found - frag - merge) / found if found else 0}


if SWEEP:
    # Tuned on the CALIBRATION sequences only, so the held-out set stays clean.
    print()
    print("=" * 70)
    print("RADIUS SWEEP  (on calibration sequences only)")
    print("=" * 70)
    print(f"{'radius':>7s} {'rows':>6s} {'1to1':>6s} {'prec':>7s} {'recall':>7s} "
          f"{'F1':>7s} {'frag':>5s} {'merge':>6s} {'err_med':>8s}")
    best, best_score = None, -1.0
    for radius in (10, 15, 20, 30, 40, 50, 60, 80, 100):
        s = score(calib_names, radius)
        med = percentile(s["errors"], 0.5) or float("nan")
        flag = ""
        if s["f1"] > best_score:
            best_score, best, flag = s["f1"], radius, "  <-"
        print(f"{radius:6d}m {s['rows']:6d} {s['correct']:6d} {s['precision']:6.1f}% "
              f"{s['recall_1to1']:6.1f}% {s['f1']:6.1f}% {s['frag']:5d} "
              f"{s['merge']:6d} {med:7.1f}m{flag}")
    print(f"\nbest radius on calibration data: {best} m")
    RADIUS_M = float(best)

# ---------------------------------------------------------------- evaluate
print()
print("=" * 70)
print(f"PER-SEQUENCE RESULTS  (radius {RADIUS_M:.0f} m)")
print("=" * 70)
print(f"{'sequence':<28s} {'signs':>6s} {'rows':>5s} {'recall':>7s} "
      f"{'dedupe':>7s} {'err_med':>8s}")

all_errors, all_baseline = [], []
tot_gt = tot_rows = tot_found = tot_frag = tot_merge = 0
rows_by_seq = {}

for name in eval_names:
    info = per_seq[name]
    tracks = pipeline_core.associate_geometric(
        info["dets"], hfov_deg=hfov, size_k=size_k, radius_m=RADIUS_M
    )
    rows_by_seq[name] = tracks

    errors, baseline = [], []
    by_gt, by_track = defaultdict(set), defaultdict(set)
    for t in tracks.values():
        if t.get("gt_sign_lat") is None:
            continue
        errors.append(pipeline_core.haversine_m(
            t["lat"], t["lon"], t["gt_sign_lat"], t["gt_sign_lon"]))
        if t.get("camera_lat") is not None:
            baseline.append(pipeline_core.haversine_m(
                t["camera_lat"], t["camera_lon"], t["gt_sign_lat"], t["gt_sign_lon"]))
        for gid in (t.get("member_gt_ids")
                    or ([str(t["gt_sign_id"])] if t.get("gt_sign_id") else [])):
            gid = info["canon"].get(gid, gid)
            by_gt[gid].add(t["track_id"])
            by_track[t["track_id"]].add(gid)

    n_gt = len(info["gt_ids"])
    found = len(by_gt)
    frag = sum(1 for v in by_gt.values() if len(v) > 1)
    merge = sum(1 for v in by_track.values() if len(v) > 1)

    all_errors += errors
    all_baseline += baseline
    tot_gt += n_gt
    tot_rows += len(tracks)
    tot_found += found
    tot_frag += frag
    tot_merge += merge

    recall = 100 * found / n_gt if n_gt else 0.0
    dedupe = 100 * (found - frag - merge) / found if found else 0.0
    med = percentile(errors, 0.5)
    print(f"{name:<28s} {n_gt:6d} {len(tracks):5d} {recall:6.1f}% "
          f"{dedupe:6.1f}% {med if med else float('nan'):7.1f}m")

# ---------------------------------------------------------------- pooled
print()
print("=" * 70)
print(f"POOLED OVER {len(eval_names)} HELD-OUT SEQUENCES")
print("=" * 70)
print(f"ground-truth physical signs : {tot_gt}")
print(f"inventory rows produced     : {tot_rows}")
print(f"signs detected              : {tot_found}  ({100*tot_found/max(1,tot_gt):.1f}% recall)")
print(f"fragmented / merged         : {tot_frag} / {tot_merge}")
if tot_found:
    print(f"dedupe accuracy             : "
          f"{100*(tot_found-tot_frag-tot_merge)/tot_found:.1f}%")

if all_errors:
    print()
    print(f"geo-localization error over {len(all_errors)} signs")
    print(f"  mean   : {statistics.mean(all_errors):.1f} m")
    print(f"  median : {percentile(all_errors, 0.5):.1f} m")
    print(f"  p90    : {percentile(all_errors, 0.9):.1f} m")
    if all_baseline:
        b = percentile(all_baseline, 0.5)
        a = percentile(all_errors, 0.5)
        print(f"  camera pin baseline (median) : {b:.1f} m")
        print(f"  improvement                  : {100*(b-a)/b:+.0f}%")

# ------------------------------------------------- classification accuracy
# Measured on detections matched to a ground-truth box, so it isolates "did we
# name it correctly" from "did we find it". Reported per family, since the whole
# point of the second stage is the multi-member ones.
print()
print("=" * 70)
print("CLASSIFICATION ACCURACY  (detections matched to ground truth)")
print("=" * 70)
fam_stats = defaultdict(lambda: {"n": 0, "s1": 0, "s2": 0})
overall = {"n": 0, "s1": 0, "s2": 0}
for name in eval_names:
    for d in per_seq[name]["dets"]:
        truth = d.get("gt_class")
        if not truth:
            continue
        fam = pipeline_core.family_of(truth)
        stage1 = d.get("cls_name_stage1", d["cls_name"])
        st = fam_stats[fam]
        st["n"] += 1
        st["s1"] += int(stage1 == truth)
        st["s2"] += int(d["cls_name"] == truth)
        overall["n"] += 1
        overall["s1"] += int(stage1 == truth)
        overall["s2"] += int(d["cls_name"] == truth)

multi_fams = set(pipeline_core.build_families(list(model.names.values())))
if overall["n"]:
    print(f"{'family':<8s} {'n':>6s} {'stage1':>8s} {'stage2':>8s} {'delta':>8s}")
    for fam, st in sorted(fam_stats.items(), key=lambda kv: -kv[1]["n"]):
        if fam not in multi_fams:
            continue
        a = 100 * st["s1"] / st["n"]
        b = 100 * st["s2"] / st["n"]
        print(f"{fam:<8s} {st['n']:6d} {a:7.1f}% {b:7.1f}% {b - a:+7.1f}")
    conf_n = sum(st["n"] for f, st in fam_stats.items() if f in multi_fams)
    conf_s1 = sum(st["s1"] for f, st in fam_stats.items() if f in multi_fams)
    conf_s2 = sum(st["s2"] for f, st in fam_stats.items() if f in multi_fams)
    print()
    if conf_n:
        print(f"confusable families only : {100*conf_s1/conf_n:.1f}% -> "
              f"{100*conf_s2/conf_n:.1f}%  ({conf_n} detections)")
    print(f"all detections           : {100*overall['s1']/overall['n']:.1f}% -> "
          f"{100*overall['s2']/overall['n']:.1f}%  ({overall['n']} detections)")
else:
    print("no detections matched to ground truth")

report = {
    "classification": {
        "n": overall["n"],
        "stage1_acc": overall["s1"] / overall["n"] if overall["n"] else None,
        "stage2_acc": overall["s2"] / overall["n"] if overall["n"] else None,
        "by_family": {f: dict(st) for f, st in fam_stats.items()},
    },
    "calibration": {"sequences": calib_names, "hfov_deg": hfov,
                    "size_k_m": size_k, "matches": n_used},
    "evaluated_sequences": eval_names,
    "gt_signs": tot_gt, "rows": tot_rows, "detected": tot_found,
    "fragmented": tot_frag, "merged": tot_merge,
    "geo_error_m": {
        "n": len(all_errors),
        "mean": statistics.mean(all_errors) if all_errors else None,
        "median": percentile(all_errors, 0.5),
        "p90": percentile(all_errors, 0.9),
        "baseline_median": percentile(all_baseline, 0.5) if all_baseline else None,
    },
}
out = ROOT / "data" / "batch_evaluation.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print()
print("saved ->", out)

# ---------------------------------------------------------------- upload
if UPLOAD:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key or "YOUR_PROJECT" in url:
        print("\nUPLOAD=1 but Supabase is not configured; skipping.")
    else:
        from supabase import create_client
        supabase = create_client(url, key)
        print()
        print("uploading held-out sequences to Supabase...")
        total = 0
        for name in eval_names:
            info = per_seq[name]
            # re-detect this sequence WITH crops (the cache stores geometry only)
            # re-detect WITH the second stage and with crops: the cache stores
            # geometry only, and rows written without stage-2 classes would not
            # match the numbers reported above
            fresh = pipeline_core.detect_all(model, info["frames"], conf=CONF,
                                             imgsz=IMGSZ, meta=info["meta"],
                                             cls_model=cls_model, families=families,
                                             cls_min_conf=CLS_MIN_CONF)
            tracks = pipeline_core.associate_geometric(
                fresh, hfov_deg=hfov, size_k=size_k, radius_m=RADIUS_M)
            pipeline_core.save_crops(tracks, str(ROOT / "data" / "pipeline_crops"), name)
            inserted = pipeline_core.upload_and_insert(
                supabase, tracks, name, upload_frames=False
            )
            total += len(inserted)
            print(f"  {name:<28s} {len(inserted):4d} rows")
        print(f"inserted {total} rows across {len(eval_names)} sequences")
