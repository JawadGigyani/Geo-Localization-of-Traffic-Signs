# Run the pipeline over many sequences and pool the results.
#
#   python scripts/07_batch_evaluate.py                    # with the crop classifier
#   CLS_WEIGHTS=none python scripts/07_batch_evaluate.py   # detector labels only
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
# Writes (all tracked in git, so a re-run can be checked with `git diff`):
#   results/batch_evaluation_stage2.json   (stage1 when CLS_WEIGHTS=none)
#   results/sequences_manifest.json        which sequences calibrated / were scored
#   weights/calibration.json               the fitted constants the pipeline uses
#
# Env:
#   SEQ_ROOT=data/sequences        folder of folders, each a sequence
#   CALIB_FRACTION=0.4             share of sequences reserved for calibration
#   CONF=0.25                      detector confidence threshold
#   CLS_WEIGHTS / CLS_MIN_CONF     crop classifier and its overrule threshold (0.8)
#   RADIUS_M                       clustering radius; default from weights/calibration.json
#   SWEEP=1                        choose the radius on the calibration sequences
#   CANON_M=2.0                    same-class ids within this distance = one post
#   SAVE_CALIBRATION=0             do not rewrite weights/calibration.json
#   UPLOAD=1                       also write the evaluation sequences to Supabase
#   RESULTS_DIR, CACHE_DIR, CALIBRATION_PATH   output locations (for testing)

import os
import sys
import copy
import json
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

SEQ_ROOT = Path(os.environ.get("SEQ_ROOT", ROOT / "data" / "sequences"))
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", str(ROOT / "weights" / "best.pt"))
CONF = float(os.environ.get("CONF", "0.25"))
CALIB_FRACTION = float(os.environ.get("CALIB_FRACTION", "0.4"))
UPLOAD = os.environ.get("UPLOAD", "0") == "1"
SWEEP = os.environ.get("SWEEP", "0") == "1"
CLS_WEIGHTS = os.environ.get("CLS_WEIGHTS", str(ROOT / "weights" / "crop_classifier.pt"))
CLS_MIN_CONF = float(os.environ.get("CLS_MIN_CONF", "0.8"))
CANON_M = float(os.environ.get("CANON_M", "2.0"))
SAVE_CALIBRATION = os.environ.get("SAVE_CALIBRATION", "1") == "1"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", ROOT / "data"))
CALIBRATION_PATH = Path(os.environ.get("CALIBRATION_PATH", ROOT / "weights" / "calibration.json"))

# The clustering radius is a pipeline setting stored with the camera constants.
RADIUS_M = float(os.environ.get("RADIUS_M")
                 or pipeline_core.load_calibration(CALIBRATION_PATH)["radius_m"])

if not SEQ_ROOT.is_dir():
    raise SystemExit(
        f"No sequence folder at {SEQ_ROOT}\n"
        "Create it with scripts/05_export_sequences.py (or notebook Phase F)."
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
          f"{len(families)} confusable families, overrules at >= {CLS_MIN_CONF}")
else:
    print("second stage: disabled")


def r4(x):
    return None if x is None else round(float(x), 4)


def library_versions():
    versions = {"python": platform.python_version()}
    for name in ("ultralytics", "torch", "numpy", "cv2"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


# ---------------------------------------------------------------- detect once
print()
print("=" * 70)
print("DETECTION")
print("=" * 70)

# Detection is the expensive part (CPU, 1280 px). Cache it so parameter sweeps
# are instant instead of a fresh half-hour inference run each time. The cache is
# made with no stage-2 threshold, so it holds the classifier's own pick and
# confidence; the threshold is applied afterwards, to a copy, and any threshold
# can be evaluated without re-running inference.
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE = CACHE_DIR / f"detections_cache_conf{CONF}_{CLS_TAG}.json"
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
        raw = cached[seq.name]["dets"]
        source = "cached"
    else:
        raw = pipeline_core.detect_all(model, frames, conf=CONF, imgsz=IMGSZ,
                                       meta=meta, keep_crops=False,
                                       cls_model=cls_model, families=families,
                                       cls_min_conf=0.0)
        cached[seq.name] = {"frames": [os.path.basename(f) for f in frames],
                            "dets": raw}
        dirty = True
        source = "detected"

    # Work on a copy so the cache keeps the unthresholded stage-2 picks.
    dets = copy.deepcopy(raw)
    for d in dets:
        d["cls_name_pick"] = d["cls_name"]       # the classifier's own choice
        c = d.get("cls_stage2_conf")
        if c is not None and c < CLS_MIN_CONF and d.get("cls_refined"):
            d["cls_name"] = d.get("cls_name_stage1", d["cls_name"])
            d["cls_refined"] = False

    canon = pipeline_core.canonical_gt_map(meta, CANON_M)
    raw_ids = {str(o["sign_id"]) for m in meta.values()
               for o in m.get("objects", []) if o.get("sign_id")}
    posts = {canon.get(g, g) for g in raw_ids}
    per_seq[seq.name] = {"dir": seq, "frames": frames, "meta": meta, "dets": dets,
                         "canon": canon, "raw_ids": raw_ids, "posts": posts}
    print(f"  {seq.name:<28s} {len(frames):4d} frames  {len(dets):4d} detections  "
          f"{len(raw_ids):3d} ids -> {len(posts):3d} posts  [{source}]")

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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
manifest = {
    "calibration_fraction": CALIB_FRACTION,
    "sequences": [
        {"name": n,
         "role": "calibration" if n in calib_names else "evaluation",
         "frames": len(per_seq[n]["frames"]),
         "sign_ids": len(per_seq[n]["raw_ids"])}
        for n in calib_names + eval_names
    ],
}
with open(RESULTS_DIR / "sequences_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

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


def score(names, radius):
    """Pooled metrics at a given clustering radius, over the given sequences."""
    scores = []
    for name in names:
        info = per_seq[name]
        tracks = pipeline_core.associate_geometric(
            info["dets"], hfov_deg=hfov, size_k=size_k, radius_m=radius)
        scores.append(pipeline_core.score_rows(tracks, info["meta"], canon=info["canon"]))
    return pipeline_core.pool_scores(scores)


if SWEEP:
    # Tuned on the CALIBRATION sequences only, so the held-out set stays clean.
    # Ranked by strict one-to-one F1: plain recall can be gamed by merging.
    print()
    print("=" * 70)
    print("RADIUS SWEEP  (on calibration sequences only)")
    print("=" * 70)
    print(f"{'radius':>7s} {'rows':>6s} {'1to1':>6s} {'prec':>7s} {'recall':>7s} "
          f"{'F1':>7s} {'frag':>5s} {'merge':>6s} {'err_med':>8s}")
    best, best_score = None, -1.0
    for radius in (10, 15, 20, 30, 40, 50, 60, 80, 100):
        s = score(calib_names, radius)
        med = pipeline_core.percentile(s["errors"], 0.5) or float("nan")
        flag = ""
        if s["f1_1to1"] > best_score:
            best_score, best, flag = s["f1_1to1"], radius, "  <-"
        print(f"{radius:6d}m {s['rows']:6d} {s['one_to_one']:6d} "
              f"{100 * s['precision_1to1']:6.1f}% {100 * s['recall_1to1']:6.1f}% "
              f"{100 * s['f1_1to1']:6.1f}% {s['fragmented']:5d} {s['merged']:6d} "
              f"{med:7.1f}m{flag}")
    print(f"\nbest radius on calibration data: {best} m")
    RADIUS_M = float(best)

if SAVE_CALIBRATION:
    pipeline_core.save_calibration(
        CALIBRATION_PATH, hfov, size_k, RADIUS_M,
        fitted_by="scripts/07_batch_evaluate.py",
        calibration_sequences=len(calib_names),
        matched_detections=n_used,
    )
    print(f"  saved -> {CALIBRATION_PATH}")

# ---------------------------------------------------------------- evaluate
print()
print("=" * 70)
print(f"PER-SEQUENCE RESULTS  (radius {RADIUS_M:.0f} m)")
print("=" * 70)
print(f"{'sequence':<28s} {'posts':>6s} {'rows':>5s} {'recall':>7s} "
      f"{'dedupe':>7s} {'err_med':>8s}")

seq_scores, per_sequence = [], {}
for name in eval_names:
    info = per_seq[name]
    tracks = pipeline_core.associate_geometric(
        info["dets"], hfov_deg=hfov, size_k=size_k, radius_m=RADIUS_M)
    s = pipeline_core.score_rows(tracks, info["meta"], canon=info["canon"])
    seq_scores.append(s)
    one = pipeline_core.pool_scores([s])
    med = pipeline_core.percentile(s["errors"], 0.5)
    per_sequence[name] = {
        "posts": s["posts"], "rows": s["rows"], "found": s["found"],
        "fragmented": s["fragmented"], "merged": s["merged"],
        "recall": r4(one["recall"]), "dedupe": r4(one["dedupe"]),
        "median_error_m": r4(med),
    }
    print(f"{name:<28s} {s['posts']:6d} {s['rows']:5d} {100 * one['recall']:6.1f}% "
          f"{100 * one['dedupe']:6.1f}% "
          f"{med if med is not None else float('nan'):7.1f}m")

pooled = pipeline_core.pool_scores(seq_scores)
errors, baseline = pooled["errors"], pooled["baseline"]

print()
print("=" * 70)
print(f"POOLED OVER {len(eval_names)} HELD-OUT SEQUENCES")
print("=" * 70)
print(f"ground-truth ids / posts    : {pooled['raw_ids']} / {pooled['posts']}")
print(f"inventory rows produced     : {pooled['rows']}")
print(f"posts found                 : {pooled['found']}  ({100 * pooled['recall']:.1f}% recall)")
print(f"fragmented / merged         : {pooled['fragmented']} / {pooled['merged']}")
if pooled["found"]:
    print(f"dedupe score                : {100 * pooled['dedupe']:.1f}%   "
          "(found - fragmented - merged) / found")
print(f"strict one-to-one           : {pooled['one_to_one']} posts  "
      f"(precision {100 * pooled['precision_1to1']:.1f}%, "
      f"recall {100 * pooled['recall_1to1']:.1f}%)")

geo = {"n": len(errors), "mean": None, "median": None, "p90": None,
       "baseline_median": None, "improvement": None}
if errors:
    geo.update(mean=statistics.mean(errors),
               median=pipeline_core.percentile(errors, 0.5),
               p90=pipeline_core.percentile(errors, 0.9))
    print()
    print(f"geo-localization error over {len(errors)} rows")
    print(f"  mean   : {geo['mean']:.1f} m")
    print(f"  median : {geo['median']:.1f} m")
    print(f"  p90    : {geo['p90']:.1f} m")
    if baseline:
        geo["baseline_median"] = pipeline_core.percentile(baseline, 0.5)
        geo["improvement"] = (geo["baseline_median"] - geo["median"]) / geo["baseline_median"]
        print(f"  camera pin baseline (median) : {geo['baseline_median']:.1f} m")
        print(f"  improvement                  : {100 * geo['improvement']:+.0f}%")

# ------------------------------------------------- classification accuracy
# Measured on detections matched to a ground-truth box, so it isolates "did we
# name it correctly" from "did we find it". Reported per family, since the whole
# point of the second stage is the multi-member ones.
print()
print("=" * 70)
print("CLASSIFICATION ACCURACY  (detections matched to ground truth, IoU >= 0.3)")
print("=" * 70)
multi_fams = set(pipeline_core.build_families(list(model.names.values())))
fam_stats = defaultdict(lambda: {"n": 0, "s1": 0, "s2": 0})
overall = {"n": 0, "s1": 0, "s2": 0}
conf_right, conf_wrong = [], []
for name in eval_names:
    for d in per_seq[name]["dets"]:
        truth = d.get("gt_class")
        if not truth:
            continue
        fam = pipeline_core.family_of(truth)
        stage1 = d.get("cls_name_stage1", d["cls_name"])
        for st in (fam_stats[fam], overall):
            st["n"] += 1
            st["s1"] += int(stage1 == truth)
            st["s2"] += int(d["cls_name"] == truth)
        # how sure the classifier was, split by whether its own pick was right
        if fam in multi_fams and d.get("cls_stage2_conf") is not None:
            (conf_right if d["cls_name_pick"] == truth else conf_wrong).append(
                d["cls_stage2_conf"])

classification = {"n": overall["n"], "stage1_acc": None, "stage2_acc": None,
                  "confusable": None, "by_family": {}, "stage2_confidence": None}
if overall["n"]:
    print(f"{'family':<8s} {'n':>6s} {'stage1':>8s} {'stage2':>8s} {'delta':>8s}")
    for fam, st in sorted(fam_stats.items(), key=lambda kv: -kv[1]["n"]):
        classification["by_family"][fam] = {
            "n": st["n"], "stage1_acc": r4(st["s1"] / st["n"]),
            "stage2_acc": r4(st["s2"] / st["n"]), "confusable": fam in multi_fams}
        if fam not in multi_fams:
            continue
        a, b = 100 * st["s1"] / st["n"], 100 * st["s2"] / st["n"]
        print(f"{fam:<8s} {st['n']:6d} {a:7.1f}% {b:7.1f}% {b - a:+7.1f}")
    conf_n = sum(st["n"] for f, st in fam_stats.items() if f in multi_fams)
    conf_s1 = sum(st["s1"] for f, st in fam_stats.items() if f in multi_fams)
    conf_s2 = sum(st["s2"] for f, st in fam_stats.items() if f in multi_fams)
    classification["stage1_acc"] = r4(overall["s1"] / overall["n"])
    classification["stage2_acc"] = r4(overall["s2"] / overall["n"])
    print()
    if conf_n:
        classification["confusable"] = {"n": conf_n, "stage1_acc": r4(conf_s1 / conf_n),
                                        "stage2_acc": r4(conf_s2 / conf_n)}
        print(f"confusable families only : {100 * conf_s1 / conf_n:.1f}% -> "
              f"{100 * conf_s2 / conf_n:.1f}%  ({conf_n} detections)")
    print(f"all detections           : {100 * overall['s1'] / overall['n']:.1f}% -> "
          f"{100 * overall['s2'] / overall['n']:.1f}%  ({overall['n']} detections)")
    if conf_right or conf_wrong:
        classification["stage2_confidence"] = {
            "mean_when_right": r4(statistics.mean(conf_right)) if conf_right else None,
            "mean_when_wrong": r4(statistics.mean(conf_wrong)) if conf_wrong else None,
            "n_right": len(conf_right), "n_wrong": len(conf_wrong)}
        cr = classification["stage2_confidence"]
        print(f"stage-2 confidence       : {cr['mean_when_right']} when right "
              f"({cr['n_right']}), {cr['mean_when_wrong']} when wrong ({cr['n_wrong']})")
else:
    print("no detections matched to ground truth")

report = {
    "stage": CLS_TAG,
    "settings": {"conf": CONF, "cls_min_conf": CLS_MIN_CONF if cls_model else None,
                 "radius_m": RADIUS_M, "calib_fraction": CALIB_FRACTION,
                 "canon_m": CANON_M, "imgsz": IMGSZ, "sweep": SWEEP,
                 "weights": os.path.basename(WEIGHTS_PATH),
                 "classifier": os.path.basename(CLS_WEIGHTS) if cls_model else None},
    "versions": library_versions(),
    "calibration": {"sequences": calib_names, "hfov_deg": hfov,
                    "size_k_m": r4(size_k), "matches": n_used},
    "evaluated_sequences": eval_names,
    "pooled": {k: (r4(v) if isinstance(v, float) else v)
               for k, v in pooled.items() if k not in ("errors", "baseline")},
    "geo_error_m": {k: (r4(v) if isinstance(v, float) else v) for k, v in geo.items()},
    "classification": classification,
    "per_sequence": per_sequence,
}
out = RESULTS_DIR / f"batch_evaluation_{CLS_TAG}.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
    f.write("\n")
print()
print("saved ->", out)
print("saved ->", RESULTS_DIR / "sequences_manifest.json")

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
            # re-detect WITH crops (the cache stores geometry only) and with the
            # stage-2 threshold, so the rows match the numbers reported above
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
