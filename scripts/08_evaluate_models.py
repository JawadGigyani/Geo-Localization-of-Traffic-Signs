# Re-measure both model scores from the shipped weights.
#
#   python scripts/08_evaluate_models.py
#
# Run after scripts/02_convert_arts_to_yolo.py. No training, no GPU needed.
#   1. Detector: mAP@50, mAP@50-95, precision, recall on the TEST split
#      (the same call as notebook cell E1)
#   2. Crop classifier: top-1 accuracy on validation crops, cut exactly the way
#      notebook cell G1 cut them (ground-truth boxes, 15% padding, <=128 px, JPEG)
#
# Writes results/model_metrics.json (merged, so each part can be re-run alone).
#
# Env:
#   ARTS_YOLO_OUT=data/arts_yolo
#   WEIGHTS_PATH=weights/best.pt   CLS_WEIGHTS=weights/crop_classifier.pt
#   DETECTOR=1   CLASSIFIER=1      set either to 0 to skip that part
#   RESULTS_DIR=results

import io
import os
import re
import sys
import json
import platform
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from ultralytics import YOLO

DATASET = Path(os.environ.get("ARTS_YOLO_OUT", ROOT / "data" / "arts_yolo"))
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", str(ROOT / "weights" / "best.pt"))
CLS_WEIGHTS = os.environ.get("CLS_WEIGHTS", str(ROOT / "weights" / "crop_classifier.pt"))
RUN_DETECTOR = os.environ.get("DETECTOR", "1") == "1"
RUN_CLASSIFIER = os.environ.get("CLASSIFIER", "1") == "1"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))

# notebook cell G1
PAD_FRAC = 0.15
OUT_PX = 128
BATCH = 64

if not (DATASET / "data.yaml").is_file():
    raise SystemExit(f"No converted dataset at {DATASET}. "
                     "Run scripts/02_convert_arts_to_yolo.py first.")


def r4(x):
    return None if x is None else round(float(x), 4)


def local_data_yaml():
    """data.yaml records the absolute folder the dataset was converted in (on
    Colab, /content/dataset/arts_yolo). Write a copy that points at where the
    dataset is now, so a dataset converted elsewhere can be evaluated here."""
    lines = (DATASET / "data.yaml").read_text(encoding="utf-8").splitlines()
    fixed = [f"path: {DATASET.resolve().as_posix()}" if line.startswith("path:") else line
             for line in lines]
    target = DATASET / "data_local.yaml"
    target.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    return target


def library_versions():
    versions = {"python": platform.python_version()}
    for name in ("ultralytics", "torch", "numpy", "PIL"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out = RESULTS_DIR / "model_metrics.json"
report = {}
if out.is_file():
    with open(out, "r", encoding="utf-8") as f:
        report = json.load(f)
report["versions"] = library_versions()

# ------------------------------------------------------------ 1. detector
if RUN_DETECTOR:
    print("=" * 70)
    print("DETECTOR on the test split")
    print("=" * 70)
    model = YOLO(WEIGHTS_PATH)
    m = model.val(data=str(local_data_yaml()), split="test", plots=False)
    names = model.names
    per_class = {}
    for i, cid in enumerate(m.box.ap_class_index):
        per_class[names[int(cid)]] = {"ap50": r4(m.box.ap50[i]), "ap50_95": r4(m.box.ap[i])}
    report["detector_test"] = {
        "weights": os.path.basename(WEIGHTS_PATH),
        "map50": r4(m.box.map50), "map50_95": r4(m.box.map),
        "precision": r4(m.box.mp), "recall": r4(m.box.mr),
        "per_class": dict(sorted(per_class.items())),
    }
    d = report["detector_test"]
    print(f"\nmAP50 {d['map50']}   mAP50-95 {d['map50_95']}   "
          f"precision {d['precision']}   recall {d['recall']}")

# ------------------------------------------------------------ 2. classifier
if RUN_CLASSIFIER:
    print()
    print("=" * 70)
    print("CROP CLASSIFIER on validation crops")
    print("=" * 70)
    classes = [l.strip() for l in open(DATASET / "classes.txt", encoding="utf-8") if l.strip()]
    fam = defaultdict(list)
    for c in classes:
        mt = re.match(r"^([A-Za-z]+\d*)-", c)
        fam[mt.group(1) if mt else c].append(c)
    confusable = {c for members in fam.values() if len(members) > 1 for c in members}

    with open(DATASET / "meta" / "frames_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)["frames"]

    cls_model = YOLO(CLS_WEIGHTS)
    cls_names = cls_model.names

    def gt_crop(im, box):
        """Exactly notebook cell G1: pad, clip, shrink, JPEG quality 92."""
        x1, y1, x2, y2 = box
        pad = max(4.0, PAD_FRAC * max(x2 - x1, y2 - y1))
        crop = im.crop((max(0, x1 - pad), max(0, y1 - pad),
                        min(im.width, x2 + pad), min(im.height, y2 + pad)))
        if crop.width < 4 or crop.height < 4:
            return None
        if max(crop.size) > OUT_PX:
            crop.thumbnail((OUT_PX, OUT_PX), Image.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=92)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    crops, truths = [], []
    for stem, m in meta.items():
        if m["split"] != "val":
            continue
        objs = [o for o in m.get("objects", []) if o["name"] in confusable]
        if not objs:
            continue
        src = DATASET / "images" / "val" / f"{stem}.jpg"
        if not src.is_file():
            continue
        with Image.open(src) as raw:
            im = raw.convert("RGB")
        for o in objs:
            crop = gt_crop(im, o["bbox"])
            if crop is not None:
                crops.append(crop)
                truths.append(o["name"])
    print(f"validation crops: {len(crops)} across {len(set(truths))} classes")

    correct = 0
    by_class = defaultdict(lambda: [0, 0])
    for start in range(0, len(crops), BATCH):
        batch = crops[start:start + BATCH]
        for res, truth in zip(cls_model.predict(source=batch, verbose=False),
                              truths[start:start + BATCH]):
            ok = cls_names[int(res.probs.top1)] == truth
            correct += int(ok)
            by_class[truth][0] += int(ok)
            by_class[truth][1] += 1
    top1 = correct / len(crops) if crops else None
    report["classifier_val"] = {
        "weights": os.path.basename(CLS_WEIGHTS),
        "crops": len(crops), "top1": r4(top1),
        "per_class": {c: r4(k / n) for c, (k, n) in sorted(by_class.items())},
    }
    print(f"top-1 accuracy: {r4(top1)}")

with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
    f.write("\n")
print("\nsaved ->", out)
