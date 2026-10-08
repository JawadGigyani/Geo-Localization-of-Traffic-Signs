# Convert ARTS / ARTSv2 (PASCAL VOC-like XML) to YOLO detection format.
#
#   python scripts/02_convert_arts_to_yolo.py
#
# What it does, in order:
#   1. finds every image + XML pair under ARTS_ROOT
#   2. auto-detects the real GPS / heading / sign-id tag names in the XML
#      and prints what it resolved, for verification
#   3. groups frames into sequences and splits train/val/test BY GROUP, so
#      consecutive frames of the same physical sign never straddle a split
#   4. writes the YOLO dataset, plus meta/frames_meta.json holding camera GPS,
#      heading and ground-truth sign GPS for the evaluation scripts
#   5. optionally exports one held-out sequence as a ready-to-run demo folder
#
# Env overrides:
#   ARTS_ROOT       the downloaded ARTSv2 folder (Annotations/, JPEGImages/, ImageSets/)
#                   default: data/arts_v2
#   ARTS_YOLO_OUT   where the YOLO dataset is written, default: data/arts_yolo
#   TOP_N_CLASSES (50), MAX_SIDE, VAL_RATIO, TEST_RATIO, SPLIT_MODE, USE_LOOKUPS,
#   EXPORT_SEQUENCE=1
#
# The defaults reproduce the dataset the shipped weights were trained on. The
# Colab notebook (cell C1) sets every value explicitly to the same settings.

import os
import re
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# --- CONFIG ---
ARTS_ROOT = os.environ.get("ARTS_ROOT", str(ROOT / "data" / "arts_v2"))
# Absolute, because data.yaml records it and YOLO resolves relative paths
# against its own datasets folder. On Colab, use local disk, not Drive.
OUT_DIR = os.path.abspath(os.environ.get("ARTS_YOLO_OUT", str(ROOT / "data" / "arts_yolo")))

TOP_N_CLASSES = int(os.environ.get("TOP_N_CLASSES", "50"))
# ARTSv2 signs are TINY -- a real example is 117x18 px inside a 1920x1080 frame.
# Downscaling would erase them, so the default keeps the native resolution.
MAX_SIDE = int(os.environ.get("MAX_SIDE", "1920"))   # 0 = never resize
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.15"))
TEST_RATIO = float(os.environ.get("TEST_RATIO", "0.15"))
SEED = 42
EXPORT_SEQUENCE = os.environ.get("EXPORT_SEQUENCE", "1") == "1"
# "geographic" = leak-free split we build; "official" = the authors' ImageSets split
SPLIT_MODE = os.environ.get("SPLIT_MODE", "geographic").strip().lower()
# ARTSv2 ships ImageSets/Layout/lookups.txt mapping sign-catalog variants onto
# the 171 canonical labels, with UNKNOWN for codes the authors excluded.
USE_LOOKUPS = os.environ.get("USE_LOOKUPS", "1") == "1"

# Tag-name hints for auto-detection (substring match, case-insensitive).
# Confirmed against a real ARTSv2 annotation, which looks like this:
#   <Location><Latitude>..</Latitude><Longitude>..</Longitude>
#             <Altitude>..</Altitude><Cam_DOM>..</Cam_DOM></Location>
#   <object><id>D3-1_44.1679_-73.2520</id><name>D3-1</name>
#           <location><latitude>..</latitude><longitude>..</longitude></location>
#           <bndbox>..</bndbox>
#           <attributes><assembly>False</assembly><side>Right</side></attributes></object>
# Cam_DOM is the camera's direction of motion, i.e. the heading.
HINTS = {
    "camera_lat": ["camera_lat", "cam_lat", "image_lat", "gps_lat", "latitude", "lat"],
    "camera_lon": ["camera_lon", "camera_lng", "cam_lon", "image_lon", "gps_lon", "longitude", "lon", "lng"],
    "heading":    ["cam_dom", "_dom", "heading", "bearing", "yaw", "course", "direction"],
    "sign_lat":   ["sign_lat", "object_lat", "gps_lat", "latitude", "lat"],
    "sign_lon":   ["sign_lon", "sign_lng", "object_lon", "gps_lon", "longitude", "lon", "lng"],
    "sign_id":    ["sign_id", "unique_id", "object_id", "track_id", "uid", "id"],
    "side":       ["side"],
    "assembly":   ["assembl"],
}


def is_lat(value):
    return value is not None and -90.0 <= value <= 90.0


def is_lon(value):
    return value is not None and -180.0 <= value <= 180.0


def as_float(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def collect_fields(node, skip_tags=()):
    """Flatten a node into {lowercased tag: text}, RECURSING through container
    tags. ARTSv2 nests its GPS two levels down (<Location><Latitude>, and
    <object><location><latitude>), so a direct-children-only scan finds nothing.

    skip_tags prunes whole subtrees -- at image level we must skip <object>,
    otherwise a sign's latitude would be mistaken for the camera's."""
    out = {}
    skip = {t.lower() for t in skip_tags}

    def walk(current):
        for key, value in (current.attrib or {}).items():
            out.setdefault(key.lower(), value)
        for child in current:
            tag = (child.tag or "").lower()
            if tag in skip:
                continue
            if len(child) == 0:
                if child.text and child.text.strip():
                    out.setdefault(tag, child.text.strip())
            else:
                walk(child)

    walk(node)
    return out


def pick_tag(field_names, hint_list, exclude=()):
    """First hint that matches an actual tag name present in the XML."""
    for hint in hint_list:
        for name in field_names:
            if name in exclude:
                continue
            if hint in name:
                return name
    return None


def find_pairs(root_dir):
    pairs = []
    jpeg_dir = os.path.join(root_dir, "JPEGImages")
    ann_dir = os.path.join(root_dir, "Annotations")
    search_dirs = []
    if os.path.isdir(jpeg_dir) and os.path.isdir(ann_dir):
        search_dirs.append((ann_dir, jpeg_dir))
    else:
        search_dirs = None

    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

    if search_dirs:
        ann_dir, jpeg_dir = search_dirs[0]
        for name in sorted(os.listdir(ann_dir)):
            if not name.lower().endswith(".xml"):
                continue
            stem = os.path.splitext(name)[0]
            for ext in exts:
                img = os.path.join(jpeg_dir, stem + ext)
                if os.path.isfile(img):
                    pairs.append((img, os.path.join(ann_dir, name)))
                    break
        return pairs

    for dirpath, _, filenames in os.walk(root_dir):
        for name in sorted(filenames):
            if not name.lower().endswith(".xml"):
                continue
            stem = os.path.splitext(name)[0]
            for ext in exts:
                img = os.path.join(dirpath, stem + ext)
                if os.path.isfile(img):
                    pairs.append((img, os.path.join(dirpath, name)))
                    break
    return pairs


print("ARTS_ROOT =", ARTS_ROOT)
print("OUT_DIR   =", OUT_DIR)

if not os.path.isdir(ARTS_ROOT):
    raise SystemExit(
        f"ARTS_ROOT not found: {ARTS_ROOT}\n"
        "Download ARTSv2 (challenging_dom) and point ARTS_ROOT at it, or put it in\n"
        "data/arts_v2 (see scripts/01_download_arts_notes.md)."
    )

pairs = find_pairs(ARTS_ROOT)
print("Found image/XML pairs:", len(pairs))
if not pairs:
    raise SystemExit(
        "No image/XML pairs found. Check the layout under ARTS_ROOT -- either\n"
        "  JPEGImages/ + Annotations/   or   */frame.jpg next to */frame.xml"
    )

# ---------------------------------------------------------------- STEP 1
# Auto-detect the real tag names by sampling XMLs.
print("\n--- detecting XML field names ---")
image_field_names = Counter()
object_field_names = Counter()
sample = pairs[: min(200, len(pairs))]

for _, xml_path in sample:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        continue
    for key in collect_fields(root, skip_tags=("object",)):
        image_field_names[key] += 1
    for obj in root.findall(".//object"):
        for key in collect_fields(obj):
            object_field_names[key] += 1

print("image-level tags:", ", ".join(sorted(image_field_names)) or "(none)")
print("object-level tags:", ", ".join(sorted(object_field_names)) or "(none)")

TAG = {}
TAG["camera_lat"] = pick_tag(image_field_names, HINTS["camera_lat"])
TAG["camera_lon"] = pick_tag(image_field_names, HINTS["camera_lon"], exclude={TAG["camera_lat"]})
TAG["heading"] = pick_tag(image_field_names, HINTS["heading"])
TAG["sign_lat"] = pick_tag(object_field_names, HINTS["sign_lat"])
TAG["sign_lon"] = pick_tag(object_field_names, HINTS["sign_lon"], exclude={TAG["sign_lat"]})
TAG["sign_id"] = pick_tag(object_field_names, HINTS["sign_id"])
TAG["side"] = pick_tag(object_field_names, HINTS["side"])
TAG["assembly"] = pick_tag(object_field_names, HINTS["assembly"])

print("\nresolved mapping (VERIFY THIS against one real XML):")
for key in ("camera_lat", "camera_lon", "heading", "sign_lat", "sign_lon", "sign_id", "side", "assembly"):
    print(f"  {key:12s} -> {TAG[key]}")
if not TAG["camera_lat"] or not TAG["camera_lon"]:
    print("\n  WARNING: no camera GPS tag found. The map will have no coordinates.")
    print("  Open one XML, find the GPS field, and add its name to HINTS above.")
if not TAG["sign_lat"]:
    print("  NOTE: no per-sign GPS tag found -- geo-localization evaluation will be skipped.")

# ---------------------------------------------------------------- STEP 1b
# Load the authors' class normalisation table, if present.
LOOKUPS = {}
if USE_LOOKUPS:
    for folder in ("Layout", "Main"):
        lookup_path = os.path.join(ARTS_ROOT, "ImageSets", folder, "lookups.txt")
        if os.path.isfile(lookup_path):
            with open(lookup_path, encoding="utf-8") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 2:
                        LOOKUPS[parts[0]] = parts[1]
            print(f"\nloaded {len(LOOKUPS)} class mappings from ImageSets/{folder}/lookups.txt")
            unknown = sum(1 for v in LOOKUPS.values() if v.upper() == "UNKNOWN")
            print(f"  {unknown} codes map to UNKNOWN and will be skipped")
            break


def normalise_class(name):
    """Map a raw sign code onto its canonical label. Returns None to skip."""
    if not LOOKUPS:
        return name
    mapped = LOOKUPS.get(name, name)
    if mapped.upper() == "UNKNOWN":
        return None
    return mapped


# ---------------------------------------------------------------- STEP 2
# Parse everything once.
print("\n--- parsing annotations ---")
frames = []
class_counter = Counter()
parse_errors = 0
skipped_unknown = 0
remapped = 0

for img_path, xml_path in pairs:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        parse_errors += 1
        continue

    size = root.find("size")
    width = height = None
    if size is not None:
        width = as_float(size.findtext("width"))
        height = as_float(size.findtext("height"))
    if not width or not height:
        try:
            with Image.open(img_path) as im:
                width, height = float(im.width), float(im.height)
        except OSError:
            parse_errors += 1
            continue
    if width <= 0 or height <= 0:
        continue

    img_fields = collect_fields(root, skip_tags=("object",))
    camera_lat = as_float(img_fields.get(TAG["camera_lat"])) if TAG["camera_lat"] else None
    camera_lon = as_float(img_fields.get(TAG["camera_lon"])) if TAG["camera_lon"] else None
    heading = as_float(img_fields.get(TAG["heading"])) if TAG["heading"] else None
    if not is_lat(camera_lat):
        camera_lat = None
    if not is_lon(camera_lon):
        camera_lon = None

    objects = []
    for obj in root.findall(".//object"):
        raw_name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not raw_name or box is None:
            continue
        name = normalise_class(raw_name)
        if name is None:
            skipped_unknown += 1
            continue
        if name != raw_name:
            remapped += 1
        xmin = as_float(box.findtext("xmin"))
        ymin = as_float(box.findtext("ymin"))
        xmax = as_float(box.findtext("xmax"))
        ymax = as_float(box.findtext("ymax"))
        if None in (xmin, ymin, xmax, ymax):
            continue
        xmin = max(0.0, min(xmin, width - 1))
        xmax = max(0.0, min(xmax, width - 1))
        ymin = max(0.0, min(ymin, height - 1))
        ymax = max(0.0, min(ymax, height - 1))
        if xmax <= xmin or ymax <= ymin:
            continue

        obj_fields = collect_fields(obj)
        sign_lat = as_float(obj_fields.get(TAG["sign_lat"])) if TAG["sign_lat"] else None
        sign_lon = as_float(obj_fields.get(TAG["sign_lon"])) if TAG["sign_lon"] else None
        objects.append({
            "name": name,
            "bbox": [xmin, ymin, xmax, ymax],
            "sign_lat": sign_lat if is_lat(sign_lat) else None,
            "sign_lon": sign_lon if is_lon(sign_lon) else None,
            "sign_id": obj_fields.get(TAG["sign_id"]) if TAG["sign_id"] else None,
            "side": obj_fields.get(TAG["side"]) if TAG["side"] else None,
            "assembly": obj_fields.get(TAG["assembly"]) if TAG["assembly"] else None,
        })
        class_counter[name] += 1

    if not objects:
        continue

    frames.append({
        "img_path": img_path,
        "width": width,
        "height": height,
        "camera_lat": camera_lat,
        "camera_lon": camera_lon,
        "heading": heading,
        "objects": objects,
    })

if remapped:
    print(f"annotations remapped to a canonical label: {remapped}")
if skipped_unknown:
    print(f"annotations skipped as UNKNOWN by lookups.txt: {skipped_unknown}")
print("frames with at least one annotation:", len(frames))
print("total annotations:", sum(class_counter.values()))
print("distinct classes:", len(class_counter))
if parse_errors:
    print("unreadable files skipped:", parse_errors)
if not frames:
    raise SystemExit("Nothing parsed. Check the <object>/<bndbox>/<name> tags in the XML.")

# ---------------------------------------------------------------- STEP 3
# Assign train/val/test.
#
# SPLIT_MODE=official   reproduce the authors' split from ImageSets/, so numbers
#                       are comparable with published results
# SPLIT_MODE=geographic (default) build a geographically disjoint split, so no
#                       physical sign is both trained on and evaluated on
#
# Comparing the two shows how much the official split's leakage inflates scores.
print("\n--- assigning splits ---")
rel_dirs = {os.path.relpath(os.path.dirname(f["img_path"]), ARTS_ROOT) for f in frames}
has_gps = sum(1 for f in frames if f["camera_lat"] is not None)

if SPLIT_MODE == "official":
    official = {}
    for split, fname in (("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")):
        for folder in ("Main", "Layout"):
            path = os.path.join(ARTS_ROOT, "ImageSets", folder, fname)
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        stem = line.strip().split()[0] if line.strip() else ""
                        if stem:
                            official[stem] = split
                break
    if not official:
        raise SystemExit("SPLIT_MODE=official but no ImageSets/{Main,Layout}/train.txt found.")

    strategy = "official ARTSv2 split from ImageSets/"
    missing = 0
    for f in frames:
        stem = os.path.splitext(os.path.basename(f["img_path"]))[0]
        split = official.get(stem)
        if split is None:
            missing += 1
            f["drop"] = True
            f["group"] = "__unlisted__"
        else:
            f["group"] = f"{split}::{stem}"
    print(f"listed in ImageSets: {len(frames) - missing}/{len(frames)}"
          + (f"   ({missing} unlisted, dropped)" if missing else ""))

elif len(rel_dirs) > 3:
    strategy = "directory (one group per sequence folder)"
    for f in frames:
        f["group"] = os.path.relpath(os.path.dirname(f["img_path"]), ARTS_ROOT)
elif has_gps > len(frames) * 0.8:
    # ~0.005 deg is roughly a 500 m tile: a true geographic hold-out
    strategy = "geographic tile (~500 m), because all images share one folder"
    for f in frames:
        if f["camera_lat"] is None:
            f["group"] = "no_gps"
        else:
            f["group"] = f"{round(f['camera_lat'] / 0.005)}_{round(f['camera_lon'] / 0.005)}"
else:
    strategy = "filename prefix (digits stripped)"
    for f in frames:
        stem = os.path.splitext(os.path.basename(f["img_path"]))[0]
        prefix = re.sub(r"[\d]+$", "", stem).rstrip("_-") or "all"
        f["group"] = prefix

groups = defaultdict(list)
for f in frames:
    groups[f["group"]].append(f)
group_names = sorted(groups)
print(f"strategy: {strategy}")
print(f"groups: {len(group_names)}  (median size {sorted(len(v) for v in groups.values())[len(groups)//2]})")

if len(group_names) < 5:
    print("  WARNING: very few groups -- the split will be coarse. Inspect the layout.")

split_of = {}
if SPLIT_MODE == "official":
    for name in group_names:
        split_of[name] = name.split("::")[0] if "::" in name else "train"
else:
    random.seed(SEED)
    random.shuffle(group_names)
    n_test = max(1, int(len(group_names) * TEST_RATIO))
    n_val = max(1, int(len(group_names) * VAL_RATIO))
    for i, name in enumerate(group_names):
        if i < n_test:
            split_of[name] = "test"
        elif i < n_test + n_val:
            split_of[name] = "val"
        else:
            split_of[name] = "train"

# ---- leakage guard -------------------------------------------------------
# Grouping alone is not quite enough. ARTSv2 gives every physical sign a stable
# id (class + its own lat/lon), and a sign photographed near a tile boundary can
# land in two groups. Any training frame that shows a sign also present in val or
# test is dropped, so no physical sign is ever both trained on and evaluated on.
ids_by_split = defaultdict(set)
for f in frames:
    for o in f["objects"]:
        if o["sign_id"]:
            ids_by_split[split_of[f["group"]]].add(o["sign_id"])

test_ids = ids_by_split["test"]
eval_ids = test_ids | ids_by_split["val"]

shared = len(ids_by_split["train"] & eval_ids)
print(f"physical signs appearing in both train and val/test: {shared}")

if SPLIT_MODE == "official":
    # Reproducing the authors' split faithfully means NOT altering it. Report
    # the contamination instead, so the numbers can be read with it in mind.
    contaminated = sum(
        1 for f in frames
        if split_of[f["group"]] in ("val", "test")
        and {o["sign_id"] for o in f["objects"] if o["sign_id"]} & ids_by_split["train"]
    )
    n_eval = sum(1 for f in frames if split_of[f["group"]] in ("val", "test"))
    if n_eval:
        print(f"leakage guard: DISABLED (official split kept intact)")
        print(f"  {contaminated}/{n_eval} eval frames ({100*contaminated/n_eval:.1f}%) contain a "
              f"physical sign that also appears in train")
        print("  -> treat metrics from this split as optimistic")
else:
    dropped = Counter()
    for f in frames:
        split = split_of[f["group"]]
        ids = {o["sign_id"] for o in f["objects"] if o["sign_id"]}
        if split == "train" and ids & eval_ids:
            f["drop"] = True
            dropped["train"] += 1
        elif split == "val" and ids & test_ids:
            f["drop"] = True
            dropped["val"] += 1

    if dropped:
        print(f"leakage guard: dropped {dict(dropped)} frames that shared a physical "
              f"sign with an evaluation split")
    else:
        print("leakage guard: no shared physical signs across splits")

# ---------------------------------------------------------------- STEP 4
# Keep the top-N classes and write the dataset.
keep_classes = [name for name, _ in class_counter.most_common(TOP_N_CLASSES)]
class_to_id = {name: i for i, name in enumerate(keep_classes)}
kept = sum(class_counter[c] for c in keep_classes)
print(f"\nkeeping top {len(keep_classes)} classes "
      f"({kept}/{sum(class_counter.values())} = {100*kept/sum(class_counter.values()):.1f}% of annotations)")

for split in ("train", "val", "test"):
    os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "meta"), exist_ok=True)

print("\n--- writing dataset (this is the slow part) ---")
written = Counter()
frames_meta = {}
per_split_classes = defaultdict(Counter)

for index, f in enumerate(frames):
    if f.get("drop"):
        continue
    split = split_of[f["group"]]
    lines = []
    kept_objects = []
    for obj in f["objects"]:
        if obj["name"] not in class_to_id:
            continue
        xmin, ymin, xmax, ymax = obj["bbox"]
        cx = ((xmin + xmax) / 2.0) / f["width"]
        cy = ((ymin + ymax) / 2.0) / f["height"]
        bw = (xmax - xmin) / f["width"]
        bh = (ymax - ymin) / f["height"]
        lines.append(f"{class_to_id[obj['name']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        kept_objects.append(obj)
        per_split_classes[split][obj["name"]] += 1

    if not lines:
        continue

    group_tag = re.sub(r"[^A-Za-z0-9]+", "-", f["group"])[:40]
    stem = os.path.splitext(os.path.basename(f["img_path"]))[0]
    out_stem = f"{group_tag}__{stem}"
    out_img = os.path.join(OUT_DIR, "images", split, out_stem + ".jpg")

    try:
        with Image.open(f["img_path"]) as im:
            im = im.convert("RGB")
            scale = 1.0
            if MAX_SIDE and max(im.width, im.height) > MAX_SIDE:
                scale = MAX_SIDE / max(im.width, im.height)
                im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR)
            im.save(out_img, "JPEG", quality=90)
    except OSError:
        continue

    with open(os.path.join(OUT_DIR, "labels", split, out_stem + ".txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    frames_meta[out_stem] = {
        "source": f["img_path"],
        "split": split,
        "group": f["group"],
        "camera_lat": f["camera_lat"],
        "camera_lon": f["camera_lon"],
        "heading": f["heading"],
        "scale": scale,
        "objects": [{
            "name": o["name"],
            # bbox rescaled to match the written image
            "bbox": [round(v * scale, 2) for v in o["bbox"]],
            "sign_lat": o["sign_lat"],
            "sign_lon": o["sign_lon"],
            "sign_id": o["sign_id"],
            "side": o["side"],
            "assembly": o["assembly"],
        } for o in kept_objects],
    }
    written[split] += 1

    if index % 1000 == 0:
        print(f"  {index}/{len(frames)} ...")

print("written:", dict(written))
if written["train"] == 0 or written["val"] == 0:
    raise SystemExit("A split came out empty. Lower VAL_RATIO/TEST_RATIO or check grouping.")

# classes.txt / data.yaml / frames_meta.json
with open(os.path.join(OUT_DIR, "classes.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(keep_classes) + "\n")

with open(os.path.join(OUT_DIR, "data.yaml"), "w", encoding="utf-8") as fh:
    fh.write(f"path: {OUT_DIR}\n")
    fh.write("train: images/train\n")
    fh.write("val: images/val\n")
    fh.write("test: images/test\n")
    fh.write("names:\n")
    for name in keep_classes:
        safe = name.replace(":", "-")
        fh.write(f"  {class_to_id[name]}: {safe}\n")

with open(os.path.join(OUT_DIR, "meta", "frames_meta.json"), "w", encoding="utf-8") as fh:
    json.dump({"tags": TAG, "split_strategy": strategy, "frames": frames_meta}, fh)

summary = {
    "images": dict(written),
    "classes": keep_classes,
    "class_counts": {c: class_counter[c] for c in keep_classes},
    "per_split_class_counts": {s: dict(c) for s, c in per_split_classes.items()},
    "groups": len(group_names),
    "split_strategy": strategy,
    "split_mode": SPLIT_MODE,
    "tags": TAG,
    "frames_with_camera_gps": has_gps,
    "frames_with_sign_gps": sum(
        1 for m in frames_meta.values() if any(o["sign_lat"] for o in m["objects"])
    ),
}
with open(os.path.join(OUT_DIR, "meta", "summary.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)

# ---------------------------------------------------------------- STEP 5
# Export one held-out test sequence as a runnable demo folder.
if EXPORT_SEQUENCE:
    # The demo needs a run of frames along one stretch of road, so the tracker
    # has something to track. In official mode each group is a single image, so
    # regroup the test frames geographically just for this export.
    if SPLIT_MODE == "official":
        geo = defaultdict(list)
        for stem, meta in frames_meta.items():
            if meta["split"] != "test" or meta["camera_lat"] is None:
                continue
            tile = f"{round(meta['camera_lat'] / 0.005)}_{round(meta['camera_lon'] / 0.005)}"
            geo[tile].append(stem)
        best_tile = max(geo, key=lambda t: len(geo[t])) if geo else None
        demo_stems = set(geo.get(best_tile, []))
        best_group = best_tile
    else:
        test_groups = [g for g in group_names if split_of[g] == "test"]
        best_group = max(test_groups, key=lambda g: len(groups[g])) if test_groups else None
        demo_stems = None

    if best_group:
        demo_dir = os.path.join(OUT_DIR, "demo_sequence")
        os.makedirs(demo_dir, exist_ok=True)
        demo_meta = {}
        count = 0
        for stem, meta in frames_meta.items():
            if demo_stems is not None:
                if stem not in demo_stems:
                    continue
            elif meta["group"] != best_group:
                continue
            src = os.path.join(OUT_DIR, "images", meta["split"], stem + ".jpg")
            if not os.path.isfile(src):
                continue
            with open(src, "rb") as rf, open(os.path.join(demo_dir, stem + ".jpg"), "wb") as wf:
                wf.write(rf.read())
            demo_meta[stem] = meta
            count += 1
        with open(os.path.join(demo_dir, "frames_meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"tags": TAG, "frames": demo_meta}, fh)
        print(f"\ndemo sequence: {count} frames from group '{best_group}' -> {demo_dir}")

print("\ndata.yaml  ->", os.path.join(OUT_DIR, "data.yaml"))
print("summary    ->", os.path.join(OUT_DIR, "meta", "summary.json"))
print("Done.")
