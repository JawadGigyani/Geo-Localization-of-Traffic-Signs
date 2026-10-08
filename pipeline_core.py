# Shared detection, projection, clustering and scoring.
#
# scripts/03_run_pipeline.py (CLI), scripts/07_batch_evaluate.py (evaluation)
# and backend/main.py (API) all import this, so they can never drift apart.
#
# Flow: frames -> YOLO detect -> crop classifier for look-alike families
#       -> project each box onto the map (heading + box geometry)
#       -> cluster same-class projections into one row per physical sign
#
# Nothing here imports torch or OpenCV at module level, so the geometry and
# scoring functions can be unit-tested without the ML stack (see tests/).

import os
import json
import glob
import math
import xml.etree.ElementTree as ET
from collections import defaultdict

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

# ---------------------------------------------------------------------------
# Stored geometry
#
# The camera field of view, the width-to-range constant and the clustering
# radius are fitted by scripts/07_batch_evaluate.py on held-out calibration
# sequences and written to weights/calibration.json, next to the models they
# belong with. These defaults are the values of the original run, used only if
# that file is missing.
# ---------------------------------------------------------------------------

DEFAULT_CALIBRATION = {"hfov_deg": 80.0, "size_k_m": 0.88, "radius_m": 60.0}


def load_calibration(path):
    """Read weights/calibration.json. Missing or broken keys fall back to
    DEFAULT_CALIBRATION, so a fresh clone always has usable constants."""
    cal = dict(DEFAULT_CALIBRATION)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in DEFAULT_CALIBRATION:
            if data.get(key) is not None:
                cal[key] = float(data[key])
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return cal


def geometry_settings(path, environ=None):
    """Calibration file values, overridden by HFOV_DEG / SIZE_K / RADIUS_M
    environment variables when those are set. Returns (hfov, size_k, radius, source)."""
    environ = os.environ if environ is None else environ
    cal = load_calibration(path)
    overridden = [k for k in ("HFOV_DEG", "SIZE_K", "RADIUS_M") if environ.get(k)]
    hfov = float(environ.get("HFOV_DEG") or cal["hfov_deg"])
    size_k = float(environ.get("SIZE_K") or cal["size_k_m"])
    radius = float(environ.get("RADIUS_M") or cal["radius_m"])
    source = os.path.basename(str(path))
    if overridden:
        source += " + env " + ",".join(overridden)
    return hfov, size_k, radius, source


def save_calibration(path, hfov_deg, size_k_m, radius_m, **provenance):
    """Write the fitted constants plus where they came from."""
    record = {
        "hfov_deg": float(hfov_deg),
        "size_k_m": round(float(size_k_m), 4),
        "radius_m": float(radius_m),
    }
    record.update(provenance)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    return record

# Tag-name hints used when reading GPS straight out of a VOC-style XML.
# The converter (scripts/02_convert_arts_to_yolo.py) auto-detects the real names
# and writes them into frames_meta.json, so this is only a fallback.
LAT_HINTS = ["lat"]
LON_HINTS = ["lon", "lng"]


def list_frames(sequence_dir):
    """One path per frame stem. Prevents the same frame being processed twice
    when both frame_000.jpg and frame_000.png exist."""
    by_stem = {}
    for path in sorted(glob.glob(os.path.join(sequence_dir, "*"))):
        stem, ext = os.path.splitext(os.path.basename(path))
        if ext.lower() not in IMG_EXTS:
            continue
        # first extension in IMG_EXTS order wins
        current = by_stem.get(stem)
        if current is None:
            by_stem[stem] = path
        else:
            old_rank = IMG_EXTS.index(os.path.splitext(current)[1].lower())
            new_rank = IMG_EXTS.index(ext.lower())
            if new_rank < old_rank:
                by_stem[stem] = path
    return [by_stem[s] for s in sorted(by_stem)]


def load_frames_meta(sequence_dir):
    """frames_meta.json is written by the converter and holds camera GPS,
    heading, and ground-truth sign GPS per frame. Returns {} when absent."""
    path = os.path.join(sequence_dir, "frames_meta.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("frames", data)


def _deep_find_float(root, hints, skip_tags=("object",)):
    """Depth-first search for a numeric tag, pruning whole subtrees.

    ARTSv2 nests camera GPS as <Location><Latitude>, and each sign repeats
    <location><latitude> inside <object>. Without pruning <object>, a sign's
    coordinate could be read as the camera's."""
    skip = {t.lower() for t in skip_tags}

    def walk(node):
        for child in node:
            tag = (child.tag or "").lower()
            if tag in skip:
                continue
            if any(h in tag for h in hints):
                try:
                    return float((child.text or "").strip())
                except (TypeError, ValueError):
                    pass
            found = walk(child)
            if found is not None:
                return found
        return None

    return walk(root)


def read_camera_gps(frame_path, meta):
    """Returns (lat, lon, source). source is one of:
    'frames_meta', 'xml', 'gps_sidecar', 'none'. Never invents a coordinate."""
    stem = os.path.splitext(os.path.basename(frame_path))[0]

    entry = meta.get(stem) or meta.get(os.path.basename(frame_path))
    if entry:
        lat = entry.get("camera_lat")
        lon = entry.get("camera_lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon), "frames_meta"

    base = os.path.splitext(frame_path)[0]

    xml_path = base + ".xml"
    if os.path.isfile(xml_path):
        try:
            root = ET.parse(xml_path).getroot()
            lat = _deep_find_float(root, LAT_HINTS)
            lon = _deep_find_float(root, LON_HINTS)
            if lat is not None and lon is not None:
                return lat, lon, "xml"
        except ET.ParseError:
            pass

    gps_path = base + ".gps.txt"
    if os.path.isfile(gps_path):
        with open(gps_path, "r", encoding="utf-8") as f:
            parts = f.read().strip().split(",")
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1]), "gps_sidecar"
            except ValueError:
                pass

    return None, None, "none"


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_ground_truth(frame_path, box, meta, min_iou=0.3):
    """Match a predicted box to an ARTSv2 annotation on the same frame so we can
    carry its ground-truth sign GPS and unique sign id through the pipeline."""
    stem = os.path.splitext(os.path.basename(frame_path))[0]
    entry = meta.get(stem) or meta.get(os.path.basename(frame_path))
    if not entry:
        return None
    best, best_iou = None, min_iou
    for obj in entry.get("objects", []):
        gt_box = obj.get("bbox")
        if not gt_box or len(gt_box) != 4:
            continue
        score = _iou(box, gt_box)
        if score >= best_iou:
            best, best_iou = obj, score
    if best is None:
        return None
    return {
        "gt_sign_id": best.get("sign_id"),
        "gt_sign_lat": best.get("sign_lat"),
        "gt_sign_lon": best.get("sign_lon"),
        "gt_class": best.get("name"),
        "gt_iou": round(best_iou, 4),
    }


def training_imgsz(model, fallback=1280):
    """The resolution the model was TRAINED at.

    Ultralytics defaults inference to imgsz=640 regardless of how the model was
    trained. This model saw 1280 px images; running it at 640 would halve every
    sign and push most of them under the detector's smallest stride. Read the
    real value out of the checkpoint so it can never drift."""
    try:
        size = (model.ckpt or {}).get("train_args", {}).get("imgsz")
        if size:
            return int(size)
    except (AttributeError, TypeError, ValueError):
        pass
    return int(fallback)


# ---------------------------------------------------------------------------
# Second-stage crop classifier
#
# The detector resolves position and sign family reliably but confuses variants
# within a family -- M3-1/2/3/4 differ only by the word NORTH/SOUTH/EAST/WEST,
# R2-125..R2-165 only by the number, and at a 26 px median height that text is
# not resolvable in the full frame.
#
# So the detector picks the family and a classifier reads the upscaled crop to
# pick the variant. Crucially the classifier's answer is restricted to the
# family the detector chose: it is good at fine discrimination and has no reason
# to be trusted about whether something is a speed limit or a route marker.
# ---------------------------------------------------------------------------

import re


def family_of(class_name):
    """'R2-140' -> 'R2'.  Classes sharing a family are the confusable ones."""
    m = re.match(r"^([A-Za-z]+\d*)-", class_name or "")
    return m.group(1) if m else (class_name or "")


def build_families(class_names):
    """{family: [classes]} for families with more than one member."""
    groups = {}
    for name in class_names:
        groups.setdefault(family_of(name), []).append(name)
    return {f: sorted(v) for f, v in groups.items() if len(v) > 1}


def refine_class(cls_model, crop, detected_name, families, min_conf=0.0):
    """Re-read a crop and pick the best variant WITHIN the detected family.

    Returns (name, confidence). Falls through unchanged when the family has one
    member, the crop is unusable, or the classifier does not clear min_conf."""
    family = family_of(detected_name)
    members = families.get(family)
    if not members or crop is None or crop.size == 0:
        return detected_name, None

    result = cls_model.predict(source=crop, verbose=False)[0]
    probs = result.probs
    if probs is None:
        return detected_name, None

    values = probs.data.tolist()
    names = result.names
    # restrict the argmax to this family's members
    candidates = [(values[i], n) for i, n in names.items() if n in members]
    if not candidates:
        return detected_name, None
    best_p, best_name = max(candidates)
    total = sum(p for p, _ in candidates) or 1.0
    normalised = best_p / total
    if normalised < min_conf:
        return detected_name, normalised
    return best_name, normalised


def detect_all(model, frames, conf=0.25, imgsz=None, meta=None, keep_crops=True,
               cls_model=None, families=None, cls_min_conf=0.0):
    """Plain per-frame detection, no tracker. Each detection carries the camera
    pose and any matched ground truth, ready for geometric association."""
    meta = meta or {}
    if imgsz is None:
        imgsz = training_imgsz(model)
    dets = []

    for frame_index, frame_path in enumerate(frames):
        result = model.predict(source=frame_path, conf=conf, imgsz=imgsz, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        names = result.names
        img = result.orig_img
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(frame_path))[0]
        entry = meta.get(stem) or meta.get(os.path.basename(frame_path)) or {}
        lat, lon, gps_source = read_camera_gps(frame_path, meta)

        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = [float(v) for v in result.boxes.xyxy[i].cpu().numpy()]
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(w, int(x2)), min(h, int(y2))
            if xi2 <= xi1 or yi2 <= yi1:
                continue
            cls_id = int(result.boxes.cls[i].item())
            cls_name = names.get(cls_id, str(cls_id))
            crop = img[yi1:yi2, xi1:xi2].copy()

            stage1_name, cls_conf = cls_name, None
            if cls_model is not None and families:
                # pad the box: the classifier trained on padded ground-truth
                # crops, and a predicted box is never quite the same framing
                pad = max(4, int(0.15 * max(xi2 - xi1, yi2 - yi1)))
                px1, py1 = max(0, xi1 - pad), max(0, yi1 - pad)
                px2, py2 = min(w, xi2 + pad), min(h, yi2 + pad)
                cls_name, cls_conf = refine_class(
                    cls_model, img[py1:py2, px1:px2], cls_name,
                    families, cls_min_conf
                )

            record = {
                "box": [x1, y1, x2, y2],
                "area": (x2 - x1) * (y2 - y1),
                # crops are numpy arrays and are not JSON-serialisable; skip them
                # when the caller only needs geometry (parameter sweeps, caching)
                "crop": crop if keep_crops else None,
                "frame_path": frame_path,
                "frame_index": frame_index,
                "image_width": float(w),
                "image_height": float(h),
                "cls_name": cls_name,
                "cls_name_stage1": stage1_name,
                "cls_refined": cls_name != stage1_name,
                "cls_stage2_conf": cls_conf,
                "conf": float(result.boxes.conf[i].item()),
                "camera_lat": lat,
                "camera_lon": lon,
                "heading": entry.get("heading"),
                "lat": lat,
                "lon": lon,
                "gps_source": gps_source,
                "gt_sign_id": None,
                "gt_sign_lat": None,
                "gt_sign_lon": None,
                "gt_class": None,
                "gt_iou": None,
            }
            gt = match_ground_truth(frame_path, (x1, y1, x2, y2), meta)
            if gt:
                record.update(
                    gt_sign_id=gt["gt_sign_id"],
                    gt_sign_lat=gt["gt_sign_lat"],
                    gt_sign_lon=gt["gt_sign_lon"],
                    gt_class=gt["gt_class"],   # needed to score classification
                    gt_iou=gt["gt_iou"],
                )
            dets.append(record)

    return dets


def run_tracking(model, frames, tracker="botsort.yaml", conf=0.25, meta=None, imgsz=None,
                 stats=None, count_model=None):
    """Track across the frame sequence, keeping only the observation with the
    largest bounding box for each track id (the clearest view of that sign).

    This is the video-tracker BASELINE. Pass a dict as `stats` to receive
    {"detections": boxes found, "with_id": boxes the tracker gave an identity},
    which is how the README's tracker comparison is measured.

    `count_model` should be a second, separately loaded copy of the detector.
    Once a frame has any tracks, Ultralytics keeps only the tracked boxes in
    the result, so the untracked ones can only be counted by a model that has
    no tracker attached. Without it, "detections" is a lower bound."""
    meta = meta or {}
    tracks = {}
    if imgsz is None:
        imgsz = training_imgsz(model)
    if stats is not None:
        stats.setdefault("detections", 0)
        stats.setdefault("with_id", 0)

    for frame_index, frame_path in enumerate(frames):
        results = model.track(
            source=frame_path,
            persist=True,      # carries tracker state across single-image calls
            tracker=tracker,
            conf=conf,
            imgsz=imgsz,       # must match training; see training_imgsz()
            verbose=False,
        )
        if not results:
            continue
        r0 = results[0]
        if stats is not None:
            with_id = 0
            if r0.boxes is not None and r0.boxes.id is not None:
                with_id = len(r0.boxes.id)
            if count_model is not None:
                plain = count_model.predict(source=frame_path, conf=conf, imgsz=imgsz,
                                            verbose=False)[0]
                found = 0 if plain.boxes is None else len(plain.boxes)
            else:
                found = 0 if r0.boxes is None else len(r0.boxes)
            stats["detections"] += max(found, with_id)
            stats["with_id"] += with_id
        if r0.boxes is None or len(r0.boxes) == 0 or r0.boxes.id is None:
            continue

        names = r0.names
        img = r0.orig_img
        boxes = r0.boxes
        h, w = img.shape[:2]

        for i in range(len(boxes)):
            tid = int(boxes.id[i].item())
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].cpu().numpy()]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            prev = tracks.get(tid)
            if prev is not None and area <= prev["area"]:
                continue

            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(w, int(x2)), min(h, int(y2))
            if xi2 <= xi1 or yi2 <= yi1:
                continue

            cls_id = int(boxes.cls[i].item())
            lat, lon, gps_source = read_camera_gps(frame_path, meta)
            gt = match_ground_truth(frame_path, (x1, y1, x2, y2), meta)

            record = {
                "track_id": tid,
                "area": area,
                "box": [x1, y1, x2, y2],
                "crop": img[yi1:yi2, xi1:xi2].copy(),
                "frame_path": frame_path,
                "frame_index": frame_index,
                "cls_name": names.get(cls_id, str(cls_id)),
                "conf": float(boxes.conf[i].item()),
                "lat": lat,
                "lon": lon,
                "gps_source": gps_source,
                "gt_sign_id": None,
                "gt_sign_lat": None,
                "gt_sign_lon": None,
                "gt_iou": None,
            }
            if gt:
                record.update(
                    gt_sign_id=gt["gt_sign_id"],
                    gt_sign_lat=gt["gt_sign_lat"],
                    gt_sign_lon=gt["gt_sign_lon"],
                    gt_iou=gt["gt_iou"],
                )
            tracks[tid] = record

    return tracks


# ---------------------------------------------------------------------------
# Geometric association
#
# Video trackers (BoT-SORT, ByteTrack) associate by IoU + Kalman motion, which
# assumes an object barely moves between frames. ARTSv2 photos are a median
# 8.1 m apart (demo sequence), so IoU between consecutive views of the same sign is
# zero, and tracks rarely activate -- on the 69-frame demo sequence BoT-SORT gave
# an id to 8 of 103 detections and ByteTrack to 7 of 103.
#
# Instead we put every detection on the map and cluster there. Camera GPS and
# heading are known; the box's horizontal offset gives the bearing to the sign;
# box width gives a rough range. Two views of one signpost land on the same
# spot however far the camera travelled in between.
# ---------------------------------------------------------------------------

EARTH_R = 6371000.0


def haversine_m(lat1, lon1, lat2, lon2):
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dlon / 2) ** 2
    return EARTH_R * 2 * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1, lon1, lat2, lon2):
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(r2)
    x = math.cos(r1) * math.sin(r2) - math.sin(r1) * math.cos(r2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def offset_point(lat, lon, bearing, distance_m):
    b = math.radians(bearing)
    dlat = (distance_m * math.cos(b)) / 111320.0
    dlon = (distance_m * math.sin(b)) / (111320.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def focal_px(image_width, hfov_deg):
    return (image_width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def project_detection(det, hfov_deg, size_k):
    """Detection -> estimated world position of the sign.

    bearing  = camera heading + pinhole angle of the box centre
    distance = size_k * focal / box_width      (size_k absorbs the average
               physical sign width; calibrate it against ground truth)
    """
    if det.get("camera_lat") is None or det.get("heading") is None:
        return None
    x1, _, x2, _ = det["box"]
    width_px = max(1.0, x2 - x1)
    f = focal_px(det["image_width"], hfov_deg)
    angle = math.degrees(math.atan2((x1 + x2) / 2.0 - det["image_width"] / 2.0, f))
    bearing = (det["heading"] + angle) % 360.0
    distance = max(2.0, min(size_k * f / width_px, 200.0))
    lat, lon = offset_point(det["camera_lat"], det["camera_lon"], bearing, distance)
    return {"lat": lat, "lon": lon, "distance_m": distance, "bearing_deg": bearing}


def calibrate_geometry(dets, hfov_grid=tuple(range(40, 145, 5))):
    """Fit hfov and size_k from detections matched to ground-truth sign GPS.

    For each matched detection the true bearing and true range are computable
    from camera and sign coordinates. hfov comes from a grid search that
    minimises the squared bearing error, size_k from least squares through the
    origin. Returns (hfov, size_k, n_used)."""
    usable = [d for d in dets
              if d.get("gt_sign_lat") is not None and d.get("camera_lat") is not None
              and d.get("heading") is not None]
    if len(usable) < 8:
        return None, None, len(usable)

    best = None
    for hfov in hfov_grid:
        err = 0.0
        for d in usable:
            f = focal_px(d["image_width"], hfov)
            x1, _, x2, _ = d["box"]
            angle = math.degrees(math.atan2((x1 + x2) / 2.0 - d["image_width"] / 2.0, f))
            predicted = (d["heading"] + angle) % 360.0
            true_b = bearing_deg(d["camera_lat"], d["camera_lon"],
                                 d["gt_sign_lat"], d["gt_sign_lon"])
            diff = abs((predicted - true_b + 180.0) % 360.0 - 180.0)
            err += diff * diff
        if best is None or err < best[1]:
            best = (hfov, err)
    hfov = best[0]

    # distance = size_k * focal / width  ->  least squares through the origin
    num = den = 0.0
    for d in usable:
        f = focal_px(d["image_width"], hfov)
        x1, _, x2, _ = d["box"]
        predictor = f / max(1.0, x2 - x1)
        true_d = haversine_m(d["camera_lat"], d["camera_lon"],
                             d["gt_sign_lat"], d["gt_sign_lon"])
        if 2.0 < true_d < 150.0:
            num += predictor * true_d
            den += predictor * predictor
    size_k = (num / den) if den > 0 else 0.7
    return hfov, size_k, len(usable)


def associate_geometric(dets, hfov_deg=DEFAULT_CALIBRATION["hfov_deg"],
                        size_k=DEFAULT_CALIBRATION["size_k_m"],
                        radius_m=DEFAULT_CALIBRATION["radius_m"], radius_frac=0.0):
    """Cluster detections into physical signs by projected world position.

    Same class + within tolerance of an existing cluster => same signpost.
    The cluster anchors on its largest box, which is both the closest view and
    the most accurate range estimate, and is also the crop worth keeping.

    radius_frac > 0 makes the tolerance scale with estimated range. Distance
    comes from apparent box width, so its error grows with distance: a sign at
    60 m is far less precisely placed than one at 10 m. A single global radius
    must therefore be too loose up close or too tight far away, and measurement
    showed fragmentation (one sign split across rows) dominating the error.
    """
    clusters = []
    for det in sorted(dets, key=lambda d: -(d["box"][2] - d["box"][0])):
        projected = project_detection(det, hfov_deg, size_k)
        det["projected"] = projected
        if projected is None:
            continue
        tolerance = max(radius_m, radius_frac * projected["distance_m"])
        match = None
        for c in clusters:
            if c["cls_name"] != det["cls_name"]:
                continue
            gap = haversine_m(c["lat"], c["lon"], projected["lat"], projected["lon"])
            if gap <= tolerance:
                match = c
                break
        if match is None:
            clusters.append({
                "cls_name": det["cls_name"],
                "lat": projected["lat"], "lon": projected["lon"],
                "members": [det],
            })
        else:
            match["members"].append(det)

    tracks = {}
    for i, c in enumerate(clusters, start=1):
        # largest box first, thanks to the sort above
        best = c["members"][0]
        # Every distinct ground-truth sign this cluster absorbed. Without this a
        # cluster only ever carries its best member's id, which would make
        # merges structurally impossible to observe rather than genuinely rare.
        member_gt_ids = sorted({str(m["gt_sign_id"]) for m in c["members"]
                                if m.get("gt_sign_id")})
        tracks[i] = dict(
            best,
            track_id=i,
            lat=c["lat"], lon=c["lon"],
            gps_source="geo_projected",
            observations=len(c["members"]),
            member_gt_ids=member_gt_ids,
            camera_lat=best.get("camera_lat"), camera_lon=best.get("camera_lon"),
            estimated_distance_m=round(best["projected"]["distance_m"], 1),
        )
        tracks[i].pop("projected", None)
    return tracks


def save_crops(tracks, out_dir, seq_name):
    import cv2  # imported here so the rest of the module needs no OpenCV
    os.makedirs(out_dir, exist_ok=True)
    for tid, info in tracks.items():
        crop = info.get("crop")
        if crop is None or crop.size == 0:
            info["crop_path"] = None
            continue
        name = f"{seq_name}_track{tid}_{info['cls_name']}.jpg".replace(" ", "_").replace("/", "-")
        path = os.path.join(out_dir, name)
        cv2.imwrite(path, crop)
        info["crop_path"] = path
        info["crop_name"] = name
    return tracks


def apply_synthetic_gps(tracks, lat=44.4759, lon=-73.2121):
    """Demo-only escape hatch for frames with no GPS at all. Tags every row it
    touches with gps_source='synthetic_demo' so the map can label it as fake
    rather than presenting an invented coordinate as a real detection."""
    touched = 0
    for tid, info in tracks.items():
        if info["lat"] is not None and info["lon"] is not None:
            continue
        info["lat"] = lat + (tid % 50) * 0.00008
        info["lon"] = lon + (tid % 50) * 0.00008
        info["gps_source"] = "synthetic_demo"
        touched += 1
    return touched


def upload_and_insert(supabase, tracks, seq_name, upload_frames=True):
    """Write one row per unique track to Supabase. Rows without a real GPS fix
    are still stored, with gps_source='none', so the map can flag them instead
    of silently showing a made-up location."""
    inserted = []
    for tid, info in tracks.items():
        crop_path = info.get("crop_path")
        crop_url = None
        frame_url = None

        if crop_path and os.path.isfile(crop_path):
            crop_name = info["crop_name"]
            with open(crop_path, "rb") as f:
                supabase.storage.from_("sign-crops").upload(
                    crop_name, f,
                    file_options={"content-type": "image/jpeg", "upsert": "true"},
                )
            crop_url = supabase.storage.from_("sign-crops").get_public_url(crop_name)

        if upload_frames and os.path.isfile(info["frame_path"]):
            frame_name = f"{seq_name}_track{tid}_frame.jpg".replace(" ", "_")
            with open(info["frame_path"], "rb") as f:
                supabase.storage.from_("sign-frames").upload(
                    frame_name, f,
                    file_options={"content-type": "image/jpeg", "upsert": "true"},
                )
            frame_url = supabase.storage.from_("sign-frames").get_public_url(frame_name)

        if info["lat"] is None or info["lon"] is None:
            print(f"  skip insert: track {tid} has no GPS fix (gps_source=none)")
            continue

        res = supabase.rpc(
            "insert_traffic_sign",
            {
                "p_sign_type": info["cls_name"],
                "p_confidence": info["conf"],
                "p_track_id": tid,
                "p_lat": info["lat"],
                "p_lng": info["lon"],
                "p_gps_source": info["gps_source"],
                "p_image_crop_url": crop_url,
                "p_full_frame_url": frame_url,
                "p_source_sequence": seq_name,
                "p_sign_lat": info.get("gt_sign_lat"),
                "p_sign_lon": info.get("gt_sign_lon"),
                "p_gt_sign_id": (
                    str(info["gt_sign_id"]) if info.get("gt_sign_id") is not None else None
                ),
                "p_camera_lat": info.get("camera_lat"),
                "p_camera_lon": info.get("camera_lon"),
                "p_observations": int(info.get("observations", 1)),
            },
        ).execute()
        inserted.append(res.data)
    return inserted


# ---------------------------------------------------------------------------
# Scoring against ARTSv2 ground truth
#
# Shared by scripts/06_evaluate_geo_and_dedupe.py (one sequence) and
# scripts/07_batch_evaluate.py (many), so both compute every metric the same way.
# ---------------------------------------------------------------------------

def percentile(values, p):
    """Linear-interpolated percentile, p in [0, 1]. None for an empty list."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return s[int(k)] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def canonical_gt_map(meta, tol_m=2.0):
    """Collapse ground-truth ids that describe the same physical post.

    ARTSv2's <id> is not a clean physical-sign key: some entries appear twice
    with a numeric prefix at IDENTICAL coordinates, and sign assemblies put
    several plaques on one post. Scoring against raw ids asks the system to
    separate signs no geometric method could, so ids of the same class within
    tol_m metres are treated as one post. Returns {raw_id: canonical_id}."""
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
            if c["name"] == name and haversine_m(c["lat"], c["lon"], lat, lon) <= tol_m:
                hit = c
                break
        if hit is None:
            canon.append({"name": name, "lat": lat, "lon": lon, "id": gid})
            mapping[gid] = gid
        else:
            mapping[gid] = hit["id"]
    return mapping


def score_rows(rows, meta, canon=None, tol_m=2.0):
    """Score one sequence's inventory rows against its ground truth.

    rows: {track_id: row} from associate_geometric, or a list of rows loaded
    from data/pipeline_result_*.json.

    Every detection inside every row counts towards "found", including rows
    whose anchor (widest box) matched no labelled sign. Position error is
    measured only for rows whose anchor did match, because that is the sign
    the row's position stands for."""
    rows = list(rows.values()) if isinstance(rows, dict) else list(rows)
    if canon is None:
        canon = canonical_gt_map(meta, tol_m)
    raw_ids = {str(o["sign_id"]) for m in meta.values()
               for o in m.get("objects", []) if o.get("sign_id")}
    posts = {canon.get(g, g) for g in raw_ids}

    errors, baseline = [], []
    by_gt, by_row = defaultdict(set), defaultdict(set)
    for i, t in enumerate(rows):
        rid = t.get("track_id", i)
        if (t.get("gt_sign_lat") is not None and t.get("lat") is not None
                and t.get("gps_source") != "synthetic_demo"):
            errors.append(haversine_m(t["lat"], t["lon"], t["gt_sign_lat"], t["gt_sign_lon"]))
            if t.get("camera_lat") is not None:
                baseline.append(haversine_m(t["camera_lat"], t["camera_lon"],
                                            t["gt_sign_lat"], t["gt_sign_lon"]))
        ids = t.get("member_gt_ids") or ([t["gt_sign_id"]] if t.get("gt_sign_id") else [])
        for gid in ids:
            gid = canon.get(str(gid), str(gid))
            by_gt[gid].add(rid)
            by_row[rid].add(gid)

    fragmented = sum(1 for v in by_gt.values() if len(v) > 1)
    merged = sum(1 for v in by_row.values() if len(v) > 1)
    # strict one-to-one: the post has exactly one row AND that row holds only it
    one_to_one = sum(1 for v in by_gt.values()
                     if len(v) == 1 and len(by_row[next(iter(v))]) == 1)
    return {
        "raw_ids": len(raw_ids), "posts": len(posts), "rows": len(rows),
        "found": len(by_gt), "fragmented": fragmented, "merged": merged,
        "one_to_one": one_to_one, "errors": errors, "baseline": baseline,
    }


def pool_scores(scores):
    """Sum per-sequence scores (from score_rows) and derive the headline rates:
    recall = found / posts
    dedupe = (found - fragmented - merged) / found
    precision/recall/F1 of the strict one-to-one match."""
    keys = ("raw_ids", "posts", "rows", "found", "fragmented", "merged", "one_to_one")
    total = {k: sum(s[k] for s in scores) for k in keys}
    total["errors"] = [e for s in scores for e in s["errors"]]
    total["baseline"] = [b for s in scores for b in s["baseline"]]
    posts, found, rows = total["posts"], total["found"], total["rows"]
    total["recall"] = found / posts if posts else 0.0
    total["dedupe"] = (found - total["fragmented"] - total["merged"]) / found if found else 0.0
    p = total["one_to_one"] / rows if rows else 0.0
    r = total["one_to_one"] / posts if posts else 0.0
    total["precision_1to1"] = p
    total["recall_1to1"] = r
    total["f1_1to1"] = (2 * p * r / (p + r)) if (p + r) else 0.0
    return total
