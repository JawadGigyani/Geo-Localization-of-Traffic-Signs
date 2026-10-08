# Export the held-out road sequences the evaluation runs on, without Colab.
#
#   python scripts/05_export_sequences.py
#
# Run after scripts/02_convert_arts_to_yolo.py. This is notebook cell F1 as a
# script, with the same selection rules, so it rebuilds the same 30 sequences:
#   - test-split map tiles with at least MIN_FRAMES photos
#   - the MAX_SEQS largest of them (ties keep the converter's order)
#   - frames inside a sequence in driving order (by original file name)
#
# Writes:
#   data/sequences/seq_<tile>/         photos + frames_meta.json, one per sequence
#   data/demo_sequence/                the converter's demo sequence, if missing
#   results/split_manifest.csv         every photo's split and tile (file names only,
#                                      no imagery), so anyone can check they rebuilt
#                                      the same split: `git diff results/`
#   results/split_check.json           leakage between splits, per sign id and per
#                                      physical post
#
# Env:
#   ARTS_YOLO_OUT=data/arts_yolo   output of the converter
#   SEQ_ROOT=data/sequences
#   MIN_FRAMES=20  MAX_SEQS=30
#   CANON_M=2.0                    same-class ids within this distance = one post
#   RESULTS_DIR=results

import os
import sys
import csv
import json
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline_core

CANON_M = float(os.environ.get("CANON_M", "2.0"))
DATASET = Path(os.environ.get("ARTS_YOLO_OUT", ROOT / "data" / "arts_yolo"))
SEQ_ROOT = Path(os.environ.get("SEQ_ROOT", ROOT / "data" / "sequences"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
MIN_FRAMES = int(os.environ.get("MIN_FRAMES", "20"))
MAX_SEQS = int(os.environ.get("MAX_SEQS", "30"))

meta_path = DATASET / "meta" / "frames_meta.json"
if not meta_path.is_file():
    raise SystemExit(f"No {meta_path}. Run scripts/02_convert_arts_to_yolo.py first.")

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
frames = meta["frames"]
print(f"converted dataset: {DATASET}  ({len(frames)} photos)")

# ------------------------------------------------------------ split manifest
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
manifest_path = RESULTS_DIR / "split_manifest.csv"
counts = defaultdict(int)
with open(manifest_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["image", "split", "tile"])
    for stem in sorted(frames, key=lambda s: s.split("__")[-1]):
        m = frames[stem]
        writer.writerow([stem.split("__")[-1], m["split"], m["group"]])
        counts[m["split"]] += 1
print(f"split manifest -> {manifest_path}  {dict(sorted(counts.items()))}")

# ------------------------------------------------------------ leakage check
# The converter's leakage guard works per sign id. Some physical posts carry two
# or three ids (duplicates at identical coordinates), so leakage is also counted
# per post, with the evaluation's rule: same-class ids within CANON_M are one post.
canon = pipeline_core.canonical_gt_map(frames, CANON_M)
ids_by_split, posts_by_split = defaultdict(set), defaultdict(set)
for m in frames.values():
    for o in m.get("objects", []):
        gid = o.get("sign_id")
        if gid:
            ids_by_split[m["split"]].add(str(gid))
            posts_by_split[m["split"]].add(canon.get(str(gid), str(gid)))
pairs = (("train", "val"), ("train", "test"), ("val", "test"))
split_check = {
    "canon_m": CANON_M,
    "ids": {s: len(v) for s, v in sorted(ids_by_split.items())},
    "posts": {s: len(v) for s, v in sorted(posts_by_split.items())},
    "ids_shared": {f"{a}_{b}": len(ids_by_split[a] & ids_by_split[b]) for a, b in pairs},
    "posts_shared": {f"{a}_{b}": len(posts_by_split[a] & posts_by_split[b]) for a, b in pairs},
}
with open(RESULTS_DIR / "split_check.json", "w", encoding="utf-8") as f:
    json.dump(split_check, f, indent=2)
    f.write("\n")
print(f"leakage check -> {RESULTS_DIR / 'split_check.json'}  "
      f"ids shared {split_check['ids_shared']}  posts shared {split_check['posts_shared']}")

# ------------------------------------------------------------ sequences
groups = defaultdict(list)
for stem, m in frames.items():
    if m["split"] == "test":
        groups[m["group"]].append(stem)

ranked = sorted(((g, s) for g, s in groups.items() if len(s) >= MIN_FRAMES),
                key=lambda kv: -len(kv[1]))[:MAX_SEQS]
print(f"test tiles with >= {MIN_FRAMES} photos: "
      f"{sum(1 for s in groups.values() if len(s) >= MIN_FRAMES)}  (exporting {len(ranked)})")

# Stale sequence folders would change which ones calibrate, so start clean.
SEQ_ROOT.mkdir(parents=True, exist_ok=True)
for old in SEQ_ROOT.glob("seq_*"):
    if old.is_dir():
        shutil.rmtree(old)

total_frames = total_ids = 0
for group, stems in ranked:
    safe = group.replace("/", "-")
    seq_dir = SEQ_ROOT / f"seq_{safe}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted(stems, key=lambda s: s.split("__")[-1])   # driving order
    seq_meta = {}
    for stem in stems:
        src = DATASET / "images" / "test" / f"{stem}.jpg"
        if not src.is_file():
            continue
        shutil.copy2(src, seq_dir / f"{stem}.jpg")
        seq_meta[stem] = frames[stem]
    with open(seq_dir / "frames_meta.json", "w", encoding="utf-8") as f:
        json.dump({"tags": meta.get("tags", {}), "frames": seq_meta}, f)
    ids = {o["sign_id"] for m in seq_meta.values()
           for o in m["objects"] if o.get("sign_id")}
    total_frames += len(seq_meta)
    total_ids += len(ids)
    print(f"  seq_{safe:<16s} {len(seq_meta):4d} frames  {len(ids):3d} distinct signs")

print(f"\nTOTAL: {len(ranked)} sequences  {total_frames} frames  {total_ids} distinct signs")
print(f"-> {SEQ_ROOT}")

# ------------------------------------------------------------ demo sequence
demo_src = DATASET / "demo_sequence"
demo_dst = ROOT / "data" / "demo_sequence"
if demo_src.is_dir() and not demo_dst.exists():
    shutil.copytree(demo_src, demo_dst)
    print(f"demo sequence -> {demo_dst}")
