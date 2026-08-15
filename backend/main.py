"""
FastAPI backend for the Traffic Sign Inventory.

Detection/tracking logic is NOT duplicated here -- it comes from pipeline_core,
the same module scripts/03_run_pipeline.py uses.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import pipeline_core
from ultralytics import YOLO

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", str(ROOT / "weights" / "best.pt"))
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")
CONF = float(os.environ.get("CONF", "0.25"))
TRACKER = os.environ.get("TRACKER", "botsort.yaml")
ALLOW_SYNTHETIC_GPS = os.environ.get("ALLOW_SYNTHETIC_GPS", "0") == "1"

app = FastAPI(title="Traffic Sign Inventory API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the detector once -- reloading per request would blow up an 8GB machine.
_weights = WEIGHTS_PATH if os.path.isfile(WEIGHTS_PATH) else "yolo11n.pt"
_using_finetuned = os.path.isfile(WEIGHTS_PATH)
model = YOLO(_weights)
IMGSZ = int(os.environ.get("IMGSZ", "0")) or pipeline_core.training_imgsz(model)
print("Loaded weights:", _weights, "(fine-tuned)" if _using_finetuned else "(COCO fallback)")
print("Inference imgsz:", IMGSZ)

supabase = None
if SUPABASE_URL and SUPABASE_KEY and "YOUR_PROJECT" not in SUPABASE_URL:
    from supabase import create_client

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client ready")
else:
    print("Supabase not configured -- /signs will return 503")


class PipelineBody(BaseModel):
    sequence_dir: str


@app.get("/health")
def health():
    return {
        "ok": True,
        "weights": _weights,
        "fine_tuned": _using_finetuned,
        "supabase": supabase is not None,
    }


@app.get("/signs")
def list_signs(
    bbox: str = Query(
        ...,
        description="minLng,minLat,maxLng,maxLat",
        examples=["-73.3,44.4,-73.1,44.6"],
    )
):
    if supabase is None:
        raise HTTPException(503, "Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")
    parts = [p.strip() for p in bbox.split(",")]
    if len(parts) != 4:
        raise HTTPException(400, "bbox must be minLng,minLat,maxLng,maxLat")
    try:
        min_lng, min_lat, max_lng, max_lat = [float(x) for x in parts]
    except ValueError:
        raise HTTPException(400, "bbox values must be numbers")

    res = supabase.rpc(
        "signs_in_bbox",
        {"min_lng": min_lng, "min_lat": min_lat, "max_lng": max_lng, "max_lat": max_lat},
    ).execute()
    return {"signs": res.data or []}


@app.get("/signs/{sign_id}")
def get_sign(sign_id: str):
    if supabase is None:
        raise HTTPException(503, "Supabase not configured")
    res = supabase.table("traffic_signs").select("*").eq("id", sign_id).limit(1).execute()
    rows = res.data or []
    if not rows:
        raise HTTPException(404, "Sign not found")
    return rows[0]


@app.get("/stats")
def stats():
    """Headline numbers for the dashboard: counts by class and by GPS source."""
    if supabase is None:
        raise HTTPException(503, "Supabase not configured")
    res = supabase.table("traffic_signs").select("sign_type,gps_source,source_sequence").execute()
    rows = res.data or []
    by_type, by_source, by_seq = {}, {}, {}
    for r in rows:
        by_type[r["sign_type"]] = by_type.get(r["sign_type"], 0) + 1
        by_source[r["gps_source"]] = by_source.get(r["gps_source"], 0) + 1
        by_seq[r["source_sequence"]] = by_seq.get(r["source_sequence"], 0) + 1
    return {"total": len(rows), "by_type": by_type, "by_gps_source": by_source, "by_sequence": by_seq}


@app.post("/pipeline/run")
def run_pipeline(body: PipelineBody):
    sequence_dir = os.path.abspath(body.sequence_dir)

    # Only allow folders inside the project's data/ directory.
    data_root = os.path.abspath(ROOT / "data")
    try:
        inside = os.path.commonpath([sequence_dir, data_root]) == data_root
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        raise HTTPException(400, "sequence_dir must be inside the project's data/ folder")
    if not os.path.isdir(sequence_dir):
        raise HTTPException(400, f"sequence_dir not found: {sequence_dir}")

    frames = pipeline_core.list_frames(sequence_dir)
    if not frames:
        raise HTTPException(400, "No images in sequence_dir")

    meta = pipeline_core.load_frames_meta(sequence_dir)
    tracks = pipeline_core.run_tracking(
        model, frames, tracker=TRACKER, conf=CONF, meta=meta, imgsz=IMGSZ
    )

    synthetic = 0
    if ALLOW_SYNTHETIC_GPS:
        synthetic = pipeline_core.apply_synthetic_gps(tracks)

    seq_name = os.path.basename(sequence_dir)
    pipeline_core.save_crops(tracks, str(ROOT / "data" / "pipeline_crops"), seq_name)

    gps_sources = {}
    for info in tracks.values():
        gps_sources[info["gps_source"]] = gps_sources.get(info["gps_source"], 0) + 1

    payload = {
        "ok": True,
        "frames": len(frames),
        "tracks": len(tracks),
        "gps_sources": gps_sources,
        "synthetic_gps_applied": synthetic,
        "types": sorted({t["cls_name"] for t in tracks.values()}),
    }

    if supabase is None:
        payload["inserted"] = 0
        payload["warning"] = "Supabase not configured; detection only"
        return payload

    inserted = pipeline_core.upload_and_insert(supabase, tracks, seq_name)
    payload["inserted"] = len(inserted)
    return payload
