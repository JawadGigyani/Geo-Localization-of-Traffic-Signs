# Server-side geo-error report: reads the `geo_error` view created by
# scripts/04_seed_supabase.sql, so PostGIS does the distance maths.
#
#   python scripts/09a_gps_error_report.py
#
# This is the database view of the same metric that
# scripts/06_evaluate_geo_and_dedupe.py computes offline. Use 06 while
# iterating; use this once rows are in Supabase.

import os
import statistics
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

url = os.environ.get("SUPABASE_URL", "")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
if not url or not key or "YOUR_PROJECT" in url:
    raise SystemExit("Configure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env first")

from supabase import create_client

supabase = create_client(url, key)
rows = (supabase.table("geo_error").select("*").execute().data) or []
rows = [r for r in rows if r.get("error_m") is not None and r.get("gps_source") != "synthetic_demo"]

print("rows with ground-truth sign GPS:", len(rows))
if not rows:
    raise SystemExit(
        "Nothing to report.\n"
        "Rows need sign_lat/sign_lon, which come from frames_meta.json produced by\n"
        "scripts/02_convert_arts_to_yolo.py. If your dataset has no per-sign GPS,\n"
        "this metric is not available."
    )

errors = sorted(r["error_m"] for r in rows)
print(f"mean   : {statistics.mean(errors):.1f} m")
print(f"median : {statistics.median(errors):.1f} m")
print(f"p90    : {errors[int(len(errors) * 0.9)]:.1f} m")
print(f"max    : {errors[-1]:.1f} m")

by_type = defaultdict(list)
for r in rows:
    by_type[r["sign_type"]].append(r["error_m"])
print("\nmedian error by sign type:")
for name in sorted(by_type, key=lambda n: -len(by_type[n]))[:15]:
    values = by_type[name]
    print(f"  {name:28s} n={len(values):4d}  median={statistics.median(values):7.1f} m")

by_source = defaultdict(list)
for r in rows:
    by_source[r["gps_source"]].append(r["error_m"])
print("\nmedian error by GPS source:")
for name, values in by_source.items():
    print(f"  {name:20s} n={len(values):4d}  median={statistics.median(values):7.1f} m")
