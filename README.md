# Traffic Sign Inventory

Turn GPS-tagged road photographs into a searchable map database where one row is one physical signpost.

![Dashboard showing traffic signs clustered across Vermont](docs/screenshots/01-overview.png)

---

## Features

- **Detects road signs** in survey photographs and classifies them into 50 Manual on Uniform Traffic Control Devices (MUTCD) types.
- **Locates each sign**, not the camera. Median error is 4.7 m, against 10.7 m for the raw vehicle GPS.
- **Deduplicates automatically.** A sign photographed 13 times becomes one database row, not 13.
- **Two-stage classification.** A second model re-reads each cropped sign, lifting accuracy on look-alike sign families from 78.7% to 94.3%.
- **Stores results in PostGIS**, so map queries like "every sign in this rectangle" run in the database.
- **Ships a dashboard** with a clustered map, per-sign crops, and the measured error for every point.
- **Scores itself.** Every headline number is reproducible with one command against held-out data.

## Tech stack

| Layer | Choice |
|---|---|
| Detection | YOLO11n (Ultralytics), 1280 px |
| Classification | YOLO11n-cls, 96 px crops |
| Backend | FastAPI, Python 3.10+ |
| Database | Supabase — PostgreSQL + PostGIS |
| Frontend | Next.js 14, MapLibre GL, Tailwind CSS |
| Training | Google Colab, free T4 GPU |
| Dataset | ARTSv2 (University of Vermont) |

---

## Quick start

### Prerequisites

- Python 3.10 or newer
- Node.js 18 or newer
- A free [Supabase](https://supabase.com) project
- About 15 GB of free disk space

Trained weights are included in `weights/`, so no training is needed to run the
project. No GPU is needed either.

### 1. Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
npm install --prefix frontend
```

Use `.venv/bin/python` instead on macOS and Linux. Without a GPU, install the CPU
build of PyTorch first to save about 2 GB:

```bash
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Create the database

1. Create a project at [supabase.com](https://supabase.com).
2. Open the SQL Editor and run [`scripts/04_seed_supabase.sql`](scripts/04_seed_supabase.sql).
   This creates the table, the PostGIS functions, and the row-level security
   policy. It is safe to run more than once.
3. Under **Storage**, create two public buckets: `sign-crops` and `sign-frames`.
   Cropped sign images and full frames are uploaded here.
4. Open **Project Settings → API** and copy three values:

   | Value | Where it appears | What it is for |
   |---|---|---|
   | Project URL | top of the page | the address of the database |
   | `service_role` key | under Project API keys | full access, used by the backend only |
   | `anon` key | under Project API keys | public read-only key for browser-side access |

### 3. Configure

```bash
cp .env.example .env
```

Fill in the three values copied above:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOi...
SUPABASE_ANON_KEY=eyJhbGciOi...
```

The `service_role` key bypasses row-level security, so it must stay on the
server and must never reach a browser. The `anon` key is the safe one to expose;
it is limited to reading by the policy created in step 2.

Every other setting has a working default. See
[Configuration](#configuration) for the full list.

### 4. Verify

```bash
.venv/Scripts/python.exe scripts/00_preflight.py
```

Checks the Python packages, loads both models and confirms their class counts,
counts the photographs and their GPS coverage, then calls the live database to
confirm the table, the functions, and both storage buckets. Prints `READY`, or
names exactly what is missing.

### 5. Run

```bash
.venv/Scripts/python.exe scripts/03_run_pipeline.py
```

Then start the two servers, each in its own terminal:

```bash
.venv/Scripts/python.exe -m uvicorn backend.main:app --port 8000
```

```bash
npm run dev --prefix frontend
```

Open **http://localhost:3000**.

---

## Screenshots

![Map view with clustered and individual signs](docs/screenshots/02-map.png)

Green circles are clusters and split apart when clicked. Gold circles are
individual signs. The side panel summarises whatever is in view and lists
detections with their crops.

![Detail panel for a single sign](docs/screenshots/03-sign-detail.png)

Selecting a sign shows its crop, class, confidence, and how many photographs it
came from. A `W1-2` curve sign on US Route 2, seen 13 times and collapsed into
one row, sits **1.3 m from where it actually stands — against 6.2 m if the
vehicle GPS were used directly**.

Every coordinate carries its provenance. Computed positions are labelled
"Triangulated sign position"; placeholders render in red, so demo data can never
pass as a real detection.

---

## Results

Measured on 18 road sequences, 1,108 photographs, and 366 signposts, all held
out. No signpost in this set appears in training, and the projection constants
were fitted on 12 different sequences.

| Metric | Result |
|---|---|
| Detection | mAP50 **0.723** · mAP50-95 **0.495** |
| Sign classification | **94.8%** |
| Signposts found | **80.1%** (293 of 366) |
| Position error | **4.7 m** median · 17.2 m at p90 |
| Same, using raw vehicle GPS | 10.7 m |
| Improvement from projection | **56%** |
| Exactly one row per signpost | ~52% |

### Two-stage classification

Some sign families are visually identical apart from small text. `M3-1` to `M3-4`
are the same rectangle reading NORTH, SOUTH, EAST, or WEST. `R2-125` to `R2-165`
differ only by the printed number. At a 26-pixel median sign height that text is
unreadable, so a second model re-reads each crop enlarged, with its answer
restricted to the family the detector already chose.

| Family | Detector | With classifier |
|---|---|---|
| M3 | 50.4% | **94.2%** |
| R2 | 84.3% | **99.1%** |
| M1 | 86.2% | 97.0% |
| M6 | 80.2% | 88.8% |
| R3 | 80.0% | 86.7% |
| D1 | 96.6% | 93.2% |
| **Confusable families** | **78.7%** | **94.3%** |
| All detections | 85.5% | **94.8%** |

This also improved the map results, which was not the goal. Clustering groups
detections by class name, so wrong labels were splitting one signpost across
several rows. Correcting the labels cut splitting by more than half and lifted
deduplication from 52.7% to 72.4%.

### Why these numbers are conservative

**Strict one-to-one scoring.** A signpost counts only if it produced exactly one
row, and that row refers to exactly one signpost. The looser "was it found at
all" measure reaches 95% simply by widening the clustering radius until one blob
swallows everything.

**Cleaned ground truth.** Some dataset entries appear twice under different ids
at identical coordinates, and sign assemblies place several plates on one post
2–3 m apart. Ids of the same class within 2 m are treated as one post, which is
why the count is 366 rather than 451.

**A custom split.** The dataset ships a split in which 86% of test signs also
appear in training — the same signpost, same drive, both sides. Splitting by
geography instead holds out whole stretches of road. These results are therefore
not comparable to published numbers on the official split.

---

## How it works

```mermaid
flowchart TB
    A["Road photographs<br/>1920 x 1080<br/>camera GPS + heading"]

    subgraph vision ["1 &nbsp; Vision"]
        direction TB
        B["Detector: YOLO11n at 1280 px<br/>finds where the sign is<br/>and which family it belongs to"]
        C["Classifier: YOLO11n-cls at 96 px<br/>re-reads the crop enlarged<br/>to pick the exact variant"]
        B --> C
    end

    subgraph geo ["2 &nbsp; Geolocation"]
        direction TB
        D["Project onto the map<br/>heading + box offset gives a bearing<br/>box width gives a range"]
        E["Cluster by position<br/>same class within 60 m<br/>= one physical signpost"]
        D --> E
    end

    subgraph serve ["3 &nbsp; Serve"]
        direction TB
        F[("Supabase<br/>PostgreSQL + PostGIS")]
        G["FastAPI<br/>bounding-box queries"]
        H["Next.js + MapLibre<br/>clustered map, crops, error"]
        F --> G --> H
    end

    A --> vision
    vision --> geo
    geo --> serve

    style A fill:#ffffff,stroke:#0a0a0a,stroke-width:3px
    style vision fill:#e8f0eb,stroke:#00693e,stroke-width:2px
    style geo fill:#fff8e6,stroke:#ffb81c,stroke-width:2px
    style serve fill:#eef2f6,stroke:#334155,stroke-width:2px
```

Three properties of this data rule out the standard approach.

**Signs are small.** The median sign is 26 pixels on its shorter side in a
1920×1080 frame, and 60% are under 32 px. Detecting at the usual 640 px shrinks
that to 9 px, below the network's smallest feature stride. Everything runs at
1280 px.

**The input is stills, not video.** Trackers such as BoT-SORT link objects
between frames by box overlap, but the camera advances about 8 m between
consecutive photographs, so two views of one sign do not overlap at all.

| Association method | Detections given an identity |
|---|---|
| BoT-SORT | 8 / 95 |
| ByteTrack | 7 / 96 |
| **Position clustering** | **95 / 95** |

**GPS records the vehicle, not the sign.** A sign photographed from 40 m away is
40 m from its recorded position.

The pipeline projects instead of tracking. Camera heading plus the box's
horizontal offset gives a bearing; box width gives a rough range; together they
place the sign itself on the map. Repeat sightings then group by location, which
holds no matter how far the vehicle moved between frames.

The camera field of view (80°) and the width-to-range constant (0.88 m) are
fitted from ground-truth coordinates by least squares, using 771 matched
detections across sequences excluded from every reported result. A fit from a
single sequence gave 95°, so fitting across many sequences matters.

---

## API

The backend exposes a small read-mostly API on port 8000. The dashboard uses it
for everything; it is also usable directly.

### `GET /health`

Service status, which weights loaded, and whether the database is connected. Use
it to confirm the backend started correctly.

```bash
curl http://localhost:8000/health
```

```json
{"ok": true, "weights": "weights/best.pt", "fine_tuned": true, "supabase": true}
```

### `GET /signs?bbox=minLng,minLat,maxLng,maxLat`

Every sign inside a map rectangle, given as four comma-separated numbers:
west, south, east, north. This is the endpoint the map calls each time it moves.
The rectangle below covers the Burlington area.

```bash
curl "http://localhost:8000/signs?bbox=-73.3,44.4,-73.1,44.6"
```

Each row includes the sign type, confidence, computed position, the URL of its
cropped image, how many photographs it was seen in, and — where ground truth
exists — the true position and the error in metres.

### `GET /signs/{id}`

One full record, by its UUID.

### `GET /stats`

Totals across the whole database: counts by sign type, by position source, and
by road sequence. Useful for a quick check that a pipeline run landed.

```bash
curl http://localhost:8000/stats
```

### `POST /pipeline/run`

Processes a folder of photographs and writes the results to the database. The
folder must sit inside `data/`, which keeps the endpoint from reading arbitrary
paths on the machine.

```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"sequence_dir": "data/demo_sequence"}'
```

---

## Configuration

All settings live in `.env`. Only the first three have no default.

| Variable | Default | Purpose |
|---|---|---|
| `SUPABASE_URL` | — | Project URL from Supabase |
| `SUPABASE_SERVICE_ROLE_KEY` | — | Full-access key. Backend only, never sent to a browser |
| `SUPABASE_ANON_KEY` | — | Public read-only key, for browser-side access |
| `WEIGHTS_PATH` | `weights/best.pt` | The detector |
| `CLS_WEIGHTS` | `weights/crop_classifier.pt` | The crop classifier. Set to `none` to disable the second stage |
| `CLS_MIN_CONF` | `0.8` | How sure the classifier must be before it overrules the detector |
| `CONF` | `0.25` | Minimum detection confidence to keep a box |
| `TRACKER` | `botsort.yaml` | Only used when running the tracker baseline |
| `FRONTEND_ORIGIN` | `http://localhost:3000` | Allowed origin for browser requests to the API |
| `SEQUENCE_DIR` | `data/demo_sequence` | Which folder of photographs to process |
| `ALLOW_SYNTHETIC_GPS` | `0` | Demo only. When `1`, signs with no GPS get a placeholder position, flagged in red in the dashboard |

The pipeline scripts accept a few more as one-off overrides, listed in
[Reproducing the results](#reproducing-the-results).

---

## Repository structure

```
├── notebooks/
│   └── 01_train_yolo_colab.ipynb   Colab notebook, phases A–G
├── scripts/
│   ├── 00_preflight.py             verifies every prerequisite
│   ├── 01_download_arts_notes.md   dataset source and annotation format
│   ├── 02_convert_arts_to_yolo.py  ARTSv2 → YOLO, geographic split
│   ├── 03_run_pipeline.py          photographs → signs → database
│   ├── 04_seed_supabase.sql        schema, functions, row-level security
│   ├── 06_evaluate_geo_and_dedupe.py   scores one sequence
│   ├── 07_batch_evaluate.py        scores many, with held-out calibration
│   ├── 09a_gps_error_report.py     position error computed inside PostGIS
│   ├── 09c_privacy_blur.py         blurs sensitive regions in crops
│   └── 10_screenshots.py           regenerates the images above
├── pipeline_core.py                detection, projection, clustering
├── backend/main.py                 FastAPI service
├── frontend/                       Next.js + MapLibre dashboard
├── weights/                        trained models, tracked in git
├── data/                           photographs and results (not in git)
└── docs/screenshots/
```

---

## Dataset

**ARTSv2 — Automotive Repository of Traffic Signs**, VaiL group, University of
Vermont. Subset `challenging_dom`.

| | |
|---|---|
| Photographs | 19,914 at 1920×1080 |
| Labelled signs | 35,979 across 171 classes |
| Distinct signposts | 7,773 |
| Coverage | 42.7°N–45.0°N, 73.3°W–71.6°W — the State of Vermont |

Every photograph carries the camera's latitude, longitude, and heading. Every
labelled sign carries its own coordinates and an identifier, which is what makes
position error and deduplication measurable at all.

Download it from the [Google Drive folder](https://drive.google.com/drive/folders/1u_nx38M0_owB0cR-qA6IOWgZhGpb9sWU)
linked from the paper. It arrives already unpacked in PASCAL VOC layout —
`Annotations/`, `JPEGImages/`, `ImageSets/`. Notes on the annotation format,
including the exact XML fields, are in
[`scripts/01_download_arts_notes.md`](scripts/01_download_arts_notes.md).

The dataset is only needed to retrain or to reproduce the evaluation. Running
the project uses the included weights.

---

## Training

Both models train on a free Colab T4 using
[`notebooks/01_train_yolo_colab.ipynb`](notebooks/01_train_yolo_colab.ipynb),
phases A–G.

| | Detector | Crop classifier |
|---|---|---|
| Architecture | YOLO11n | YOLO11n-cls |
| Parameters | 2.6 M | 1.6 M |
| Input | 1280 px | 96 px |
| Classes | 50 | 34 |
| Size | 5.6 MB | 3.3 MB |
| Training time | ~9 hours | ~15 minutes |

**Detector.** 40 epochs, batch 8, AdamW, early stop after 12 flat epochs. Around
14 minutes per epoch. Free Colab allows a few GPU-hours per day, so checkpoints
write to Google Drive every epoch and a resume cell continues from the exact
epoch a session ended.

**Crop classifier.** 11,232 crops cut from training photographs using
ground-truth boxes with 15% padding, so the framing resembles a predicted box.
Reaches 96.6% top-1 on held-out crops, 94.3% on real predicted boxes.

Only 50 of 171 classes are trained. The top 50 cover 75% of all labelled signs,
and the rarest of them still has around 200 examples. Below that there is too
little data.

One detail worth knowing before running the notebook: photographs are copied to
Colab's local disk before training. Reading them from Google Drive one at a time
runs at roughly one file per second because each read is a separate network
request. Copying with 32 parallel threads reaches about 700 files per second,
turning 11 hours into 15 minutes.

### Reproducing the results

```bash
.venv/Scripts/python.exe scripts/07_batch_evaluate.py
```

Reads every folder under `data/sequences/`, fits the projection constants on 40%
of them, applies those constants unchanged to the rest, and reports pooled scores
plus a per-family classification table.

| Override | Effect |
|---|---|
| `SWEEP=1` | Tries several clustering radii, on the calibration sequences only |
| `CLS_WEIGHTS=none` | Disables the crop classifier, to measure its contribution |
| `ASSOCIATION=tracker` | Runs the BoT-SORT baseline instead of position clustering |
| `UPLOAD=1` | Also writes the evaluated sequences to the database |

Detection results are cached, so repeat runs with different settings are instant.

---

## Limitations

- **Splitting is the main weakness.** One signpost still produces several rows
  often enough that strict one-to-one accuracy sits near 52%. A deployment would
  need a review step to merge duplicates. Merging distinct signs into one row is
  the rarer error, which is the safer direction.
- **Distance is estimated from apparent size**, assuming signs are roughly
  standard. Direction is far more reliable than range, so most remaining error
  lies along the line of sight.
- **The crop classifier is overconfident when wrong** — 0.89 average confidence
  on mistakes against 1.00 when correct — so confidence is a weak filter. A 0.8
  threshold is applied but cannot do more.
- **121 of 171 classes are unsupported**, having too few examples.
- **Recall is 80% of signposts.** Signs photographed once, or only from a
  distance, are missed most often.


## Credits

### Dataset

ARTSv2 was collected and released by the Vermont Artificial Intelligence Lab (VaiL) at the
University of Vermont. 

> Wilson, D., Alshaabi, T., Van Oort, C., Zhang, X., Nelson, J., & Wshah, S.
> (2022). Object Tracking and Geo-Localization from Street Images.
> *Remote Sensing*, 14(11), 2575. https://doi.org/10.3390/rs14112575

