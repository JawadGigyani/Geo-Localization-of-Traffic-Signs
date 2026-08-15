# ARTSv2 / ARTS — download and format notes

## The dataset

**ARTS — Automotive Repository of Traffic Signs**, from Professor Safwan Wshah's
VaiL group at the University of Vermont.

- ARTS (v1) ships in three configurations: **Easy**, **Challenging**, **Video-Logs**.
  The VaiL site states the annotations are *"very similar to the PASCAL VOC format"*.
- **ARTSv2** accompanies *Object Tracking and Geo-Localization from Street Images*
  (Remote Sensing, 2022): ~25,544 images, ~47,589 annotations, 199 classes aligned
  with the Vermont VCI sign catalog. Each annotation carries the sign's class,
  GPS, an assembly flag, sign side (left / right / other), and an integer sign id.
- Images come from road video logs, so consecutive frames are roughly a second
  apart and show the same physical signs repeatedly. **This is why the split must
  be by sequence.**

## Links — VERIFIED 13 Aug 2026

| Source | URL | What is actually there |
|---|---|---|
| VaiL datasets page | https://www.wshahaigroup.com/datasets | "Dataset here" button → the v1 Drive below |
| **ARTS v1** Drive | https://drive.google.com/drive/folders/1bMWnDfAOuaRf2WEb4YfQGbkB00hbVfbj | `ARTS-V1/` with `Easy/easy-dev.tar.gz` (4.35 GB) and `Challenging/challenging-dev.tar.gz` (7.61 GB), plus the MUTCD manual and the VCI Sign Catalog PDFs. **No GPS confirmed, and no Video-Logs folder despite the website text.** |
| **ARTSv2** Drive (from the 2022 paper) | https://drive.google.com/drive/folders/1u_nx38M0_owB0cR-qA6IOWgZhGpb9sWU | Folder `challenging_dom/` in ready-to-use PASCAL VOC layout: `Annotations/`, `JPEGImages/`, `ImageSets/Main/`. **This is the one to use** — the XML carries camera GPS, per-sign GPS, heading, and a stable sign id. |
| Paper | https://doi.org/10.3390/rs14112575 | |
| Related code | https://gitlab.com/vail-uvm/VTrans-AI | |

**Use the ARTSv2 link, not the one on the website.** The website's button leads
to v1, which ships as multi-GB tarballs and lacks the geospatial fields the
whole inventory half of this project depends on. ARTSv2 is already unpacked into
individual files, so a Drive shortcut works without any download.

`ImageSets/Main/` holds VOC-style per-class split lists (`D1-1_train.txt`,
`_val`, `_test`, `_trainval`). Our converter builds its own geographically
disjoint split instead — see below for why.

License is research/non-commercial — fine for a portfolio, but say so.

## Target Drive layout

```text
MyDrive/traffic-sign-inventory/
  datasets/
    arts_v2/            raw download, unzipped
    arts_yolo.zip       converted dataset, cached by the notebook
  repo/                 a copy of this project folder
  runs/yolo11n_arts/    training output + checkpoints
  export/               best.pt, classes.txt, demo_sequence.zip
```

## The real annotation format (confirmed, not assumed)

A genuine ARTSv2 annotation file, read directly from the Drive folder:

```xml
<annotation>
  <folder>Annotations</folder>
  <filename>12000000005054.jpg</filename>
  <size><width>1920</width><height>1080</height><depth>3</depth></size>
  <Location>
    <Latitude>44.16793702366667</Latitude>
    <Longitude>-73.25184413725</Longitude>
    <Altitude>67.164</Altitude>
    <Cam_DOM>234.98</Cam_DOM>
  </Location>
  <object>
    <id>D3-1_44.167954339215385_-73.25200238758067</id>
    <name>D3-1</name>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <location>
      <latitude>44.167954339215385</latitude>
      <longitude>-73.25200238758067</longitude>
    </location>
    <bndbox><xmin>1559</xmin><ymin>331</ymin><xmax>1676</xmax><ymax>349</ymax></bndbox>
    <attributes><assembly>False</assembly><side>Right</side></attributes>
  </object>
</annotation>
```

Four things this tells us, all of which shaped the code:

1. **GPS is nested two levels deep**, and the camera's `<Latitude>` has the same
   lowercased name as each sign's `<latitude>`. Any flat scan of the XML would
   either miss both or confuse one for the other. The parser recurses and prunes
   the `<object>` subtree when reading camera position.
2. **`Cam_DOM` is the heading** — camera direction of motion. Nothing named
   "heading" or "bearing" exists.
3. **`<id>` is a stable physical-sign identifier** (class + the sign's own
   lat/lon), so the same signpost keeps one id across every frame it appears in.
   That is what makes the dedupe metric and the leakage guard possible.
4. **The signs are tiny.** That example box is 117 × 18 px inside a 1920 × 1080
   frame. Do not downscale the images, and do not train at `imgsz=640`.

## You do not need to guess the field names

The converter **auto-detects** the XML tag names by sampling 200 annotation
files and reports what it resolved:

```
resolved mapping (VERIFY THIS against one real XML):
  camera_lat   -> ...
  camera_lon   -> ...
  heading      -> ...
  sign_lat     -> ...
  sign_lon     -> ...
  sign_id      -> ...
```

Notebook cell **A5** prints one complete XML so you can check that mapping with
your own eyes. Do that once, before converting.

If a field is missing from the mapping, add its real name to the `HINTS`
dictionary at the top of [02_convert_arts_to_yolo.py](02_convert_arts_to_yolo.py).

### What each field is used for

| Field | Used by | If absent |
|---|---|---|
| `size/width`, `size/height` | box normalisation | falls back to reading the image header |
| `object/name` | class label | the object is skipped |
| `object/bndbox` | the box | the object is skipped |
| camera lat/lon | the map pin | no coordinates; the map stays empty |
| camera heading | depth projection (`09b`) | derived from the GPS trail instead |
| sign lat/lon | geo-localization error | that metric is unavailable |
| sign id | dedupe accuracy | that metric is unavailable |

## Layouts the converter handles

```text
arts_v2/
  SEGMENT_ID/                 <- one folder per road segment
    img_0001.jpg
    img_0001.xml
    ...

# or classic VOC:
arts_v2/
  JPEGImages/
  Annotations/
```

ARTSv2 uses the flat VOC layout, so the converter falls back to a **geographic
split**: frames are tiled into ~500 m cells by camera GPS and whole tiles go to
train / val / test. That is a stronger hold-out than a random image split, since
consecutive frames roughly a second apart stay together.

On top of that, a **leakage guard** uses the `<id>` field: any training frame
showing a physical sign that also appears in val or test is dropped outright. It
prints how many frames that cost. The result is a split where no signpost is
ever both trained on and evaluated on — which a random VOC-style split, including
the one in `ImageSets/Main/`, does not give you.

If you want numbers comparable to the paper, run the official `ImageSets` split
as a second experiment and report both. The gap between them is itself a result.

## Class taxonomy

199 classes with a long tail. The converter keeps the top `TOP_N_CLASSES`
(default 30). Notebook cell B2 shows what share of annotations that covers and
how many classes have too few examples to learn. Raise the number if the
coverage looks poor and your GPU budget allows.

## Local demo without the real data

`data/demo_sequence/` holds eight synthetic frames for smoke-testing the API and
UI. They are flat colour rectangles — a fine-tuned model will find nothing in
them, which is expected. Replace them with the real `demo_sequence.zip` the
notebook exports.
