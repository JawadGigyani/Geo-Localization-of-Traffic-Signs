"""Unit tests for the geometry, clustering, classification and scoring code.

They need only the Python standard library: no dataset, no PyTorch, no OpenCV.

    python -m unittest discover -s tests -v
"""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_core as pc  # noqa: E402

CAMERA = (44.5, -72.6)


def det_towards(target, cls_name, camera=CAMERA, heading=90.0, width_scale=1.0,
                gt_id=None, hfov=80.0, size_k=0.88, image_width=1920.0):
    """A detection whose box projects onto `target` (lat, lon) under the given
    camera constants. width_scale > 1 makes a wider (closer-looking) box."""
    lat, lon = camera
    true_bearing = pc.bearing_deg(lat, lon, *target)
    true_dist = pc.haversine_m(lat, lon, *target)
    f = pc.focal_px(image_width, hfov)
    angle = ((true_bearing - heading + 180.0) % 360.0) - 180.0
    cx = image_width / 2.0 + f * math.tan(math.radians(angle))
    w = size_k * f / true_dist * width_scale
    return {
        "box": [cx - w / 2.0, 500.0, cx + w / 2.0, 500.0 + w],
        "image_width": image_width, "image_height": 1080.0,
        "camera_lat": lat, "camera_lon": lon, "heading": heading,
        "cls_name": cls_name, "conf": 0.9, "frame_path": "frame.jpg",
        "gt_sign_id": gt_id,
        "gt_sign_lat": target[0] if gt_id else None,
        "gt_sign_lon": target[1] if gt_id else None,
    }


class GeometryTests(unittest.TestCase):
    def test_haversine_one_degree_of_latitude(self):
        self.assertAlmostEqual(pc.haversine_m(0.0, 0.0, 1.0, 0.0),
                               math.pi / 180 * pc.EARTH_R, delta=0.01)

    def test_bearing_compass_points(self):
        lat, lon = CAMERA
        self.assertAlmostEqual(pc.bearing_deg(lat, lon, lat + 0.001, lon), 0.0, delta=0.01)
        self.assertAlmostEqual(pc.bearing_deg(lat, lon, lat, lon + 0.001), 90.0, delta=0.01)
        self.assertAlmostEqual(pc.bearing_deg(lat, lon, lat - 0.001, lon), 180.0, delta=0.01)
        self.assertAlmostEqual(pc.bearing_deg(lat, lon, lat, lon - 0.001), 270.0, delta=0.01)

    def test_offset_point_round_trip(self):
        lat, lon = pc.offset_point(*CAMERA, 116.5, 16.8)
        self.assertAlmostEqual(pc.haversine_m(*CAMERA, lat, lon), 16.8, delta=0.05)
        self.assertAlmostEqual(pc.bearing_deg(*CAMERA, lat, lon), 116.5, delta=0.1)

    def test_focal_length_at_80_degrees(self):
        self.assertAlmostEqual(pc.focal_px(1920, 80), 1144.1, delta=0.05)

    def test_project_detection_worked_example(self):
        # The worked example in the README (How it works): box 1500-1560 px,
        # heading 90 deg (east).
        det = {"camera_lat": CAMERA[0], "camera_lon": CAMERA[1], "heading": 90.0,
               "box": [1500.0, 400.0, 1560.0, 460.0], "image_width": 1920.0}
        p = pc.project_detection(det, 80.0, 0.88)
        self.assertAlmostEqual(p["distance_m"], 16.8, delta=0.05)
        self.assertAlmostEqual(p["bearing_deg"], 116.5, delta=0.05)
        north = (p["lat"] - CAMERA[0]) * 111320.0
        east = (p["lon"] - CAMERA[1]) * 111320.0 * math.cos(math.radians(CAMERA[0]))
        self.assertAlmostEqual(north, -7.5, delta=0.05)
        self.assertAlmostEqual(east, 15.0, delta=0.05)

    def test_project_detection_needs_heading_and_clamps_range(self):
        det = {"camera_lat": CAMERA[0], "camera_lon": CAMERA[1], "heading": None,
               "box": [900.0, 0.0, 1000.0, 10.0], "image_width": 1920.0}
        self.assertIsNone(pc.project_detection(det, 80.0, 0.88))
        det["heading"] = 0.0
        det["box"] = [959.9, 0.0, 960.0, 1.0]            # 0.1 px wide -> very far
        self.assertEqual(pc.project_detection(det, 80.0, 0.88)["distance_m"], 200.0)
        det["box"] = [0.0, 0.0, 1920.0, 1.0]             # whole frame -> very close
        self.assertEqual(pc.project_detection(det, 80.0, 0.88)["distance_m"], 2.0)


class CalibrationTests(unittest.TestCase):
    def test_recovers_known_constants(self):
        dets = []
        for i in range(40):
            heading = (i * 37.0) % 360.0
            bearing = heading + ((i % 9) - 4) * 8.0          # within the field of view
            distance = 10.0 + (i % 7) * 10.0                 # 10 .. 70 m
            target = pc.offset_point(*CAMERA, bearing, distance)
            dets.append(det_towards(target, "W1-2", heading=heading, gt_id=f"S{i}"))
        hfov, size_k, used = pc.calibrate_geometry(dets)
        self.assertEqual(used, 40)
        self.assertEqual(hfov, 80)
        self.assertAlmostEqual(size_k, 0.88, delta=1e-6)

    def test_too_few_matches(self):
        hfov, size_k, used = pc.calibrate_geometry([])
        self.assertIsNone(hfov)
        self.assertEqual(used, 0)


class ClusteringTests(unittest.TestCase):
    def setUp(self):
        sign = pc.offset_point(*CAMERA, 100.0, 30.0)
        near = pc.offset_point(*sign, 45.0, 3.0)
        far = pc.offset_point(*sign, 90.0, 150.0)
        self.dets = [
            det_towards(sign, "W1-2", width_scale=1.0, gt_id="A"),
            det_towards(near, "W1-2", width_scale=0.9, gt_id="A"),
            det_towards(sign, "W1-2", width_scale=1.3, gt_id="A"),   # widest: the anchor
            det_towards(sign, "R1-1", gt_id="B"),                    # other class, same post
            det_towards(far, "W1-2", gt_id="C"),                     # same class, far away
        ]
        self.widest_box = self.dets[2]["box"]

    def test_same_class_nearby_becomes_one_row(self):
        rows = pc.associate_geometric(self.dets, hfov_deg=80.0, size_k=0.88, radius_m=60.0)
        self.assertEqual(len(rows), 3)
        w12 = [r for r in rows.values() if r["cls_name"] == "W1-2" and r["observations"] == 3]
        self.assertEqual(len(w12), 1)
        self.assertEqual(w12[0]["box"], self.widest_box)
        self.assertEqual(w12[0]["member_gt_ids"], ["A"])
        self.assertEqual(w12[0]["gps_source"], "geo_projected")

    def test_defaults_are_the_stored_calibration(self):
        rows = pc.associate_geometric(self.dets)
        self.assertEqual(len(rows), 3)


class ClassifierTests(unittest.TestCase):
    CLASSES = [l.strip() for l in open(ROOT / "weights" / "classes.txt", encoding="utf-8")
               if l.strip()]

    def test_families_from_shipped_classes(self):
        fams = pc.build_families(self.CLASSES)
        self.assertEqual(len(self.CLASSES), 50)
        self.assertEqual(len(fams), 11)
        self.assertEqual(sum(len(v) for v in fams.values()), 34)
        self.assertEqual(pc.family_of("R2-140"), "R2")
        self.assertEqual(pc.family_of("VD-062"), "VD")

    def _fake(self, probs_by_name):
        names = dict(enumerate(probs_by_name))
        values = list(probs_by_name.values())

        class Tensor:
            def tolist(self):
                return values

        class Probs:
            data = Tensor()

        class Result:
            pass

        result = Result()
        result.probs = Probs()
        result.names = names

        class Model:
            def predict(self, source, verbose=False):
                return [result]

        return Model()

    class Crop:
        size = 1

    def test_answer_is_restricted_to_the_family(self):
        fams = pc.build_families(self.CLASSES)
        # R1-1 scores highest overall but is not an M3 sign, so it cannot win.
        model = self._fake({"M3-1": 0.05, "M3-2": 0.60, "M3-3": 0.05, "M3-4": 0.00, "R1-1": 0.30})
        name, conf = pc.refine_class(model, self.Crop(), "M3-3", fams, min_conf=0.8)
        self.assertEqual(name, "M3-2")
        self.assertAlmostEqual(conf, 0.60 / 0.70, places=6)

    def test_low_confidence_keeps_the_detector_label(self):
        fams = pc.build_families(self.CLASSES)
        model = self._fake({"M3-1": 0.4, "M3-2": 0.6, "M3-3": 0.0, "M3-4": 0.0})
        name, conf = pc.refine_class(model, self.Crop(), "M3-3", fams, min_conf=0.8)
        self.assertEqual(name, "M3-3")
        self.assertAlmostEqual(conf, 0.6, places=6)

    def test_classes_without_relatives_are_untouched(self):
        fams = pc.build_families(self.CLASSES)
        model = self._fake({"R1-1": 1.0})
        self.assertEqual(pc.refine_class(model, self.Crop(), "R1-1", fams), ("R1-1", None))


class ScoringTests(unittest.TestCase):
    def setUp(self):
        s1 = (44.5, -72.6)
        s2 = (44.5005, -72.6)
        s4 = pc.offset_point(*s1, 0.0, 100.0)
        obj = lambda gid, name, p: {"sign_id": gid, "name": name,
                                    "sign_lat": p[0], "sign_lon": p[1]}
        self.meta = {"frame": {"objects": [
            obj("S1", "W1-2", s1), obj("S1b", "W1-2", s1),   # duplicate id, same spot
            obj("S2", "R1-1", s2), obj("S4", "W1-2", s4)]}}
        pin = pc.offset_point(*s1, 90.0, 3.0)
        cam = pc.offset_point(*s1, 270.0, 10.0)
        self.rows = [
            {"track_id": 1, "lat": pin[0], "lon": pin[1], "gt_sign_id": "S1",
             "gt_sign_lat": s1[0], "gt_sign_lon": s1[1], "camera_lat": cam[0],
             "camera_lon": cam[1], "member_gt_ids": ["S1", "S1b"]},
            {"track_id": 2, "lat": s1[0], "lon": s1[1], "gt_sign_id": None,
             "gt_sign_lat": None, "member_gt_ids": ["S1"]},             # S1 split in two
            {"track_id": 3, "lat": s2[0], "lon": s2[1], "gt_sign_id": None,
             "gt_sign_lat": None, "member_gt_ids": ["S2", "S4"]},       # unmatched anchor, merge
            {"track_id": 4, "lat": 44.6, "lon": -72.7, "gt_sign_id": None,
             "gt_sign_lat": None, "member_gt_ids": []},                 # false positive
        ]

    def test_canonical_ids(self):
        canon = pc.canonical_gt_map(self.meta)
        self.assertEqual(canon["S1b"], "S1")
        self.assertEqual(canon["S2"], "S2")
        self.assertEqual(canon["S4"], "S4")

    def test_score_rows(self):
        s = pc.score_rows(self.rows, self.meta)
        self.assertEqual((s["raw_ids"], s["posts"], s["rows"]), (4, 3, 4))
        # rows whose anchor matched nothing still contribute the signs inside them
        self.assertEqual(s["found"], 3)
        self.assertEqual(s["fragmented"], 1)
        self.assertEqual(s["merged"], 1)
        self.assertEqual(s["one_to_one"], 0)
        self.assertEqual(len(s["errors"]), 1)
        self.assertAlmostEqual(s["errors"][0], 3.0, delta=0.05)
        self.assertAlmostEqual(s["baseline"][0], 10.0, delta=0.05)

    def test_pool_scores(self):
        p = pc.pool_scores([pc.score_rows(self.rows, self.meta)])
        self.assertAlmostEqual(p["recall"], 1.0)
        self.assertAlmostEqual(p["dedupe"], 1 / 3)
        self.assertEqual(p["f1_1to1"], 0.0)

    def test_percentile(self):
        self.assertIsNone(pc.percentile([], 0.5))
        self.assertEqual(pc.percentile([3, 1, 2], 0.5), 2)
        self.assertAlmostEqual(pc.percentile([0, 10], 0.9), 9.0)


class CalibrationFileTests(unittest.TestCase):
    def test_missing_file_gives_defaults(self):
        self.assertEqual(pc.load_calibration("does/not/exist.json"), pc.DEFAULT_CALIBRATION)

    def test_save_load_and_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "calibration.json")
            pc.save_calibration(path, 75, 0.912345, 40, fitted_by="test")
            self.assertEqual(pc.load_calibration(path),
                             {"hfov_deg": 75.0, "size_k_m": 0.9123, "radius_m": 40.0})
            hfov, size_k, radius, source = pc.geometry_settings(path, environ={"RADIUS_M": "25"})
            self.assertEqual((hfov, size_k, radius), (75.0, 0.9123, 25.0))
            self.assertIn("RADIUS_M", source)


class CommittedArtifactTests(unittest.TestCase):
    """The tracked result files agree with each other and with the README."""

    def test_calibration_file(self):
        cal = pc.load_calibration(ROOT / "weights" / "calibration.json")
        self.assertEqual(cal["hfov_deg"], 80.0)
        self.assertAlmostEqual(cal["size_k_m"], 0.88, delta=0.01)
        self.assertEqual(cal["radius_m"], 60.0)

    def test_sequence_manifest(self):
        with open(ROOT / "results" / "sequences_manifest.json", encoding="utf-8") as f:
            seqs = json.load(f)["sequences"]
        calib = [s for s in seqs if s["role"] == "calibration"]
        evals = [s for s in seqs if s["role"] == "evaluation"]
        self.assertEqual((len(calib), len(evals)), (12, 18))
        self.assertEqual(sum(s["frames"] for s in calib), 462)
        self.assertEqual(sum(s["frames"] for s in evals), 646)
        self.assertEqual(sum(s["sign_ids"] for s in evals), 451)
        names = [s["name"] for s in seqs]
        self.assertEqual(names, sorted(names))     # 07 calibrates on the first 40% by name


if __name__ == "__main__":
    unittest.main()
