"""Targeted tests for the surveillance capabilities added to this project.

Run with the standard library only:

    python -m unittest discover -s tests -v

Where OpenCV / NumPy / Ultralytics are NOT installed, this module installs
minimal stand-ins so the pure *logic* of the new modules can still be
exercised (the cv2 stand-in ships a genuine ray-casting point-in-polygon so the
virtual-fence tests are meaningful). Where the real packages ARE installed they
are used untouched, so the same suite also validates the real implementations.
Tests that depend on the Ultralytics stand-in skip themselves when real
Ultralytics is present, rather than downloading model weights.
"""

import ast
import json
import os
import re
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Which dependencies we had to stub. Anything left True means the real package
# was imported successfully.
_STUBBED = {"numpy": False, "cv2": False, "ultralytics": False}

COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
              7: "truck", 39: "bottle", 43: "knife", 76: "scissors", 34: "baseball bat"}
CUSTOM_NAMES = {0: "knife", 1: "handgun"}


def _point_in_polygon(contour, pt, measureDist=False):
    """Real ray-casting implementation behind the cv2 stand-in."""
    poly = contour.data if hasattr(contour, "data") else contour
    pts = []
    for p in poly:
        if isinstance(p, (list, tuple)) and len(p) == 1 and isinstance(p[0], (list, tuple)):
            pts.append((p[0][0], p[0][1]))
        else:
            pts.append((p[0], p[1]))
    x, y = pt
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return 1.0 if inside else -1.0


def _install_ultralytics_stub():
    pkg = types.ModuleType("ultralytics")

    class _T:
        def __init__(self, data):
            self.data = data

        def cpu(self):
            return self

        def numpy(self):
            return self.data

    class Boxes:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = _T(xyxy)
            self.conf = _T(conf)
            self.cls = _T(cls)
            self.id = None

        def __len__(self):
            return len(self.xyxy.data)

    class Result:
        def __init__(self, boxes):
            self.boxes = boxes

    class YOLO:
        def __init__(self, model_path):
            self.path = str(model_path)
            self.names = dict(COCO_NAMES) if os.path.basename(self.path).lower().startswith(
                "yolov8") else dict(CUSTOM_NAMES)
            self._results = []

        def __call__(self, *a, **k):
            return self._results

        def track(self, *a, **k):
            return self._results

    pkg.YOLO = YOLO
    pkg.Boxes = Boxes
    pkg.Result = Result
    sys.modules["ultralytics"] = pkg
    _STUBBED["ultralytics"] = True


def _install_stubs():
    try:
        import numpy  # noqa: F401
    except Exception:
        np = types.ModuleType("numpy")

        class _Arr:
            def __init__(self, data):
                self.data = data

            def reshape(self, *a):
                return self

        np.float32 = "float32"
        np.array = lambda data, dtype=None: _Arr(data)
        np.median = lambda xs: sorted(xs)[len(xs) // 2]
        sys.modules["numpy"] = np
        _STUBBED["numpy"] = True

    try:
        import cv2  # noqa: F401
    except Exception:
        cv2 = types.ModuleType("cv2")

        class _Cascade:
            def empty(self):
                return True

            def detectMultiScale(self, *a, **k):
                return []

        class _Clahe:
            def apply(self, img):
                return img

        cv2.pointPolygonTest = _point_in_polygon
        cv2.CascadeClassifier = lambda *a, **k: _Cascade()
        cv2.createCLAHE = lambda *a, **k: _Clahe()
        cv2.data = types.SimpleNamespace(haarcascades=os.path.join(ROOT, "__no_such_dir__"))
        for flag in ("FONT_HERSHEY_SIMPLEX", "MORPH_RECT", "MORPH_CLOSE", "RETR_EXTERNAL",
                     "CHAIN_APPROX_SIMPLE", "THRESH_BINARY", "THRESH_OTSU", "INTER_CUBIC",
                     "INTER_LINEAR", "INTER_NEAREST", "COLOR_BGR2GRAY", "COLOR_BGR2HSV",
                     "COLOR_BGR2RGB", "COLOR_BGR2LAB", "COLOR_LAB2BGR"):
            setattr(cv2, flag, 0)
        # Any other cv2 call becomes a no-op under test.
        cv2.__getattr__ = lambda name: (lambda *a, **k: None)
        sys.modules["cv2"] = cv2
        _STUBBED["cv2"] = True

    try:
        import ultralytics  # noqa: F401
    except Exception:
        _install_ultralytics_stub()


_install_stubs()

from anpr import ANPRReader, normalize_plate  # noqa: E402
from c2_integration import C2Client, build_event, severity_for  # noqa: E402
from event_store import EventStore  # noqa: E402
from face_detector import FaceDetector, _iou  # noqa: E402
from night_vision import NightVision  # noqa: E402
from object_detector import ObjectDetector, _is_stock_model, find_weapon_model  # noqa: E402
from vehicle_detector import VehicleDetector, _hue_to_name  # noqa: E402
from virtual_fence import VirtualFence  # noqa: E402

MODULE_NAMES = ["vehicle_detector", "face_detector", "anpr_reader", "virtual_fence",
                "night_vision", "c2_client", "event_store", "object_detector"]


def _square(x0=0.1, y0=0.1, x1=0.9, y1=0.9):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# --------------------------------------------------------------------------- C2
class C2Tests(unittest.TestCase):
    def test_event_schema_and_severity(self):
        ev = build_event("Fall Detected", 0.95, "Cam_1", "d",
                         source_id="site-9", data={"track_id": 3})
        self.assertEqual(ev["schema"], "smart-surveillance.event/1.0")
        self.assertEqual(ev["severity"], "critical")
        self.assertEqual(ev["source"]["camera_id"], "Cam_1")
        self.assertEqual(ev["source"]["site"], "site-9")
        self.assertEqual(ev["data"]["track_id"], 3)
        self.assertTrue(ev["timestamp"].endswith("Z"), ev["timestamp"])
        for alert_type, expected in [("Weapon Detected", "critical"),
                                     ("Intrusion Detected", "high"),
                                     ("Restricted Zone Dwell", "high"),
                                     ("Night-time Movement", "high"),
                                     ("Running Detected", "medium"),
                                     ("ANPR Plate Read", "info"),
                                     ("Something Unknown", "low")]:
            self.assertEqual(severity_for(alert_type), expected)

    def test_disabled_client_is_inert(self):
        client = C2Client(webhook_url="")
        self.assertFalse(client.enabled)
        self.assertFalse(client.publish(build_event("Test Alert")))
        self.assertFalse(client.status()["enabled"])

    def test_delivers_to_local_webhook_with_retry_metadata(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                received.append(json.loads(self.rfile.read(length)))
                received.append({"auth": self.headers.get("Authorization")})
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = C2Client(webhook_url=f"http://127.0.0.1:{port}/hook",
                              api_key="secret123", timeout=3)
            self.assertTrue(client.enabled)
            for i in range(5):
                self.assertTrue(client.publish(build_event("Test Alert", 1.0, "Cam_1", f"e{i}")))
            deadline = time.time() + 5
            while time.time() < deadline and client.sent < 5:
                time.sleep(0.05)
            self.assertEqual(client.sent, 5, client.status())
            self.assertEqual(client.failed, 0)
            self.assertEqual(received[0]["source"]["system"], "EnhancedSmartSurveillance")
            self.assertTrue(any(isinstance(r, dict) and r.get("auth") == "Bearer secret123"
                                for r in received))
        finally:
            server.shutdown()
            server.server_close()


# ------------------------------------------------------------------ event store
class EventStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = EventStore(base_dir=self.tmp)

    def test_alerts_events_and_reports_round_trip(self):
        self.assertTrue(self.store.append_alert({"type": "Fall Detected", "confidence": 0.9}))
        self.assertTrue(self.store.append_alert({"type": "Weapon Detected", "confidence": 0.5}))
        self.assertTrue(self.store.append_event(build_event("Test Alert")))
        path = self.store.save_report({"hello": "world"})
        self.assertTrue(path and os.path.exists(path))
        with open(path) as fh:
            self.assertEqual(json.load(fh), {"hello": "world"})

        recent = self.store.load_recent_alerts(10)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["type"], "Weapon Detected")  # newest first

        self.assertEqual(self.store.stats(),
                         {"alerts_logged": 2, "events_logged": 1, "reports_saved": 1})


# ---------------------------------------------------------------- virtual fence
class VirtualFenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = os.path.join(self.tmp, "zones.json")
        self.fence = VirtualFence(config_path=self.config, default_dwell_seconds=0.05)

    def test_intrusion_dwell_exit_and_reentry(self):
        self.fence.add_zone("Restricted", _square())
        self.assertEqual([e["type"] for e in self.fence.evaluate(1, (0.5, 0.5))], ["intrusion"])
        self.assertEqual(self.fence.evaluate(1, (0.5, 0.5)), [])      # no re-fire inside
        time.sleep(0.08)
        self.assertEqual([e["type"] for e in self.fence.evaluate(1, (0.5, 0.5))], ["dwell"])
        self.assertEqual(self.fence.evaluate(1, (0.5, 0.5)), [])      # dwell fires once
        self.assertEqual([e["type"] for e in self.fence.evaluate(1, (0.95, 0.95))], ["exit"])
        self.assertEqual([e["type"] for e in self.fence.evaluate(1, (0.5, 0.5))], ["intrusion"])
        self.assertEqual(self.fence.intrusions, 2)
        self.assertEqual(self.fence.dwell_events, 1)

    def test_points_outside_produce_no_events(self):
        self.fence.add_zone("Restricted", _square())
        self.assertEqual(self.fence.evaluate(7, (0.99, 0.01)), [])

    def test_persistence_validation_and_reset(self):
        self.fence.add_zone("Gate", _square(0.2, 0.2, 0.4, 0.4))
        self.assertTrue(os.path.exists(self.config))
        with open(self.config) as fh:
            self.assertEqual(json.load(fh)["zones"][0]["name"], "Gate")
        self.assertIn("Gate", [z["name"] for z in VirtualFence(config_path=self.config).list_zones()])
        self.assertTrue(self.fence.remove_zone("Gate"))
        self.assertEqual(self.fence.list_zones(), [])
        self.assertFalse(self.fence.remove_zone("Gate"))

        self.assertFalse(VirtualFence._valid_zone({"name": "x", "points": [[0, 0]]}))
        self.assertFalse(VirtualFence._valid_zone({"points": _square()}))

        self.fence.add_zone("Z", _square())
        self.fence.evaluate(1, (0.5, 0.5))
        self.fence.reset_stats()
        self.assertEqual((self.fence.intrusions, self.fence.dwell_events), (0, 0))

    def test_drop_tracks_forgets_stale_ids(self):
        self.fence.add_zone("Z", _square())
        self.fence.evaluate(1, (0.5, 0.5))
        self.fence.evaluate(2, (0.99, 0.01))
        self.assertIn(1, self.fence.track_state)
        self.fence.drop_tracks(set())
        self.assertEqual(self.fence.track_state, {})


# --------------------------------------------------------------------- helpers
class HelperTests(unittest.TestCase):
    def test_hue_to_name(self):
        for hue, name in [(0, "red"), (5, "red"), (15, "orange"), (30, "yellow"),
                          (60, "green"), (90, "cyan"), (110, "blue"), (140, "purple"),
                          (170, "red")]:
            self.assertEqual(_hue_to_name(hue), name, hue)

    def test_iou(self):
        self.assertEqual(_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(_iou((0, 0, 10, 10), (100, 100, 5, 5)), 0.0)
        self.assertTrue(0.0 < _iou((0, 0, 10, 10), (5, 5, 10, 10)) < 1.0)

    def test_normalize_plate(self):
        self.assertEqual(normalize_plate("ab-12 cd!"), "AB12CD")
        self.assertEqual(normalize_plate(None), "")
        self.assertEqual(normalize_plate("   "), "")

    def test_vehicle_detector_stats_reset(self):
        detector = VehicleDetector()
        detector.by_type["car"] = 2
        detector.by_color["blue"] = 1
        detector.total_unique = 2
        detector.reset_stats()
        stats = detector.stats()
        self.assertEqual(stats, {"in_frame": 0, "total_unique": 0, "by_type": {}, "by_color": {}})

    def test_anpr_disabled_skips_ocr_load(self):
        reader = ANPRReader(enabled=False)
        self.assertEqual(reader.backend, "disabled")
        self.assertFalse(reader.available)
        self.assertIsNone(reader.read_plate(None, (0, 0, 100, 100)))
        reader.reset_stats()
        self.assertEqual(reader.total_unique, 0)

    def test_face_detector_stats_shape(self):
        detector = FaceDetector(enabled=False)
        faces = detector.detect(None)
        self.assertEqual(faces, [])
        self.assertEqual(detector.last_faces, [])
        stats = detector.stats()
        for key in ("in_frame", "available", "privacy_blur"):
            self.assertIn(key, stats)

    def test_night_vision_disabled_is_safe(self):
        nv = NightVision(enabled=False)
        result = nv.analyze(None)
        self.assertFalse(result["night_mode"])
        self.assertFalse(result["motion"])
        self.assertIsNone(result["enhanced"])
        nv.reset_stats()
        self.assertEqual(nv.stats()["night_motion_events"], 0)


# -------------------------------------------------------------- object detector
class ObjectDetectorPathTests(unittest.TestCase):
    def test_stock_model_detection(self):
        self.assertTrue(_is_stock_model("yolov8n.pt"))
        self.assertTrue(_is_stock_model("yolov8s.pt"))
        self.assertFalse(_is_stock_model("models/weapons/best.pt"))
        self.assertFalse(_is_stock_model("/tmp/custom_yolov8.pt"))

    def test_find_weapon_model_honours_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "best.pt")
            with open(weights, "w") as fh:
                fh.write("not-really-weights")
            old = os.environ.get("WEAPON_MODEL_PATH")
            try:
                os.environ["WEAPON_MODEL_PATH"] = weights
                self.assertEqual(find_weapon_model(), weights)
                os.environ["WEAPON_MODEL_PATH"] = os.path.join(tmp, "missing.pt")
                self.assertIsNone(find_weapon_model())
            finally:
                if old is None:
                    os.environ.pop("WEAPON_MODEL_PATH", None)
                else:
                    os.environ["WEAPON_MODEL_PATH"] = old


@unittest.skipUnless(_STUBBED["ultralytics"], "requires the Ultralytics stand-in")
class ObjectDetectorInferenceTests(unittest.TestCase):
    def setUp(self):
        os.environ["HAZARDOUS_CONF_THRESHOLD"] = "0.4"

    def _result(self, xyxy, conf, cls):
        from ultralytics import Boxes, Result
        return Result(Boxes(xyxy, conf, cls))

    def test_stock_model_uses_coco_hazardous_classes(self):
        detector = ObjectDetector("yolov8n.pt")
        self.assertFalse(detector.is_custom)
        self.assertIn("knife", detector.hazardous_labels)
        self.assertIn("bottle", detector.hazardous_labels)
        self.assertNotIn("handgun", detector.hazardous_labels)

    def test_custom_model_treats_every_class_as_hazardous(self):
        detector = ObjectDetector("models/weapons/best.pt")
        self.assertTrue(detector.is_custom)
        self.assertEqual(sorted(detector.hazardous_labels), ["handgun", "knife"])

    def test_detect_objects_flags_only_hazardous_above_threshold(self):
        detector = ObjectDetector("yolov8n.pt")
        detector.model._results = [self._result(
            xyxy=[[0, 0, 50, 100], [60, 60, 90, 80], [10, 10, 20, 20]],
            conf=[0.95, 0.80, 0.30],
            cls=[0, 43, 43])]
        detections, hazardous = detector.detect_objects(object())
        self.assertEqual(len(detections), 3)
        self.assertEqual([h["class_name"] for h in hazardous], ["knife"])
        self.assertEqual(hazardous[0]["bbox"], (60, 60, 90, 80))

    def test_custom_model_uses_its_own_class_names(self):
        detector = ObjectDetector("models/weapons/best.pt")
        detector.model._results = [self._result(xyxy=[[0, 0, 10, 10]], conf=[0.7], cls=[1])]
        _, hazardous = detector.detect_objects(object())
        self.assertEqual([h["class_name"] for h in hazardous], ["handgun"])


# ------------------------------------------------------- app.py / dashboard glue
class AppWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "app.py")) as fh:
            cls.app_src = fh.read()
        with open(os.path.join(ROOT, "templates", "index.html")) as fh:
            cls.html = fh.read()

    def test_all_new_modules_are_imported_by_app(self):
        for module in ["vehicle_detector", "face_detector", "anpr", "virtual_fence",
                       "night_vision", "c2_integration", "event_store"]:
            self.assertIn(f"from {module} import", self.app_src, module)

    def test_app_only_uses_attributes_that_exist(self):
        used = {}
        for node in ast.walk(ast.parse(self.app_src)):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in MODULE_NAMES):
                used.setdefault(node.value.id, set()).add(node.attr)
        self.assertGreaterEqual(len(used), 6, f"unexpectedly few module usages: {used}")

        with tempfile.TemporaryDirectory() as tmp:
            instances = {
                "vehicle_detector": VehicleDetector(),
                "face_detector": FaceDetector(),
                "anpr_reader": ANPRReader(enabled=False),
                "virtual_fence": VirtualFence(config_path=os.path.join(tmp, "zones.json")),
                "night_vision": NightVision(),
                "c2_client": C2Client(webhook_url=""),
                "event_store": EventStore(base_dir=tmp),
            }
            for name, attrs in used.items():
                # app.py only calls ObjectDetector's methods, so the class is enough
                # (and this avoids constructing/downloading a real YOLO model here).
                target = ObjectDetector if name == "object_detector" else instances[name]
                for attr in sorted(attrs):
                    self.assertTrue(hasattr(target, attr),
                                    f"app.py uses {name}.{attr}, which does not exist")

    def test_every_endpoint_the_dashboard_calls_exists(self):
        routes = set(re.findall(r"@app\.route\('([^']+)'", self.app_src))
        for endpoint in ["/", "/video_feed", "/api/stats", "/api/alerts", "/api/detections",
                         "/api/vehicles", "/api/faces", "/api/anpr", "/api/zones",
                         "/api/zones/<name>", "/api/c2/status", "/api/c2/test",
                         "/api/report", "/api/clear_alerts", "/api/weapon_detections",
                         "/api/behavior_history", "/api/test_alert"]:
            self.assertIn(endpoint, routes, endpoint)

        called = set(re.findall(r"fetch\(\s*'(/api/[^']+)'", self.html))
        self.assertTrue(called)
        for endpoint in called:
            self.assertIn(endpoint, routes, f"dashboard fetches {endpoint}, which has no route")

    def test_every_element_id_the_dashboard_references_exists_exactly_once(self):
        referenced = set(re.findall(r"getElementById\('([^']+)'\)", self.html))
        self.assertTrue(referenced)
        for element_id in sorted(referenced):
            count = len(re.findall(r'id="%s"' % re.escape(element_id), self.html))
            self.assertEqual(count, 1, f"#{element_id} appears {count} times in the template")

    def test_dashboard_markup_is_balanced(self):
        self.assertEqual(len(re.findall(r"<div\b", self.html)), len(re.findall(r"</div>", self.html)))
        self.assertEqual(len(re.findall(r"<script\b", self.html)), len(re.findall(r"</script>", self.html)))
        self.assertEqual(len(re.findall(r"<style\b", self.html)), len(re.findall(r"</style>", self.html)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
