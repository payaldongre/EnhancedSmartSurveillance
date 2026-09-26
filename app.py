from flask import Flask, Response, render_template, jsonify, request
from ultralytics import YOLO
import mediapipe as mp
import threading
import time
import numpy as np
from datetime import datetime
import json
import os
from collections import deque, defaultdict
import math
from mediapipe.framework.formats import landmark_pb2
from speed_estimator import process_person_detections
import cv2
import torch
from ultralytics.nn.tasks import DetectionModel
import traceback
import sys

from object_detector import ObjectDetector, find_weapon_model  # FIX: use the corrected module
from vehicle_detector import VehicleDetector
from face_detector import FaceDetector
from anpr import ANPRReader
from virtual_fence import VirtualFence
from night_vision import NightVision
from c2_integration import C2Client, build_event
from event_store import EventStore
from cameras import CAMERAS

torch.serialization.add_safe_globals([DetectionModel])

# ------------------------------------------------------
# Config (tunable)
# ------------------------------------------------------
PERSON_MODEL = "yolov8s.pt"  # FIX: "s" (small) trades speed for accuracy vs "n" (nano).
                               # If lag is still your bigger problem than missed people,
                               # switch this to "yolov8n.pt" — meaningfully faster, some
                               # accuracy cost. Test both and pick per your demo needs.
PERSON_IMGSZ = 320          # inference resolution for person tracking. Raise (e.g. 480)
                             # if you're missing distant/small people and can afford the
                             # extra latency; lower it if lag is still bad on your machine.
PERSON_TRACKER = "bytetrack.yaml"  # FIX: ships with ultralytics, no extra install needed.
                                    # Lighter/faster than the previous default (botsort.yaml).

OBJECT_DETECT_EVERY_N_FRAMES = 2   # FIX: the hazardous-object pass was the single most
                                     # expensive call per frame (full-res, no imgsz cap).
                                     # Running it every other frame roughly halves that cost;
                                     # we keep drawing the last known boxes on skipped frames.

POSE_ROI_MAX_DIM = 256      # FIX: cap each person's crop before MediaPipe pose estimation.
                             # Cost multiplies with headcount (one call per person per frame),
                             # so this matters most exactly when you have several people in
                             # frame — like the 4-5 person screenshot that dropped FPS to ~3.

RUNNING_SPEED_FRACTION = 0.20
RUNNING_MOVEMENT_THRESHOLD = 0.06
RUNNING_DETECTION_FRAMES = 15
RUNNING_CONFIRM_COUNT = 10
RUNNING_HYSTERESIS_COUNT = 8
DEBUG_RUNNING_LOG = False
RUNNING_TORSO_SPEED = 0.6
EMA_ALPHA = 0.6

WEAPON_DETECTION_CONFIDENCE = 0.5
WEAPON_PROXIMITY_THRESHOLD = 200


# ------------------------------------------------------
# Feature toggles for the added capabilities (env-overridable).
# ------------------------------------------------------
def _env_flag(name, default=True):
    """Read a boolean toggle from the environment (falls back to `default`)."""
    return os.environ.get(name, "1" if default else "0").strip().lower() in (
        "1", "true", "yes", "on")


ENABLE_VEHICLE_DETECTION = _env_flag("ENABLE_VEHICLE_DETECTION", True)
ENABLE_FACE_DETECTION = _env_flag("ENABLE_FACE_DETECTION", True)
ENABLE_ANPR = _env_flag("ENABLE_ANPR", True)
ENABLE_VIRTUAL_FENCE = _env_flag("ENABLE_VIRTUAL_FENCE", True)
ENABLE_NIGHT_VISION = _env_flag("ENABLE_NIGHT_VISION", True)
FACE_PRIVACY_BLUR = _env_flag("FACE_PRIVACY_BLUR", False)
ALERT_ON_NEW_VEHICLE = _env_flag("ALERT_ON_NEW_VEHICLE", False)

# Issue 2: face detection (a Haar cascade over the full frame) and ANPR OCR are
# the CPU-bound extras that are easy to keep paying for even when nothing relevant
# is on screen. They are now throttled harder AND gated on there actually being a
# person / vehicle in frame. Read the [Timing] breakdown logged by
# video_processing_thread before tuning these further.
FACE_DETECT_EVERY_N_FRAMES = int(os.environ.get("FACE_DETECT_EVERY_N_FRAMES", "8"))
ANPR_EVERY_N_FRAMES = int(os.environ.get("ANPR_EVERY_N_FRAMES", "15"))  # used for id-less vehicles
ANPR_MAX_NEW_PER_FRAME = int(os.environ.get("ANPR_MAX_NEW_PER_FRAME", "1"))
# Camera list. The cameras live in cameras.py so you can add/remove them in
# code (no `set` command needed); the FIRST entry is the analytics camera that
# runs the full AI pipeline, and any further entries stream live to the same
# dashboard. See cameras.py for the entry format.
#
# Fallbacks, used only when cameras.py defines no cameras:
#   CAMERA_SOURCES (comma-separated `id=source` entries), then the legacy
#   CAMERA_ID / CAMERA_INDEX pair.
# Issue 3: the dashboard now shows the feed full-width, where 640x480 looked soft
# once stretched. 1280x720 is the new default; a per-camera width/height in
# cameras.py still wins, and narrower resolutions cost less CPU per frame — the
# [Timing] log shows what the extra pixels actually cost on your machine.
DEFAULT_CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "1280"))
DEFAULT_CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "720"))


def _camera_entries_from_env():
    """Build the fallback camera list from CAMERA_SOURCES / CAMERA_INDEX."""
    entries = []
    sources = os.environ.get("CAMERA_SOURCES", "").strip()
    if sources:
        for token in sources.split(","):
            token = token.strip()
            if not token:
                continue
            if "=" in token:
                cid, src = token.split("=", 1)
                cid, src = cid.strip(), src.strip()
            else:
                cid, src = "", token
            entries.append({"id": cid or f"Cam_{len(entries) + 1}", "source": src})
    if not entries:
        entries.append({"id": os.environ.get("CAMERA_ID", "Cam_1"),
                        "source": os.environ.get("CAMERA_INDEX", "0")})
    return entries


def _camera_entries():
    """Resolve every camera to a dict of id/source/width/height.

    cameras.py wins when it defines any camera; otherwise the env fallbacks
    keep the previous single-camera behavior working unchanged.
    """
    raw = list(CAMERAS) if CAMERAS else _camera_entries_from_env()
    entries = []
    for cam in raw:
        if not isinstance(cam, dict) or "source" not in cam:
            continue
        entries.append({
            "id": str(cam.get("id") or f"Cam_{len(entries) + 1}"),
            "source": cam["source"],
            "width": int(cam.get("width") or DEFAULT_CAMERA_WIDTH),
            "height": int(cam.get("height") or DEFAULT_CAMERA_HEIGHT),
        })
    if not entries:
        entries.append({"id": "Cam_1", "source": "0",
                        "width": DEFAULT_CAMERA_WIDTH, "height": DEFAULT_CAMERA_HEIGHT})
    return entries[0], entries[1:]


_PRIMARY_CAMERA, _EXTRA_CAMERAS = _camera_entries()
CAMERA_ID = _PRIMARY_CAMERA["id"]
CAMERA_INDEX = _PRIMARY_CAMERA["source"]
CAMERA_WIDTH = _PRIMARY_CAMERA["width"]
CAMERA_HEIGHT = _PRIMARY_CAMERA["height"]

# FIX: detect device BEFORE creating either model, so both can be told
# explicitly which device to use rather than relying on Ultralytics' auto-detect.
device_in_use = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
INFERENCE_DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"Inference device: {device_in_use} | torch {torch.__version__}"
      + (f" | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}"
         if torch.cuda.is_available() else ""))
if not torch.cuda.is_available():
    if "+cu" in torch.__version__:
        print("  torch is a CUDA build but no GPU is visible — check the NVIDIA driver (`nvidia-smi`).")
    else:
        print("  torch is a CPU-only build. For GPU acceleration, reinstall torch from the CUDA index (README step 4).")
print("Note: MediaPipe Pose always runs on CPU regardless of this setting — the "
      "Windows pip build has no GPU delegate. Only the two YOLO calls get the GPU boost.")

# Prefer a custom weapon checkpoint (Roboflow knife+gun best.pt) when present;
# otherwise fall back to stock COCO (knife / scissors / baseball bat / bottle).
WEAPON_MODEL_PATH = find_weapon_model()
if WEAPON_MODEL_PATH:
    print(f"Custom weapon model found: {WEAPON_MODEL_PATH}")
else:
    print("No custom weapon model found — falling back to the stock COCO model.")

object_detector = ObjectDetector(WEAPON_MODEL_PATH or "yolov8n.pt",
                                 imgsz=320, device=INFERENCE_DEVICE)
_od_stats = object_detector.stats()
if _od_stats["custom_weapon_model"]:
    print(f"Custom weapon model classes: {_od_stats['model_classes']}")
    print(f"Flagging as hazardous: {_od_stats['hazardous_classes']} "
          "(override with HAZARDOUS_LABELS=knife,gun)")

# ------------------------------------------------------
# Added capability modules (each documented in its own file)
# ------------------------------------------------------
vehicle_detector = VehicleDetector()
face_detector = FaceDetector(enabled=ENABLE_FACE_DETECTION, privacy_blur=FACE_PRIVACY_BLUR)
anpr_reader = ANPRReader(enabled=ENABLE_ANPR, gpu=torch.cuda.is_available())
virtual_fence = VirtualFence(enabled=ENABLE_VIRTUAL_FENCE)
night_vision = NightVision(enabled=ENABLE_NIGHT_VISION)
c2_client = C2Client()
event_store = EventStore()

print(f"ANPR backend: {anpr_reader.backend} | Faces: "
      f"{'on' if face_detector.available else 'unavailable'} | "
      f"Zones: {len(virtual_fence.list_zones())} | "
      f"C2: {'on' if c2_client.enabled else 'off'}")
if ENABLE_VIRTUAL_FENCE and not virtual_fence.list_zones():
    print("[VirtualFence] WARNING: no zones loaded — the virtual fence will not fire. "
          "Add one via zones.json or POST /api/zones.")


def _camera_listing():
    """Uniform list of every configured camera for the dashboard/API."""
    cams = [{
        'camera_id': CAMERA_ID,
        'source': CAMERA_INDEX,
        'detect': True,
        'running': latest_frame is not None,
        # "connecting" / "live" / "reconnecting" / "error: <reason>"
        'status': _PRIMARY_GRABBER.status if _PRIMARY_GRABBER is not None else "connecting",
        'resolution': (_PRIMARY_GRABBER.resolution() if _PRIMARY_GRABBER is not None
                       else f"{CAMERA_WIDTH}x{CAMERA_HEIGHT} (requested)"),
        'fps': round(system_stats.get('fps', 0), 1),
    }]
    cams += [cam.public_stats() for cam in _STREAM_CAMERAS]
    return cams


app = Flask(__name__)

# ------------------------------------------------------
# Global state
# ------------------------------------------------------
latest_frame = None
lock = threading.Lock()
alerts_list = []
# The analytics camera's FrameGrabber (assigned in video_processing_thread); read
# by /api/cameras so the dashboard can show a real per-camera connection status.
_PRIMARY_GRABBER = None
system_stats = {
    'total_detections': 0,
    'falls_detected': 0,
    'abnormal_behaviors': 0,
    'weapons_detected': 0,
    'system_uptime': time.time(),
    'fps': 0,
    'pose_detections': 0,
    'vehicles_detected': 0,
    'vehicles_in_frame': 0,
    'vehicles_by_type': {},
    'faces_in_frame': 0,
    'plates_read': 0,
    'intrusions': 0,
    'night_mode': False,
    'brightness': 0.0,
    'motion': False,
    'c2_enabled': False,
    'c2_sent': 0,
    'c2_queued': 0
}

print("Loading AI models... Please wait...")
model = YOLO(PERSON_MODEL)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("AI models loaded successfully!")

last_alert_time = 0
last_plate_by_track = {}   # track_id -> last plate read (avoids re-alerting the same plate)
anpr_attempted_tracks = set()  # vehicle track_ids already OCR'd; ANPR tries each once
pose_behavior_persisted = {}   # track_id -> last behaviour written to the pose-sequence store
alert_cooldown = 3
active_weapon_alerts = {}   # (track_id, weapon_label) -> last_seen_time — see FIX note below
WEAPON_ALERT_TIMEOUT = 6.0  # seconds a weapon can go undetected near this person before
                             # a reappearance counts as a genuinely new event

# FIX (critical): these were previously single global deques SHARED across every
# person in frame. With more than one person tracked, person B's "movement" would
# be computed against person A's landmarks (whoever was processed last, possibly
# in the very same frame) instead of B's own previous frame — two different
# people at different screen positions look like a huge, spurious movement no
# matter how still everyone actually is. This is why every bounding box flashed
# "Running Detected (0.80)" simultaneously regardless of headcount. Now keyed
# per track_id so each person is only ever compared against themselves.
# Each pose_history entry is (landmarks, timestamp) so movement can also be
# normalized by real elapsed time (see NOMINAL_FRAME_DT below) instead of
# assuming a fixed ~30fps frame interval that low-FPS runs don't actually hit.
NOMINAL_FRAME_DT = 1.0 / 30.0  # the frame interval the original thresholds were tuned for
pose_history_by_track = defaultdict(lambda: deque(maxlen=RUNNING_DETECTION_FRAMES))
behavior_buffer_by_track = defaultdict(lambda: deque(maxlen=RUNNING_DETECTION_FRAMES))
recent_behaviors_log = deque(maxlen=50)  # for the /api/behavior_history endpoint only
loitering_tracker = {}

# Shared state for the last hazardous-object detection pass (reused on skipped frames)
last_hazardous_detections = []


def torso_size(landmarks):
    """Return approximate torso size (used to normalize pose movement)."""
    try:
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_width = math.hypot(left_shoulder.x - right_shoulder.x, left_shoulder.y - right_shoulder.y)
        hip_center_y = (left_hip.y + right_hip.y) / 2.0
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2.0
        torso_height = abs(hip_center_y - shoulder_center_y)
        return max(shoulder_width, torso_height, 1e-3)
    except Exception:
        return 1.0


class PoseAnalyzer:
    """Class to analyze human poses and detect abnormal behaviors.

    FIX: this class's classify_behavior() is now ONLY used as the fallback
    signal that feeds the single, torso-normalized running/fall/abnormal
    decision made once per frame in video_processing_thread(). Previously,
    its raw (non-normalized) movement_intensity threshold was ALSO used a
    second time, later in the loop, to independently re-classify and
    re-draw the behavior label — overwriting the better, normalized result.
    That duplicate pass is what caused "running" to fire on stationary
    people: any bounding-box jitter shifts every landmark (nose included)
    between frames, spiking raw movement_intensity even when the person
    hasn't moved. It has been removed; there is now exactly one
    classification path per frame.
    """

    def __init__(self):
        self.normal_standing_threshold = 0.3
        self.fall_threshold = 0.4
        self.movement_threshold = 0.15
        self.running_threshold = 0.5
        self.fall_velocity_threshold = 0.1

    def calculate_pose_metrics(self, landmarks, prev_landmarks=None, dt=None):
        if not landmarks:
            return None

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        knee_center_y = (left_knee.y + right_knee.y) / 2
        ankle_center_y = (left_ankle.y + right_ankle.y) / 2

        body_height = abs(nose.y - ankle_center_y)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        if shoulder_width > 0:
            verticality = min(body_height / shoulder_width, 2.0) / 2.0
        else:
            verticality = 0.5

        head_body_ratio = abs(nose.y - shoulder_center_y) / max(abs(shoulder_center_y - ankle_center_y), 0.1)

        movement_intensity = 0.0
        vertical_velocity = 0.0
        if prev_landmarks:
            movement_intensity = self.calculate_movement(landmarks, prev_landmarks)
            prev_hip_y = (prev_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y +
                          prev_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
            curr_hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y +
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
            vertical_velocity = curr_hip_y - prev_hip_y

            # FIX: rescale both deltas by how much real time actually passed vs. the
            # ~30fps interval the fixed thresholds below were tuned for. Without this,
            # a system running at 3 FPS (10x slower) sees 10x the real motion between
            # samples and every threshold reads as triggered even when people are still.
            if dt and dt > 1e-6:
                scale = min(NOMINAL_FRAME_DT / dt, 1.0)  # only ever discount, never amplify
                movement_intensity *= scale
                vertical_velocity *= scale

        return {
            'verticality': verticality,
            'head_body_ratio': head_body_ratio,
            'movement_intensity': movement_intensity,
            'vertical_velocity': vertical_velocity,
            'shoulder_center_y': shoulder_center_y,
            'hip_center_y': hip_center_y,
            'knee_center_y': knee_center_y,
            'ankle_center_y': ankle_center_y
        }

    def calculate_movement(self, current_landmarks, previous_landmarks):
        if not current_landmarks or not previous_landmarks:
            return 0.0

        total_movement = 0.0
        key_points = [
            mp_pose.PoseLandmark.NOSE,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE
        ]
        for landmark in key_points:
            curr = current_landmarks[landmark.value]
            prev = previous_landmarks[landmark.value]
            distance = math.sqrt((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2)
            total_movement += distance
        return total_movement / len(key_points)

    def classify_behavior(self, pose_metrics):
        if not pose_metrics:
            return "unknown", 0.0

        verticality = pose_metrics['verticality']
        movement = pose_metrics['movement_intensity']
        head_body_ratio = pose_metrics['head_body_ratio']
        vertical_velocity = pose_metrics['vertical_velocity']

        if verticality < self.fall_threshold and vertical_velocity > self.fall_velocity_threshold:
            return "fall", 0.95

        if movement > self.running_threshold:
            return "running", 0.8

        abnormal_score = 0.0
        if verticality < self.normal_standing_threshold:
            abnormal_score += 0.3
        if movement > self.movement_threshold:
            abnormal_score += 0.2
        if head_body_ratio < 0.1 or head_body_ratio > 0.6:
            abnormal_score += 0.2

        if abnormal_score > 0.4:
            return "abnormal", abnormal_score
        else:
            return "normal", 1.0 - abnormal_score


pose_analyzer = PoseAnalyzer()


def log_alert(alert_type, confidence=0.0, details="", data=None):
    global alerts_list, system_stats
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_data = {
        'timestamp': timestamp,
        'type': alert_type,
        'confidence': confidence,
        'details': details,
        'id': len(alerts_list) + 1,
        'data': data or {}
    }
    alerts_list.insert(0, alert_data)
    if len(alerts_list) > 50:
        alerts_list = alerts_list[:50]

    if alert_type == "Fall Detected":
        system_stats['falls_detected'] += 1
    elif alert_type == "Abnormal Behavior":
        system_stats['abnormal_behaviors'] += 1
    elif alert_type == "Running Detected":
        system_stats['abnormal_behaviors'] += 1
    elif alert_type == "Loitering Detected":
        system_stats['abnormal_behaviors'] += 1
    elif alert_type == "Weapon Detected":
        system_stats['weapons_detected'] += 1

    print(f"ALERT: {alert_type} at {timestamp} - {details}")

    # Persist to disk and forward to any configured Command & Control system.
    event_store.append_alert(alert_data)
    try:
        event = build_event(alert_type, confidence, CAMERA_ID, details,
                            source_id=c2_client.source_id, data=data)
        event_store.append_event(event)
        c2_client.publish(event)
    except Exception as exc:  # never let integration break the pipeline
        print(f"[C2] publish failed: {exc}")


# ------------------------------------------------------
# Per-key alert throttling.
#
# Unlike the single global `last_alert_time`/`alert_cooldown` pair (where one
# alert type silences every other one for a few seconds), this keys the
# cooldown by event identity so unrelated alerts can fire together while
# repeats of the same event are still suppressed.
# ------------------------------------------------------
_throttle_last = {}


def alert_throttled(key, alert_type, confidence=0.0, details="", cooldown=5.0, data=None):
    now = time.time()
    if now - _throttle_last.get(key, 0.0) < cooldown:
        return False
    _throttle_last[key] = now
    log_alert(alert_type, confidence, details, data=data)
    return True


def _handle_zone_event(ev):
    """Translate a virtual-fence event into the appropriate alert."""
    if ev['type'] == 'intrusion':
        alert_throttled(
            ('zone-entry', ev['zone'], ev['track_id']), "Intrusion Detected", 0.9,
            f"{ev['label']} entered restricted zone '{ev['zone']}'", cooldown=5.0,
            data={'zone': ev['zone'], 'track_id': ev['track_id'], 'label': ev['label']})
    elif ev['type'] == 'dwell':
        alert_throttled(
            ('zone-dwell', ev['zone'], ev['track_id']), "Restricted Zone Dwell", 0.8,
            f"{ev['label']} loitering in '{ev['zone']}' for {ev.get('seconds', 0)}s",
            cooldown=15.0,
            data={'zone': ev['zone'], 'track_id': ev['track_id'],
                  'seconds': ev.get('seconds')})


# ------------------------------------------------------
# Capture reliability (Issue 1) + per-stage timing (Issue 2).
#
# The grabber below is still a dedicated capture thread decoupled from
# inference (so the stream keeps only the newest frame instead of building up
# lag), but it now also owns its own reconnection: the previous version opened
# cv2.VideoCapture(source) once with no retry, so a phone/IP camera that
# dropped mid-session killed its feed for the rest of the run.
# ------------------------------------------------------
CAMERA_RECONNECT_AFTER_FAILURES = int(
    os.environ.get("CAMERA_RECONNECT_AFTER_FAILURES", "20"))
CAMERA_RECONNECT_BACKOFF_START = 1.0   # seconds before the first reopen attempt
CAMERA_RECONNECT_BACKOFF_MAX = 10.0    # backoff ceiling — retries continue forever
CAMERA_OPEN_TIMEOUT_MSEC = int(os.environ.get("CAMERA_OPEN_TIMEOUT_MSEC", "8000"))

_OPEN_TIMEOUT_PROP = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
_READ_TIMEOUT_PROP = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)

# Issue 2: how often the per-stage timing breakdown is logged (in frames).
STAGE_REPORT_EVERY_N_FRAMES = int(os.environ.get("STAGE_REPORT_EVERY_N_FRAMES", "30"))
STAGE_NAMES = ("night_vision", "person_tracking", "object_detection", "vehicle",
               "anpr", "face", "pose")


def _log_stage_timing(stage_ms, frames, persons, poses):
    """Log the measured per-frame cost of every pipeline stage.

    Read THIS before tuning any throttle — it says where the time actually goes.
    `other` is the loop's own bookkeeping / drawing / JPEG-encode remainder.
    """
    frames = max(1, frames)
    total = stage_ms.get("frame_total", 0.0) / frames
    parts = []
    accounted = 0.0
    for name in STAGE_NAMES:
        ms = stage_ms.get(name, 0.0) / frames
        accounted += ms
        parts.append(f"{name}={ms:.1f}ms")
    other = max(0.0, total - accounted)
    print(f"[Timing] avg/frame over {frames} frames: total={total:.0f}ms "
          f"(throughput ~{1000.0 / max(total, 1e-6):.1f} fps) | "
          f"people={persons} poses={poses} | " + " ".join(parts) +
          f" | other={other:.1f}ms")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"[GPU] torch.cuda allocated={alloc:.0f}MB reserved={reserved:.0f}MB "
              "(non-zero / growing means this run really is inferring on CUDA)")
class FrameGrabber:
    """Continuously capture from `source`, reconnecting on its own when it drops.

    `status` is one of "connecting", "live", "reconnecting" or "error: <reason>"
    and is published per camera through /api/cameras, so the dashboard shows the
    real connection state instead of a console-only message.
    """

    def __init__(self, source=0, width=640, height=480):
        # A numeric source (or numeric string) is a local camera index; any other
        # string (a video-file path, or an RTSP/HTTP stream URL such as a phone
        # running an "IP Webcam" style app) is opened as a stream.
        if isinstance(source, str) and source.strip().isdigit():
            source = int(source.strip())
        self.source = source
        self.width = width
        self.height = height

        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.frame_size = None          # actual (w, h), known once frames arrive
        self.cap = None
        self.consecutive_failures = 0
        self.connected_once = False
        self.status = "connecting"
        self._backoff = CAMERA_RECONNECT_BACKOFF_START

        # The capture thread always runs: it owns reconnection, so a camera that
        # is absent at startup (or drops later) recovers by itself.
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------ open / close
    def _candidate_backends(self):
        """Backend hints to try, in order.

        A local device index keeps DirectShow on Windows (previous behaviour).
        Any non-numeric source is tried with the default backend first and then
        with FFMPEG explicitly — FFMPEG's HTTP/RTSP handling is the part that
        makes an IP-Webcam style stream come back reliably.
        """
        if isinstance(self.source, int):
            return [cv2.CAP_DSHOW] if sys.platform.startswith("win") else [cv2.CAP_ANY]
        return [cv2.CAP_ANY, cv2.CAP_FFMPEG]

    def _open_capture(self):
        """Open the source. Returns (cap, error_message), with cap=None on failure."""
        params = []
        if not isinstance(self.source, int):
            # Connect/read timeout: a bad IP or an unreachable phone must fail the
            # open quickly instead of hanging startup (or a reconnect) for minutes.
            if _OPEN_TIMEOUT_PROP is not None:
                params += [int(_OPEN_TIMEOUT_PROP), int(CAMERA_OPEN_TIMEOUT_MSEC)]
            if _READ_TIMEOUT_PROP is not None:
                params += [int(_READ_TIMEOUT_PROP), int(CAMERA_OPEN_TIMEOUT_MSEC)]

        last_error = f"could not open {self.source!r}"
        for backend in self._candidate_backends():
            cap = None
            try:
                if params:
                    try:
                        cap = cv2.VideoCapture(self.source, backend, params)
                    except TypeError:
                        # OpenCV built without the params overload — open plainly.
                        cap = cv2.VideoCapture(self.source, backend)
                else:
                    cap = cv2.VideoCapture(self.source, backend)
            except Exception as exc:
                last_error = str(exc)
                continue
            if cap is not None and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # shallow buffer: prefer fresh frames
                except Exception:
                    pass
                return cap, ""
            if cap is not None:
                cap.release()
            last_error = (f"backend {backend} could not open {self.source!r} "
                          "(no device / bad URL / unreachable host)")
        return None, last_error

    def _release_cap(self):
        cap, self.cap = self.cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _sleep_backoff(self):
        """Wait the current backoff, then grow it (capped) for the next attempt."""
        time.sleep(self._backoff)
        self._backoff = min(self._backoff * 2, CAMERA_RECONNECT_BACKOFF_MAX)

    # ------------------------------------------------------------------- loop
    def _loop(self):
        while self.running:
            if self.cap is None:
                self.status = "reconnecting" if self.connected_once else "connecting"
                cap, error = self._open_capture()
                if cap is None:
                    # Never give up: publish the reason and retry forever.
                    self.status = f"error: {error}"
                    print(f"[FrameGrabber] {self.source!r}: {error} — retrying in "
                          f"{self._backoff:.0f}s")
                    self._sleep_backoff()
                    continue
                self.cap = cap
                self.connected_once = True
                self.consecutive_failures = 0
                self._backoff = CAMERA_RECONNECT_BACKOFF_START
                self.status = "live"
                print(f"[FrameGrabber] {self.source!r} connected (live)")
                continue

            try:
                success, frame = self.cap.read()
            except Exception as exc:      # some backends raise instead of returning False
                if self.consecutive_failures == 0:
                    print(f"[FrameGrabber] {self.source!r}: read raised {exc}")
                success, frame = False, None

            if success and frame is not None:
                self.consecutive_failures = 0
                self._backoff = CAMERA_RECONNECT_BACKOFF_START
                self.status = "live"
                self.frame_size = (frame.shape[1], frame.shape[0])
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                continue

            # Failed read — reopen the capture after enough consecutive failures.
            self.consecutive_failures += 1
            if self.consecutive_failures >= CAMERA_RECONNECT_AFTER_FAILURES:
                print(f"[FrameGrabber] {self.source!r}: "
                      f"{self.consecutive_failures} failed reads in a row — reconnecting")
                self.status = "reconnecting"
                self.consecutive_failures = 0
                self._release_cap()
                self._sleep_backoff()
            else:
                time.sleep(0.01)

    def resolution(self):
        """Actual captured resolution, or the requested one before any frame."""
        if self.frame_size:
            return f"{self.frame_size[0]}x{self.frame_size[1]}"
        return f"{self.width}x{self.height} (requested)"

    def get_latest(self):
        with self.lock:
            return self.frame, self.frame_id

    def release(self):
        self.running = False
        self._release_cap()


class StreamCamera:
    """A lightweight live feed for an ADDITIONAL camera.

    Extra cameras stream to the dashboard but deliberately skip the AI models:
    each detection pipeline loads its own YOLO/MediaPipe instances and is
    CPU-bound, so running the full stack per camera would multiply the load (and
    the lag). Detection runs on the analytics camera — the first CAMERA_SOURCES
    entry, or CAMERA_INDEX when CAMERA_SOURCES is unset. Promote a different
    camera by putting it first.
    """

    def __init__(self, camera_id, source, width=640, height=480):
        self.id = camera_id
        self.source = str(source)
        self.detect = False
        self.latest_frame = None
        self._lock = threading.Lock()
        self.grabber = FrameGrabber(source, width, height)
        # The stream thread stays up while the grabber reconnects in the
        # background, so a camera that is absent at startup is not disabled for good.
        self.running = True
        self._last_frame_id = -1
        self.fps = 0.0
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        # The grabber reconnects on its own, so this thread simply waits while the
        # camera is absent and resumes as soon as it comes back. The live status is
        # published through /api/cameras instead of a one-off console message.
        print(f"[Camera {self.id}] streaming {self.source!r} "
              f"(live view — AI detection runs on {CAMERA_ID})")
        count = 0
        start_time = time.time()
        while True:
            frame, frame_id = self.grabber.get_latest()
            if frame is None or frame_id == self._last_frame_id:
                time.sleep(0.005)
                continue
            self._last_frame_id = frame_id
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Cam: {self.id} (live view)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            with self._lock:
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    self.latest_frame = jpeg.tobytes()
            count += 1
            if count % 10 == 0:
                now = time.time()
                self.fps = 10.0 / max(1e-6, now - start_time)
                start_time = now

    def public_stats(self):
        return {
            'camera_id': self.id,
            'source': self.source,
            'detect': False,
            'running': self.latest_frame is not None,
            'status': self.grabber.status,
            'resolution': self.grabber.resolution(),
            'fps': round(self.fps, 1),
        }


# Build the extra (live-only) cameras now that StreamCamera is defined. The
# primary camera's analytics pipeline is started from __main__ below.
_STREAM_CAMERAS = [StreamCamera(cam["id"], cam["source"], cam["width"], cam["height"])
                   for cam in _EXTRA_CAMERAS]
_STREAM_BY_ID = {cam.id: cam for cam in _STREAM_CAMERAS}


def video_processing_thread():
    global latest_frame, last_alert_time, system_stats, loitering_tracker
    global last_hazardous_detections, _PRIMARY_GRABBER

    grabber = FrameGrabber(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    _PRIMARY_GRABBER = grabber
    print(f"Capture started for camera {CAMERA_ID} ({CAMERA_INDEX!r}) — the grabber "
          "reconnects automatically if the feed drops (status on /api/cameras)")
    print("Starting video processing with pose estimation...")

    frame_count = 0
    start_time = time.time()
    last_processed_frame_id = -1
    # Issue 2: per-stage millisecond totals for the current reporting window.
    # _log_stage_timing() prints the breakdown every STAGE_REPORT_EVERY_N_FRAMES
    # frames so the real bottleneck is measured instead of guessed.
    stage_ms = defaultdict(float)
    stage_frames = 0

    while True:
        frame_start = time.time()
        current_time = frame_start

        frame, frame_id = grabber.get_latest()
        if frame is None or frame_id == last_processed_frame_id:
            time.sleep(0.005)
            continue
        last_processed_frame_id = frame_id

        frame_count += 1
        frame = cv2.flip(frame, 1)

        # Night-time / low-light assessment; when dark, `enhanced` is a
        # brightness-lifted copy used for both detection and display.
        _t = time.perf_counter()
        night_result = night_vision.analyze(frame)
        stage_ms['night_vision'] += (time.perf_counter() - _t) * 1000
        frame = night_result['enhanced']
        night_motion = night_result['motion']
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Person detection + tracking
        _t = time.perf_counter()
        yolo_results = model.track(
            frame, persist=True, imgsz=PERSON_IMGSZ,
            tracker=PERSON_TRACKER, device=INFERENCE_DEVICE, verbose=False
        )[0]
        stage_ms['person_tracking'] += (time.perf_counter() - _t) * 1000

        # FIX: throttle the heaviest call (hazardous-object detection) instead
        # of running it on every single frame; reuse last result in between.
        if frame_count % OBJECT_DETECT_EVERY_N_FRAMES == 0:
            _t = time.perf_counter()
            all_detections, hazardous_detections = object_detector.detect_objects(frame)
            stage_ms['object_detection'] += (time.perf_counter() - _t) * 1000
            last_hazardous_detections = hazardous_detections
        else:
            hazardous_detections = last_hazardous_detections

        # Cheapest person-presence signal available (the tracker's own boxes).
        # Issue 2: face detection used to run on the throttled schedule even with
        # nobody in frame — this gates it the same way ANPR is already gated on
        # vehicles actually being present.
        person_in_frame = False
        _boxes = getattr(yolo_results, "boxes", None)
        if _boxes is not None:
            try:
                for _box in _boxes:
                    if int(_box.cls) == 0 and float(_box.conf) > 0.4:
                        person_in_frame = True
                        break
            except Exception:
                person_in_frame = False

        annotated_frame = frame.copy()
        # Virtual fence zones are drawn first so detections sit on top of them.
        virtual_fence.draw(annotated_frame)
        person_detected = False
        current_behavior = "idle"
        # FIX: these two are snapshot counts for THIS frame, not running totals —
        # see note above system_stats['total_detections'] below for why.
        persons_in_frame = 0
        poses_in_frame = 0

        # ------------------------------------------------------
        # Vehicles — reuses the tracker's own boxes (no extra inference).
        # ------------------------------------------------------
        vehicle_dets = []
        if ENABLE_VEHICLE_DETECTION:
            _t = time.perf_counter()
            vehicle_dets = vehicle_detector.process_boxes(
                getattr(yolo_results, "boxes", None), frame)
            stage_ms['vehicle'] += (time.perf_counter() - _t) * 1000
            vehicle_detector.draw(annotated_frame, vehicle_dets)
            system_stats['vehicles_detected'] = vehicle_detector.total_unique
            system_stats['vehicles_in_frame'] = len(vehicle_dets)
            system_stats['vehicles_by_type'] = dict(vehicle_detector.by_type)

            for v in vehicle_dets:
                vx1, vy1, vx2, vy2 = v['bbox']
                vcx, vcy = (vx1 + vx2) // 2, (vy1 + vy2) // 2
                if ENABLE_VIRTUAL_FENCE:
                    for zone_event in virtual_fence.evaluate(
                            v['track_id'],
                            (vcx / frame.shape[1], vcy / frame.shape[0]),
                            label='vehicle'):
                        _handle_zone_event(zone_event)
                if ALERT_ON_NEW_VEHICLE and v['track_id'] >= 0:
                    alert_throttled(
                        ('vehicle', v['track_id']), "Vehicle Detected", v['conf'],
                        f"{v['class_name']} ({v['color']}) detected",
                        cooldown=1e9,
                        data={'class': v['class_name'], 'color': v['color']})

        # ------------------------------------------------------
        # ANPR — number-plate read: every new vehicle track is attempted once.
        # ------------------------------------------------------
        _t = time.perf_counter()
        if ENABLE_ANPR and vehicle_dets:
            tried_new = 0
            for v in vehicle_dets:
                tid = v['track_id']
                if tid >= 0:
                    # ByteTrack ids are stable: attempt each vehicle exactly once,
                    # so a quick drive-by is not missed by periodic sampling.
                    if tid in anpr_attempted_tracks:
                        continue
                    anpr_attempted_tracks.add(tid)
                elif frame_count % ANPR_EVERY_N_FRAMES != 0:
                    continue  # no stable id — fall back to periodic sampling
                if tried_new >= ANPR_MAX_NEW_PER_FRAME:
                    break
                tried_new += 1
                record = anpr_reader.read_plate(frame, v['bbox'], v['track_id'])
                if not record:
                    continue
                anpr_reader.draw(annotated_frame, record)
                if last_plate_by_track.get(record['track_id']) != record['plate']:
                    last_plate_by_track[record['track_id']] = record['plate']
                    log_alert("ANPR Plate Read", record['confidence'],
                              f"Plate {record['plate']} on {v['class_name']}",
                              data={'plate': record['plate'],
                                    'vehicle': v['class_name'],
                                    'track_id': record['track_id']})
        system_stats['plates_read'] = anpr_reader.total_unique
        stage_ms['anpr'] += (time.perf_counter() - _t) * 1000

        # ------------------------------------------------------
        # Faces — Haar cascade on the data bundled with OpenCV.
        # Issue 2: only when faces could actually exist (a person is in frame) and
        # only on the throttled schedule; with nobody present this now costs
        # nothing instead of a full-frame cascade scan every few frames.
        # ------------------------------------------------------
        _t = time.perf_counter()
        if (ENABLE_FACE_DETECTION and person_in_frame
                and frame_count % FACE_DETECT_EVERY_N_FRAMES == 0):
            face_detector.detect(frame, gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            system_stats['faces_in_frame'] = face_detector.last_count
        if ENABLE_FACE_DETECTION and person_in_frame:
            face_detector.annotate(annotated_frame, face_detector.last_faces)
        elif ENABLE_FACE_DETECTION:
            # Nobody in frame: there are no faces to find, so drop the stale boxes.
            face_detector.last_faces = []
            face_detector.last_count = 0
            system_stats['faces_in_frame'] = 0
        stage_ms['face'] += (time.perf_counter() - _t) * 1000

        # Draw + proximity-check hazardous objects
        for obj in hazardous_detections:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["class_name"]
            conf = obj["conf"]

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if yolo_results.boxes is not None and getattr(yolo_results.boxes, 'id', None) is not None:
                for box in yolo_results.boxes:
                    cls_id = int(box.cls)
                    if cls_id == 0:
                        track_id_p = int(box.id) if box.id is not None else -1
                        px1, py1, px2, py2 = map(int, box.xyxy[0])
                        person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                        obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        distance = math.hypot(person_center[0] - obj_center[0],
                                               person_center[1] - obj_center[1])

                        if distance < WEAPON_PROXIMITY_THRESHOLD:
                            current_time = time.time()
                            # FIX: was a global 3s cooldown shared by every alert type —
                            # a single bottle sitting near you re-fired and re-counted
                            # every 3 seconds it stayed in frame, and repositioning it
                            # made this worse (distance recompute), not better. Now
                            # keyed per (this person, this weapon label): counts once
                            # when the pairing STARTS, then just refreshes silently
                            # while it continues, and only counts again if it's been
                            # gone for WEAPON_ALERT_TIMEOUT seconds (i.e. actually a
                            # new occurrence, not the same one still sitting there).
                            weapon_key = (track_id_p, label)
                            if weapon_key not in active_weapon_alerts:
                                log_alert("Weapon Detected", conf,
                                          f"{label} detected near person (Distance: {distance:.1f}px)")
                            active_weapon_alerts[weapon_key] = current_time
                            cv2.line(annotated_frame, person_center, obj_center, (0, 0, 255), 2)
                            cv2.putText(annotated_frame, "WEAPON ALERT!",
                                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Prune stale loitering_tracker entries
        STALE_TRACK_SECONDS = 10.0
        now_ts = current_time
        for tid, v in list(loitering_tracker.items()):
            if 'positions' in v and len(v['positions']) > 0:
                last_pos_time = v['positions'][-1][2]
            else:
                last_pos_time = v.get('start_time', v.get('last_seen', now_ts))
            if now_ts - last_pos_time > STALE_TRACK_SECONDS:
                loitering_tracker.pop(tid, None)
                pose_history_by_track.pop(tid, None)
                pose_behavior_persisted.pop(tid, None)
                behavior_buffer_by_track.pop(tid, None)

        # Prune weapon-alert keys that haven't been refreshed recently — lets a
        # genuinely new occurrence (weapon reappears after being gone a while) count again
        if ENABLE_VIRTUAL_FENCE:
            virtual_fence.drop_tracks(set(loitering_tracker.keys()))

        for wkey, last_seen in list(active_weapon_alerts.items()):
            if now_ts - last_seen > WEAPON_ALERT_TIMEOUT:
                active_weapon_alerts.pop(wkey, None)

        if yolo_results.boxes is not None and getattr(yolo_results.boxes, 'id', None) is not None:
            for box in yolo_results.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                track_id = int(box.id)

                if cls_id != 0 or conf <= 0.4:
                    continue

                person_detected = True
                persons_in_frame += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Virtual fence check for this person.
                if ENABLE_VIRTUAL_FENCE:
                    for zone_event in virtual_fence.evaluate(
                            track_id,
                            (center_x / frame.shape[1], center_y / frame.shape[0]),
                            label='person'):
                        _handle_zone_event(zone_event)

                detections = [(track_id, x1, y1, x2, y2, "person")]
                annotated_frame = process_person_detections(detections, annotated_frame, camera_id=CAMERA_ID)

                if track_id not in loitering_tracker:
                    loitering_tracker[track_id] = {
                        'positions': deque(maxlen=150),
                        'start_time': current_time,
                        'ema_movement': 0.0,
                        'running_state': False,
                        'running_negative_counter': 0
                    }
                loitering_tracker[track_id]['positions'].append((center_x, center_y, current_time))
                loitering_tracker[track_id]['last_seen'] = current_time

                track_speed = 0.0
                positions = loitering_tracker[track_id]['positions']
                if len(positions) >= 2:
                    total_dist = 0.0
                    total_dt = 0.0
                    for i in range(1, len(positions)):
                        x0, y0, t0 = positions[i - 1]
                        x1p, y1p, t1p = positions[i]
                        total_dist += math.hypot(x1p - x0, y1p - y0)
                        total_dt += max(1e-3, t1p - t0)
                    if total_dt > 0:
                        track_speed = total_dist / total_dt

                h, w = rgb_frame.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w, x2), min(h, y2)
                person_roi = rgb_frame[y1c:y2c, x1c:x2c]

                pose_metrics = None
                adjusted_landmarks = None
                confidence = 1.0
                _t = time.perf_counter()

                if person_roi is not None and person_roi.size > 0:
                    # FIX: MediaPipe cost scales with input size, and cost multiplies by
                    # how many people are in frame (one pose.process() call per person,
                    # per frame). Downsizing each crop before processing is a real,
                    # low-risk speed win with multiple people — landmarks come back
                    # normalized 0-1 regardless of input size, and adjusted_landmarks
                    # below already maps them using the ORIGINAL bbox (x1,y1,x2,y2),
                    # so accuracy of the final on-frame position is unaffected.
                    roi_h, roi_w = person_roi.shape[:2]
                    max_dim = max(roi_h, roi_w)
                    if max_dim > POSE_ROI_MAX_DIM:
                        scale = POSE_ROI_MAX_DIM / max_dim
                        person_roi_for_pose = cv2.resize(
                            person_roi, (max(1, int(roi_w * scale)), max(1, int(roi_h * scale)))
                        )
                    else:
                        person_roi_for_pose = person_roi
                    pose_results = pose.process(person_roi_for_pose)

                    if pose_results and pose_results.pose_landmarks:
                        poses_in_frame += 1

                        adjusted_landmarks = []
                        for landmark in pose_results.pose_landmarks.landmark:
                            adjusted_landmarks.append(
                                landmark_pb2.NormalizedLandmark(
                                    x=(landmark.x * (x2 - x1) + x1) / frame.shape[1],
                                    y=(landmark.y * (y2 - y1) + y1) / frame.shape[0],
                                    z=landmark.z,
                                    visibility=landmark.visibility
                                )
                            )

                        for landmark in adjusted_landmarks:
                            if landmark.visibility > 0.5:
                                lx = int(landmark.x * frame.shape[1])
                                ly = int(landmark.y * frame.shape[0])
                                cv2.circle(annotated_frame, (lx, ly), 3, (0, 255, 255), -1)

                        # FIX: look up THIS track's own previous pose/time, never a
                        # shared global — see note above pose_history_by_track.
                        track_pose_hist = pose_history_by_track[track_id]
                        prev_landmarks, prev_time = (track_pose_hist[-1] if track_pose_hist else (None, None))
                        dt = (current_time - prev_time) if prev_time else None

                        pose_metrics = pose_analyzer.calculate_pose_metrics(
                            adjusted_landmarks, prev_landmarks=prev_landmarks, dt=dt
                        )

                        movement_raw = pose_metrics.get('movement_intensity', 0.0) if pose_metrics else 0.0
                        t_size = torso_size(adjusted_landmarks) if adjusted_landmarks else 1.0
                        normalized_movement = movement_raw / t_size

                        frame_width = frame.shape[1] if frame is not None else 640
                        running_speed_threshold = RUNNING_SPEED_FRACTION * frame_width

                        is_running_by_speed = track_speed > running_speed_threshold
                        is_running_by_movement = normalized_movement > RUNNING_MOVEMENT_THRESHOLD

                        if DEBUG_RUNNING_LOG:
                            print(f"TRACK {track_id}: speed={track_speed:.1f}px/s thr={running_speed_threshold:.1f} "
                                  f"norm_mv={normalized_movement:.4f} thr={RUNNING_MOVEMENT_THRESHOLD} "
                                  f"is_speed={is_running_by_speed} is_mv={is_running_by_movement}")

                        # FIX: this is now the ONLY classification pass per frame.
                        if is_running_by_speed and is_running_by_movement:
                            label, confidence = 'running', 0.9
                        else:
                            label, confidence = pose_analyzer.classify_behavior(pose_metrics)

                        track_behavior_buf = behavior_buffer_by_track[track_id]
                        track_behavior_buf.append(label)
                        track_pose_hist.append((adjusted_landmarks, current_time))
                        recent_behaviors_log.append({'track_id': track_id, 'behavior': label, 'time': current_time})

                # Issue 2: pose cost is per person per frame, so this is the stage
                # to watch in the log when several people are in frame. The ROI cap
                # (POSE_ROI_MAX_DIM) is what keeps it from scaling with person size.
                stage_ms['pose'] += (time.perf_counter() - _t) * 1000

                # Stable behavior over sliding window (single source of truth, per track)
                stable_behavior = 'normal'
                track_behavior_buf = behavior_buffer_by_track[track_id]
                if len(track_behavior_buf) == track_behavior_buf.maxlen:
                    fall_count = track_behavior_buf.count('fall')
                    running_count = track_behavior_buf.count('running')
                    abnormal_count = track_behavior_buf.count('abnormal')

                    if fall_count >= max(7, int(0.6 * track_behavior_buf.maxlen)):
                        stable_behavior = 'fall'
                    elif running_count >= RUNNING_CONFIRM_COUNT:
                        stable_behavior = 'running'
                    elif abnormal_count >= max(6, int(0.5 * track_behavior_buf.maxlen)):
                        stable_behavior = 'abnormal'

                # Persist a compact pose-sequence record when this track's confirmed
                # behaviour changes (metadata only — never raw frames), backing the
                # README's "pose sequences" storage claim.
                if poses_in_frame and pose_behavior_persisted.get(track_id) != stable_behavior:
                    pose_behavior_persisted[track_id] = stable_behavior  # behaviour changed
                    try:
                        event_store.append_pose_sequence(track_id, {
                            'behavior': stable_behavior,
                            'confidence': round(confidence, 3),
                            'samples': len(pose_history_by_track[track_id]),
                        })
                    except Exception:
                        pass

                lt = loitering_tracker[track_id]
                if stable_behavior == 'running':
                    lt['running_state'] = True
                    lt['running_negative_counter'] = 0
                    current_behavior = 'running'
                elif lt.get('running_state', False):
                    lt['running_negative_counter'] += 1
                    if lt['running_negative_counter'] >= RUNNING_HYSTERESIS_COUNT:
                        lt['running_state'] = False
                        lt['running_negative_counter'] = 0
                        current_behavior = stable_behavior
                    else:
                        current_behavior = 'running'
                else:
                    current_behavior = stable_behavior

                current_time = time.time()
                if current_behavior == 'fall' and current_time - last_alert_time > alert_cooldown:
                    log_alert("Fall Detected", 0.95, "Consistent fall detected")
                    last_alert_time = current_time
                elif current_behavior == 'running' and current_time - last_alert_time > alert_cooldown:
                    log_alert("Running Detected", 0.85, "Consistent running detected")
                    last_alert_time = current_time
                elif current_behavior == 'abnormal' and current_time - last_alert_time > alert_cooldown:
                    log_alert("Abnormal Behavior", 0.75, "Consistent abnormal behavior")
                    last_alert_time = current_time

                if current_behavior == "fall":
                    color, text = (0, 0, 255), f"FALL DETECTED! ({confidence:.2f})"
                elif current_behavior == "running":
                    color, text = (0, 100, 255), f"Running Detected ({confidence:.2f})"
                elif current_behavior == "abnormal":
                    color, text = (0, 165, 255), f"Abnormal Behavior ({confidence:.2f})"
                else:
                    color, text = (0, 255, 0), f"Normal Activity ({confidence:.2f})"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"ID: {track_id} | {text}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if pose_metrics:
                    metrics_text = f"V:{pose_metrics['verticality']:.2f} M:{pose_metrics['movement_intensity']:.2f}"
                    cv2.putText(annotated_frame, metrics_text,
                                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                if current_behavior == "fall":
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
                    cv2.putText(annotated_frame, "EMERGENCY: FALL DETECTED",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        night_vision.annotate(annotated_frame)

        # System info overlay
        info_y = frame.shape[0] - 80
        cv2.rectangle(annotated_frame, (0, info_y), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        status = "ANALYZING" if person_detected else "MONITORING"
        cv2.putText(annotated_frame, f"Status: {status}",
                    (10, info_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Behavior: {current_behavior.upper()}",
                    (10, info_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FIX: "Human Detections"/"Poses" now reflect what's in frame RIGHT NOW,
        # not an ever-growing cumulative total. The old version incremented these
        # every frame for every person, so a room with 4 people over 84 seconds
        # read "951 Human Detections" — accurate as a running sum, but not what
        # a live monitoring dashboard should show. Falls/Abnormal/Weapons remain
        # genuinely cumulative below (they're event counts, which IS the right
        # semantics for "how many times has this alert fired").
        system_stats['total_detections'] = persons_in_frame
        system_stats['pose_detections'] = poses_in_frame
        system_stats['intrusions'] = virtual_fence.intrusions
        system_stats['night_mode'] = night_vision.night_mode
        system_stats['brightness'] = round(night_vision.brightness, 1)
        system_stats['motion'] = night_motion
        system_stats['c2_enabled'] = c2_client.enabled
        system_stats['c2_sent'] = c2_client.sent
        system_stats['c2_queued'] = c2_client.queue.qsize()

        # Night-time movement alert — independent of the YOLO passes and
        # therefore still works when models struggle with a dark frame.
        if ENABLE_NIGHT_VISION and night_vision.night_mode and night_motion:
            alert_throttled('night-motion', "Night-time Movement", 0.6,
                            "Movement detected in low-light conditions", cooldown=10.0)

        if frame_count % 10 == 0:
            system_stats['fps'] = 10.0 / (time.time() - start_time)
            start_time = time.time()
        cv2.putText(annotated_frame, f"FPS: {system_stats['fps']:.1f}",
                    (10, info_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        uptime = int(time.time() - system_stats['system_uptime'])
        cv2.putText(annotated_frame, f"Uptime: {uptime // 60}:{uptime % 60:02d}",
                    (frame.shape[1] - 120, info_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Poses: {system_stats['pose_detections']}",
                    (frame.shape[1] - 120, info_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Weapons: {system_stats['weapons_detected']}",
                    (frame.shape[1] - 120, info_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        with lock:
            ret, jpeg = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                latest_frame = jpeg.tobytes()

        # Issue 2: close out this frame's timing window and log the breakdown
        # (see _log_stage_timing) so the real bottleneck is measured, not guessed.
        stage_ms['frame_total'] += (time.perf_counter() - frame_start) * 1000
        stage_frames += 1
        if stage_frames >= STAGE_REPORT_EVERY_N_FRAMES:
            _log_stage_timing(stage_ms, stage_frames, persons_in_frame, poses_in_frame)
            stage_ms.clear()
            stage_frames = 0

    grabber.release()


def generate_frames_from(cam):
    while True:
        with cam._lock:
            frame = cam.latest_frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


def generate_frames():
    global latest_frame, lock
    while True:
        with lock:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/<camera_id>')
def video_feed_camera(camera_id):
    if camera_id == CAMERA_ID:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    cam = _STREAM_BY_ID.get(camera_id)
    if cam is None:
        return jsonify({'status': 'error', 'message': f'unknown camera {camera_id}'}), 404
    return Response(generate_frames_from(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/cameras')
def get_cameras():
    return jsonify({'cameras': _camera_listing()})


@app.route('/api/alerts')
def get_alerts():
    return jsonify({'alerts': alerts_list[:10], 'total_alerts': len(alerts_list)})


@app.route('/api/stats')
def get_stats():
    payload = dict(system_stats)
    payload['camera_id'] = CAMERA_ID
    payload['cameras'] = _camera_listing()
    return jsonify(payload)


@app.route('/api/test_alert')
def test_alert():
    log_alert("Test Alert", 0.95, "Manual test triggered")
    return jsonify({'status': 'success', 'message': 'Test alert generated'})


@app.route('/api/behavior_history')
def get_behavior_history():
    total_pose_history = sum(len(h) for h in pose_history_by_track.values())
    return jsonify({
        'pose_history_length': total_pose_history,
        'active_tracks': len(pose_history_by_track),
        'recent_behaviors': list(recent_behaviors_log)[-5:]
    })


@app.route('/api/weapon_detections')
def get_weapon_detections():
    weapon_alerts = [alert for alert in alerts_list if alert['type'] == 'Weapon Detected']
    return jsonify({'weapon_alerts': weapon_alerts[:10], 'total_weapon_alerts': len(weapon_alerts)})


@app.route('/api/clear_alerts', methods=['POST'])
def clear_alerts():
    global alerts_list, system_stats
    alerts_list = []
    last_plate_by_track.clear()
    anpr_attempted_tracks.clear()
    vehicle_detector.reset_stats()
    anpr_reader.reset_stats()
    virtual_fence.reset_stats()
    night_vision.reset_stats()
    system_stats = {
        'total_detections': system_stats['total_detections'],
        'falls_detected': 0,
        'abnormal_behaviors': 0,
        'weapons_detected': 0,
        'vehicles_detected': 0,
        'vehicles_in_frame': 0,
        'vehicles_by_type': {},
        'faces_in_frame': 0,
        'plates_read': 0,
        'intrusions': 0,
        'system_uptime': system_stats['system_uptime'],
        'fps': system_stats['fps'],
        'pose_detections': system_stats['pose_detections']
    }
    return jsonify({'status': 'success', 'message': 'Alerts cleared successfully'})


@app.route('/api/detections')
def get_detections():
    """Combined snapshot of every detector: vehicles, faces, plates, zones, night."""
    return jsonify({
        'vehicles': vehicle_detector.stats(),
        'faces': face_detector.stats(),
        'anpr': anpr_reader.stats(),
        'fence': virtual_fence.stats(),
        'night': night_vision.stats(),
        'object_detector': object_detector.stats(),
    })


@app.route('/api/vehicles')
def get_vehicles():
    return jsonify(vehicle_detector.stats())


@app.route('/api/anpr')
def get_anpr():
    return jsonify(anpr_reader.stats())


@app.route('/api/faces')
def get_faces():
    return jsonify(face_detector.stats())


@app.route('/api/zones', methods=['GET', 'POST'])
def zones():
    """List zones, or add one with {"name", "points" (>=3 normalised [x,y]), "dwell_seconds"}."""
    if request.method == 'POST':
        payload = request.get_json(silent=True) or {}
        name = payload.get('name')
        points = payload.get('points')
        if not name or not points or len(points) < 3:
            return jsonify({'status': 'error',
                            'message': 'name and points (>=3 [x,y] pairs) are required'}), 400
        try:
            virtual_fence.add_zone(name, points, payload.get('dwell_seconds'))
        except Exception as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 400
        return jsonify({'status': 'success', 'zones': virtual_fence.list_zones()})
    return jsonify(virtual_fence.stats())


@app.route('/api/zones/<name>', methods=['DELETE'])
def delete_zone(name):
    removed = virtual_fence.remove_zone(name)
    return jsonify({'status': 'success' if removed else 'not_found'})


@app.route('/api/c2/status')
def c2_status():
    return jsonify(c2_client.status())


@app.route('/api/c2/test', methods=['POST'])
def c2_test():
    ok = c2_client.send_test()
    return jsonify({'status': 'success' if ok else 'disabled', 'c2': c2_client.status()})


@app.route('/api/report')
def generate_report():
    """Build a full snapshot report and persist it under data/reports/."""
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'system': 'Enhanced Smart Surveillance System v2.0',
        'camera_id': CAMERA_ID,
        'stats': dict(system_stats),
        'alerts_recent': alerts_list[:50],
        'detections': {
            'vehicles': vehicle_detector.stats(),
            'faces': face_detector.stats(),
            'anpr': anpr_reader.stats(),
            'fence': virtual_fence.stats(),
            'night': night_vision.stats(),
            'object_detector': object_detector.stats(),
        },
        'storage': event_store.stats(),
    }
    report['saved_to'] = event_store.save_report(report)
    return jsonify(report)


if __name__ == '__main__':
    print("Starting Enhanced Smart Surveillance System...")
    print(f"Camera source: {CAMERA_INDEX} @ {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    os.makedirs('templates', exist_ok=True)

    video_thread = threading.Thread(target=video_processing_thread, daemon=True)
    video_thread.start()

    # Additional cameras stream live into the same dashboard (no AI pipeline).
    for _stream_cam in _STREAM_CAMERAS:
        _stream_cam.start()

    print("Open your browser and go to: http://localhost:5000")
    app.run(host='0.0.0.0', debug=False, port=5000, use_reloader=False)