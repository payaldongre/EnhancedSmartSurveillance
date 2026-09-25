from flask import Flask, Response, render_template, jsonify
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

from object_detector import ObjectDetector  # FIX: use the corrected module

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

# FIX: detect device BEFORE creating either model, so both can be told
# explicitly which device to use rather than relying on Ultralytics' auto-detect.
device_in_use = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
INFERENCE_DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"Inference device: {device_in_use}")
print("Note: MediaPipe Pose always runs on CPU regardless of this setting — the "
      "Windows pip build has no GPU delegate. Only the two YOLO calls get the GPU boost.")

object_detector = ObjectDetector("yolov8n.pt", imgsz=320, device=INFERENCE_DEVICE)
app = Flask(__name__)

# ------------------------------------------------------
# Global state
# ------------------------------------------------------
latest_frame = None
lock = threading.Lock()
alerts_list = []
system_stats = {
    'total_detections': 0,
    'falls_detected': 0,
    'abnormal_behaviors': 0,
    'weapons_detected': 0,
    'system_uptime': time.time(),
    'fps': 0,
    'pose_detections': 0
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


def log_alert(alert_type, confidence=0.0, details=""):
    global alerts_list, system_stats
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_data = {
        'timestamp': timestamp,
        'type': alert_type,
        'confidence': confidence,
        'details': details,
        'id': len(alerts_list) + 1
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


# ------------------------------------------------------
# FIX: dedicated capture thread, decoupled from inference.
#
# Previously cap.read() happened inline in the same loop as all the AI
# inference, so if inference for a frame took (say) 200ms, the next
# cap.read() only happened after that — frames pile up in the camera's
# internal buffer and what you see on screen visibly lags real time.
# This grabber thread continuously reads frames as fast as the camera
# provides them and always keeps only the newest one; the processing loop
# below picks up whatever is freshest when it's ready, so the stream
# self-corrects instead of accumulating backlog. You'll see occasional
# dropped frames under load rather than growing lag, which is the right
# tradeoff for a live monitoring feed.
# ------------------------------------------------------
class FrameGrabber:
    def __init__(self, source=0, width=640, height=480):
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ask the driver for a shallow buffer too
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = self.cap.isOpened()
        if self.running:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def get_latest(self):
        with self.lock:
            return self.frame, self.frame_id

    def release(self):
        self.running = False
        self.cap.release()


def video_processing_thread():
    global latest_frame, last_alert_time, system_stats, loitering_tracker
    global last_hazardous_detections

    grabber = FrameGrabber(0, 640, 480)
    if not grabber.running:
        print("Error: Could not open webcam")
        return
    print("Webcam opened successfully")
    print("Starting video processing with pose estimation...")

    frame_count = 0
    start_time = time.time()
    last_processed_frame_id = -1

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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Person detection + tracking
        yolo_results = model.track(
            frame, persist=True, imgsz=PERSON_IMGSZ,
            tracker=PERSON_TRACKER, device=INFERENCE_DEVICE, verbose=False
        )[0]

        # FIX: throttle the heaviest call (hazardous-object detection) instead
        # of running it on every single frame; reuse last result in between.
        if frame_count % OBJECT_DETECT_EVERY_N_FRAMES == 0:
            all_detections, hazardous_detections = object_detector.detect_objects(frame)
            last_hazardous_detections = hazardous_detections
        else:
            hazardous_detections = last_hazardous_detections

        annotated_frame = frame.copy()
        person_detected = False
        current_behavior = "idle"
        # FIX: these two are snapshot counts for THIS frame, not running totals —
        # see note above system_stats['total_detections'] below for why.
        persons_in_frame = 0
        poses_in_frame = 0

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
                behavior_buffer_by_track.pop(tid, None)

        # Prune weapon-alert keys that haven't been refreshed recently — lets a
        # genuinely new occurrence (weapon reappears after being gone a while) count again
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

                detections = [(track_id, x1, y1, x2, y2, "person")]
                annotated_frame = process_person_detections(detections, annotated_frame, camera_id="Cam_1")

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

    grabber.release()


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


@app.route('/api/alerts')
def get_alerts():
    return jsonify({'alerts': alerts_list[:10], 'total_alerts': len(alerts_list)})


@app.route('/api/stats')
def get_stats():
    return jsonify(system_stats)


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
    system_stats = {
        'total_detections': system_stats['total_detections'],
        'falls_detected': 0,
        'abnormal_behaviors': 0,
        'weapons_detected': 0,
        'system_uptime': system_stats['system_uptime'],
        'fps': system_stats['fps'],
        'pose_detections': system_stats['pose_detections']
    }
    return jsonify({'status': 'success', 'message': 'Alerts cleared successfully'})


if __name__ == '__main__':
    print("Starting Enhanced Smart Surveillance System...")
    os.makedirs('templates', exist_ok=True)

    video_thread = threading.Thread(target=video_processing_thread, daemon=True)
    video_thread.start()

    print("Open your browser and go to: http://localhost:5000")
    app.run(host='0.0.0.0', debug=False, port=5000, use_reloader=False)