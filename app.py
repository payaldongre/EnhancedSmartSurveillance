from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
import mediapipe as mp
import threading
import time
import numpy as np
from datetime import datetime
import json
import os
from collections import deque
import math
from mediapipe.framework.formats import landmark_pb2
from speed_estimator import process_person_detections
import cv2
import torch
from ultralytics.nn.tasks import DetectionModel
import traceback

torch.serialization.add_safe_globals([DetectionModel])

# ------------------------------------------------------
# 🔹 Enhanced Object Detector with Hazardous Object Detection
# ------------------------------------------------------
class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf
        # Define hazardous objects
        self.hazardous_labels = [
            "knife", "scissors", "handgun", "rifle", "shotgun", 
            "axe", "hammer", "screwdriver", "bat", "bottle"
        ]

    def detect_objects(self, frame):
        """Run YOLO detection and return all results and hazardous objects"""
        results = self.model(frame, conf=self.conf)
        detections = []
        hazardous_detections = []
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, cf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.model.names[int(cls)]
                
                detection = {
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(cf),
                    "class": int(cls),
                    "class_name": class_name
                }
                
                detections.append(detection)
                
                # Check if it's a hazardous object
                if class_name.lower() in self.hazardous_labels and cf > 0.5:
                    hazardous_detections.append(detection)
        
        return detections, hazardous_detections

# Initialize object detector
object_detector = ObjectDetector("yolov8n.pt")
# --------------------------------------------------

# --- Running detection tuning (tunable) ---
RUNNING_SPEED_FRACTION = 0.20           # fraction of frame width -> px/sec threshold
RUNNING_MOVEMENT_THRESHOLD = 0.06      # normalized movement per frame (after torso normalization)
RUNNING_DETECTION_FRAMES = 15          # smoothing window size (frames)
RUNNING_CONFIRM_COUNT = 10             # positives within window to confirm running
RUNNING_HYSTERESIS_COUNT = 8           # negatives to clear running state
DEBUG_RUNNING_LOG = False              # set True to print debug logs for tuning
RUNNING_TORSO_SPEED = 0.6              # torso-lengths per second threshold for running (when pose available)
EMA_ALPHA = 0.6                        # EMA alpha for smoothing track speed

# Weapon detection settings
WEAPON_DETECTION_CONFIDENCE = 0.5      # Minimum confidence for weapon detection
WEAPON_PROXIMITY_THRESHOLD = 200       # Distance threshold for person-weapon proximity (pixels)

# Ensure behavior_buffer uses the configured window (if behavior_buffer defined later, update it there)
# Helper to normalize movement by torso size (shoulder width or torso height)
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

cap = cv2.VideoCapture(0)
print(cap.isOpened())
cap.release()

app = Flask(__name__)

# Global variables for thread communication
latest_frame = None
lock = threading.Lock()
alerts_list = []
system_stats = {
    'total_detections': 0,
    'falls_detected': 0,
    'abnormal_behaviors': 0,
    'weapons_detected': 0,  # New stat for weapon detection
    'system_uptime': time.time(),
    'fps': 0,
    'pose_detections': 0
}

# Load YOLOv8 model (will download automatically on first run)
print("Loading AI models... Please wait...")
model = YOLO('yolov8s.pt')  # Small version for better accuracy

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # Reduced from 1 to 0 for better performance
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("AI models loaded successfully!")

# Alert cooldown to prevent spam
last_alert_time = 0
alert_cooldown = 3  # seconds

# Behavior analysis variables
pose_history = deque(maxlen=RUNNING_DETECTION_FRAMES)  # Store last poses for analysis
behavior_buffer = deque(maxlen=RUNNING_DETECTION_FRAMES)  # Buffer for behavior classification (use configured window)
loitering_tracker = {}  # For loitering detection

class PoseAnalyzer:
    """Class to analyze human poses and detect abnormal behaviors"""
    
    def __init__(self):
        self.normal_standing_threshold = 0.3
        self.fall_threshold = 0.4  # Lowered threshold
        self.movement_threshold = 0.15
        self.running_threshold = 0.5  # Increased threshold to reduce false positives
        self.fall_velocity_threshold = 0.1  # Corrected threshold for vertical velocity (positive for downward)
        
    def calculate_pose_metrics(self, landmarks):
        """Calculate various pose metrics for behavior analysis"""
        if not landmarks:
            return None
            
        # Extract key landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        
        # Calculate body orientation
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        knee_center_y = (left_knee.y + right_knee.y) / 2
        ankle_center_y = (left_ankle.y + right_ankle.y) / 2
        
        # Body verticality (0 = horizontal, 1 = vertical)
        body_height = abs(nose.y - ankle_center_y)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        
        if shoulder_width > 0:
            verticality = min(body_height / shoulder_width, 2.0) / 2.0
        else:
            verticality = 0.5
            
        # Calculate head-to-body ratio
        head_body_ratio = abs(nose.y - shoulder_center_y) / max(abs(shoulder_center_y - ankle_center_y), 0.1)
        
        # Movement detection (requires pose history)
        movement_intensity = 0.0
        vertical_velocity = 0.0
        if len(pose_history) > 1:
            prev_pose = pose_history[-1]
            if prev_pose:
                movement_intensity = self.calculate_movement(landmarks, prev_pose)
                
                # Calculate vertical velocity of the hip
                prev_hip_y = (prev_pose[mp_pose.PoseLandmark.LEFT_HIP.value].y + 
                              prev_pose[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
                curr_hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + 
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
                vertical_velocity = curr_hip_y - prev_hip_y
        
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
        """Calculate movement intensity between two poses"""
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
            
            # Calculate Euclidean distance
            distance = math.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2)
            total_movement += distance
            
        return total_movement / len(key_points)
    
    def classify_behavior(self, pose_metrics):
        """Classify behavior as normal, abnormal, or fall"""
        if not pose_metrics:
            return "unknown", 0.0
            
        verticality = pose_metrics['verticality']
        movement = pose_metrics['movement_intensity']
        head_body_ratio = pose_metrics['head_body_ratio']
        vertical_velocity = pose_metrics['vertical_velocity']
        
        # Fall detection based on verticality and velocity
        if verticality < self.fall_threshold and vertical_velocity > self.fall_velocity_threshold:
            return "fall", 0.95
            
        # Running detection
        if movement > self.running_threshold:
            return "running", 0.8
            
        # Abnormal behavior detection
        abnormal_score = 0.0
        
        # Check for unusual postures
        if verticality < self.normal_standing_threshold:
            abnormal_score += 0.3
            
        # Check for erratic movement
        if movement > self.movement_threshold:
            abnormal_score += 0.2
            
        # Check for unusual head position
        if head_body_ratio < 0.1 or head_body_ratio > 0.6:
            abnormal_score += 0.2
            
        if abnormal_score > 0.4:
            return "abnormal", abnormal_score
        else:
            return "normal", 1.0 - abnormal_score

# Initialize pose analyzer
pose_analyzer = PoseAnalyzer()

def calculate_fps(start_time):
    """Calculate frames per second"""
    return 1.0 / max(time.time() - start_time, 0.001)

def log_alert(alert_type, confidence=0.0, details=""):
    """Log alerts with timestamp and details"""
    global alerts_list, system_stats
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_data = {
        'timestamp': timestamp,
        'type': alert_type,
        'confidence': confidence,
        'details': details,
        'id': len(alerts_list) + 1
    }
    
    alerts_list.insert(0, alert_data)  # Add to beginning for latest first
    
    # Keep only last 50 alerts to prevent memory issues
    if len(alerts_list) > 50:
        alerts_list = alerts_list[:50]
    
    # Update stats
    if alert_type == "Fall Detected":
        system_stats['falls_detected'] += 1
    elif alert_type == "Abnormal Behavior":
        system_stats['abnormal_behaviors'] += 1
    elif alert_type == "Running Detected":
        system_stats['abnormal_behaviors'] += 1  # Count running as abnormal
    elif alert_type == "Loitering Detected":
        system_stats['abnormal_behaviors'] += 1 # Count loitering as abnormal
    elif alert_type == "Weapon Detected":
        system_stats['weapons_detected'] += 1
    
    print(f"🚨 ALERT: {alert_type} at {timestamp} - {details}")

def video_processing_thread():
    """Main AI processing thread with pose estimation and behavior analysis"""
    global latest_frame, last_alert_time, system_stats, pose_history, loitering_tracker
    
    # Try to open webcam
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    # Set camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Webcam opened successfully")
    print("🎥 Starting video processing with pose estimation...")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        frame_start = time.time()
        # ensure a safe current_time is available for all code paths
        current_time = frame_start
        success, frame = cap.read()
        
        if not success:
            print("⚠️ Warning: Failed to read frame")
            continue
        
        frame_count += 1
        
        # Flip frame horizontally for mirror effect (more natural)
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference for person detection with tracking
        yolo_results = model.track(frame, persist=True, imgsz=320, verbose=False)[0]
        
        # Detect hazardous objects
        all_detections, hazardous_detections = object_detector.detect_objects(frame)
        
        # Initialize annotated frame
        annotated_frame = frame.copy()
        person_detected = False
        current_behavior = "idle"
        
        # Process hazardous objects
        for obj in hazardous_detections:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["class_name"]
            conf = obj["conf"]
            
            # Draw bounding box for hazardous object
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"{label} {conf:.2f}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Check if a person is near the hazardous object
            if yolo_results.boxes is not None and getattr(yolo_results.boxes, 'id', None) is not None:
                for box in yolo_results.boxes:
                    cls_id = int(box.cls)
                    if cls_id == 0:  # Person class
                        px1, py1, px2, py2 = map(int, box.xyxy[0])
                        
                        # Calculate distance between person and object
                        person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                        obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        distance = math.sqrt((person_center[0] - obj_center[0])**2 + 
                                            (person_center[1] - obj_center[1])**2)
                        
                        # If person is close to the object, trigger alert
                        if distance < WEAPON_PROXIMITY_THRESHOLD:
                            current_time = time.time()
                            if current_time - last_alert_time > alert_cooldown:
                                log_alert("Weapon Detected", conf, 
                                         f"{label} detected near person (Distance: {distance:.1f}px)")
                                last_alert_time = current_time
                                
                                # Add visual indicator
                                cv2.line(annotated_frame, person_center, obj_center, (0, 0, 255), 2)
                                cv2.putText(annotated_frame, "WEAPON ALERT!", 
                                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Process YOLO detections
        try:
            # Prune stale loitering_tracker entries (no updates for STALE_TRACK_SECONDS)
            STALE_TRACK_SECONDS = 10.0
            now_ts = current_time
            stale_keys = [tid for tid, v in loitering_tracker.items()
                          if ('positions' not in v or len(v['positions']) == 0) and (now_ts - v.get('start_time', now_ts) > STALE_TRACK_SECONDS)]
            # also prune tracks with last update older than threshold
            for tid, v in list(loitering_tracker.items()):
                last_pos_time = None
                if 'positions' in v and len(v['positions']) > 0:
                    last_pos_time = v['positions'][-1][2]
                else:
                    last_pos_time = v.get('start_time', v.get('last_seen', None))

                if last_pos_time is not None and now_ts - last_pos_time > STALE_TRACK_SECONDS:
                    try:
                        print(f"🧹 Pruning stale track {tid} (last update {now_ts - last_pos_time:.1f}s ago)")
                        del loitering_tracker[tid]
                    except KeyError:
                        pass

            if yolo_results.boxes is not None and getattr(yolo_results.boxes, 'id', None) is not None:
                for box in yolo_results.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    track_id = int(box.id)

                    # Check if detected object is a person (class 0 in COCO dataset)
                    if cls_id == 0 and conf > 0.4:
                        person_detected = True
                        system_stats['total_detections'] += 1

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Loitering detection logic
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        # current_time is initialized at frame start

                        detections = [(track_id, x1, y1, x2, y2, "person")]
                        annotated_frame = process_person_detections(detections, annotated_frame, camera_id="Cam_1")

                        # Ensure track entry exists
                        if track_id not in loitering_tracker:
                            loitering_tracker[track_id] = {
                                'positions': deque(maxlen=150),  # keep ~5 sec at 30fps
                                'start_time': current_time,
                                'ema_movement': 0.0,
                                'running_state': False,
                                'running_negative_counter': 0
                            }

                        # Append latest center with timestamp
                        try:
                            loitering_tracker[track_id]['positions'].append((center_x, center_y, current_time))
                            # update last seen timestamp for pruning logic
                            loitering_tracker[track_id]['last_seen'] = current_time
                        except Exception:
                            pass

                        # Compute smoothed track speed if sufficient history
                        track_speed = 0.0
                        positions = loitering_tracker[track_id]['positions']
                        if len(positions) >= 2:
                            total_dist = 0.0
                            total_dt = 0.0
                            for i in range(1, len(positions)):
                                x0, y0, t0 = positions[i - 1]
                                x1p, y1p, t1p = positions[i]
                                dist = math.hypot(x1p - x0, y1p - y0)
                                dt = max(1e-3, t1p - t0)
                                total_dist += dist
                                total_dt += dt
                            if total_dt > 0:
                                track_speed = total_dist / total_dt

                        # Extract person region for pose estimation (ensure coordinates are sane)
                        h, w = rgb_frame.shape[:2]
                        x1c, y1c = max(0, x1), max(0, y1)
                        x2c, y2c = min(w, x2), min(h, y2)
                        person_roi = rgb_frame[y1c:y2c, x1c:x2c]

                        if person_roi is not None and person_roi.size > 0:
                            # Run pose estimation on person ROI
                            pose_results = pose.process(person_roi)

                            if pose_results and pose_results.pose_landmarks:
                                system_stats['pose_detections'] += 1

                                # Adjust landmarks to full frame coordinates
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

                                # Draw pose landmarks
                                for i, landmark in enumerate(adjusted_landmarks):
                                    if landmark.visibility > 0.5:
                                        x = int(landmark.x * frame.shape[1])
                                        y = int(landmark.y * frame.shape[0])
                                        cv2.circle(annotated_frame, (x, y), 3, (0, 255, 255), -1)

                                # Analyze pose for behavior classification
                                pose_metrics = pose_analyzer.calculate_pose_metrics(adjusted_landmarks)

                                # Ensure track entry exists (positions should be a deque of (x,y,t) tuples)
                                if track_id not in loitering_tracker:
                                    loitering_tracker[track_id] = {
                                        'positions': deque(maxlen=150),
                                        'start_time': time.time(),
                                        'running_state': False,
                                        'running_negative_counter': 0
                                    }

                                # append timestamped center for this track
                                try:
                                    loitering_tracker[track_id]['positions'].append((center_x, center_y, current_time))
                                except Exception:
                                    pass

                                # compute smoothed track speed in pixels/sec (recompute if needed)
                                track_speed = 0.0
                                positions = loitering_tracker[track_id]['positions']
                                if len(positions) >= 2:
                                    total_dist = 0.0
                                    total_dt = 0.0
                                    for i in range(1, len(positions)):
                                        x0, y0, t0 = positions[i - 1]
                                        x1p, y1p, t1p = positions[i]
                                        dist = math.hypot(x1p - x0, y1p - y0)
                                        dt = max(1e-3, t1p - t0)
                                        total_dist += dist
                                        total_dt += dt
                                    if total_dt > 0:
                                        track_speed = total_dist / total_dt

                                # If pose_metrics missing, set safe defaults
                                movement_raw = pose_metrics.get('movement_intensity', 0.0) if pose_metrics else 0.0

                                # Normalize movement by torso size (landmarks are normalized 0..1)
                                t_size = torso_size(adjusted_landmarks) if adjusted_landmarks else 1.0
                                normalized_movement = movement_raw / t_size

                                # frame pixel threshold
                                frame_width = frame.shape[1] if frame is not None else 640
                                running_speed_threshold = RUNNING_SPEED_FRACTION * frame_width

                                # boolean tests
                                is_running_by_speed = track_speed > running_speed_threshold
                                is_running_by_movement = normalized_movement > RUNNING_MOVEMENT_THRESHOLD

                                if DEBUG_RUNNING_LOG:
                                    print(f"TRACK {track_id}: track_speed={track_speed:.1f}px/s, frame_w={frame_width}, "
                                          f"speed_thr={running_speed_threshold:.1f}, movement_raw={movement_raw:.4f}, "
                                          f"norm_mv={normalized_movement:.4f}, mv_thr={RUNNING_MOVEMENT_THRESHOLD}, "
                                          f"is_speed={is_running_by_speed}, is_mv={is_running_by_movement}")

                                # initial label decision: require both signals for an immediate 'running' label
                                if is_running_by_speed and is_running_by_movement:
                                    label = 'running'
                                    confidence = 0.9
                                else:
                                    label, confidence = pose_analyzer.classify_behavior(pose_metrics)

                                # push label into buffer for smoothing
                                behavior_buffer.append(label)

                                # stable detection over sliding window
                                stable_behavior = 'normal'
                                if len(behavior_buffer) == behavior_buffer.maxlen:
                                    running_count = behavior_buffer.count('running')
                                    fall_count = behavior_buffer.count('fall')
                                    abnormal_count = behavior_buffer.count('abnormal')

                                    # Confirm fall if many falls
                                    if fall_count >= max(7, int(0.6 * behavior_buffer.maxlen)):
                                        stable_behavior = 'fall'
                                    elif running_count >= RUNNING_CONFIRM_COUNT:
                                        stable_behavior = 'running'
                                    elif abnormal_count >= max(6, int(0.5 * behavior_buffer.maxlen)):
                                        stable_behavior = 'abnormal'
                                    else:
                                        stable_behavior = 'normal'

        except Exception as e:
            # Log the exception with traceback and continue processing next frames
            print("Exception in person processing loop:", str(e))
            traceback.print_exc()
            # Make sure we don't let locals used later remain unset
            if 'stable_behavior' not in locals():
                stable_behavior = 'normal'
            # continue to next frame iteration (skip remaining person logic)
            continue

        # Only run per-track hysteresis and alert/drawing logic if we have per-track variables
        if 'track_id' in locals():
            # Ensure a loitering_tracker entry exists for this track to avoid KeyError
            if track_id not in loitering_tracker:
                print(f"⚠️ Warning: missing loitering_tracker entry for track {track_id}, creating default.")
                loitering_tracker[track_id] = {
                    'positions': deque(maxlen=150),
                    'start_time': current_time,
                    'ema_movement': 0.0,
                    'running_state': False,
                    'running_negative_counter': 0
                }

            # Use a local reference to simplify access
            lt = loitering_tracker[track_id]

            # Per-track hysteresis: preserve running_state until several negatives
            if stable_behavior == 'running':
                lt['running_state'] = True
                lt['running_negative_counter'] = 0
                current_behavior = 'running'
            elif lt.get('running_state', False):
                # track was running, require hysteresis to clear
                lt['running_negative_counter'] += 1
                if lt['running_negative_counter'] >= RUNNING_HYSTERESIS_COUNT:
                    lt['running_state'] = False
                    lt['running_negative_counter'] = 0
                    current_behavior = stable_behavior  # now allow other state
                else:
                    # keep running displayed until hysteresis expires
                    current_behavior = 'running'
            else:
                current_behavior = stable_behavior

            # Trigger alert (stable) - keep your existing alert cooldown usage
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

            if 'pose_metrics' in locals() and pose_metrics:
                # Add to pose history
                pose_history.append(adjusted_landmarks)

                # Classify behavior and add to buffer for smoothing
                behavior, confidence = pose_analyzer.classify_behavior(pose_metrics)
                behavior_buffer.append(behavior)

                # Use the latest behavior for immediate display, but alert on stable behavior
                current_behavior = behavior

                # Determine stable behavior for alerting and display smoothing
                if len(behavior_buffer) == behavior_buffer.maxlen:
                    fall_counts = behavior_buffer.count("fall")
                    running_counts = behavior_buffer.count("running")
                    abnormal_counts = behavior_buffer.count("abnormal")

                    stable_behavior = "normal"
                    # Check for stable abnormal behaviors (e.g., 7 out of 10 frames)
                    if fall_counts >= 7:
                        stable_behavior = "fall"
                        current_behavior = "fall"  # Lock display to fall
                    elif running_counts >= 7:
                        stable_behavior = "running"
                        current_behavior = "running"  # Lock display to running
                    elif abnormal_counts >= 7:
                        stable_behavior = "abnormal"
                        current_behavior = "abnormal" # Lock display to abnormal

                    # Handle alerts for stable behaviors
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        if stable_behavior == "fall":
                            log_alert("Fall Detected", 0.95, "Consistent fall detected")
                            last_alert_time = current_time
                        elif stable_behavior == "running":
                            log_alert("Running Detected", 0.85, "Consistent running detected")
                            last_alert_time = current_time
                        elif stable_behavior == "abnormal":
                            log_alert("Abnormal Behavior", 0.75, "Consistent abnormal behavior")
                            last_alert_time = current_time

                # Color coding based on behavior
                if current_behavior == "fall":
                    color = (0, 0, 255)  # Red
                    text = f"FALL DETECTED! ({confidence:.2f})"
                elif current_behavior == "running":
                    color = (0, 100, 255) # Dark Orange
                    text = f"Running Detected ({confidence:.2f})"
                elif current_behavior == "abnormal":
                    color = (0, 165, 255)  # Orange
                    text = f"Abnormal Behavior ({confidence:.2f})"
                else:
                    color = (0, 255, 0)  # Green
                    text = f"Normal Activity ({confidence:.2f})"

                # Draw bounding box and track ID
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"ID: {track_id} | {text}", 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add pose metrics display
                metrics_text = f"V:{pose_metrics['verticality']:.2f} M:{pose_metrics['movement_intensity']:.2f}"
                cv2.putText(annotated_frame, metrics_text, 
                          (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                # Fallback: draw simple bounding box if pose estimation fails
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID: {track_id} | Person ({conf:.2f})", 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add emergency overlay for critical alerts
        if current_behavior == "fall":
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            cv2.putText(annotated_frame, "🚨 EMERGENCY: FALL DETECTED 🚨", 
                      (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Add system information overlay
        info_y = frame.shape[0] - 80
        cv2.rectangle(annotated_frame, (0, info_y), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        
        # System status
        status = "ANALYZING" if person_detected else "MONITORING"
        cv2.putText(annotated_frame, f"Status: {status}", 
                   (10, info_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(annotated_frame, f"Behavior: {current_behavior.upper()}", 
                   (10, info_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calculate and display FPS
        if frame_count % 10 == 0:  # Update every 10 frames
            system_stats['fps'] = 10.0 / (time.time() - start_time)
            start_time = time.time()
        
        cv2.putText(annotated_frame, f"FPS: {system_stats['fps']:.1f}", 
                   (10, info_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Uptime
        uptime = int(time.time() - system_stats['system_uptime'])
        cv2.putText(annotated_frame, f"Uptime: {uptime//60}:{uptime%60:02d}", 
                   (frame.shape[1] - 120, info_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection counts
        cv2.putText(annotated_frame, f"Poses: {system_stats['pose_detections']}", 
                   (frame.shape[1] - 120, info_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Weapon detection count
        cv2.putText(annotated_frame, f"Weapons: {system_stats['weapons_detected']}", 
                   (frame.shape[1] - 120, info_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Update the latest frame for web streaming
        with lock:
            ret, jpeg = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]) # Lowered quality for performance
            if ret:
                latest_frame = jpeg.tobytes()
        
        # Small delay to prevent overwhelming the system
        time.sleep(0.01)
    
    cap.release()

def generate_frames():
    """Generate frames for video streaming"""
    global latest_frame, lock
    
    while True:
        with lock:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.033)  # ~30 FPS limit

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/alerts')
def get_alerts():
    """API endpoint to get latest alerts"""
    return jsonify({
        'alerts': alerts_list[:10],  # Return latest 10 alerts
        'total_alerts': len(alerts_list)
    })

@app.route('/api/stats')
def get_stats():
    """API endpoint to get system statistics"""
    return jsonify(system_stats)

@app.route('/api/test_alert')
def test_alert():
    """Test endpoint to simulate an alert"""
    log_alert("Test Alert", 0.95, "Manual test triggered")
    return jsonify({'status': 'success', 'message': 'Test alert generated'})

@app.route('/api/behavior_history')
def get_behavior_history():
    """API endpoint to get behavior analysis history"""
    return jsonify({
        'pose_history_length': len(pose_history),
        'recent_behaviors': list(behavior_buffer)[-5:] if behavior_buffer else []
    })

@app.route('/api/weapon_detections')
def get_weapon_detections():
    """API endpoint to get weapon detection history"""
    weapon_alerts = [alert for alert in alerts_list if alert['type'] == 'Weapon Detected']
    return jsonify({
        'weapon_alerts': weapon_alerts[:10],  # Return latest 10 alerts
        'total_weapon_alerts': len(weapon_alerts)
    })

@app.route('/api/clear_alerts', methods=['POST'])
def clear_alerts():
    """API endpoint to clear all alerts and reset counters"""
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
    print("🚀 Starting Enhanced Smart Surveillance System...")
    print("📹 Initializing camera and AI processing...")
    print("🏃 Loading pose estimation and behavior analysis...")
    print("🔪 Weapon detection enabled...")
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Start the video processing thread
    video_thread = threading.Thread(target=video_processing_thread, daemon=True)
    video_thread.start()
    
    print("🌐 Starting web server...")
    print("🔗 Open your browser and go to: http://localhost:5000")
    print("📊 Features: Human Detection | Pose Estimation | Behavior Analysis | Fall Detection | Weapon Detection")
    print("⚠️  Press CTRL+C to stop the system")
    
    # Run Flask app
    app.run(host='0.0.0.0', debug=False, port=5000, use_reloader=False)