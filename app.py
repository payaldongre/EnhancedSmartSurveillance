from flask import Flask, Response, render_template, jsonify
import cv2
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

app = Flask(__name__)

# Global variables for thread communication
latest_frame = None
lock = threading.Lock()
alerts_list = []
system_stats = {
    'total_detections': 0,
    'falls_detected': 0,
    'abnormal_behaviors': 0,
    'system_uptime': time.time(),
    'fps': 0,
    'pose_detections': 0
}

# Load YOLOv8 model (will download automatically on first run)
print("Loading AI models... Please wait...")
model = YOLO('yolov8n.pt')  # Nano version for speed

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("AI models loaded successfully!")

# Alert cooldown to prevent spam
last_alert_time = 0
alert_cooldown = 3  # seconds

# Behavior analysis variables
pose_history = deque(maxlen=30)  # Store last 30 poses for analysis
behavior_buffer = deque(maxlen=10)  # Buffer for behavior classification

class PoseAnalyzer:
    """Class to analyze human poses and detect abnormal behaviors"""
    
    def __init__(self):
        self.normal_standing_threshold = 0.3
        self.fall_threshold = 0.7
        self.movement_threshold = 0.15
        
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
        if len(pose_history) > 1:
            prev_pose = pose_history[-1]
            if prev_pose:
                movement_intensity = self.calculate_movement(landmarks, prev_pose)
        
        return {
            'verticality': verticality,
            'head_body_ratio': head_body_ratio,
            'movement_intensity': movement_intensity,
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
        
        # Fall detection
        if verticality < self.fall_threshold and head_body_ratio > 0.8:
            return "fall", 0.9
            
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
    
    print(f"🚨 ALERT: {alert_type} at {timestamp} - {details}")

def video_processing_thread():
    """Main AI processing thread with pose estimation and behavior analysis"""
    global latest_frame, last_alert_time, system_stats, pose_history
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    
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
        success, frame = cap.read()
        
        if not success:
            print("⚠️ Warning: Failed to read frame")
            continue
        
        frame_count += 1
        
        # Flip frame horizontally for mirror effect (more natural)
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference for person detection
        yolo_results = model(frame, imgsz=320, verbose=False)[0]
        
        # Initialize annotated frame
        annotated_frame = frame.copy()
        person_detected = False
        current_behavior = "idle"
        
        # Process YOLO detections
        if yolo_results.boxes is not None:
            for box in yolo_results.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                
                # Check if detected object is a person (class 0 in COCO dataset)
                if cls_id == 0 and conf > 0.4:
                    person_detected = True
                    system_stats['total_detections'] += 1
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Extract person region for pose estimation
                    person_roi = rgb_frame[y1:y2, x1:x2]
                    
                    if person_roi.size > 0:
                        # Run pose estimation on person ROI
                        pose_results = pose.process(person_roi)
                        
                        if pose_results.pose_landmarks:
                            system_stats['pose_detections'] += 1
                            
                            # Draw pose landmarks on the frame
                            # First, adjust landmarks to full frame coordinates
                            adjusted_landmarks = []
                            for landmark in pose_results.pose_landmarks.landmark:
                                adjusted_landmark = mp_pose.PoseLandmark()
                                adjusted_landmark.x = (landmark.x * (x2 - x1) + x1) / frame.shape[1]
                                adjusted_landmark.y = (landmark.y * (y2 - y1) + y1) / frame.shape[0]
                                adjusted_landmark.z = landmark.z
                                adjusted_landmark.visibility = landmark.visibility
                                adjusted_landmarks.append(adjusted_landmark)
                            
                            # Draw pose landmarks
                            for i, landmark in enumerate(adjusted_landmarks):
                                if landmark.visibility > 0.5:
                                    x = int(landmark.x * frame.shape[1])
                                    y = int(landmark.y * frame.shape[0])
                                    cv2.circle(annotated_frame, (x, y), 3, (0, 255, 255), -1)
                            
                            # Analyze pose for behavior classification
                            pose_metrics = pose_analyzer.calculate_pose_metrics(adjusted_landmarks)
                            
                            if pose_metrics:
                                # Add to pose history
                                pose_history.append(adjusted_landmarks)
                                
                                # Classify behavior
                                behavior, confidence = pose_analyzer.classify_behavior(pose_metrics)
                                current_behavior = behavior
                                
                                # Handle alerts
                                current_time = time.time()
                                if current_time - last_alert_time > alert_cooldown:
                                    if behavior == "fall":
                                        log_alert("Fall Detected", confidence, 
                                                f"Verticality: {pose_metrics['verticality']:.2f}")
                                        last_alert_time = current_time
                                    elif behavior == "abnormal":
                                        log_alert("Abnormal Behavior", confidence,
                                                f"Movement: {pose_metrics['movement_intensity']:.2f}")
                                        last_alert_time = current_time
                                
                                # Color coding based on behavior
                                if behavior == "fall":
                                    color = (0, 0, 255)  # Red
                                    text = f"FALL DETECTED! ({confidence:.2f})"
                                elif behavior == "abnormal":
                                    color = (0, 165, 255)  # Orange
                                    text = f"Abnormal Behavior ({confidence:.2f})"
                                else:
                                    color = (0, 255, 0)  # Green
                                    text = f"Normal Activity ({confidence:.2f})"
                                
                                # Draw bounding box
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(annotated_frame, text, 
                                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                                # Add pose metrics display
                                metrics_text = f"V:{pose_metrics['verticality']:.2f} M:{pose_metrics['movement_intensity']:.2f}"
                                cv2.putText(annotated_frame, metrics_text, 
                                          (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    else:
                        # Fallback: draw simple bounding box if pose estimation fails
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Person ({conf:.2f})", 
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
        
        # Update the latest frame for web streaming
        with lock:
            ret, jpeg = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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

if __name__ == '__main__':
    print("🚀 Starting Enhanced Smart Surveillance System...")
    print("📹 Initializing camera and AI processing...")
    print("🏃 Loading pose estimation and behavior analysis...")
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Start the video processing thread
    video_thread = threading.Thread(target=video_processing_thread, daemon=True)
    video_thread.start()
    
    print("🌐 Starting web server...")
    print("🔗 Open your browser and go to: http://localhost:5000")
    print("📊 Features: Human Detection | Pose Estimation | Behavior Analysis | Fall Detection")
    print("⚠️  Press CTRL+C to stop the system")
    
    # Run Flask app
    app.run(host='0.0.0.0', debug=False, port=5000, use_reloader=False)