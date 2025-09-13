# Enhanced Smart Surveillance System Requirements
# ================================================

# Core Dependencies
flask==2.3.3
opencv-python==4.8.1.78
ultralytics==8.0.196
mediapipe==0.10.7
numpy==1.24.3
Pillow==10.0.1
python-dateutil==2.8.2

# Additional Dependencies for Enhanced Features
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
pandas==2.0.3

# System Requirements:
# - Python 3.8 or higher
# - Webcam or USB camera
# - Minimum 4GB RAM (8GB recommended)
# - GPU support optional but recommended for better performance

# Installation Instructions:
# ========================
# 1. Create virtual environment:
#    python -m venv surveillance_env
#
# 2. Activate environment:
#    # Windows:
#    surveillance_env\Scripts\activate
#    # Linux/Mac:
#    source surveillance_env/bin/activate
#
# 3. Install requirements:
#    pip install -r requirements.txt
#
# 4. Create templates folder:
#    mkdir templates
#
# 5. Save the HTML file as templates/index.html
#
# 6. Run the application:
#    python app.py
#
# 7. Open browser and go to:
#    http://localhost:5000

# Hardware Setup (for future ESP32-CAM integration):
# =================================================
# 1. ESP32-CAM module with OV2640 camera
# 2. FTDI programmer or ESP32-CAM-MB board
# 3. MicroSD card (optional for local storage)
# 4. Power supply (5V/2A recommended)
# 5. WiFi network for streaming

# ESP32-CAM Code (Arduino IDE):
# ============================
"""
#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// WiFi credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Camera configuration for AI OV2640
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

void setup() {
  Serial.begin(115200);
  
  // Camera configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if(psramFound()){
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  // WiFi connection
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected");
  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");

  startCameraServer();
}

void loop() {
  delay(10000);
}
"""

# Project Structure:
# ==================
# surveillance_project/
# ├── app.py                 # Main Flask application
# ├── requirements.txt       # This file
# ├── templates/
# │   └── index.html        # Dashboard HTML template
# ├── models/               # AI models (auto-downloaded)
# ├── data/                 # Data storage
# │   ├── alerts/          # Alert logs
# │   └── reports/         # Generated reports
# └── static/              # Static files (optional)
#     ├── css/
#     ├── js/
#     └── images/

# Features Implemented:
# ====================
# ✅ Real-time video capture from webcam
# ✅ Human detection using YOLOv8
# ✅ Pose estimation using MediaPipe
# ✅ Fall detection algorithm
# ✅ Abnormal behavior analysis
# ✅ Web-based dashboard
# ✅ Real-time alerts system
# ✅ Statistics tracking
# ✅ Responsive UI design

# Future Enhancements (when hardware is available):
# ================================================
# 🔄 ESP32-CAM integration
# 🔄 Multiple camera support
# 🔄 Database storage (SQLite/MySQL)
# 🔄 Email/SMS notifications
# 🔄 Motion tracking
# 🔄 Object detection beyond humans
# 🔄 Night vision support
# 🔄 Cloud storage integration
# 🔄 Mobile app companion
# 🔄 Advanced behavior patterns

# Troubleshooting:
# ===============
# 1. Camera not detected:
#    - Check if camera is connected and not used by other apps
#    - Try different camera index: cv2.VideoCapture(1) or cv2.VideoCapture(2)
#
# 2. Low FPS:
#    - Reduce camera resolution in the code
#    - Use GPU acceleration if available
#    - Close other applications
#
# 3. High CPU usage:
#    - Increase sleep time in video processing loop
#    - Use smaller YOLO model (yolov8n.pt instead of yolov8s.pt)
#    - Reduce MediaPipe model complexity
#
# 4. Installation issues:
#    - Use Python 3.8-3.11 (avoid 3.12 for compatibility)
#    - Update pip: python -m pip install --upgrade pip
#    - Install Visual C++ Build Tools (Windows)

# Performance Optimization:
# ========================
# - Use GPU acceleration: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# - Enable CUDA support for OpenCV if available
# - Optimize frame processing rate
# - Use threading for better performance
# - Implement frame skipping for real-time processing

# Security Considerations:
# =======================
# - Change default Flask secret key
# - Implement user authentication
# - Use HTTPS in production
# - Secure camera access
# - Implement rate limiting
# - Add input validation