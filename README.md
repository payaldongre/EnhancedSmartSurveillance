# 🛡️ Intelligent Smart Surveillance System

An AI-powered real-time surveillance system that integrates:

- 🧍 Human detection
- 🤸 Pose-based behavior analysis
- 🔪 Hazardous object (weapon) detection
- 🏃 Speed and motion estimation

All combined into a **live monitoring dashboard** for intelligent surveillance.

> This project demonstrates building a real-time AI system with backend integration and automated decision-making, aligned with production AI workflows.

---

## 🚀 Features

- 🎥 Real-time video streaming (Webcam)
- 🧍 Human detection + persistent tracking using YOLOv8 (ByteTrack)
- 🤸 Pose estimation using MediaPipe, tracked per person (not shared across people in frame)
- 🚨 Fall detection & abnormal behavior detection
- 🏃 Running / high-speed movement detection, normalized against frame rate to avoid false positives at low FPS
- 🔪 Hazardous object detection (knife, tools, etc.), with duplicate-alert suppression for a single object that stays in view
- 📏 Speed estimation using motion tracking
- 📊 Live dashboard with alerts and system stats
- 🔔 Alert logging system
- ⚡ Optional GPU acceleration for YOLO inference
- 📁 Data storage:
  - Pose sequences
  - Alerts
  - Reports

---

## 📸 Demo Preview

### 🧠 Live AI Detection (Normal Activity)

[![Normal Activity]](/assets/demo1.png)

---

### 🏃 Behavior Detection (Running)

[![Running Detection]](/assets/demo2.png)

---

### 🔪 Weapon Detection Alert

[![Weapon Detection]](/assets/demo3.png)

---

### 📊 System Statistics Dashboard

[![System Stats]](/assets/stats1.png)

---

### ⚠️ Weapon Detection Logs

[![Weapon Logs]](/assets/stats2.png)

---

### 🤸 Behavior Analysis Panel

[![Behavior Analysis]](/assets/stats3.png)

---

### 🚨 Security Alerts Panel

[![Security Alerts]](/assets/stats4.png)

---

## 🧠 Tech Stack

- Python 3.10 (any standard install — Anaconda is optional, not required)
- Flask (Backend + API)
- OpenCV (Video processing)
- YOLOv8 (Ultralytics), with ByteTrack for tracking
- MediaPipe (Pose estimation — CPU-only, no GPU delegate available)
- PyTorch (CPU or CUDA, see setup below)
- NumPy, Pandas, Scikit-learn
- HTML, CSS, JavaScript (Frontend)

---

## 📁 Project Structure

```
EnhancedSmartSurveillance/
│
├── app.py
├── object_detector.py
├── speed_estimator.py
├── label_generator.py
├── knife.yaml
│
├── templates/
│   └── index.html
│
├── static/
│
├── assets/
│   └── (demo & dashboard screenshots)
│
├── data/
│   ├── pose_sequences/
│   ├── alerts/
│   └── reports/
│
├── models/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

These steps work on **Windows, macOS, and Linux**, and do **not** require Anaconda. If you already use Anaconda/Miniconda, an equivalent conda-based path is given as an alternative in each step.

### 0️⃣ Prerequisites

- Python **3.10** installed and available on your system (avoid 3.13). Download from [python.org](https://www.python.org/downloads/) if you don't already have it, or use Anaconda if you prefer.
- Git installed.
- A webcam.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/payaldongre/EnhancedSmartSurveillance.git
cd EnhancedSmartSurveillance
```

### 2️⃣ Create a Virtual Environment

**Option A — using `venv` (recommended, built into Python, no Anaconda needed)**

Windows (PowerShell or CMD):
```bash
python -m venv env
env\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv env
source env/bin/activate
```

**Option B — using conda (only if you already have Anaconda/Miniconda)**

```bash
conda create --prefix ./env python=3.10
conda activate ./env
```

Either way, you should see `(env)` at the start of your terminal prompt once it's active.

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` intentionally does **not** pin `torch` / `torchvision` / `torchaudio` — install those separately in the next step, since the correct build depends on your hardware.

### 4️⃣ Install PyTorch (CPU or GPU)

PyTorch is only needed to speed up YOLO inference if your machine has an NVIDIA GPU — it's not tied to Anaconda in any way and installs identically inside a `venv` or a conda env.

Check whether you have an NVIDIA GPU and what CUDA version your driver supports:

```bash
nvidia-smi
```

**If you have a compatible NVIDIA GPU** (recommended — CPU-only inference is noticeably laggy), install the matching CUDA build. For CUDA 12.1, for example:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Confirm it worked:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**If you don't have a GPU**, install the CPU-only build instead:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

`app.py` prints `Inference device: GPU (CUDA)` or `Inference device: CPU` on startup, so you can confirm which one is actually active. Note that MediaPipe Pose always runs on CPU regardless of this setting — only the two YOLO passes (person tracking + object detection) benefit from GPU.

---

## ▶️ Run the Project

```bash
# Navigate to the project folder (adjust the path to wherever you cloned it)
cd EnhancedSmartSurveillance

# Activate your environment
# venv — Windows:      env\Scripts\activate
# venv — macOS/Linux:  source env/bin/activate
# conda:                conda activate ./env

# Run the application
python app.py
```

---

## 🌐 Access Dashboard

After running, open:

- Local:
```
http://localhost:5000
```

- Same network (optional):
```
http://<your-ip-address>:5000
```

Example:
```
http://192.168.1.5:5000
```

---

## 📡 API Endpoints

| Endpoint                 | Description        |
| ------------------------ | ------------------ |
| `/`                      | Dashboard          |
| `/video_feed`            | Live video stream  |
| `/api/stats`             | System statistics  |
| `/api/alerts`            | Alert logs         |
| `/api/weapon_detections` | Weapon alerts      |
| `/api/behavior_history`  | Behavior tracking  |
| `/api/test_alert`        | Trigger test alert |
| `/api/clear_alerts`      | Reset alerts       |

---

## 🎯 Detection Capabilities

### 🧍 Human Detection

- YOLOv8-based detection with persistent tracking (ByteTrack) — each person keeps a stable ID across frames

### 🤸 Behavior Analysis

- Normal activity
- Abnormal behavior
- Running detection
- Fall detection

Each person's behavior history is tracked independently, so one person's movement never affects another person's classification even when several people are in frame at once.

### 🔪 Hazard Detection

- Out of the box (stock, COCO-pretrained model): detects **knife, scissors, baseball bat, bottle** near a person
- Detecting firearms or an axe requires a custom-trained model, since those classes don't exist in the stock model's training data — this is in progress
- A detected weapon only triggers a new alert once per occurrence — moving the same object around, or it staying in view, no longer inflates the alert count
- Confidence can drop under partial occlusion (e.g. an object partly covered by a hand) — a known limitation of general-purpose pretrained models, tunable via `hazardous_conf_threshold` in `object_detector.py`

### 🏃 Speed Estimation

- Motion tracking across frames
- Based on:
  - Pixel displacement
  - Pose normalization
  - Temporal smoothing, normalized against actual elapsed time so it stays accurate even when FPS varies

---

## ⚠️ Important Notes

- Python **3.10 recommended** (avoid 3.13)
- Webcam must be available
- Close other apps using camera
- An NVIDIA GPU is recommended but not required — CPU-only mode works, just with lower FPS
- Anaconda is entirely optional — a plain `venv` works the same
- 👉 **YOLO weights will be downloaded automatically on first run.**

---

## 🧪 Testing

- Start the system and open dashboard
- Try:
  - Fast movement → running detection
  - Sudden fall → fall detection
  - Object in hand → weapon detection

---

## 📌 Future Scope

- Virtual fencing (zone-based intrusion alerts)
- Multi-camera support
- ESP32-CAM integration
- Cloud deployment
- SMS / Email alerts
- Database integration
- Expanded weapon-detection classes via a custom-trained model

---

## 👩‍💻 Contributors

- Payal Dongre
- Priyanka Jadhav

---

## 🏁 Summary

A real-time intelligent surveillance system combining:

- Object detection
- Pose estimation
- Behavior analysis
- Motion tracking

into a unified monitoring solution.

---

> Note: Large files, model weights, and environment folders are excluded using `.gitignore` for efficient repository management.