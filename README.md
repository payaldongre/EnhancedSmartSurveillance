# 🛡️ Intelligent Smart Surveillance System

An AI-powered real-time surveillance system that integrates:

- 🧍 Human detection & tracking
- 🚗 Vehicle detection & classification
- 🙂 Face detection
- 🪪 Automatic Number Plate Recognition (ANPR)
- 🤸 Pose-based behavior analysis
- 🔪 Hazardous object (weapon) detection
- 🚧 Virtual fence / restricted-zone intrusion detection
- 🌙 Night-time movement detection
- 🏃 Speed and motion estimation
- 🔗 Command & Control (C2) integration

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
- 🚗 Vehicle detection & classification (type + dominant colour, reuses the tracker's boxes — no extra inference)
- 🙂 Face detection (OpenCV's bundled Haar cascades, optional privacy blur)
- 🪪 Automatic Number Plate Recognition (plate localisation + pluggable OCR)
- 🚧 Virtual fence intrusion detection (polygon zones, dwell/loiter alerts)
- 🌙 Night-time movement detection (low-light enhancement + frame-difference motion)
- 🔗 Command & Control (C2) integration (standard event schema over webhook, optional MQTT)
- 💾 Persistent storage of alerts, events and reports (JSONL / JSON under `data/`)
- 📁 Data storage:
  - Pose sequences
  - Alerts
  - Events (C2 payloads)
  - Reports

---

## 📸 Demo Preview

### 🧠 Live AI Detection (Normal Activity)

[![Normal Activity](assets/demo1.png)](assets/demo1.png)

---

### 🏃 Behavior Detection (Running)

[![Running Detection](assets/demo2.png)](assets/demo2.png)

---

### 🔪 Weapon Detection Alert

[![Weapon Detection](assets/demo3.png)](assets/demo3.png)

---

### 📊 System Statistics Dashboard

[![System Stats](assets/stats1.png)](assets/stats1.png)

---

### ⚠️ Weapon Detection Logs

[![Weapon Logs](assets/stats2.png)](assets/stats2.png)

---

### 🤸 Behavior Analysis Panel

[![Behavior Analysis](assets/stats3.png)](assets/stats3.png)

---

### 🚨 Security Alerts Panel

[![Security Alerts](assets/stats4.png)](assets/stats4.png)

---

## 🧠 Tech Stack

- Python 3.10 (any standard install — Anaconda is optional, not required)
- Flask (Backend + API)
- OpenCV (Video processing)
- YOLOv8 (Ultralytics), with ByteTrack for tracking
- MediaPipe (Pose estimation — CPU-only, no GPU delegate available)
- PyTorch (CPU or CUDA, see setup below)
- NumPy, Pandas, Scikit-learn
- OpenCV Haar cascades (face detection) + the bundled plate cascade (ANPR localisation)
- easyocr / pytesseract (optional — ANPR OCR backend)
- HTTP webhooks / MQTT (Command & Control integration)
- HTML, CSS, JavaScript (Frontend)

---

## 📁 Project Structure

```
EnhancedSmartSurveillance/
│
├── app.py
├── object_detector.py        # hazardous-object detection (custom weapon model aware)
├── vehicle_detector.py       # vehicle detection + type/colour classification
├── face_detector.py          # face detection (Haar cascades, optional blur)
├── anpr.py                   # number-plate localisation + OCR
├── virtual_fence.py          # polygon zones, intrusion / dwell detection
├── night_vision.py           # low-light enhancement + motion detection
├── c2_integration.py         # Command & Control event publishing (webhook/MQTT)
├── event_store.py            # persistence for alerts, events and reports
├── zones.json                # virtual fence zone definitions
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
| `/api/detections`        | Combined snapshot of every detector |
| `/api/vehicles`          | Vehicle detection stats |
| `/api/faces`             | Face detection stats |
| `/api/anpr`              | Number-plate reads   |
| `/api/zones`             | List / add (POST) virtual fence zones |
| `/api/zones/<name>`      | Delete a virtual fence zone (DELETE) |
| `/api/c2/status`         | Command & Control integration status |
| `/api/c2/test`           | Queue a test event to the C2 system (POST) |
| `/api/report`            | Build + persist a full snapshot report |

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
- A **custom-trained weapon model is supported and auto-detected**. Drop your trained YOLOv8 weights at `models/weapons/best.pt` (or set `WEAPON_MODEL_PATH`), and the system uses them instead of the stock model — every class the model was trained on is then treated as hazardous, so a knife+gun model lights up both. Candidate paths checked, in order: `WEAPON_MODEL_PATH`, `models/weapons/best.pt`, `models/best.pt`, `best.pt`, `runs/detect/train/weights/best.pt`. The startup log prints which model is in use, and `/api/detections` reports it too
- Fine-tune the model with the included `label_generator.py` annotator and `knife.yaml` dataset config (`dataset/images/{train,val,test}` + `dataset/labels/{train,val,test}`)
- A detected weapon only triggers a new alert once per occurrence — moving the same object around, or it staying in view, no longer inflates the alert count
- Confidence can drop under partial occlusion (e.g. an object partly covered by a hand) — a known limitation of general-purpose pretrained models, tunable via `HAZARDOUS_CONF_THRESHOLD`

### 🏃 Speed Estimation

- Motion tracking across frames
- Based on:
  - Pixel displacement
  - Pose normalization
  - Temporal smoothing, normalized against actual elapsed time so it stays accurate even when FPS varies

### 🚗 Vehicle Detection & Classification

- Reuses the person tracker's own YOLO boxes, so enabling it adds essentially no inference cost
- Classifies **type** (car, truck, bus, motorcycle, bicycle) and **dominant paint colour** (red, blue, silver/gray, black, ...)
- Cumulative counts are de-duplicated per tracked vehicle, so a vehicle that stays in frame is only counted once

### 🙂 Face Detection

- Uses the Haar cascades bundled inside `opencv-python` — no extra download or dependency, works offline
- Optional pixelation of detected faces for privacy (`FACE_PRIVACY_BLUR=1`)
- Only the face **count** is stored or forwarded; no face recognition or identity data is kept

### 🪪 Automatic Number Plate Recognition (ANPR)

- Locates the plate inside each vehicle box (bundled plate cascade, then a shape/aspect-ratio fallback)
- Reads it with a pluggable OCR backend: **easyocr** (recommended, reuses the torch install) or **pytesseract**
- De-duplicates per vehicle track, so the same plate only raises a new alert when it actually changes
- If no OCR backend is installed the rest of the system keeps running and ANPR simply reports itself unavailable

### 🚧 Virtual Fence / Restricted Zones

- Polygons defined in `zones.json` using **normalised** coordinates (0-1), so they are camera-resolution independent
- Emits **intrusion**, **dwell** (loitering past the zone's dwell time) and **exit** events for people and vehicles
- Zones can also be added at runtime via `POST /api/zones`
- State is per (track, zone), so standing inside a zone does not re-fire the alert every frame

### 🌙 Night-time Movement Detection

- Estimates scene brightness every frame and switches to **night mode** below the threshold
- In night mode the frame is enhanced (CLAHE + gamma) for both display and detection
- Detects movement with frame differencing — independent of YOLO, so it still works when the models struggle in the dark

### 🔗 Command & Control (C2) Integration

- Every alert is normalised into one stable JSON schema (`smart-surveillance.event/1.0`) with a severity level
- Delivered over an HTTP webhook (`C2_WEBHOOK_URL`, optional `C2_API_KEY`) and/or MQTT (`C2_MQTT_HOST`)
- Publishing is queued and asynchronous with retries, so a slow or unreachable C2 system never stalls the video loop
- Live delivery stats (sent / queued / failed) are shown on the dashboard

---

## ⚙️ Configuration (environment variables)

All optional — sensible defaults apply when unset.

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `CAMERA_ID` | `Cam_1` | Identifier included in alerts / C2 events |
| `WEAPON_MODEL_PATH` | auto-detected | Path to a custom weapon `best.pt`; otherwise COCO is used |
| `HAZARDOUS_CONF_THRESHOLD` | `0.4` | Confidence needed to flag a hazardous object |
| `ENABLE_VEHICLE_DETECTION` | `1` | Toggle vehicle detection/classification |
| `ENABLE_FACE_DETECTION` | `1` | Toggle face detection |
| `FACE_PRIVACY_BLUR` | `0` | Pixelate detected faces |
| `ENABLE_ANPR` | `1` | Toggle number-plate recognition |
| `ENABLE_VIRTUAL_FENCE` | `1` | Toggle zone intrusion detection |
| `ZONES_CONFIG` | `zones.json` | Zone definition file |
| `ENABLE_NIGHT_VISION` | `1` | Toggle night-mode enhancement + motion detection |
| `C2_ENABLED` | auto | Force-enable C2 (auto-on when a webhook URL is set) |
| `C2_WEBHOOK_URL` | — | C2 HTTP endpoint |
| `C2_API_KEY` | — | Bearer token for the webhook |
| `C2_SOURCE_ID` | `site-1` | Site identifier in events |
| `C2_MQTT_HOST` / `C2_MQTT_PORT` | — / `1883` | Optional MQTT broker |

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
  - Hold a printed/on-screen number plate up to the camera (with an OCR backend installed) → ANPR plate read in the dashboard
  - Walk through the shaded zone on the video feed → *Intrusion Detected* alert (edit `zones.json` to move the zone)
  - Dim the lights → the dashboard switches to **Night Mode** and moving objects raise *Night-time Movement*
- Inspect the API directly, e.g. `curl http://localhost:5000/api/detections` and `curl http://localhost:5000/api/c2/status`
- Persisted output lands in `data/alerts/alerts.jsonl`, `data/events/events.jsonl` and `data/reports/`

### Automated checks

Runs on the standard library alone (it installs lightweight OpenCV / NumPy / Ultralytics
stand-ins only when those packages are absent, so it works before you install anything):

```bash
python -m unittest discover -s tests -v
```

It covers the C2 event schema and real webhook delivery, alert/event/report persistence,
the virtual-fence state machine (intrusion → dwell → exit → re-entry) and zone validation,
stock-vs-custom weapon model class handling, and static consistency of `app.py` routes with
the dashboard's endpoints and element ids.

---

## 📌 Future Scope

- Multi-camera support
- ESP32-CAM integration
- Cloud deployment
- SMS / Email alerts (beyond the current webhook/MQTT C2 delivery)
- Database integration (beyond the current file-based store)
- Add weapons to the custom model by extending `knife.yaml` with your own annotated data

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