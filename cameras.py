"""Camera configuration for the surveillance dashboard.

Edit the CAMERAS list below to add or remove cameras, then just run
``python app.py`` — no ``set`` command and no environment variable needed.

Each entry is a dict:

    "id"     : short label shown on the dashboard camera tab (must be unique)
    "source" : 0/1/...            -> local USB / onboard camera index
               "http://.../video" -> IP camera or phone running an
                                     "IP Webcam" style app
               "rtsp://user:pass@host:554/stream1" -> RTSP camera
               "samples/clip.mp4"  -> a recorded video file
    "width"  : optional capture width  (defaults to CAMERA_WIDTH, 1280)
    "height" : optional capture height (defaults to CAMERA_HEIGHT, 720)

Capture resolution: the dashboard shows the feed full-width now, where 640x480
looked soft once stretched, so the default is 1280x720. Raise it for wide-area
cameras, or lower it per camera if inference time matters more than detail — the
[Timing] log printed by app.py shows what the extra pixels actually cost (the
models themselves are resolution-capped separately by PERSON_IMGSZ and
POSE_ROI_MAX_DIM, so mostly night-vision, drawing and JPEG encoding scale here).

Connection reliability: every camera reconnects on its own. If a feed drops, the
grabber releases the capture and retries with backoff (1s, doubling up to 10s)
for as long as the app runs; URL sources are retried with the FFMPEG backend too,
and opening a URL has an 8s connect timeout so a bad IP cannot hang startup.
GET /api/cameras reports each camera's state ("connecting", "live",
"reconnecting", "error: <reason>") and the actual captured resolution, which the
dashboard shows on every camera tab.

How multiple cameras behave on the one dashboard:

* The FIRST entry is the analytics camera. It runs the full AI pipeline
  (YOLO person tracking, pose/behaviour analysis, ANPR, vehicle & face
  detection, virtual fence, night vision) and is the feed served at
  ``/video_feed``.
* Every further entry is streamed live to the same dashboard as an extra
  camera tab, but does NOT run the AI models. Each detection pipeline is
  CPU-bound, so analysing many streams at once multiplies the load and the
  lag; keeping the extras as live-only feeds is what makes several cameras
  practical on one machine.

To change which camera is analysed, reorder the list so the camera you want
is first.
"""

CAMERAS = [
    # Primary camera — runs the full AI pipeline and feeds the dashboard.
    {"id": "Cam_1", "source": 0, "width": 1280, "height": 720},

    # --- Add more cameras below as needed, then restart `python app.py`. ---
    # Each one shows up as an extra tab on the dashboard (live view, no AI) and
    # reconnects automatically like the primary one.
    # {"id": "phone",    "source": "http://192.168.1.50:8080/video"},
    # {"id": "gate",     "source": "rtsp://admin:pass@192.168.1.60:554/stream1"},
    # {"id": "backyard", "source": 1},
    # {"id": "clip",     "source": "samples/demo.mp4"},
]
