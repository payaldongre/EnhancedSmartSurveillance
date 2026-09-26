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
    "width"  : optional capture width  (defaults to CAMERA_WIDTH, 640)
    "height" : optional capture height (defaults to CAMERA_HEIGHT, 480)

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
    {"id": "Cam_1", "source": 0, "width": 640, "height": 480},

    # --- Add more cameras below as needed, then restart `python app.py`. ---
    # Each one shows up as an extra tab on the dashboard (live view, no AI).
    # {"id": "phone",    "source": "http://192.168.1.50:8080/video"},
    # {"id": "gate",     "source": "rtsp://admin:pass@192.168.1.60:554/stream1"},
    # {"id": "backyard", "source": 1},
    # {"id": "clip",     "source": "samples/demo.mp4"},
]
