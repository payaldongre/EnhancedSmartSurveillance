# import time
# import math
# from collections import deque, defaultdict
# import numpy as np
# import cv2

# # Per-track history (deque of (cx, cy, t, bbox_h, ppm_est))
# track_histories = defaultdict(lambda: deque(maxlen=8))
# # Exponential moving average of speed (km/h)
# speed_ema = {}

# # Configurable defaults (tune per site)
# DEFAULT_PERSON_HEIGHT_M = 1.70     # assume average person height (m)
# MIN_SPEED_KMPH = 0.5               # below this => treat as 0 (remove jitter)
# RUNNING_KMPH_THRESHOLD = 10.0      # > this => running (adjustable)
# EMA_ALPHA = 0.45                   # smoothing factor for speed EMA (0..1)

# def compute_ppm_from_bbox(bbox_h_pixels, expected_height_m=DEFAULT_PERSON_HEIGHT_M):
#     """Estimate pixels-per-meter from the bbox height in pixels."""
#     if bbox_h_pixels <= 0 or expected_height_m <= 0:
#         return None
#     return float(bbox_h_pixels) / float(expected_height_m)

# def _estimate_velocity_regression(history):
#     """
#     Given a list of (cx, cy, t, ppm), estimate speed:
#       - use linear regression (polyfit degree=1) on x(t) and y(t)
#       - compute resultant speed (px/s) then convert to km/h with median ppm
#     Returns: (speed_kmph_or_None, px_per_s, used_ppm)
#     """
#     if len(history) < 2:
#         return None, 0.0, None

#     # arrays
#     ts = np.array([h[2] for h in history], dtype=np.float64)
#     xs = np.array([h[0] for h in history], dtype=np.float64)
#     ys = np.array([h[1] for h in history], dtype=np.float64)
#     ppms = [h[4] for h in history if h[4] is not None]

#     # center times to improve numerical stability
#     t0 = ts.mean()
#     T = ts - t0

#     try:
#         # fit linear slope: x = a_x * T + b_x
#         a_x, b_x = np.polyfit(T, xs, 1)
#         a_y, b_y = np.polyfit(T, ys, 1)
#         vx = float(a_x)  # px/s
#         vy = float(a_y)
#         px_per_s = math.hypot(vx, vy)
#     except Exception:
#         # fallback to last-two-samples difference
#         x1, y1, t1 = xs[-2], ys[-2], ts[-2]
#         x2, y2, t2 = xs[-1], ys[-1], ts[-1]
#         dt = max(1e-3, t2 - t1)
#         px_per_s = math.hypot(x2 - x1, y2 - y1) / dt
#         vx = vy = 0.0

#     used_ppm = float(np.median(ppms)) if len(ppms) else None

#     if used_ppm and used_ppm > 1e-6:
#         m_per_s = px_per_s / used_ppm
#         kmph = m_per_s * 3.6
#         return float(kmph), float(px_per_s), used_ppm
#     else:
#         # no ppm available -> cannot convert to km/h reliably
#         return None, float(px_per_s), None

# def process_person_detections(detections, frame,
#                               camera_id="Cam_1",
#                               landmarks_map=None,
#                               expected_person_height_m=DEFAULT_PERSON_HEIGHT_M,
#                               min_history_len=3):
#     """
#     detections: list of (track_id, x1, y1, x2, y2, label)
#     landmarks_map: optional dict {track_id: (hip_x_pix, hip_y_pix)} to use torso center instead of bbox center
#     Returns annotated frame. Also prints/logs alerts for running.
#     """
#     now = time.time()
#     for det in detections:
#         try:
#             track_id, x1, y1, x2, y2, label = det
#         except Exception:
#             continue
#         if label != "person":
#             continue

#         # prefer hip center if available (more stable than face)
#         if landmarks_map and track_id in landmarks_map:
#             cx, cy = landmarks_map[track_id]
#         else:
#             cx = (x1 + x2) / 2.0
#             cy = (y1 + y2) / 2.0

#         bbox_h = max(1.0, float(y2 - y1))
#         ppm = compute_ppm_from_bbox(bbox_h, expected_person_height_m)

#         # push into history
#         hist = track_histories[track_id]
#         hist.append((float(cx), float(cy), now, float(bbox_h), ppm))

#         # only compute if we have enough samples (reduces false spikes)
#         speed_kmph, pxps, used_ppm = _estimate_velocity_regression(list(hist))  # km/h or None

#         # update EMA smoothing
#         prev = speed_ema.get(track_id)
#         if speed_kmph is not None:
#             if prev is None:
#                 smoothed = speed_kmph
#             else:
#                 smoothed = (1 - EMA_ALPHA) * prev + EMA_ALPHA * speed_kmph
#             speed_ema[track_id] = float(smoothed)
#         else:
#             # no conversion available; keep prior smoothed speed if any
#             smoothed = prev if prev is not None else None

#         # fallback: convert px/s to km/h using last ppm if possible
#         if smoothed is None and pxps and used_ppm:
#             smoothed = (pxps / used_ppm) * 3.6

#         # treat micro-speeds as zero
#         if smoothed is None:
#             display_speed = 0.0
#         else:
#             display_speed = float(smoothed) if smoothed >= MIN_SPEED_KMPH else 0.0

#         # Annotate frame
#         text = f"ID:{track_id} {display_speed:.1f} km/h"
#         cv2.putText(frame, text, (int(x1), max(20, int(y1)-10)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

#         # Decide "running" alert using km/h threshold (use display_speed)
#         if display_speed >= RUNNING_KMPH_THRESHOLD:
#             # you can replace print() with log_alert(...) from your app
#             print(f"[ALERT] Running detected on {camera_id} | ID:{track_id} | {display_speed:.1f} km/h")

#     return frame

# # Optional helper to clear old tracks (call periodically)
# def cleanup_stale_tracks(max_age_sec=5.0):
#     now = time.time()
#     remove = []
#     for tid, hist in list(track_histories.items()):
#         if not hist:
#             remove.append(tid)
#             continue
#         # last timestamp:
#         if now - hist[-1][2] > max_age_sec:
#             remove.append(tid)
#     for tid in remove:
#         track_histories.pop(tid, None)
#         speed_ema.pop(tid, None)

# speed_estimator.py
import time
import math
from collections import deque, defaultdict
import numpy as np
import cv2

# Per-track history (deque of (cx, cy, t, bbox_h, ppm_est))
track_histories = defaultdict(lambda: deque(maxlen=8))
# Exponential moving average of speed (km/h)
speed_ema = {}

# Configurable defaults (tune per site)
DEFAULT_PERSON_HEIGHT_M = 1.70     # assume average person height (m)
MIN_SPEED_KMPH = 0.5               # below this => treat as 0 (remove jitter)
RUNNING_KMPH_THRESHOLD = 10.0      # > this => running (adjustable)
EMA_ALPHA = 0.45                   # smoothing factor for speed EMA (0..1)

def compute_ppm_from_bbox(bbox_h_pixels, expected_height_m=DEFAULT_PERSON_HEIGHT_M):
    """Estimate pixels-per-meter from the bbox height in pixels."""
    if bbox_h_pixels <= 0 or expected_height_m <= 0:
        return None
    return float(bbox_h_pixels) / float(expected_height_m)

def _estimate_velocity_regression(history):
    """
    Given a list of (cx, cy, t, ppm), estimate speed:
      - use linear regression (polyfit degree=1) on x(t) and y(t)
      - compute resultant speed (px/s) then convert to km/h with median ppm
    Returns: (speed_kmph_or_None, px_per_s, used_ppm)
    """
    if len(history) < 2:
        return None, 0.0, None

    # arrays
    ts = np.array([h[2] for h in history], dtype=np.float64)
    xs = np.array([h[0] for h in history], dtype=np.float64)
    ys = np.array([h[1] for h in history], dtype=np.float64)
    ppms = [h[4] for h in history if h[4] is not None]

    # center times to improve numerical stability
    t0 = ts.mean()
    T = ts - t0

    try:
        # fit linear slope: x = a_x * T + b_x
        a_x, b_x = np.polyfit(T, xs, 1)
        a_y, b_y = np.polyfit(T, ys, 1)
        vx = float(a_x)  # px/s
        vy = float(a_y)
        px_per_s = math.hypot(vx, vy)
    except Exception:
        # fallback to last-two-samples difference
        x1, y1, t1 = xs[-2], ys[-2], ts[-2]
        x2, y2, t2 = xs[-1], ys[-1], ts[-1]
        dt = max(1e-3, t2 - t1)
        px_per_s = math.hypot(x2 - x1, y2 - y1) / dt
        vx = vy = 0.0

    used_ppm = float(np.median(ppms)) if len(ppms) else None

    if used_ppm and used_ppm > 1e-6:
        m_per_s = px_per_s / used_ppm
        kmph = m_per_s * 3.6
        return float(kmph), float(px_per_s), used_ppm
    else:
        # no ppm available -> cannot convert to km/h reliably
        return None, float(px_per_s), None

def process_person_detections(detections, frame,
                              camera_id="Cam_1",
                              landmarks_map=None,
                              expected_person_height_m=DEFAULT_PERSON_HEIGHT_M,
                              min_history_len=3):
    """
    detections: list of (track_id, x1, y1, x2, y2, label)
    landmarks_map: optional dict {track_id: (hip_x_pix, hip_y_pix)} to use torso center instead of bbox center
    Returns annotated frame. Also prints/logs alerts for running.
    """
    now = time.time()
    for det in detections:
        try:
            track_id, x1, y1, x2, y2, label = det
        except Exception:
            continue
        if label != "person":
            continue

        # prefer hip center if available (more stable than face)
        if landmarks_map and track_id in landmarks_map:
            cx, cy = landmarks_map[track_id]
        else:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

        bbox_h = max(1.0, float(y2 - y1))
        ppm = compute_ppm_from_bbox(bbox_h, expected_person_height_m)

        # push into history
        hist = track_histories[track_id]
        hist.append((float(cx), float(cy), now, float(bbox_h), ppm))

        # only compute if we have enough samples (reduces false spikes)
        speed_kmph, pxps, used_ppm = _estimate_velocity_regression(list(hist))  # km/h or None

        # update EMA smoothing
        prev = speed_ema.get(track_id)
        if speed_kmph is not None:
            if prev is None:
                smoothed = speed_kmph
            else:
                smoothed = (1 - EMA_ALPHA) * prev + EMA_ALPHA * speed_kmph
            speed_ema[track_id] = float(smoothed)
        else:
            # no conversion available; keep prior smoothed speed if any
            smoothed = prev if prev is not None else None

        # fallback: convert px/s to km/h using last ppm if possible
        if smoothed is None and pxps and used_ppm:
            smoothed = (pxps / used_ppm) * 3.6

        # treat micro-speeds as zero
        if smoothed is None:
            display_speed = 0.0
        else:
            display_speed = float(smoothed) if smoothed >= MIN_SPEED_KMPH else 0.0

        # Annotate frame
        text = f"ID:{track_id} {display_speed:.1f} km/h"
        cv2.putText(frame, text, (int(x1), max(20, int(y1)-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        # Decide "running" alert using km/h threshold (use display_speed)
        if display_speed >= RUNNING_KMPH_THRESHOLD:
            # you can replace print() with log_alert(...) from your app
            print(f"[ALERT] Running detected on {camera_id} | ID:{track_id} | {display_speed:.1f} km/h")

    return frame

# Optional helper to clear old tracks (call periodically)
def cleanup_stale_tracks(max_age_sec=5.0):
    now = time.time()
    remove = []
    for tid, hist in list(track_histories.items()):
        if not hist:
            remove.append(tid)
            continue
        # last timestamp:
        if now - hist[-1][2] > max_age_sec:
            remove.append(tid)
    for tid in remove:
        track_histories.pop(tid, None)
        speed_ema.pop(tid, None)