"""Vehicle detection & classification.

Completes the "Vehicle detection and classification" objective.

This module deliberately does NOT load its own YOLO model. The person tracker
in app.py already runs `model.track(...)` on every frame with a full COCO model,
which detects cars/trucks/buses/motorcycles/bicycles for free. Re-using those
boxes means vehicle detection adds essentially zero inference cost.

Classification is two-fold:
  * type  — taken straight from the COCO class (car / truck / bus / ...)
  * colour — dominant paint colour of the bounding-box region, computed in HSV

Counts of unique tracks are de-duplicated the same way weapon alerts are, so a
vehicle that stays in frame does not inflate the totals every frame.
"""

from collections import defaultdict

import cv2
import numpy as np

# COCO class ids that are vehicles (stock yolov8n/s weights).
VEHICLE_CLASS_IDS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def _hue_to_name(hue: int) -> str:
    """Map an OpenCV hue (0-179) to a coarse colour name."""
    if hue < 10 or hue >= 165:
        return "red"
    if hue < 22:
        return "orange"
    if hue < 34:
        return "yellow"
    if hue < 79:
        return "green"
    if hue < 100:
        return "cyan"
    if hue < 131:
        return "blue"
    if hue < 150:
        return "purple"
    return "magenta"


def dominant_color(bgr_region) -> str:
    """Return a coarse colour name for a BGR image region."""
    if bgr_region is None or bgr_region.size == 0:
        return "unknown"
    try:
        hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        chromatic = (s > 45) & (v > 45)
        total = h.size
        if total == 0:
            return "unknown"

        if np.count_nonzero(chromatic) < 0.15 * total:
            # Achromatic paint: decide by brightness.
            mean_v = float(v.mean())
            if mean_v < 55:
                return "black"
            if mean_v > 175:
                return "white"
            return "silver/gray"

        hues = h[chromatic]
        hist, _ = np.histogram(hues, bins=18, range=(0, 180))
        peak_hue = int(np.argmax(hist)) * 10 + 5
        return _hue_to_name(peak_hue)
    except Exception:
        return "unknown"


class VehicleDetector:
    """Turns the tracker's raw boxes into classified vehicle detections."""

    def __init__(self, conf_threshold=0.4, enable_color=True):
        self.conf_threshold = conf_threshold
        self.enable_color = enable_color

        # Cumulative, de-duplicated counters.
        self.by_type = defaultdict(int)
        self.by_color = defaultdict(int)
        self.total_unique = 0
        self._seen_ids = set()
        self.last_frame = []  # detections from the most recent processed frame

    def process_boxes(self, boxes, frame) -> list:
        """Classify vehicle boxes from an ultralytics `results.boxes` object.

        `boxes` may be None when tracking has not produced ids yet.
        Returns a list of detection dicts.
        """
        detections = []
        if boxes is None or len(boxes) == 0 or frame is None:
            self.last_frame = []
            return detections

        has_ids = getattr(boxes, "id", None) is not None
        h, w = frame.shape[:2]

        for box in boxes:
            try:
                cls_id = int(box.cls)
            except Exception:
                continue
            vehicle_type = VEHICLE_CLASS_IDS.get(cls_id)
            if vehicle_type is None:
                continue

            conf = float(box.conf)
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            track_id = -1
            if has_ids:
                try:
                    track_id = int(box.id) if box.id is not None else -1
                except Exception:
                    track_id = -1

            color = dominant_color(frame[y1:y2, x1:x2]) if self.enable_color else "unknown"

            detections.append({
                "track_id": track_id,
                "class_name": vehicle_type,
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "color": color,
            })

            # De-duplicate cumulative stats on first sighting of a track id.
            if track_id >= 0 and track_id not in self._seen_ids:
                self._seen_ids.add(track_id)
                self.by_type[vehicle_type] += 1
                self.by_color[color] += 1
                self.total_unique += 1
            elif track_id < 0:
                # No stable id available — count the sighting but mark as new.
                self.by_type[vehicle_type] += 1
                self.by_color[color] += 1
                self.total_unique += 1

        self.last_frame = detections
        return detections

    def draw(self, frame, detections=None) -> int:
        """Annotate vehicles on the frame. Returns the count drawn."""
        detections = self.last_frame if detections is None else detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name'].upper()} {det['color']} {det['conf']:.2f}"
            tid = det["track_id"]
            if tid >= 0:
                label = f"V{tid} {label}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 191, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (255, 191, 0), -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return len(detections)

    def reset_stats(self):
        self.by_type.clear()
        self.by_color.clear()
        self._seen_ids.clear()
        self.total_unique = 0
        self.last_frame = []

    def stats(self) -> dict:
        return {
            "in_frame": len(self.last_frame),
            "total_unique": self.total_unique,
            "by_type": dict(self.by_type),
            "by_color": dict(self.by_color),
        }
