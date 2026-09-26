"""Face detection.

Completes the "Face detection" objective.

Uses the Haar cascade that ships *inside* opencv-python (cv2.data.haarcascades),
so there is no new dependency, no model download, and it works fully offline.

Privacy note: the detections are used only for counting / on-frame boxes. If
`privacy_blur` is enabled the face region is pixelated on the displayed stream,
and nothing but the count is ever logged or sent to a C2 system.
"""

import os

import cv2

CASCADE_FILE = "haarcascade_frontalface_default.xml"
# A second, smaller cascade catches some profile/partial faces at a small cost.
CASCADE_FILE_ALT = "haarcascade_profileface.xml"


class FaceDetector:
    def __init__(self, enabled=True, scale_factor=1.1, min_neighbors=5,
                 min_size=(30, 30), privacy_blur=False):
        self.enabled = enabled
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.privacy_blur = privacy_blur

        self.cascade = self._load(CASCADE_FILE)
        self.cascade_alt = self._load(CASCADE_FILE_ALT)
        self.available = self.cascade is not None

        self.last_count = 0
        self.last_faces = []

    @staticmethod
    def _load(name):
        try:
            path = os.path.join(cv2.data.haarcascades, name)
            if not os.path.exists(path):
                return None
            cascade = cv2.CascadeClassifier(path)
            return cascade if not cascade.empty() else None
        except Exception:
            return None

    def detect(self, frame, gray=None) -> list:
        """Return a list of (x, y, w, h) face boxes."""
        if not self.enabled or not self.available or frame is None:
            self.last_count = 0
            self.last_faces = []
            return []
        try:
            if gray is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors, minSize=self.min_size,
            )
            boxes = [tuple(int(v) for v in f) for f in faces]
            if self.cascade_alt is not None:
                try:
                    profile = self.cascade_alt.detectMultiScale(
                        gray, scaleFactor=self.scale_factor,
                        minNeighbors=self.min_neighbors, minSize=self.min_size,
                    )
                    # Keep only profile boxes that don't heavily overlap a frontal one.
                    for p in profile:
                        p = tuple(int(v) for v in p)
                        if not any(_iou(p, b) > 0.4 for b in boxes):
                            boxes.append(p)
                except Exception:
                    pass
            self.last_count = len(boxes)
            self.last_faces = boxes
            return boxes
        except Exception:
            self.last_count = 0
            self.last_faces = []
            return []

    def annotate(self, frame, faces) -> None:
        for (x, y, w, h) in faces:
            if self.privacy_blur:
                region = frame[y:y + h, x:x + w]
                if region.size:
                    small = cv2.resize(region, (max(1, w // 12), max(1, h // 12)),
                                       interpolation=cv2.INTER_LINEAR)
                    frame[y:y + h, x:x + w] = cv2.resize(
                        small, (w, h), interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "FACE", (x, max(12, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def stats(self) -> dict:
        return {
            "in_frame": self.last_count,
            "available": self.available,
            "privacy_blur": self.privacy_blur,
        }


def _iou(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
