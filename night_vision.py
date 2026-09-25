"""Night-time / low-light movement detection.

Completes the "Night-time movement detection" objective.

Two things happen here:

1. Light-level assessment. The mean brightness of the frame (V channel of HSV)
   decides whether the camera is in "night mode". When it is, the frame is
   enhanced with CLAHE + a gamma lift so downstream detection stays usable, and
   a NIGHT chip is drawn on the overlay. `night_mode` is also published in the
   dashboard stats.

2. Movement detection that is independent of YOLO. Frame differencing over the
   grayscale (enhanced) frames finds moving regions cheaply. In night mode a
   movement event above the pixel threshold raises a "Night-time Movement"
   alert. This also acts as a fallback for detecting motion at all when the
   main models are struggling with a dark, noisy frame.
"""

import cv2
import numpy as np


class NightVision:
    def __init__(self, enabled=True, brightness_threshold=60,
                 enhance=True, min_motion_area=800, motion_pixel_ratio=0.008):
        self.enabled = enabled
        self.brightness_threshold = brightness_threshold
        self.enhance = enhance
        self.min_motion_area = min_motion_area
        self.motion_pixel_ratio = motion_pixel_ratio

        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._prev_gray = None

        self.night_mode = False
        self.brightness = 255.0
        self.motion_regions = []
        self.motion_active = False
        self.night_motion_events = 0

    def analyze(self, frame):
        """Assess light level and motion. Returns a small result dict."""
        if not self.enabled or frame is None:
            return {"night_mode": False, "brightness": 255.0,
                    "motion": False, "enhanced": frame}

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.brightness = float(hsv[:, :, 2].mean())
        self.night_mode = self.brightness < self.brightness_threshold

        if self.enhance and self.night_mode:
            enhanced = self._apply_low_light(frame)
        else:
            enhanced = frame

        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        motion, regions = self._detect_motion(gray)

        self.motion_active = motion
        self.motion_regions = regions
        if motion and self.night_mode:
            self.night_motion_events += 1

        return {
            "night_mode": self.night_mode,
            "brightness": self.brightness,
            "motion": motion,
            "motion_regions": regions,
            "enhanced": enhanced,
        }

    # ------------------------------------------------------------------ internals
    def _apply_low_light(self, frame):
        """CLAHE on the L channel of LAB + a mild gamma lift."""
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            merged = cv2.merge((l, a, b))
            out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            # Gamma lift brightens shadows without blowing out highlights.
            inv = 1.0 / 1.3
            table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
            return cv2.LUT(out, table)
        except Exception:
            return frame

    def _detect_motion(self, gray):
        if self._prev_gray is None:
            self._prev_gray = gray
            return False, []

        try:
            delta = cv2.absdiff(self._prev_gray, gray)
            self._prev_gray = gray
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regions = []
            moved_pixels = 0
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_motion_area:
                    continue
                regions.append(tuple(int(v) for v in cv2.boundingRect(c)))
                moved_pixels += area

            frame_area = gray.shape[0] * gray.shape[1]
            ratio = moved_pixels / float(max(1, frame_area))
            return (len(regions) > 0 and ratio >= self.motion_pixel_ratio), regions
        except Exception:
            return False, []

    def annotate(self, frame):
        """Draw night-mode chip and movement boxes."""
        if not self.enabled or frame is None:
            return frame
        if self.night_mode:
            cv2.putText(frame, "NIGHT MODE", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for (x, y, w, h) in self.motion_regions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
        return frame

    def reset_stats(self):
        self.night_motion_events = 0

    def stats(self):
        return {
            "night_mode": self.night_mode,
            "brightness": round(self.brightness, 1),
            "motion": self.motion_active,
            "night_motion_events": self.night_motion_events,
        }
