"""Automatic Number Plate Recognition (ANPR / ALPR).

Completes the "Automatic Number Plate Recognition (ANPR)" objective.

Pipeline:  vehicle bbox -> plate localisation -> de-skew / threshold -> OCR -> normalise.

Plate localisation tries, in order:
  1. the `haarcascade_russian_plate_number.xml` cascade bundled with
     opencv-python (good on the rectangular plates used in most regions), then
  2. a contour/shape heuristic over the lower half of the vehicle box.

OCR backends are pluggable and optional, so the app still runs without them:
  * easyocr      — pip install easyocr      (reuses torch, already installed for YOLO)
  * pytesseract  — pip install pytesseract  (+ the Tesseract binary on PATH)

Both are listed as optional in requirements.txt. If neither is present the
module reports itself unavailable instead of crashing the video pipeline.
"""

import os
import re

import cv2
import numpy as np

PLATE_CASCADE_FILE = "haarcascade_russian_plate_number.xml"

# Characters that are actually meaningful on a plate.
_PLATE_RE = re.compile(r"[^A-Z0-9]")
# Common OCR confusions, corrected only inside digit/letter runs.
_DIGIT_FIX = {"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}


def normalize_plate(text: str) -> str:
    """Uppercase, strip noise, and drop obviously non-plate strings."""
    if not text:
        return ""
    text = _PLATE_RE.sub("", text.upper())
    return text


class ANPRReader:
    def __init__(self, enabled=True, gpu=False, min_text_len=4, max_text_len=10):
        self.enabled = enabled
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len

        self.plate_cascade = self._load_plate_cascade()
        # Don't pay the (large) OCR model load when the feature is switched off.
        if enabled:
            self.ocr, self.backend = self._init_ocr(gpu)
        else:
            self.ocr, self.backend = None, "disabled"

        # track_id -> last plate text (avoids re-alerting on the same vehicle)
        self.last_read_by_track = {}
        self.reads = []  # list of {"plate","confidence","track_id","time"}
        self.total_unique = 0
        self._seen_plates = set()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _load_plate_cascade():
        try:
            path = os.path.join(cv2.data.haarcascades, PLATE_CASCADE_FILE)
            if not os.path.exists(path):
                return None
            cascade = cv2.CascadeClassifier(path)
            return cascade if not cascade.empty() else None
        except Exception:
            return None

    @staticmethod
    def _init_ocr(gpu):
        # Preferred: easyocr (already pulls in torch, which YOLOv8 needs).
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=bool(gpu), verbose=False)
            return reader, "easyocr"
        except Exception:
            pass
        # Fallback: pytesseract (needs the Tesseract binary too).
        try:
            import pytesseract  # noqa: F401
            return pytesseract, "pytesseract"
        except Exception:
            return None, "unavailable"

    @property
    def available(self) -> bool:
        return self.enabled and self.ocr is not None

    # ------------------------------------------------------------- detection
    def _locate_plate(self, vehicle_bgr):
        """Return (x, y, w, h) of the best plate candidate inside a vehicle crop."""
        if vehicle_bgr is None or vehicle_bgr.size == 0:
            return None
        h, w = vehicle_bgr.shape[:2]
        if h < 20 or w < 40:
            return None

        # 1) cascade, if it is available.
        if self.plate_cascade is not None:
            gray = cv2.cvtColor(vehicle_bgr, cv2.COLOR_BGR2GRAY)
            plates = self.plate_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 10))
            if len(plates):
                return max(plates, key=lambda p: p[2] * p[3])

        # 2) shape heuristic over the lower half (plates sit low on a vehicle).
        roi = vehicle_bgr[int(h * 0.45):h, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray, 60, 200)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0.0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch < 8 or cw < 25:
                continue
            aspect = cw / float(ch)
            if not (1.8 <= aspect <= 7.0):
                continue
            rect_fill = cv2.contourArea(c) / float(max(1, cw * ch))
            if rect_fill < 0.35:
                continue
            score = aspect * rect_fill * (cw * ch) ** 0.35
            if score > best_score:
                best_score = score
                best = (x, y + int(h * 0.45), cw, ch)
        return best

    @staticmethod
    def _preprocess(plate_bgr):
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        scale = 3.0
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    # ------------------------------------------------------------------- OCR
    def _ocr_text(self, plate_bgr):
        """Return (text, confidence) or ("", 0.0)."""
        if self.backend == "easyocr":
            try:
                results = self.ocr.readtext(
                    plate_bgr, detail=1,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ")
                if not results:
                    return "", 0.0
                confs = [float(r[2]) for r in results]
                texts = [r[1] for r in results]
                return " ".join(texts), float(sum(confs) / len(confs))
            except Exception:
                return "", 0.0

        if self.backend == "pytesseract":
            try:
                prepped = self._preprocess(plate_bgr)
                data = self.ocr.image_to_data(
                    prepped, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    output_type=self.ocr.Output.DICT)
                words, confs = [], []
                for txt, conf in zip(data.get("text", []), data.get("conf", [])):
                    try:
                        c = float(conf)
                    except Exception:
                        continue
                    if txt and txt.strip() and c >= 0:
                        words.append(txt.strip())
                        confs.append(c)
                if not words:
                    return "", 0.0
                return "".join(words), float(sum(confs) / len(confs)) / 100.0
            except Exception:
                return "", 0.0

        return "", 0.0

    # ---------------------------------------------------------------- public
    def read_plate(self, frame, vehicle_bbox, track_id=-1):
        """Attempt to read a plate from one vehicle. Returns a dict or None."""
        if not self.available or frame is None:
            return None
        try:
            x1, y1, x2, y2 = map(int, vehicle_bbox)
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 < 40 or y2 - y1 < 30:
                return None
            vehicle_bgr = frame[y1:y2, x1:x2]

            plate_box = self._locate_plate(vehicle_bgr)
            if plate_box is None:
                return None
            px, py, pw, ph = plate_box
            plate_bgr = vehicle_bgr[py:py + ph, px:px + pw]
            if plate_bgr.size == 0:
                return None

            raw_text, conf = self._ocr_text(plate_bgr)
            plate = normalize_plate(raw_text)
            # Require both a plausible length and a decent OCR confidence.
            if not (self.min_text_len <= len(plate) <= self.max_text_len) or conf < 0.35:
                return None

            gx1, gy1 = x1 + px, y1 + py
            record = {
                "plate": plate,
                "confidence": round(conf, 3),
                "track_id": int(track_id),
                "plate_bbox": (gx1, gy1, gx1 + pw, gy1 + ph),
                "backend": self.backend,
            }
            self.last_read_by_track[track_id] = plate
            self.reads.insert(0, record)
            self.reads = self.reads[:50]
            if plate not in self._seen_plates:
                self._seen_plates.add(plate)
                self.total_unique += 1
            return record
        except Exception:
            return None

    def draw(self, frame, record):
        if not record:
            return
        x1, y1, x2, y2 = record["plate_bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"PLATE: {record['plate']}", (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def reset_stats(self):
        self.last_read_by_track.clear()
        self.reads = []
        self.total_unique = 0
        self._seen_plates.clear()

    def stats(self) -> dict:
        return {
            "backend": self.backend,
            "available": self.available,
            "plates_read": self.total_unique,
            "recent": self.reads[:5],
        }
