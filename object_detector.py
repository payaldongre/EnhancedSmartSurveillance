import os

from ultralytics import YOLO

# Hazardous objects that ACTUALLY exist in the COCO-80 class set that a stock
# yolov8n/s.pt is trained on. Anything not in COCO (handgun, rifle, shotgun,
# axe, ...) can never be detected by the stock model no matter the threshold.
COCO_HAZARDOUS = ["knife", "scissors", "baseball bat", "bottle"]

# Keywords used to auto-detect weapon classes when a CUSTOM model is loaded.
WEAPON_KEYWORDS = [
    "knife", "gun", "handgun", "pistol", "rifle", "shotgun", "firearm",
    "weapon", "axe", "machete", "sword", "scissors", "bat", "blade", "grenade",
]

# Where a trained weapon model is looked for, in priority order.
CUSTOM_MODEL_CANDIDATES = [
    "models/weapons/best.pt",
    "models/best.pt",
    "best.pt",
    "runs/detect/train/weights/best.pt",
]


def find_weapon_model():
    """Return the path of a custom weapon model if one is present, else None."""
    env_path = os.environ.get("WEAPON_MODEL_PATH", "").strip()
    if env_path:
        return env_path if os.path.exists(env_path) else None
    for candidate in CUSTOM_MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def _is_stock_model(model_path):
    name = os.path.basename(str(model_path)).lower()
    return name.startswith("yolov8") and name.endswith(".pt") and "cust" not in name


class ObjectDetector:
    """Detect hazardous objects using YOLOv8.

    FIX NOTES (see PATCH_NOTES.md for full explanation):
    - The previous hazardous_labels list included "handgun", "rifle", "shotgun",
      "axe", "hammer", "screwdriver" — none of these exist in the COCO-80 class
      set that a stock yolov8n.pt is trained on, so they could NEVER be detected
      no matter the confidence threshold. They are removed here (dead weight).
    - "bat" never matched COCO's actual class name, which is "baseball bat".
      Fixed to the correct string.
    - Added `imgsz` param (default 320) — the old code ran this model at full
      frame resolution with no imgsz cap, which was actually the single most
      expensive call in the whole per-frame pipeline. Capping it here is one
      of the biggest lag fixes.

    CUSTOM WEAPON MODEL SUPPORT:
    - Pass `model_path` pointing at a weapon-trained checkpoint (e.g. the
      Roboflow knife+gun `best.pt`). When a non-stock model is detected, every
      class the model was trained on is treated as hazardous automatically
      (that model exists only to find weapons), so adding a class to the
      training set needs no code change here.
    - Use `find_weapon_model()` to locate the weights; falls back to the stock
      COCO model (knife / scissors / baseball bat / bottle only) when absent.
    """

    def __init__(self, model_path="yolov8n.pt", conf=0.4, imgsz=320, device="cpu",
                 hazardous_labels=None):
        self.model = YOLO(model_path)
        self.model_path = str(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device  # FIX: explicit device instead of relying on auto-detect
        self.is_custom = not _is_stock_model(model_path)
        self.class_names = list(self.model.names.values()) if hasattr(self.model, "names") else []

        if hazardous_labels is not None:
            self.hazardous_labels = [l.lower() for l in hazardous_labels]
        elif self.is_custom:
            # A weapon-specific checkpoint: every class is a hazard.
            self.hazardous_labels = [n.lower() for n in self.class_names]
        else:
            self.hazardous_labels = list(COCO_HAZARDOUS)

        # Separate, tunable threshold for flagging as hazardous vs. just detecting.
        # If a weapon/object disappears when partially occluded (hand/body covering
        # part of it), try lowering this — tradeoff is more false positives.
        self.hazardous_conf_threshold = float(
            os.environ.get("HAZARDOUS_CONF_THRESHOLD", "0.4"))

    def _is_hazardous(self, class_name, conf):
        return (class_name.lower() in self.hazardous_labels
                and conf > self.hazardous_conf_threshold)

    def detect_objects(self, frame):
        """Run YOLO detection and return all results and hazardous objects."""
        results = self.model(frame, conf=self.conf, imgsz=self.imgsz,
                             device=self.device, verbose=False)
        detections = []
        hazardous_detections = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, cf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.model.names[int(cls)]

                detection = {
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(cf),
                    "class": int(cls),
                    "class_name": class_name
                }
                detections.append(detection)

                if self._is_hazardous(class_name, float(cf)):
                    hazardous_detections.append(detection)

        return detections, hazardous_detections

    def stats(self):
        return {
            "model": os.path.basename(self.model_path),
            "custom_weapon_model": self.is_custom,
            "hazardous_classes": self.hazardous_labels,
        }
