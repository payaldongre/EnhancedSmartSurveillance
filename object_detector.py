from ultralytics import YOLO


class ObjectDetector:
    """Detect hazardous objects using YOLOv8.

    FIX NOTES (see PATCH_NOTES.md for full explanation):
    - The previous hazardous_labels list included "handgun", "rifle", "shotgun",
      "axe", "hammer", "screwdriver" — none of these exist in the COCO-80 class
      set that a stock yolov8n.pt is trained on, so they could NEVER be detected
      no matter the confidence threshold. They are removed here (dead weight).
    - "bat" never matched COCO's actual class name, which is "baseball bat".
      Fixed to the correct string.
    - Until a custom-trained weapon model replaces this one (see PATCH_NOTES.md
      / label_generator.py), only "knife", "scissors", "baseball bat" and
      "bottle" are real, detectable hazardous classes from stock YOLOv8.
    - Added `imgsz` param (default 320) — the old code ran this model at full
      frame resolution with no imgsz cap, which was actually the single most
      expensive call in the whole per-frame pipeline. Capping it here is one
      of the biggest lag fixes.
    """

    def __init__(self, model_path="yolov8n.pt", conf=0.4, imgsz=320, device="cpu"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device  # FIX: explicit device instead of relying on auto-detect

        # Hazardous objects — only classes that ACTUALLY exist in COCO-80.
        # Once you have a custom-trained weapon model (knife.yaml dataset),
        # point model_path at those weights and this list can expand to
        # gun/rifle/etc. for real.
        self.hazardous_labels = [
            "knife", "scissors", "baseball bat", "bottle"
        ]
        # Separate, tunable threshold for flagging as hazardous vs. just detecting.
        # If a weapon/object disappears when partially occluded (hand/body covering
        # part of it), try lowering this — tradeoff is more false positives.
        self.hazardous_conf_threshold = 0.4

    def detect_objects(self, frame):
        """Run YOLO detection and return all results and hazardous objects."""
        results = self.model(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
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

                if class_name.lower() in self.hazardous_labels and cf > self.hazardous_conf_threshold:
                    hazardous_detections.append(detection)

        return detections, hazardous_detections