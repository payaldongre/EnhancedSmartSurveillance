from ultralytics import YOLO

class ObjectDetector:
    """Detect hazardous objects using YOLOv8"""

    def __init__(self, model_path="yolov8n.pt", hazardous_labels=None):
        # Load YOLOv8 model
        self.model = YOLO(model_path)

        # Hazardous objects (expandable)
        self.hazardous_labels = hazardous_labels or [
            "knife", "scissors", "hammer", "handgun", "gun", "rifle", "fork"
        ]

    def detect_objects(self, frame):
        """Run object detection and return hazardous detections"""
        results = self.model.predict(frame, verbose=False)
        hazardous_detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id].lower()
                conf = float(box.conf[0])

                if label in self.hazardous_labels and conf > 0.5:
                    hazardous_detections.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": box.xyxy[0].tolist()
                    })
        return hazardous_detections