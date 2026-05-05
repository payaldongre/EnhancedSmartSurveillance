import cv2
import os

# Paths
image_path = "knife_01.jpg"   # change this to your image
label_path = "knife_01.txt"   # output YOLO label file

# Global vars
drawing = False
x1, y1, x2, y2 = -1, -1, -1, -1

def draw_bbox(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # start drawing
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = img.copy()
        cv2.rectangle(img_copy, (x1, y1), (x, y), (0, 255, 0), 2)
        cv2.imshow("Draw Bounding Box", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:  # finish drawing
        drawing = False
        x2, y2 = x, y

        # Normalize values for YOLO format
        h, w, _ = img.shape
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        bbox_width = abs(x2 - x1) / w
        bbox_height = abs(y2 - y1) / h

        # Class 0 = knife
        yolo_format = f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"

        with open(label_path, "w") as f:
            f.write(yolo_format + "\n")

        print("YOLO Label Saved:", yolo_format)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Draw Bounding Box", img)

# Load image
img = cv2.imread(image_path)
cv2.imshow("Draw Bounding Box", img)
cv2.setMouseCallback("Draw Bounding Box", draw_bbox)
cv2.waitKey(0)
cv2.destroyAllWindows()