"""
Batch YOLO bounding-box annotation tool.

The original label_generator.py only handled a single hardcoded image
("knife_01.jpg") and a single hardcoded class (0 = knife). To actually build
a custom weapon-detection dataset (needed because stock yolov8n.pt has no
"handgun"/"rifle"/"axe" classes — see PATCH_NOTES.md), you need to annotate
many images across multiple weapon classes. This version does that.

Usage:
    python label_generator.py --images ./dataset/images --labels ./dataset/labels

Controls (shown in the window title bar too):
    Left-click + drag : draw a box
    0-9               : select class id (edit CLASS_NAMES below to match knife.yaml)
    s                 : save current image's boxes and move to next image
    u                 : undo last box on current image
    n                 : skip to next image without saving
    q / Esc           : quit

Each image gets one label file with the same stem, one line per box:
    <class_id> <x_center> <y_center> <width> <height>   (all normalized 0-1)
"""

import argparse
import os
import cv2

# Edit this to match the classes in your knife.yaml (order matters — index = class id)
CLASS_NAMES = ["knife", "handgun", "rifle", "axe"]
CLASS_COLORS = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (255, 0, 255)
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class Annotator:
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        os.makedirs(labels_dir, exist_ok=True)

        self.image_files = sorted(
            f for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        )
        if not self.image_files:
            raise SystemExit(f"No images found in {images_dir}")

        self.idx = 0
        self.class_id = 0
        self.boxes = []  # list of (class_id, x1, y1, x2, y2) in pixel coords
        self.drawing = False
        self.x1 = self.y1 = self.x2 = self.y2 = -1
        self.img = None
        self.window = "Weapon Dataset Annotator"

    def load_image(self):
        path = os.path.join(self.images_dir, self.image_files[self.idx])
        self.img = cv2.imread(path)
        self.boxes = []
        title = (f"[{self.idx+1}/{len(self.image_files)}] {self.image_files[self.idx]} "
                  f"| class={CLASS_NAMES[self.class_id]} | "
                  f"s=save n=skip u=undo 0-9=class q=quit")
        cv2.setWindowTitle(self.window, title)

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x1, self.y1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.redraw(preview=(x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.x2, self.y2 = x, y
            if abs(self.x2 - self.x1) > 3 and abs(self.y2 - self.y1) > 3:
                self.boxes.append((self.class_id, self.x1, self.y1, self.x2, self.y2))
            self.redraw()

    def redraw(self, preview=None):
        disp = self.img.copy()
        for cid, bx1, by1, bx2, by2 in self.boxes:
            color = CLASS_COLORS[cid % len(CLASS_COLORS)]
            cv2.rectangle(disp, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(disp, CLASS_NAMES[cid], (bx1, max(15, by1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if self.drawing and preview:
            cv2.rectangle(disp, (self.x1, self.y1), preview,
                          CLASS_COLORS[self.class_id % len(CLASS_COLORS)], 2)
        cv2.imshow(self.window, disp)

    def save_labels(self):
        h, w = self.img.shape[:2]
        stem = os.path.splitext(self.image_files[self.idx])[0]
        label_path = os.path.join(self.labels_dir, stem + ".txt")
        lines = []
        for cid, bx1, by1, bx2, by2 in self.boxes:
            xc = ((bx1 + bx2) / 2) / w
            yc = ((by1 + by2) / 2) / h
            bw = abs(bx2 - bx1) / w
            bh = abs(by2 - by1) / h
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"Saved {len(lines)} box(es) -> {label_path}")

    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.mouse_cb)
        self.load_image()
        self.redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), 27):  # q or Esc
                break
            elif key == ord('s'):
                self.save_labels()
                self.idx += 1
                if self.idx >= len(self.image_files):
                    print("All images processed.")
                    break
                self.load_image()
                self.redraw()
            elif key == ord('n'):
                self.idx += 1
                if self.idx >= len(self.image_files):
                    print("Reached last image.")
                    break
                self.load_image()
                self.redraw()
            elif key == ord('u') and self.boxes:
                self.boxes.pop()
                self.redraw()
            elif ord('0') <= key <= ord('9'):
                cid = key - ord('0')
                if cid < len(CLASS_NAMES):
                    self.class_id = cid
                    self.load_image.__self__  # no-op, keeps boxes
                    title = (f"[{self.idx+1}/{len(self.image_files)}] {self.image_files[self.idx]} "
                              f"| class={CLASS_NAMES[self.class_id]} | "
                              f"s=save n=skip u=undo 0-9=class q=quit")
                    cv2.setWindowTitle(self.window, title)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="dataset/images", help="Folder of images to annotate")
    parser.add_argument("--labels", default="dataset/labels", help="Output folder for YOLO .txt labels")
    args = parser.parse_args()

    Annotator(args.images, args.labels).run()

# """
# Batch YOLO bounding-box annotation tool.

# The original label_generator.py only handled a single hardcoded image
# ("knife_01.jpg") and a single hardcoded class (0 = knife). To actually build
# a custom weapon-detection dataset (needed because stock yolov8n.pt has no
# "handgun"/"rifle"/"axe" classes — see PATCH_NOTES.md), you need to annotate
# many images across multiple weapon classes. This version does that.

# Usage:
#     python label_generator.py --images ./dataset/images --labels ./dataset/labels

# Controls (shown in the window title bar too):
#     Left-click + drag : draw a box
#     0-9               : select class id (edit CLASS_NAMES below to match knife.yaml)
#     s                 : save current image's boxes and move to next image
#     u                 : undo last box on current image
#     n                 : skip to next image without saving
#     q / Esc           : quit

# Each image gets one label file with the same stem, one line per box:
#     <class_id> <x_center> <y_center> <width> <height>   (all normalized 0-1)
# """

# import argparse
# import os
# import cv2

# # Edit this to match the classes in your knife.yaml (order matters — index = class id)
# CLASS_NAMES = ["knife", "handgun", "rifle", "axe", "blunt_weapon"]
# CLASS_COLORS = [
#     (0, 0, 255), (0, 165, 255), (0, 255, 255), (255, 0, 255), (255, 0, 0)
# ]

# IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# class Annotator:
#     def __init__(self, images_dir, labels_dir):
#         self.images_dir = images_dir
#         self.labels_dir = labels_dir
#         os.makedirs(labels_dir, exist_ok=True)

#         self.image_files = sorted(
#             f for f in os.listdir(images_dir)
#             if os.path.splitext(f)[1].lower() in IMG_EXTS
#         )
#         if not self.image_files:
#             raise SystemExit(f"No images found in {images_dir}")

#         self.idx = 0
#         self.class_id = 0
#         self.boxes = []  # list of (class_id, x1, y1, x2, y2) in pixel coords
#         self.drawing = False
#         self.x1 = self.y1 = self.x2 = self.y2 = -1
#         self.img = None
#         self.window = "Weapon Dataset Annotator"

#     def load_image(self):
#         path = os.path.join(self.images_dir, self.image_files[self.idx])
#         self.img = cv2.imread(path)
#         self.boxes = []
#         title = (f"[{self.idx+1}/{len(self.image_files)}] {self.image_files[self.idx]} "
#                   f"| class={CLASS_NAMES[self.class_id]} | "
#                   f"s=save n=skip u=undo 0-9=class q=quit")
#         cv2.setWindowTitle(self.window, title)

#     def mouse_cb(self, event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             self.drawing = True
#             self.x1, self.y1 = x, y
#         elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
#             self.redraw(preview=(x, y))
#         elif event == cv2.EVENT_LBUTTONUP:
#             self.drawing = False
#             self.x2, self.y2 = x, y
#             if abs(self.x2 - self.x1) > 3 and abs(self.y2 - self.y1) > 3:
#                 self.boxes.append((self.class_id, self.x1, self.y1, self.x2, self.y2))
#             self.redraw()

#     def redraw(self, preview=None):
#         disp = self.img.copy()
#         for cid, bx1, by1, bx2, by2 in self.boxes:
#             color = CLASS_COLORS[cid % len(CLASS_COLORS)]
#             cv2.rectangle(disp, (bx1, by1), (bx2, by2), color, 2)
#             cv2.putText(disp, CLASS_NAMES[cid], (bx1, max(15, by1 - 5)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         if self.drawing and preview:
#             cv2.rectangle(disp, (self.x1, self.y1), preview,
#                           CLASS_COLORS[self.class_id % len(CLASS_COLORS)], 2)
#         cv2.imshow(self.window, disp)

#     def save_labels(self):
#         h, w = self.img.shape[:2]
#         stem = os.path.splitext(self.image_files[self.idx])[0]
#         label_path = os.path.join(self.labels_dir, stem + ".txt")
#         lines = []
#         for cid, bx1, by1, bx2, by2 in self.boxes:
#             xc = ((bx1 + bx2) / 2) / w
#             yc = ((by1 + by2) / 2) / h
#             bw = abs(bx2 - bx1) / w
#             bh = abs(by2 - by1) / h
#             lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
#         with open(label_path, "w") as f:
#             f.write("\n".join(lines) + ("\n" if lines else ""))
#         print(f"Saved {len(lines)} box(es) -> {label_path}")

#     def run(self):
#         cv2.namedWindow(self.window)
#         cv2.setMouseCallback(self.window, self.mouse_cb)
#         self.load_image()
#         self.redraw()

#         while True:
#             key = cv2.waitKey(20) & 0xFF

#             if key in (ord('q'), 27):  # q or Esc
#                 break
#             elif key == ord('s'):
#                 self.save_labels()
#                 self.idx += 1
#                 if self.idx >= len(self.image_files):
#                     print("All images processed.")
#                     break
#                 self.load_image()
#                 self.redraw()
#             elif key == ord('n'):
#                 self.idx += 1
#                 if self.idx >= len(self.image_files):
#                     print("Reached last image.")
#                     break
#                 self.load_image()
#                 self.redraw()
#             elif key == ord('u') and self.boxes:
#                 self.boxes.pop()
#                 self.redraw()
#             elif ord('0') <= key <= ord('9'):
#                 cid = key - ord('0')
#                 if cid < len(CLASS_NAMES):
#                     self.class_id = cid
#                     self.load_image.__self__  # no-op, keeps boxes
#                     title = (f"[{self.idx+1}/{len(self.image_files)}] {self.image_files[self.idx]} "
#                               f"| class={CLASS_NAMES[self.class_id]} | "
#                               f"s=save n=skip u=undo 0-9=class q=quit")
#                     cv2.setWindowTitle(self.window, title)

#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--images", default="dataset/images", help="Folder of images to annotate")
#     parser.add_argument("--labels", default="dataset/labels", help="Output folder for YOLO .txt labels")
#     args = parser.parse_args()

#     Annotator(args.images, args.labels).run()