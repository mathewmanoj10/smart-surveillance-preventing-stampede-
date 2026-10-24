import cv2
import os
import glob
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

IMAGE_FOLDER = "frames"
LABEL_FOLDER = "annotations"

# Pretrained model proposes person boxes.
# This model uses COCO class 0 = person.
MODEL_NAME = "yolo11n.pt"

CONFIDENCE = 0.10

os.makedirs(LABEL_FOLDER, exist_ok=True)

model = YOLO(MODEL_NAME)

image_files = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
    image_files.extend(
        glob.glob(os.path.join(IMAGE_FOLDER, ext))
    )

image_files = sorted(image_files)

if not image_files:
    raise SystemExit("No images found in the frames folder.")

# ---------------- ANNOTATION TOOL ----------------

boxes = []
selected_box = -1
drawing = False
start_x = start_y = 0
current_image = None
display_image = None
scale = 1.0
offset_x = offset_y = 0


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y
    global boxes, selected_box

    # Convert displayed coordinates to original image coordinates
    ox = int((x - offset_x) / scale)
    oy = int((y - offset_y) / scale)

    h, w = current_image.shape[:2]

    if not (0 <= ox < w and 0 <= oy < h):
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        # Select an existing box if clicked inside it
        selected_box = -1

        for i in range(len(boxes) - 1, -1, -1):
            x1, y1, x2, y2 = boxes[i]

            if x1 <= ox <= x2 and y1 <= oy <= y2:
                selected_box = i
                return

        # Otherwise start drawing a new box
        drawing = True
        start_x, start_y = ox, oy

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False

        x1 = max(0, min(start_x, ox))
        y1 = max(0, min(start_y, oy))
        x2 = min(w - 1, max(start_x, ox))
        y2 = min(h - 1, max(start_y, oy))

        # Ignore tiny boxes
        if x2 - x1 >= 5 and y2 - y1 >= 5:
            boxes.append([x1, y1, x2, y2])


def save_labels(image_path):
    h, w = current_image.shape[:2]

    name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    label_path = os.path.join(
        LABEL_FOLDER, name + ".txt"
    )

    with open(label_path, "w") as f:
        for x1, y1, x2, y2 in boxes:
            # YOLO format: class x_center y_center width height
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            f.write(
                f"0 {xc:.6f} {yc:.6f} "
                f"{bw:.6f} {bh:.6f}\n"
            )

    print(f"Saved {len(boxes)} boxes: {label_path}")


window = "Crowd Annotation Tool"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window, mouse_callback)

for index, image_path in enumerate(image_files):
    current_image = cv2.imread(image_path)

    if current_image is None:
        continue

    # Get initial person proposals
    results = model.predict(
        current_image,
        classes=[0],
        conf=CONFIDENCE,
        imgsz=1280,
        verbose=False
    )

    boxes = []

    if results[0].boxes is not None:
        for b in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, b)
            boxes.append([x1, y1, x2, y2])

    selected_box = -1

    print(f"\nImage {index + 1}/{len(image_files)}")
    print(os.path.basename(image_path))
    print("Review the proposed boxes and correct them.")

    while True:
        h, w = current_image.shape[:2]

        # Fit the image to the screen
        screen_w, screen_h = 1200, 750
        scale = min(screen_w / w, screen_h / h, 1.0)

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(
            current_image, (new_w, new_h)
        )

        canvas = resized.copy()
        offset_x = offset_y = 0

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            p1 = (int(x1 * scale), int(y1 * scale))
            p2 = (int(x2 * scale), int(y2 * scale))

            color = (0, 255, 0)
            if i == selected_box:
                color = (0, 0, 255)

            cv2.rectangle(canvas, p1, p2, color, 2)

        help_text = (
            f"Image {index+1}/{len(image_files)} "
            f"| Boxes: {len(boxes)}"
        )

        cv2.putText(
            canvas, help_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2
        )

        cv2.imshow(window, canvas)
        cv2.resizeWindow(window, new_w, new_h)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("s"):
            save_labels(image_path)
            break

        elif key == ord("n"):
            print("Skipped without saving.")
            break

        elif key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Annotation stopped.")

        elif key == 127 or key == 8 or key == ord("d"):
            if selected_box >= 0:
                boxes.pop(selected_box)
                selected_box = -1

        elif key == ord("c"):
            boxes = []
            selected_box = -1

cv2.destroyAllWindows()
print("\nAnnotation session complete!")